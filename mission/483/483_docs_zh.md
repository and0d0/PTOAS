# Issue 483 PTODSL RMSNorm 前端接口草案

本文档描述用于表达 RMSNorm SimtVF kernel 的 PTODSL 前端接口草案。目标是在尽量复用现有 PTODSL 概念的前提下，只补齐缺失的通用能力：连续 vector 访存、DSL 层 vector 值、SIMT all-reduce，以及 lane-local 指针/局部存储。

## 1. 扩展 scalar memory access

`scalar.load` 和 `scalar.store` 继续作为用户层面的内存访问 API。它们从“只支持标量读写”扩展为：当用户显式指定连续元素个数时，可以读写一段连续 vector。

`offset` 的单位是元素，不是字节。

#### `scalar.load(ptr, offset=0, *, contiguous=None) -> ScalarType | VecType`

**描述**：从 typed pointer 读取一个标量元素，或者读取 `contiguous` 个连续元素。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr` | `PtrType` | 源 typed pointer。它可以是 kernel 入参中的 typed pointer，例如 `pto.ptr(T, "ub")`；也可以是 `pto.alloc_buffer(...)` 返回的 typed pointer。 |
| `offset` | index-like PTO 标量或 Python 整数 | 从 `ptr` 开始计算的元素偏移。默认值是 `0`。 |
| `contiguous` | `int` 或 `None` | 要读取的连续元素个数。`None` 和 `1` 表示标量 load；`N > 1` 表示 vector load。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `value` | `ScalarType` | 当 `contiguous` 是 `None` 或 `1` 时返回；类型和源 pointer 的元素类型一致。 |
| `value` | `pto.vec(T, N)` | 当 `contiguous=N` 且 `N > 1` 时返回；`T` 是源 pointer 的元素类型。 |

**示例**：

```python
# 现有标量访问仍然有效。
x = scalar.load(x_ub, offset)

# 新增连续 vector 访问：从 UB pointer 连续读取 4 个 f32。
x4 = scalar.load(x_ub, offset, contiguous=4)

# 从 lane-local pointer 连续读取 4 个 f32。
w4 = scalar.load(w_frag, frag_offset, contiguous=4)
```

**约束**：

| 规则 | 说明 |
|------|------|
| 元素单位 | `offset` 是元素偏移，不是字节偏移。 |
| 元素类型 | vector 的元素类型从源 typed pointer 的元素类型推导。 |
| `contiguous` 取值 | 如果提供 `contiguous`，它必须是编译期已知的正整数。 |

---

#### `scalar.store(value, ptr, offset=0, *, contiguous=None) -> None`

**描述**：向 typed pointer 写入一个标量值，或者写入一个 vector 值。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | `ScalarType` 或 `VecType` | 要写入的值。标量值写 1 个元素；vector 值写连续多个元素。 |
| `ptr` | `PtrType` | 目标 typed pointer。它可以是 kernel pointer，也可以是 `pto.alloc_buffer(...)` 返回的 typed pointer。 |
| `offset` | index-like PTO 标量或 Python 整数 | 从 `ptr` 开始计算的元素偏移。默认值是 `0`。 |
| `contiguous` | `int` 或 `None` | 可选的显式宽度校验。如果 `value` 是 vector，并且提供了 `contiguous=N`，则 `N` 必须和 vector lane 数一致。 |

**返回值**：无。

**示例**：

```python
# 现有标量 store 仍然有效。
scalar.store(rstd, rstd_ub, ping)

# 新增 vector store：向 UB pointer 连续写入 4 个 f32。
scalar.store(y4, y_ub, y_offset)

# 可选显式宽度校验。
scalar.store(y4, y_ub, y_offset, contiguous=4)

# 把 vector 写入 lane-local pointer。
scalar.store(x4, x_frag, frag_offset)
```

**约束**：

| 规则 | 说明 |
|------|------|
| 标量 store | 如果 `value` 是标量，则只写入 1 个元素。 |
| vector store | 如果 `value` 是 `pto.vec(T, N)`，则连续写入 `N` 个元素。 |
| 类型匹配 | 写入的标量类型或 vector 元素类型必须和目标 pointer 的元素类型一致。 |
| 显式宽度 | 如果对 vector store 提供 `contiguous=N`，则 `N` 必须和 vector lane 数一致。 |
| 元素单位 | `offset` 是元素偏移，不是字节偏移。 |

**RMSNorm 使用模式**：

```python
x4 = scalar.load(x_ub, x_offset, contiguous=4)
scalar.store(x4, x_frag, r * lanes)

for lane in pto.static_range(0, lanes):
    x = scalar.load(x_frag, r * lanes + lane)
    w = scalar.load(w_frag, r * lanes + lane)
    y = x * rstd * w
    scalar.store(y, y_ub, y_offset + lane)
```

后续下标语法糖应支持把上面的标量 load 写成
`x_frag[r * lanes + lane]` 和 `w_frag[r * lanes + lane]`。

## 2. DSL vector 类型

PTODSL 需要一个前端 vector 抽象，用来表达固定宽度的 builtin vector 值，例如 `vector<4xf32>`。这个抽象用于承接连续内存 load 的返回值，也就是之前提到的 `load_contiguous(ptr, offset, lanes=4)` 和 `fragment_load_contiguous(fragment, offset, lanes=4)` 的返回类型。

这个 vector 抽象和 PTO 硬件 vector register 类型不同，例如 `!pto.vreg<NxT>`。

#### `pto.vec(dtype, lanes) -> VecType`

**描述**：构造一个 DSL builtin vector 类型，包含 `lanes` 个 `dtype` 元素。连续 load 可以根据 pointer 元素类型和 `contiguous` 自动推导这个类型；显式形式可用于类型标注、校验和 vector 值构造。

```python
f32x4 = pto.vec(pto.f32, 4)
```

这表示：

```text
DSL 类型：pto.vec(pto.f32, 4)
lowering 后类型：vector<4xf32>
```

在实际 RMSNorm kernel 中，通常不需要显式写这个类型。连续 load 产生的值会隐式得到该类型。例如：

```python
x4 = scalar.load(x_ub, x_offset, contiguous=4)
```

如果 `x_ub` 指向 `f32`，那么：

```text
x4 是一个 VecValue
x4 的 DSL 类型是 pto.vec(pto.f32, 4)
x4 lowering 后是 vector<4xf32>
```

lane-local storage 也是一样：

```python
w4 = scalar.load(w_frag, frag_offset, contiguous=4)
```

如果 `w_frag` 指向 `f32`，那么 `w4` 也具有同样的 DSL 类型 `pto.vec(pto.f32, 4)`，并 lower 成 `vector<4xf32>`。RMSNorm x128 kernel 当前使用的 float2 模式也遵循同样规则：`contiguous=2` 产生 `pto.vec(pto.f32, 2)`，并 lower 成 `vector<2xf32>`。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `dtype` | PTODSL 标量 dtype | vector 元素类型。RMSNorm 使用 `pto.f32`。 |
| `lanes` | `int` | vector 元素个数。RMSNorm 使用它表示 float2 或 float4 连续访问。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `vec_type` | `VecType` | builtin vector 类型对象，例如 `pto.vec(pto.f32, 4)`。 |

---

#### `pto.vec(dtype, lanes, *, init=value) -> VecValue`

**描述**：构造一个类型为 `pto.vec(dtype, lanes)` 的 vector 值。当 `init` 是标量时，把这个标量广播到所有 vector lane；当 `init` 已经是兼容 vector 值时，对它执行类型校验。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `dtype` | PTODSL 标量 dtype | vector 元素类型。 |
| `lanes` | `int` | vector lane 数。 |
| `init` | 标量值或兼容 vector 值 | 初始化值。如果是标量，则广播到所有 lane。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `result` | `VecValue` | 运行时 vector 值，DSL 类型为 `pto.vec(dtype, lanes)`。 |

**示例**：

```python
# 把一个标量 rstd 广播成 [rstd, rstd, rstd, rstd]。
rstd4 = pto.vec(pto.f32, 4, init=rstd)

# 逐元素 vector 计算。
y4 = x4 * rstd4 * w4
```

**约束**：

| 规则 | 说明 |
|------|------|
| 类型形式 | `pto.vec(dtype, lanes)` 构造 builtin vector 类型。 |
| 值形式 | `pto.vec(dtype, lanes, init=value)` 构造 `VecValue`。 |
| 广播 | 标量 `init` 会广播到 vector 的每个 lane。 |
| 算术 | 两个兼容 `VecValue` 对象上的 Python 算术是逐元素算术。 |
| 类型区分 | `pto.vec(dtype, lanes)` 是 builtin vector 类型，不等同于 `pto.vreg_type(lanes, dtype)`。 |

### Vector 算术运算符重载

RMSNorm 输出阶段可以对 `VecValue` 操作数执行逐元素乘法，例如 float2 风格的 `x_vec * rstd_vec * w_vec`。初始运算符重载范围刻意收窄：要求支持兼容 vector 操作数上的 Python `*`。

#### `lhs * rhs -> VecValue`

**描述**：当两个操作数都是兼容的 `VecValue`，或者其中一个操作数可以被显式广播/转换成兼容 vector 值时，执行逐元素乘法。

**示例**：

```python
x4 = scalar.load(x_ub, x_offset, contiguous=4)
w4 = scalar.load(w_frag, frag_offset, contiguous=4)
rstd4 = pto.vec(pto.f32, 4, init=rstd)

sq4 = x4 * x4
y4 = x4 * rstd4 * w4
```

**语义**：

```text
(x4 * w4)[i] = x4[i] * w4[i]
```

**Lowering 目标**：builtin vector 类型上的逐元素乘法，例如 `vector<4xf32>` 上的 `arith.mulf`。

除显式 `pto.vec(..., init=...)` 之外的 scalar-vector 隐式广播，不属于 RMSNorm 初始需求。

## 3. SIMT all-reduce sum

RMSNorm 还需要把每个参与的 SIMT workitem 各自产生的一个标量做求和，并把同一个求和结果返回给每个 workitem。每个 workitem 会先在本地形成这个标量，再调用 `pto.simt_allreduce_sum`。

#### `pto.simt_allreduce_sum(value, scratch=None, *, threads, scale=1, thread_offset=0) -> ScalarType`

**描述**：对每个参与的 SIMT workitem 输入的一个标量做求和，并把总和返回给每个参与的 workitem。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `value` | PTO 标量 | 当前 workitem 贡献给 all-reduce 的标量值。RMSNorm 中通常是当前 lane 算出的局部平方和。 |
| `scratch` | typed UB pointer 或 `None` | 可选临时空间。某些实现方式可能需要用它暂存中间值。它是 C++ helper 中 `red_buf` 参数在 PTODSL 层的命名。 |
| `threads` | `int` | 参与规约的 SIMT workitem 数量。RMSNorm 常见值是 `128`。 |
| `scale` | `int` | group 内携带的独立 reduction lane 数。默认值是 `1`；更大时，逻辑 lane id 满足相同 `lane % scale` 的 workitem 归约到同一组。 |
| `thread_offset` | `int` | 逻辑 workitem 偏移。默认值是 `0`。 |

PTODSL 接口对应下面的 helper 形态：

```cpp
template <class Reducer, int threads, int scale = 1, int thread_offset = 0>
struct AscendAllReduce {
  template <typename T>
  static __simt_callee__ T run(T x, __ubuf__ T *red_buf = nullptr);
};
```

`pto.simt_allreduce_sum` 固定 `Reducer` 为 sum。运行时数据作为 SSA operand 传入；类似模板参数的值作为 attribute 表达：

| C++ helper 参数 | PTODSL 参数 | PTO/VPTO IR 表达 | 传递方式 |
|-----------------|-------------|------------------|----------|
| `T x` | `value` | `%value` operand | runtime SSA |
| `__ubuf__ T *red_buf` | `scratch` | `%scratch` operand | runtime SSA |
| `Reducer` | 固定为 sum | `reducer = #pto<reduce_op sum>` | attribute |
| `threads` | `threads` | `threads = ... : i32` | attribute |
| `scale` | `scale` | `scale = ... : i32` | attribute |
| `thread_offset` | `thread_offset` | `thread_offset = ... : i32` | attribute |

PTO/VPTO IR 是后续还会继续 lower 的中间 IR，不是最终 lowered MLIR。预期 op 形态是：

```mlir
%result = pto.all_reduce %value, %scratch {
  reducer = #pto<reduce_op sum>,
  threads = 128 : i32,
  scale = 1 : i32,
  thread_offset = 0 : i32
} : f32, !pto.ptr<f32, ub> -> f32
```

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `result` | `ScalarType` | 所有参与 workitem 的 `value` 求和结果。每个参与 workitem 都拿到同一个结果。 |

**示例**：

```python
local_sum = 0.0
for lane in pto.static_range(0, lanes):
    x = scalar.load(x_frag, frag_offset + lane)
    local_sum = local_sum + x * x

sum_sq = pto.simt_allreduce_sum(
    local_sum,
    scratch=reduce_scratch,
    threads=128,
    scale=1,
    thread_offset=0,
)
```

**语义**：

```text
对于每个参与的 workitem：
  result = sum(所有参与 workitem 的 value)
```

**约束**：

| 规则 | 说明 |
|------|------|
| 标量输入 | `value` 是每个 workitem 的一个标量，不是 vector。 |
| 参与线程数 | `threads` 必须和当前 SIMT launch/body 的规约范围匹配。 |
| 编译期参数 | `threads`、`scale` 和 `thread_offset` 是编译期 Python 整数或 `pto.const_expr` 值。它们在 IR 中表达为 attribute，不是 SSA value。 |
| 规约形状 | `threads >= 1`、`scale >= 1`，且 `threads % scale == 0`。 |
| 临时空间 | 如果传入 `scratch`，它必须是 UB pointer，元素类型需要和 `value` 一致，并且必须有足够空间支撑对应的实现方式和 `threads` 数量。 |

## 4. Lane-local pointer / alloc_buffer

RMSNorm 需要每个 workitem 私有的局部数组，例如 `x_frag[32]` 和 `w_frag[32]`。这些数组属于当前 SIMT workitem 自己，不在不同 workitem 之间共享。接口上，`pto.alloc_buffer(...)` 返回 typed pointer，但它背后的分配语义仍然可以表示“每个 workitem 私有的一段局部存储”。

#### `pto.alloc_buffer(shape, dtype, *, scope, persistent=False) -> PtrType`

**描述**：在指定作用域分配一块线性可寻址存储，并返回指向这块存储的 typed pointer。对于 RMSNorm，`scope="ub"` 用于 UB scratch，例如 `x_ub` / `y_ub`；`scope="local"` 用于 lane-local pointer，例如 `x_frag` / `w_frag`。它替代之前提出的 lane-local 存储接口名 `alloc_fragment`。

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `shape` | `tuple[int, ...]` 或 `list[int]` | 分配形状。例如对 `scope="local"`，`(32,)` 表示每个 workitem 拥有 32 个元素。 |
| `dtype` | PTODSL 标量 dtype | 元素类型，例如 `pto.f32`。 |
| `scope` | `"ub"` 或 `"local"` | 存储作用域。`"ub"` 表示 UB scratch storage；`"local"` 表示每个 SIMT workitem 私有的 storage instance。 |
| `persistent` | `bool` | local storage 是否需要跨多次 SIMT launch 保留内容。默认值是 `False`。 |

**返回值**：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `ptr` | `PtrType` | 指向已分配存储的 typed pointer。元素类型来自 `dtype`；地址空间来自 `scope`。 |

**示例**：

```python
# UB scratch pointer，用来放一个或两个 token row。
x_ub = pto.alloc_buffer((2, 4096), pto.f32, scope="ub")
y_ub = pto.alloc_buffer((2, 4096), pto.f32, scope="ub")

# 当前 token 的 x 临时值。每个 SIMT workitem 私有。
x_frag = pto.alloc_buffer((32,), pto.f32, scope="local")

# 权重 w 的临时值。每个 workitem 私有，并且希望跨 token 循环保留。
w_frag = pto.alloc_buffer((32,), pto.f32, scope="local", persistent=True)

# 通过 scalar.load/store 做 vector 访问。
scalar.store(x4, x_frag, r * 4)
x4 = scalar.load(x_frag, r * 4, contiguous=4)
```

**可选下标语法糖**：

```python
# 标量元素访问语法糖。
x_frag[i] = value
value = x_frag[i]
weight = w_frag[i]

# 连续 vector 访问语法糖。
x_frag[i : i + 4] = x4
x4 = x_frag[i : i + 4]
```

这些下标形式本质上是 `scalar.load` 和 `scalar.store` 的语法糖：

```python
value = x_frag[i]                  # scalar.load(x_frag, i)
weight = w_frag[i]                 # scalar.load(w_frag, i)
x_frag[i] = value                  # scalar.store(value, x_frag, i)
x4 = x_frag[i : i + 4]             # scalar.load(x_frag, i, contiguous=4)
x_frag[i : i + 4] = x4             # scalar.store(x4, x_frag, i, contiguous=4)
```

**约束**：

| 规则 | 说明 |
|------|------|
| 作用域所有权 | `scope="ub"` 返回的 pointer 指向 UB scratch storage；`scope="local"` 返回的 pointer 指向当前 SIMT workitem 私有的 storage instance。 |
| 形状 | `shape` 必须在编译期已知。 |
| persistent storage | `persistent=True` 主要用于 local storage，例如 RMSNorm 权重，希望一次读入后跨多次 SIMT launch 复用。 |
| 访问方式 | 返回的 pointer 通过 `scalar.load/store` 或等价下标语法糖访问。 |

**前端与 lowering 计划**：

| 场景 | 当前 / 计划 lowering | 原因 |
|------|----------------------|------|
| `scope="ub", persistent=False` | lower 成现有 `alloc_tile` 加 `tile_buf_addr` / pointer extraction。 | 复用当前 PTO IR，避免在 IR 层新增 allocation op。 |
| `scope="local", persistent=False` | PTODSL 前端先生成 local carrier，例如 `memref.alloca() : memref<NxT>`，并把它包装成带 `alloc_buffer_scope = "local"` 元数据的 `AddressValue`。后续 load/store lowering 再把访问转换成 LLVM pointer-backed local load/store。 | 让每个 SIMT workitem 都有自己的 `x_frag[32]`，同时保持用户层 API 仍然是 pointer-shaped。 |
| `scope="local", persistent=True` | PTODSL 前端接受该标志，并在 local carrier 上标记 `pto.persistent`；后续 lowering 负责把这个标记转移到最终的 `llvm.alloca`。keep/resume 或等价状态保存恢复 pass 仍是后续工作。 | 让 `w_frag` 可以一次加载后跨 token loop 复用。 |

初始 `scope="local"` 前端改动刻意不修改 `scalar.load` 或 `scalar.store`。
后续 load/store 扩展需要识别 local `AddressValue` 元数据，并通过 local carrier
lower 标量或连续 vector 访问，而不是把它当成普通 PTO address-space pointer。
对于 persistent fragment，allocation carrier 上已经有标记；驻留 keep/resume 生成仍由后续 pass 完成。

**Tile 访问说明**：

如果实现复用 `alloc_tile`，那么 tile value 仍然需要先转换成 typed pointer，才能用于 pointer 风格的 `scalar.load/store`。显式形式是：

```python
x = scalar.load(tile.as_ptr(), offset)
```

也可以保留更友好的标量 tile 元素访问形式：

```python
x = scalar.load(tile[row, col])
```

连续 vector 访问仍建议使用 `alloc_buffer(...)` 或 `tile.as_ptr()` 产生的 typed pointer，因为它需要线性元素偏移。

## 5. RMSNorm 数据流示例

下面的草图展示这些接口如何组合起来表达一个 RMSNorm SIMT body。在这个示例中，`x_ub` 和 `y_ub` 是通过 `pto.alloc_buffer(..., scope="ub")` 分配的 UB pointer；`x_frag` 和 `w_frag` 是通过 `scope="local"` 分配的 lane-local pointer。

```python
@pto.simt
def rmsnorm_lane_body(
    x_ub,
    y_ub,
    reduce_scratch: pto.ptr(pto.f32, "ub"),
    x_frag,
    w_frag,
    eps: pto.f32,
    *,
    threads: pto.const_expr,
    rounds: pto.const_expr,
    lanes: pto.const_expr = 4,
    hidden_size: pto.const_expr = 4096,
):
    tx = pto.get_tid_x()
    local_sum = 0.0

    for r in pto.static_range(0, rounds):
        offset = r * threads * lanes + tx * lanes
        x4 = scalar.load(x_ub, offset, contiguous=lanes)
        scalar.store(x4, x_frag, r * lanes)

        for lane in pto.static_range(0, lanes):
            x = scalar.load(x_frag, r * lanes + lane)
            local_sum = local_sum + x * x

    sum_sq = pto.simt_allreduce_sum(
        local_sum,
        scratch=reduce_scratch,
        threads=threads,
        scale=1,
        thread_offset=0,
    )

    rstd = 1.0 / scalar.sqrt(sum_sq / hidden_size + eps)

    for r in pto.static_range(0, rounds):
        offset = r * threads * lanes + tx * lanes
        for lane in pto.static_range(0, lanes):
            x = scalar.load(x_frag, r * lanes + lane)
            w = scalar.load(w_frag, r * lanes + lane)
            y = x * rstd * w
            scalar.store(y, y_ub, offset + lane)
```

这里显式写出的 `x_frag` 和 `w_frag` 标量 load 是当前草案的规范形式。后续语法糖层应支持等价的
`x_frag[i]` 和 `w_frag[i]` 标量元素读取形式。

## 6. 接口汇总

| 接口 | 用途 |
|------|------|
| `scalar.load(ptr, offset=0, *, contiguous=None)` | 从 typed pointer 读取一个标量或一段连续 vector。 |
| `scalar.store(value, ptr, offset=0, *, contiguous=None)` | 向 typed pointer 写入一个标量或一段连续 vector。 |
| `pto.vec(dtype, lanes)` | 定义 DSL builtin vector 类型。 |
| `pto.vec(dtype, lanes, *, init=None)` | 构造 vector 值，包括把标量广播成 vector。 |
| `pto.simt_allreduce_sum(value, scratch=None, *, threads, scale=1, thread_offset=0)` | 把每个参与 SIMT workitem 的一个标量求和，并把结果广播给所有参与 workitem。 |
| `pto.alloc_buffer(shape, dtype, *, scope, persistent=False)` | 分配线性 UB 或 lane-local storage，并返回 typed pointer。 |
