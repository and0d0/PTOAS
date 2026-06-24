# `pto.simt_allreduce_sum` Interface Design

This document defines the updated PTODSL and PTO/VPTO IR interface for SIMT
all-reduce sum used by the RMSNorm SIMT kernel.

## PTODSL Interface

```python
pto.simt_allreduce_sum(
    value,
    scratch=None,
    *,
    threads,
    scale=1,
    thread_offset=0,
) -> ScalarType
```

**Description**: Reduces one scalar value from each participating SIMT workitem
with sum reduction and returns the reduced sum to every participating workitem.

`scratch` is an optional UB temporary buffer. It corresponds to the
device-helper parameter named `red_buf` in the C++ helper:

```cpp
static __simt_callee__ T run(T x, __ubuf__ T *red_buf = nullptr);
```

The PTODSL name is `scratch` because it describes the user-visible role of the
buffer: temporary UB storage used by implementations that cannot complete the
all-reduce only with register or hardware reduction paths.

## Parameter Mapping

The interface mirrors the device-side all-reduce template:

```cpp
template <class Reducer, int threads, int scale = 1, int thread_offset = 0>
struct AscendAllReduce;
```

| C++ helper parameter | PTODSL parameter | PTO/VPTO IR representation | Passing mode |
|----------------------|------------------|----------------------------|--------------|
| `T x` | `value` | `%value` operand | runtime SSA |
| `__ubuf__ T *red_buf` | `scratch` | `%scratch` operand | runtime SSA |
| `Reducer` | fixed sum | `reducer = #pto<reduce_op sum>` | attribute |
| `threads` | `threads` | `threads = ... : i32` | attribute |
| `scale` | `scale` | `scale = ... : i32` | attribute |
| `thread_offset` | `thread_offset` | `thread_offset = ... : i32` | attribute |

Only runtime values are SSA operands. Template-like parameters are attributes.
`threads`, `scale`, and `thread_offset` must be compile-time Python values or
`pto.const_expr` values; they must not be produced by runtime SSA computation.

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `value` | PTO scalar | Yes | Per-workitem scalar value to reduce. RMSNorm passes the lane-local sum of squares. |
| `scratch` | `pto.ptr(T, "ub")` or `None` | No | Optional UB scratch buffer. Corresponds to `.h` `red_buf`. The element type must match `value`. |
| `threads` | compile-time `int` | Yes | Number of participating SIMT workitems. Must be at least `1`. |
| `scale` | compile-time `int` | No | Number of independent reduction slots inside the participating group. Defaults to `1`. |
| `thread_offset` | compile-time `int` | No | Logical offset subtracted from the SIMT x-thread id before grouping and slot calculations. Defaults to `0`. |

## Return Value

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | Same scalar type as `value` | Reduced sum for the current workitem's reduction slot. |

## Constraints

| Rule | Description |
|------|-------------|
| Scalar input | `value` must be a scalar value. Vector values are not accepted. |
| Compile-time parameters | `threads`, `scale`, and `thread_offset` are attributes, not SSA operands. |
| Positive threads | `threads >= 1`. |
| Positive scale | `scale >= 1`. |
| Divisible grouping | `threads % scale == 0`. |
| Non-negative offset | `thread_offset >= 0`. |
| Scratch element type | If `scratch` is provided, its element type must match `value`. |
| Scratch capacity | If `scratch` is provided, it must contain enough elements for the selected implementation and participating thread group. |

## Semantics

For `scale == 1`, all participating workitems contribute one scalar:

```text
result = sum(value from each participating workitem)
```

Every participating workitem receives `result`.

For `scale > 1`, participating workitems are partitioned into independent
reduction slots:

```text
tx = threadIdx.x - thread_offset
slot = tx % scale
result[slot] = sum(value from participating workitems with the same slot)
```

Each workitem receives the reduced value for its own slot.

## PTO/VPTO IR Interface

The PTODSL call lowers to a PTO/VPTO dialect operation:

```mlir
%result = pto.all_reduce %value, %scratch {
  reducer = #pto<reduce_op sum>,
  threads = 128 : i32,
  scale = 1 : i32,
  thread_offset = 0 : i32
} : f32, !pto.ptr<f32, ub> -> f32
```

This is an intermediate PTO/VPTO IR operation. It is not the final lowered
MLIR. Later lowering converts it to the target backend representation.

The IR operand list contains only runtime values:

```text
%value
%scratch
```

The template-like parameters are printed only in the attribute dictionary:

```text
reducer
threads
scale
thread_offset
```

They must not appear as SSA operands such as `%threads`, `%scale`, or
`%thread_offset`.

## RMSNorm Usage

```python
reduce_scratch = pto.alloc_buffer((threads,), pto.f32, scope="ub")

local_sum = 0.0
for r in pto.static_range(0, rounds):
    frag_offset = r * lanes
    for lane in pto.static_range(0, lanes):
        x = scalar.load(x_frag, frag_offset + lane)
        local_sum = local_sum + x * x

sum_sq = pto.simt_allreduce_sum(
    local_sum,
    scratch=reduce_scratch,
    threads=threads,
    scale=1,
    thread_offset=0,
)

rstd = 1.0 / scalar.sqrt(sum_sq / hidden_size + eps)
```
