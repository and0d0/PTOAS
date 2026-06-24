# Issue 483 PTODSL RMSNorm Frontend Interface Draft

This document describes the proposed PTODSL frontend surface for expressing the
RMSNorm SimtVF kernel. The focus is to reuse existing PTODSL concepts where
possible and add only the missing general-purpose pieces: contiguous vector
memory access, DSL vector values, SIMT all-reduce, and lane-local pointers.

## 1. Extended scalar memory access

`scalar.load` and `scalar.store` remain the user-facing memory access APIs. They
are extended from scalar-only access to support contiguous vector access when the
user explicitly requests multiple adjacent elements.

Offsets are counted in elements, not bytes.

#### `scalar.load(ptr, offset=0, *, contiguous=None) -> ScalarType | VecType`

**Description**: Loads one scalar element or `contiguous` adjacent elements from
a typed pointer.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Source typed pointer. It may be a kernel pointer such as `pto.ptr(T, "ub")`, or the typed pointer returned by `pto.alloc_buffer(...)`. |
| `offset` | index-like PTO scalar or Python integer | Element displacement from `ptr`. Defaults to `0`. |
| `contiguous` | `int` or `None` | Number of adjacent elements to load. `None` and `1` mean scalar load. `N > 1` means vector load. |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `value` | `ScalarType` | Returned when `contiguous` is `None` or `1`; type matches the source element type. |
| `value` | `pto.vec(T, N)` | Returned when `contiguous=N` and `N > 1`; `T` is the source element type. |

**Examples**:

```python
# Existing scalar access remains valid.
x = scalar.load(x_ub, offset)

# New contiguous vector access: read 4 adjacent f32 values from a UB pointer.
x4 = scalar.load(x_ub, offset, contiguous=4)

# New contiguous vector access from a lane-local pointer.
w4 = scalar.load(w_frag, frag_offset, contiguous=4)
```

**Constraints**:

| Rule | Description |
|------|-------------|
| Element unit | `offset` is an element offset, not a byte offset. |
| Element type | The vector element type is inferred from the source typed pointer element type. |
| `contiguous` value | `contiguous` must be a positive compile-time integer when provided. |

---

#### `scalar.store(value, ptr, offset=0, *, contiguous=None) -> None`

**Description**: Stores one scalar value or a vector value to a typed pointer.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `ScalarType` or `VecType` | Value to store. Scalar values store one element. Vector values store adjacent elements. |
| `ptr` | `PtrType` | Destination typed pointer. It may be a kernel pointer or the typed pointer returned by `pto.alloc_buffer(...)`. |
| `offset` | index-like PTO scalar or Python integer | Element displacement from `ptr`. Defaults to `0`. |
| `contiguous` | `int` or `None` | Optional explicit width check for vector stores. If provided for a vector value, it must match the vector lane count. |

**Returns**: None.

**Examples**:

```python
# Existing scalar store remains valid.
scalar.store(rstd, rstd_ub, ping)

# New vector store: write 4 adjacent f32 values to a UB pointer.
scalar.store(y4, y_ub, y_offset)

# Optional explicit width check.
scalar.store(y4, y_ub, y_offset, contiguous=4)

# Store a vector into a lane-local pointer.
scalar.store(x4, x_frag, frag_offset)
```

**Constraints**:

| Rule | Description |
|------|-------------|
| Scalar store | If `value` is scalar, exactly one element is stored. |
| Vector store | If `value` is `pto.vec(T, N)`, `N` adjacent elements are stored. |
| Type match | The stored scalar type or vector element type must match the destination element type. |
| Explicit width | If `contiguous=N` is provided with a vector value, `N` must match the vector lane count. |
| Element unit | `offset` is an element offset, not a byte offset. |

**RMSNorm usage pattern**:

```python
x4 = scalar.load(x_ub, x_offset, contiguous=4)
scalar.store(x4, x_frag, r * lanes)

for lane in pto.static_range(0, lanes):
    x = scalar.load(x_frag, r * lanes + lane)
    w = scalar.load(w_frag, r * lanes + lane)
    y = x * rstd * w
    scalar.store(y, y_ub, y_offset + lane)
```

Future indexing sugar should allow the scalar loads above to be written as
`x_frag[r * lanes + lane]` and `w_frag[r * lanes + lane]`.

## 2. DSL vector type

PTODSL needs a frontend vector abstraction for fixed-width builtin vector values
such as `vector<4xf32>`. This abstraction is used to type the values returned by
contiguous memory loads, including the cases that were previously described as
`load_contiguous(ptr, offset, lanes=4)` and
`fragment_load_contiguous(fragment, offset, lanes=4)`.

This vector abstraction is separate from PTO hardware vector-register values
such as `!pto.vreg<NxT>`.

#### `pto.vec(dtype, lanes) -> VecType`

**Description**: Constructs a DSL builtin vector type with `lanes` elements of
`dtype`. Contiguous loads infer this type automatically from the pointer element
type and `contiguous`, but the explicit form is available for annotations,
validation, and value construction.

```python
f32x4 = pto.vec(pto.f32, 4)
```

This means:

```text
DSL type: pto.vec(pto.f32, 4)
Lowering type: vector<4xf32>
```

In the actual RMSNorm kernel, this explicit type spelling is usually unnecessary. Values produced from contiguous loads get the type implicitly. For example:

```python
x4 = scalar.load(x_ub, x_offset, contiguous=4)
```

If `x_ub` points to `f32`, then:

```text
x4 is a VecValue
x4 has DSL type pto.vec(pto.f32, 4)
x4 lowers as vector<4xf32>
```

The same applies to lane-local storage:

```python
w4 = scalar.load(w_frag, frag_offset, contiguous=4)
```

If `w_frag` points to `f32`, then `w4` has the same DSL type
`pto.vec(pto.f32, 4)` and lowers as `vector<4xf32>`. The same rule applies to
the float2 pattern used by the RMSNorm x128 kernel: `contiguous=2` produces
`pto.vec(pto.f32, 2)` and lowers as `vector<2xf32>`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | PTODSL scalar dtype | Vector element type. RMSNorm uses `pto.f32`. |
| `lanes` | `int` | Number of vector elements. RMSNorm uses this for float2 or float4 contiguous access. |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec_type` | `VecType` | Builtin vector type object, for example `pto.vec(pto.f32, 4)`. |

---

#### `pto.vec(dtype, lanes, *, init=value) -> VecValue`

**Description**: Constructs a vector value of type `pto.vec(dtype, lanes)`. When
`init` is a scalar, the scalar is broadcast to every vector lane. When `init` is
already a compatible vector value, it is checked against the requested vector
type.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | PTODSL scalar dtype | Vector element type. |
| `lanes` | `int` | Number of vector lanes. |
| `init` | scalar value or compatible vector value | Initializer. Scalar input is broadcast to every lane. |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VecValue` | Runtime vector value with DSL type `pto.vec(dtype, lanes)`. |

**Example**:

```python
# Broadcast one scalar rstd to [rstd, rstd, rstd, rstd].
rstd4 = pto.vec(pto.f32, 4, init=rstd)

# Elementwise vector arithmetic.
y4 = x4 * rstd4 * w4
```

**Constraints**:

| Rule | Description |
|------|-------------|
| Type form | `pto.vec(dtype, lanes)` constructs a builtin vector type. |
| Value form | `pto.vec(dtype, lanes, init=value)` constructs a `VecValue`. |
| Broadcast | Scalar `init` is broadcast to every vector lane. |
| Arithmetic | Python arithmetic on two compatible `VecValue` objects is elementwise. |
| Type separation | `pto.vec(dtype, lanes)` is a builtin vector type and is not the same as `pto.vreg_type(lanes, dtype)`. |

### Vector arithmetic operator overloading

RMSNorm output can use elementwise multiplication on `VecValue` operands, for
example `float2`-style `x_vec * rstd_vec * w_vec`. The initial
operator-overloading scope is intentionally narrow: Python `*` is required for
compatible vector operands.

#### `lhs * rhs -> VecValue`

**Description**: Performs elementwise multiplication when both operands are
compatible `VecValue` objects, or when one operand can be broadcast/converted to
a compatible vector value.

**Example**:

```python
x4 = scalar.load(x_ub, x_offset, contiguous=4)
w4 = scalar.load(w_frag, frag_offset, contiguous=4)
rstd4 = pto.vec(pto.f32, 4, init=rstd)

sq4 = x4 * x4
y4 = x4 * rstd4 * w4
```

**Semantics**:

```text
(x4 * w4)[i] = x4[i] * w4[i]
```

**Lowering target**: elementwise multiply on builtin vector types, for example
`arith.mulf` on `vector<4xf32>`.

Mixed scalar-vector implicit broadcasting beyond the explicit
`pto.vec(..., init=...)` form is not part of the initial RMSNorm requirement.

## 3. SIMT all-reduce sum

RMSNorm also needs to reduce one scalar from every participating SIMT workitem
and return the same total to every workitem. Each workitem forms that scalar
locally before calling `pto.simt_allreduce_sum`.

#### `pto.simt_allreduce_sum(value, *, threads, scale=1, thread_offset=0, scratch=None, scratch_offset=0) -> ScalarType`

**Description**: Sums one scalar value from each participating SIMT workitem and
returns the reduced sum to every participating workitem.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | PTO scalar | Per-workitem scalar value to reduce. RMSNorm uses the lane-local sum of squares. |
| `threads` | `int` | Number of participating SIMT workitems. RMSNorm commonly uses `128`. |
| `scale` | `int` | Optional scale factor matching all-reduce-style interfaces. Defaults to `1`. |
| `thread_offset` | `int` | Logical workitem offset. Defaults to `0`. |
| `scratch` | typed UB pointer or `None` | Optional scratch storage for implementations that need temporary memory. |
| `scratch_offset` | index-like PTO scalar or Python integer | Element offset into `scratch`. Defaults to `0`. |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `ScalarType` | Sum of `value` across all participating workitems. Every participating workitem receives the same result. |

**Example**:

```python
local_sum = 0.0
for lane in pto.static_range(0, lanes):
    x = scalar.load(x_frag, frag_offset + lane)
    local_sum = local_sum + x * x

sum_sq = pto.simt_allreduce_sum(
    local_sum,
    threads=128,
    scratch=reduce_scratch,
)
```

**Semantics**:

```text
for each participating workitem lane:
  result = sum(value from all participating workitems)
```

**Constraints**:

| Rule | Description |
|------|-------------|
| Scalar input | `value` is a scalar per workitem, not a vector. |
| Participating threads | `threads` must match the SIMT launch/body contract for the reduction. |
| Scratch | If `scratch` is provided, it must have enough elements for the selected implementation and `threads`. |

## 4. Lane-local pointer

RMSNorm needs per-workitem local arrays such as `x_frag[32]` and `w_frag[32]`.
These arrays are private to each SIMT workitem. They are not shared across
workitems.

#### `pto.alloc_buffer(shape, dtype, *, scope, persistent=False) -> PtrType`

**Description**: Allocates linear addressable storage in the requested scope and returns a typed pointer to it. For RMSNorm, `scope="ub"` is used for UB scratch such as `x_ub` / `y_ub`, and `scope="local"` is used for lane-local pointers such as `x_frag` / `w_frag`. This replaces the previous proposed `alloc_fragment` name for lane-local storage.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` or `list[int]` | Allocation shape. For example, `(32,)` means each workitem owns 32 elements for `scope="local"`. |
| `dtype` | PTODSL scalar dtype | Element type, for example `pto.f32`. |
| `scope` | `"ub"` or `"local"` | Storage scope. `"ub"` means shared UB scratch storage; `"local"` means each SIMT workitem owns a private storage instance. |
| `persistent` | `bool` | Whether local storage should be preserved across multiple SIMT launches in the same kernel. Defaults to `False`. |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `ptr` | `PtrType` | Typed pointer to the allocated storage. The element type is `dtype`; the address space is derived from `scope`. |

**Examples**:

```python
# UB scratch pointers for one or two token rows.
x_ub = pto.alloc_buffer((2, 4096), pto.f32, scope="ub")
y_ub = pto.alloc_buffer((2, 4096), pto.f32, scope="ub")

# Current token values. Private to each SIMT workitem.
x_frag = pto.alloc_buffer((32,), pto.f32, scope="local")

# Weight values. Private to each workitem and intended to persist across token loops.
w_frag = pto.alloc_buffer((32,), pto.f32, scope="local", persistent=True)

# Vector access through scalar.load/store.
scalar.store(x4, x_frag, r * 4)
x4 = scalar.load(x_frag, r * 4, contiguous=4)
```

**Optional indexing sugar**:

```python
# Scalar element access sugar.
x_frag[i] = value
value = x_frag[i]
weight = w_frag[i]

# Slice access sugar for contiguous vector access.
x_frag[i : i + 4] = x4
x4 = x_frag[i : i + 4]
```

The indexing forms are syntax sugar over `scalar.load` and `scalar.store`:

```python
value = x_frag[i]                  # scalar.load(x_frag, i)
weight = w_frag[i]                 # scalar.load(w_frag, i)
x_frag[i] = value                  # scalar.store(value, x_frag, i)
x4 = x_frag[i : i + 4]             # scalar.load(x_frag, i, contiguous=4)
x_frag[i : i + 4] = x4             # scalar.store(x4, x_frag, i, contiguous=4)
```

**Constraints**:

| Rule | Description |
|------|-------------|
| Scope ownership | With `scope="ub"`, the pointer addresses UB scratch storage. With `scope="local"`, the pointer addresses the current SIMT workitem's private storage instance. |
| Shape | Shape must be compile-time known. |
| Persistent storage | `persistent=True` is intended for local values such as RMSNorm weights that should be loaded once and reused across multiple SIMT launches. |
| Access | Returned pointers are accessed through `scalar.load/store` or equivalent indexing sugar. |

**Lowering plan**:

| Case | Suggested lowering | Reason |
|------|--------------------|--------|
| `scope="ub", persistent=False` | Lower to existing `alloc_tile` plus `tile_buf_addr` / pointer extraction. | Reuses current PTO IR and avoids adding a new allocation op in the IR layer. |
| `scope="local", persistent=False` | Lower to lane-local storage, ultimately `llvm.alloca` or an equivalent local allocation. | Gives each SIMT workitem private storage such as `x_frag[32]`. |
| `scope="local", persistent=True` | Lower to the persistent-fragment path for keep/resume or equivalent state preservation. | Enables `w_frag` to be loaded once and reused across token loops. |

**Tile access note**:

If the implementation reuses `alloc_tile`, tile values still need to become
typed pointers before pointer-style `scalar.load/store` can use them. The
explicit form is:

```python
x = scalar.load(tile.as_ptr(), offset)
```

A friendlier scalar-only tile element form can also remain supported:

```python
x = scalar.load(tile[row, col])
```

Contiguous vector access should still use a typed pointer produced by
`alloc_buffer(...)` or `tile.as_ptr()`, because it needs a linear element offset.

## 5. RMSNorm dataflow example

The following sketch shows how these interfaces work together for one RMSNorm
SIMT body. In this sketch, `x_ub` and `y_ub` are UB pointers allocated with
`pto.alloc_buffer(..., scope="ub")`, while `x_frag` and `w_frag` are lane-local
pointers allocated with `scope="local"`.

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
        threads=threads,
        scratch=reduce_scratch,
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

The explicit scalar loads from `x_frag` and `w_frag` are the canonical form for
this draft. A later syntax-sugar layer should support the equivalent
`x_frag[i]` and `w_frag[i]` forms for scalar element reads.

## 6. Interface summary

| Interface | Purpose |
|-----------|---------|
| `scalar.load(ptr, offset=0, *, contiguous=None)` | Load one scalar or a contiguous vector from a typed pointer. |
| `scalar.store(value, ptr, offset=0, *, contiguous=None)` | Store one scalar or a contiguous vector to a typed pointer. |
| `pto.vec(dtype, lanes)` | Define a DSL builtin vector type. |
| `pto.vec(dtype, lanes, *, init=None)` | Construct a vector value, including scalar broadcast. |
| `pto.simt_allreduce_sum(value, *, threads, scale=1, thread_offset=0, scratch=None, scratch_offset=0)` | Sum one scalar from each participating SIMT workitem and broadcast the result. |
| `pto.alloc_buffer(shape, dtype, *, scope, persistent=False)` | Allocate linear UB or lane-local storage and return a typed pointer to it. |
