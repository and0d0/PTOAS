# alloc_buffer scope="local" PTODSL Frontend Plan

This note records the current implementation boundary for
`pto.alloc_buffer(..., scope="local")`.

## Goal

`scope="local"` represents per-workitem local fragment storage for kernels such
as RMSNorm:

```python
x_frag = pto.alloc_buffer((frag_elems,), pto.f32, scope="local")
w_frag = pto.alloc_buffer((frag_elems,), pto.f32, scope="local", persistent=True)
```

The returned object is an author-facing `AddressValue`, so user code can keep
using the same address-shaped surface as UB pointers:

```python
scalar.store(x_vec, x_frag, frag_offset)
x = scalar.load(x_frag, frag_offset + lane)
```

## Current Implementation

The first implementation is intentionally limited to the PTODSL-to-PTO-IR
frontend layer:

- `scope="ub"` keeps the existing lowering through `pto.alloc_tile` and
  `pto.tile_buf_addr`.
- `scope="local"` emits a local carrier:

  ```mlir
  %frag = memref.alloca() : memref<Nxf32>
  ```

  When `persistent=True`, the carrier is tagged:

  ```mlir
  %frag = memref.alloca() {pto.persistent} : memref<Nxf32>
  ```

- The local carrier is wrapped as `AddressValue` with metadata:

  ```python
  {
      "alloc_buffer_scope": "local",
      "shape": (...),
      "dtype": ...,
      "element_type": ...,
      "address_backend": "memref_alloca",
      "persistent": persistent,
  }
  ```

This metadata is the handoff point for the later load/store implementation.

## Deliberately Not Implemented Here

This change does not modify `scalar.load` or `scalar.store`.

The later load/store branch should recognize `AddressValue.surface_metadata`
with `alloc_buffer_scope == "local"` and lower accesses through the local
carrier instead of emitting ordinary `pto.load` / `pto.store`.

Expected lowering direction:

```python
scalar.store(x_vec, x_frag, frag_offset)
x = scalar.load(x_frag, frag_offset + lane)
```

should become equivalent to local pointer-backed access, ultimately similar to:

```mlir
%ptr = llvm.getelementptr %x_frag[%offset] : ...
llvm.store %x_vec, %ptr : vector<Nxf32>, ...
%x = llvm.load %ptr : ... -> f32
```

The exact memref-to-LLVM conversion is owned by the load/store work, not this
frontend allocation change.

## Persistent Local Storage

`persistent=True` is accepted for `scope="local"` and rejected for `scope="ub"`.
The intended final meaning is persistent per-workitem fragment storage. RMSNorm
uses this for `w_frag`, so that each workitem can cache its assigned weight
fragment once and reuse it across token iterations.

Persistent local storage needs a separate design for lifetime and reuse across
SIMT launches or token loops. This frontend step only emits the marker and
metadata; the keep/resume or equivalent residency transform is deferred.

## Constraints

- `shape` must be a non-empty list or tuple.
- For `scope="local"`, every dimension must be a positive static integer.
- `scope="local"` currently returns a local `AddressValue`, not a public
  `!pto.ptr<T, local>` type.
- MTE operations should continue to use UB or GM pointers; local fragments are
  intended for scalar/vector load-store paths inside SIMT bodies.
