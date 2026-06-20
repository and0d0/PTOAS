# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Scalar arithmetic helpers – exposed as top-level ``scalar.*`` from the
``ptodsl`` package (for example ``from ptodsl import scalar``).

Arithmetic helpers operate on raw ``mlir.ir.Value`` objects and emit the
corresponding arith dialect operations at the active insertion point.
Scalar memory helpers (`load` / `store`) also accept PTODSL surface-level
address views such as `tile[row, col]` and `tile.as_ptr() + offset`.
"""

from ._bootstrap import make_context  # ensure MLIR is on sys.path  # noqa: F401
from ._scalar_coercion import coerce_scalar_to_type
from ._scalar_adaptation import classify_runtime_scalar_type
from ._runtime_scalar_ops import (
    emit_runtime_abs,
    emit_runtime_binary_op,
    emit_runtime_max,
    emit_runtime_min,
)
from ._surface_values import resolve_address_access, unwrap_surface_value, wrap_surface_value
from ._types import _resolve

from mlir.dialects import arith
from mlir.dialects import math
from mlir.ir import IndexType, MemRefType, Operation, VectorType
from mlir.dialects import pto as _pto


def muli(lhs, rhs):
    """arith.muli"""
    return wrap_surface_value(emit_runtime_binary_op("mul", unwrap_surface_value(lhs), unwrap_surface_value(rhs)))


def addi(lhs, rhs):
    """arith.addi"""
    return wrap_surface_value(emit_runtime_binary_op("add", unwrap_surface_value(lhs), unwrap_surface_value(rhs)))


def subi(lhs, rhs):
    """arith.subi"""
    return wrap_surface_value(emit_runtime_binary_op("sub", unwrap_surface_value(lhs), unwrap_surface_value(rhs)))


def index_cast(type_or_val, val=None):
    """
    arith.index_cast.

    Two calling conventions::

        index_cast(result_type, value)   # explicit result type
        index_cast(value)                # result type = index (1-arg shorthand)
    """
    if val is None:
        # 1-arg form: cast to index
        return wrap_surface_value(arith.IndexCastOp(IndexType.get(), unwrap_surface_value(type_or_val)).result)
    return wrap_surface_value(arith.IndexCastOp(_resolve(type_or_val), unwrap_surface_value(val)).result)


def select(cond, true_val, false_val):
    """arith.select"""
    return wrap_surface_value(arith.SelectOp(
        unwrap_surface_value(cond),
        unwrap_surface_value(true_val),
        unwrap_surface_value(false_val),
    ).result)


def max(lhs, rhs):
    """Runtime scalar maximum across float / integer / index values."""
    return wrap_surface_value(emit_runtime_max(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    ))


def min(lhs, rhs):
    """Runtime scalar minimum across float / integer / index values."""
    return wrap_surface_value(emit_runtime_min(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    ))


def exp(value):
    """Runtime scalar exponential for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind != "float":
        raise TypeError(f"scalar.exp(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(math.ExpOp(raw_value).result)


def log(value):
    """Runtime scalar natural logarithm for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind != "float":
        raise TypeError(f"scalar.log(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(math.LogOp(raw_value).result)


def sqrt(value):
    """Runtime scalar square root for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind != "float":
        raise TypeError(f"scalar.sqrt(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(math.SqrtOp(raw_value).result)


def abs(value):
    """Runtime scalar absolute value across float / integer / index values."""
    return wrap_surface_value(emit_runtime_abs(unwrap_surface_value(value)))


def load(ptr_or_ref, offset=None, *, contiguous=None):
    """Load one scalar element or one contiguous vector from a PTODSL address view."""
    buffer_value, index_value = resolve_address_access(ptr_or_ref, offset)
    result_type = _infer_load_result_type(buffer_value.type, contiguous)
    return wrap_surface_value(Operation.create(
        "pto.load",
        results=[result_type],
        operands=[buffer_value, index_value],
    ).results[0])


def store(value, ptr_or_ref, offset=None, *, contiguous=None):
    """Store one scalar element or one contiguous vector to a PTODSL address view."""
    buffer_value, index_value = resolve_address_access(ptr_or_ref, offset)
    elem_type = _infer_buffer_element_type(buffer_value.type)
    raw_value = unwrap_surface_value(value)
    raw_value = _coerce_store_value(raw_value, elem_type, contiguous)
    Operation.create(
        "pto.store",
        operands=[buffer_value, index_value, raw_value],
    )


def _infer_buffer_element_type(buffer_type):
    try:
        return _pto.PtrType(buffer_type).element_type
    except Exception:
        return MemRefType(buffer_type).element_type


def _infer_load_result_type(buffer_type, contiguous):
    elem_type = _infer_buffer_element_type(buffer_type)
    lanes = _normalize_contiguous(contiguous)
    if lanes is None or lanes == 1:
        return elem_type
    return VectorType.get([lanes], elem_type)


def _coerce_store_value(raw_value, elem_type, contiguous):
    lanes = _normalize_contiguous(contiguous)
    if hasattr(raw_value, "type") and VectorType.isinstance(raw_value.type):
        vec_type = VectorType(raw_value.type)
        if vec_type.rank != 1:
            raise TypeError(f"scalar.store(...) expects a rank-1 vector value, got {raw_value.type}")
        if vec_type.element_type != elem_type:
            raise TypeError(
                "scalar.store(...) vector element type must match pointer element type: "
                f"got {vec_type.element_type}, expected {elem_type}"
            )
        if lanes is not None and lanes != vec_type.shape[0]:
            raise ValueError(
                "scalar.store(...) contiguous width must match vector lane count: "
                f"got contiguous={lanes}, value has {vec_type.shape[0]} lanes"
            )
        return raw_value

    if lanes is not None and lanes != 1:
        raise TypeError(
            "scalar.store(...) with contiguous > 1 expects a vector value; "
            f"got scalar value for contiguous={lanes}"
        )
    return coerce_scalar_to_type(raw_value, elem_type, context="scalar.store(...)")


def _normalize_contiguous(contiguous):
    if contiguous is None:
        return None
    if isinstance(contiguous, bool) or not isinstance(contiguous, int):
        raise TypeError("scalar.load/store contiguous must be a positive compile-time integer")
    if contiguous <= 0:
        raise ValueError("scalar.load/store contiguous must be positive")
    return contiguous


__all__ = [
    "muli", "addi", "subi",
    "index_cast",
    "select",
    "max", "min", "exp", "log", "sqrt", "abs",
    "load", "store",
]
