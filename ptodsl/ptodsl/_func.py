# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""``@pto.func`` decorator and reusable callable handle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import update_wrapper
import inspect

from ._ast_rewrite import rewrite_jit_function
from ._cache_signature import closure_cache_signature
from ._tracing import current_runtime


@dataclass(frozen=True)
class FuncSpec:
    """Declarative metadata for a rewrite-capable reusable PTODSL callable."""

    symbol_name: str


class FuncTemplate:
    """Callable decorated PTODSL helper surface."""

    def __init__(self, spec: FuncSpec, py_fn, *, ast_rewrite: bool = True):
        self.spec = spec
        self.py_fn = py_fn
        self._ast_rewrite = ast_rewrite
        self.signature = inspect.signature(py_fn)
        update_wrapper(self, py_fn)

    def emit_body(self, *args, **kwargs):
        """Emit this helper body into the currently active trace."""
        py_fn = rewrite_jit_function(self.py_fn) if self._ast_rewrite else self.py_fn
        return py_fn(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        runtime = current_runtime()
        if runtime is None:
            raise RuntimeError(
                "@pto.func helpers may only be called while tracing a compatible PTODSL kernel"
            )
        return runtime.dispatch_ptodsl_func_call(self, *args, **kwargs)

    def __ptodsl_cache_signature__(self):
        return (
            type(self).__name__,
            self.spec.symbol_name,
            id(self.py_fn),
            self._ast_rewrite,
            closure_cache_signature(self.py_fn),
        )


def func(fn=None, *, name: str | None = None, ast_rewrite: bool = True):
    """Decorate a Python function as a reusable PTODSL callable helper."""

    def decorator(py_fn):
        return FuncTemplate(
            FuncSpec(symbol_name=name or py_fn.__name__),
            py_fn,
            ast_rewrite=ast_rewrite,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


__all__ = [
    "FuncSpec",
    "FuncTemplate",
    "func",
]
