#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Probe for mixed ``@pto.func`` AST rewrite and trace-time expansion behavior."""

from ptodsl import pto


def plain_trace_helper():
    if True:
        for _ in range(2):
            pto.pipe_barrier(pto.Pipe.ALL)


@pto.func(returns=pto.i32)
def dyn_loop_helper(limit: pto.i32, value: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    total = value
    for _ in range(limit):
        total = total + one
    return total


@pto.func(returns=pto.i32)
def dyn_if_helper(lhs: pto.i32, rhs: pto.i32):
    if lhs > rhs:
        chosen = lhs - rhs
    else:
        chosen = rhs - lhs
    return chosen


@pto.func(ast_rewrite=False, returns=pto.i32)
def no_rewrite_static_helper(value: pto.i32):
    total = value
    if True:
        for _ in range(2):
            total = total + pto.const(1, dtype=pto.i32)
    return total


@pto.func(returns=pto.i32)
def chain_mid(limit: pto.i32, seed: pto.i32):
    looped = dyn_loop_helper(limit, seed)
    branched = dyn_if_helper(looped, seed)
    static_expanded = no_rewrite_static_helper(branched)
    return static_expanded


@pto.func(returns=(pto.i32, pto.i32))
def multi_return_helper(limit: pto.i32, seed: pto.i32):
    value = chain_mid(limit, seed)
    return value, value + pto.const(1, dtype=pto.i32)


@pto.jit(target="a5")
def func_chain_probe(limit: pto.i32):
    zero = pto.const(0, dtype=pto.i32)
    first, second = multi_return_helper(limit, zero)
    merged = dyn_if_helper(first, second)
    _ = merged
    plain_trace_helper()


def main():
    text = func_chain_probe.compile().mlir_text()
    print(text)
    print("\n=== COUNTS ===")
    for needle in [
        "scf.for",
        "scf.if",
        "func.func @dyn_loop_helper__ptodsl_",
        "func.func @dyn_if_helper__ptodsl_",
        "func.func @no_rewrite_static_helper__ptodsl_",
        "func.func @chain_mid__ptodsl_",
        "func.func @multi_return_helper__ptodsl_",
        "call @dyn_if_helper__ptodsl_",
        "pto.barrier <PIPE_ALL>",
    ]:
        print(f"{needle}: {text.count(needle)}")


if __name__ == "__main__":
    main()
