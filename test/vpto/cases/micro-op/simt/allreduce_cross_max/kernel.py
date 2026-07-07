#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys

import numpy as np


def _bootstrap_dsl_st_common() -> None:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        common_dir = candidate / "test" / "dsl-st"
        if (common_dir / "common.py").exists():
            sys.path.insert(0, str(common_dir))
            return
    raise RuntimeError("Unable to locate test/dsl-st/common.py from allreduce_cross_max/kernel.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case  # noqa: E402
from ptodsl import pto, scalar  # noqa: E402


THREADS = 128
SEED = 20260707


@pto.simt
def allreduce_cross_max_body(
    inp: pto.ptr(pto.f32, "gm"),
    out: pto.ptr(pto.f32, "gm"),
    scratch: pto.ptr(pto.f32, "ub"),
    *,
    threads: pto.const_expr = THREADS,
):
    tid = pto.get_tid_x()
    idx = scalar.index_cast(tid)
    value = scalar.load(inp, idx)
    reduced = pto.simt_allreduce_max(
        value,
        threads=threads,
        scale=1,
        thread_offset=0,
        scratch=scratch,
    )
    scalar.store(reduced, out, idx)


@pto.jit(
    name="allreduce_cross_max_kernel",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def allreduce_cross_max_kernel(
    inp: pto.ptr(pto.f32, "gm"),
    out: pto.ptr(pto.f32, "gm"),
):
    scratch = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
    allreduce_cross_max_body[THREADS, 1, 1](inp, out, scratch)
    pto.pipe_barrier(pto.Pipe.ALL)


def make_inputs():
    rng = np.random.default_rng(SEED)
    values = rng.uniform(-8.0, 8.0, size=THREADS).astype(np.float32)
    values[97] = np.float32(16.0)
    return [values]


def make_expected(values):
    return np.full((THREADS,), np.max(values), dtype=np.float32)


CASES = [
    golden_output_case(
        "allreduce_cross_max",
        allreduce_cross_max_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
