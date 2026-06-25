#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ptodsl"))

from mlir.ir import Module
from ptodsl._bootstrap import make_context


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_parse_roundtrip_and_verify(text: str, label: str) -> None:
    with make_context() as ctx:
        parsed = Module.parse(text, ctx)
        parsed.operation.verify()
        roundtrip_text = str(parsed)
    expect(
        roundtrip_text == text,
        f"{label} should survive Module.parse(...) round-trip without textual drift",
    )


def load_rmsnorm_example():
    example_path = REPO_ROOT / "ptodsl" / "examples" / "rmsnorm_alloc_buffer_simt.py"
    expect(example_path.is_file(), f"RMSNorm example is missing: {example_path}")

    spec = spec_from_file_location("ptodsl_rmsnorm_alloc_buffer_simt", example_path)
    expect(spec is not None and spec.loader is not None, f"unable to create import spec for {example_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_variant(compiled, *, label: str, vector_type: str, helper_name_fragment: str, ub_size: int) -> None:
    compiled.verify()
    text = compiled.mlir_text()
    expect_parse_roundtrip_and_verify(text, f"RMSNorm {label} MLIR")

    expect("func.func @rmsnorm_4096_alloc_buffer_simt_context_kernel" in text, f"{label}: missing entry")
    expect(f"dyn_shared_memory_buf = {ub_size} : i64" in text, f"{label}: unexpected UB scratch size")
    expect("scf.for" in text, f"{label}: tokens_per_core loop should lower to scf.for")
    expect(text.count("scf.for") >= 4, f"{label}: SIMT inner loops should lower to compact scf.for ops")
    expect("pto.mte_gm_ub" in text, f"{label}: missing GM->UB transfer")
    expect("pto.mte_ub_gm" in text, f"{label}: missing UB->GM transfer")
    expect(vector_type in text, f"{label}: missing contiguous vector access type {vector_type}")
    expect(helper_name_fragment in text, f"{label}: missing allreduce helper")
    expect("func.call @__tl_allreduce_sum" in text or "call @__tl_allreduce_sum" in text,
           f"{label}: allreduce should remain helper-call based")

    expect(
        text.count("pto.mte_gm_ub") == 2,
        f"{label}: expected compact transfer structure with 2 GM->UB ops",
    )
    expect(
        text.count("pto.mte_ub_gm") == 2,
        f"{label}: expected compact transfer structure with 2 UB->GM ops",
    )
    expect(
        text.count("pto.castptr") <= 12,
        f"{label}: SIMT inner loops should not be trace-time expanded into many castptr ops",
    )
    expect(
        text.count("pto.store ") <= 8,
        f"{label}: SIMT inner loops should not be trace-time expanded into many scalar stores",
    )


def main() -> None:
    example = load_rmsnorm_example()

    expect(hasattr(example, "build_x128"), "RMSNorm example should export build_x128()")
    expect(hasattr(example, "build_x64"), "RMSNorm example should export build_x64()")

    check_variant(
        example.build_x128(),
        label="x128",
        vector_type="vector<2xf32>",
        helper_name_fragment="__tl_allreduce_sum_f32_t128_s1_o0",
        ub_size=82464,
    )
    check_variant(
        example.build_x64(),
        label="x64",
        vector_type="vector<4xf32>",
        helper_name_fragment="__tl_allreduce_sum_f32_t64_s1_o0",
        ub_size=82208,
    )

    print("ptodsl_rmsnorm_example_compile: PASS")


if __name__ == "__main__":
    main()
