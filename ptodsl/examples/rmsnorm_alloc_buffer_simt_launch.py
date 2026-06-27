# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Launch and validate the RMSNorm alloc_buffer/SIMT example on an Ascend NPU.

The test compares the kernel outputs written to GM against a NumPy RMSNorm
reference. It also fills output buffers with sentinels and checks guard regions
after the logical outputs, so missed writes and simple over-writes are caught by
the same host-side validation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np


if __package__ in {None, ""}:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "ptodsl" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError(
            "Unable to locate the PTODSL Python package root from rmsnorm_alloc_buffer_simt_launch.py"
        )


from rmsnorm_alloc_buffer_simt import rmsnorm_4096_alloc_buffer_simt_context_kernel


_DEVICE = "npu:0"
_HIDDEN_SIZE = 4096
_THREADS = 128
_ROUNDS = 8
_LANES = 4
_EPS = np.float32(1.0e-6)
_Y_GUARD_ELEMS = 1024
_RSTD_GUARD_ELEMS = 64
_SENTINEL = np.float32(123456.0)


@dataclass(frozen=True)
class Case:
    name: str
    n_cores: int
    tokens_per_core: int
    seed: int
    rtol: float = 1.0e-4
    y_atol: float = 1.0e-4
    rstd_atol: float = 1.0e-5

    @property
    def tokens(self) -> int:
        return self.n_cores * self.tokens_per_core


CASES = [
    Case("one_core_one_token", n_cores=1, tokens_per_core=1, seed=0x483001),
    Case("one_core_four_tokens", n_cores=1, tokens_per_core=4, seed=0x483004),
    Case("four_cores_two_tokens_each", n_cores=4, tokens_per_core=2, seed=0x483402),
]

FULL_CASE = Case("full_64_cores_64_tokens_each", n_cores=64, tokens_per_core=64, seed=0x483640)


def init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def make_inputs(case: Case) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(case.seed)
    x = rng.uniform(-0.75, 0.75, size=(case.tokens, _HIDDEN_SIZE)).astype(np.float32)
    w = rng.uniform(0.5, 1.5, size=(_HIDDEN_SIZE,)).astype(np.float32)

    # Make token/core addressing mistakes obvious in the output comparison.
    token_offsets = (np.arange(case.tokens, dtype=np.float32)[:, None] * np.float32(0.001))
    x = (x + token_offsets).astype(np.float32)
    return x, w


def rmsnorm_reference(x: np.ndarray, w: np.ndarray, eps: np.float32) -> tuple[np.ndarray, np.ndarray]:
    sum_sq = np.sum(x * x, axis=1, dtype=np.float32)
    rstd = (np.float32(1.0) / np.sqrt(sum_sq / np.float32(x.shape[1]) + eps)).astype(np.float32)
    y = (x * rstd[:, None] * w[None, :]).astype(np.float32)
    return y, rstd


def compile_kernel(case: Case):
    return rmsnorm_4096_alloc_buffer_simt_context_kernel.compile(
        threads=_THREADS,
        rounds=_ROUNDS,
        lanes=_LANES,
        hidden_size=_HIDDEN_SIZE,
        n_cores=case.n_cores,
        tokens_per_core=case.tokens_per_core,
    )


def assert_guard_unchanged(name: str, guard: np.ndarray) -> None:
    if not np.all(guard == _SENTINEL):
        bad = np.nonzero(guard != _SENTINEL)[0]
        first = int(bad[0])
        raise AssertionError(
            f"{name} guard overwritten at guard index {first}: got {guard[first]!r}, expected {_SENTINEL!r}"
        )


def run_case(case: Case, torch) -> None:
    x, w = make_inputs(case)
    y_ref, rstd_ref = rmsnorm_reference(x, w, _EPS)

    x_t = torch.from_numpy(x).to(_DEVICE)
    w_t = torch.from_numpy(w).to(_DEVICE)

    y_storage = torch.full(
        (case.tokens * _HIDDEN_SIZE + _Y_GUARD_ELEMS,),
        float(_SENTINEL),
        dtype=torch.float32,
        device=_DEVICE,
    )
    rstd_storage = torch.full(
        (case.tokens + _RSTD_GUARD_ELEMS,),
        float(_SENTINEL),
        dtype=torch.float32,
        device=_DEVICE,
    )

    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled = compile_kernel(case)
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[case.n_cores, stream](
        x_t.data_ptr(),
        y_storage.data_ptr(),
        w_t.data_ptr(),
        rstd_storage.data_ptr(),
        float(_EPS),
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    y_out = y_storage[: case.tokens * _HIDDEN_SIZE].cpu().numpy().reshape(case.tokens, _HIDDEN_SIZE)
    rstd_out = rstd_storage[: case.tokens].cpu().numpy()
    y_guard = y_storage[case.tokens * _HIDDEN_SIZE :].cpu().numpy()
    rstd_guard = rstd_storage[case.tokens :].cpu().numpy()

    np.testing.assert_allclose(rstd_out, rstd_ref, rtol=case.rtol, atol=case.rstd_atol)
    np.testing.assert_allclose(y_out, y_ref, rtol=case.rtol, atol=case.y_atol)
    assert_guard_unchanged("Y", y_guard)
    assert_guard_unchanged("RSTD", rstd_guard)

    y_diff = float(np.max(np.abs(y_out - y_ref))) if y_out.size else 0.0
    rstd_diff = float(np.max(np.abs(rstd_out - rstd_ref))) if rstd_out.size else 0.0
    print(
        f"PASS {case.name}  "
        f"grid={case.n_cores} tokens={case.tokens} "
        f"compile={compile_s:.3f}s launch={launch_s:.3f}s "
        f"max|Y|={y_diff:.3e} max|RSTD|={rstd_diff:.3e}"
    )


def emit_mlir(case: Case) -> str:
    return compile_kernel(case).mlir_text()


def main(argv=None) -> int:
    global _DEVICE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=_DEVICE, help="torch NPU device, default: npu:0")
    parser.add_argument("--case", choices=[case.name for case in CASES] + [FULL_CASE.name, "all"], default="all")
    parser.add_argument("--include-full", action="store_true", help="include the 64-core x 64-token full case")
    parser.add_argument("--emit-mlir", action="store_true", help="print MLIR for the selected case and exit")
    args = parser.parse_args(argv)

    _DEVICE = args.device

    selected = list(CASES)
    if args.include_full:
        selected.append(FULL_CASE)
    if args.case != "all":
        all_cases = {case.name: case for case in selected + [FULL_CASE]}
        selected = [all_cases[args.case]]

    if args.emit_mlir:
        if len(selected) != 1:
            parser.error("--emit-mlir expects one concrete --case")
        print(emit_mlir(selected[0]))
        return 0

    torch = init_runtime()
    for case in selected:
        run_case(case, torch)
    print("All RMSNorm cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
