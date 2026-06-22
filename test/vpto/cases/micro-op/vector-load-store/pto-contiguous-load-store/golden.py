#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import struct
from pathlib import Path

ELEMS = 64


def write_f32(path: Path, values: list[float]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


def write_i32(path: Path, values: list[int]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}i", *values))


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    src_f32 = [float(i) * 1.25 + 3.0 for i in range(ELEMS)]
    dst_f32 = [-7.0 for _ in range(ELEMS)]
    golden_f32 = list(dst_f32)
    golden_f32[0:4] = src_f32[4:8]
    golden_f32[8:12] = src_f32[16:20]
    golden_f32[32:36] = src_f32[32:36]

    src_i32 = [i * 17 - 9 for i in range(ELEMS)]
    dst_i32 = [-11 for _ in range(ELEMS)]
    golden_i32 = list(dst_i32)
    golden_i32[0:4] = src_i32[4:8]
    golden_i32[8:12] = src_i32[16:20]

    write_f32(output_dir / "src_f32.bin", src_f32)
    write_f32(output_dir / "dst_f32.bin", dst_f32)
    write_f32(output_dir / "golden_dst_f32.bin", golden_f32)
    write_i32(output_dir / "src_i32.bin", src_i32)
    write_i32(output_dir / "dst_i32.bin", dst_i32)
    write_i32(output_dir / "golden_dst_i32.bin", golden_i32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
