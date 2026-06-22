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


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    src = [float(i) * 1.25 + 3.0 for i in range(ELEMS)]
    dst = [-7.0 for _ in range(ELEMS)]
    golden = list(dst)
    golden[0:4] = src[4:8]
    golden[8:12] = src[16:20]
    golden[32:36] = src[32:36]
    write_f32(output_dir / "src.bin", src)
    write_f32(output_dir / "dst.bin", dst)
    write_f32(output_dir / "golden_dst.bin", golden)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
