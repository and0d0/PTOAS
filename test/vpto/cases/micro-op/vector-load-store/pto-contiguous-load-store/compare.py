#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import math
import struct
import sys
from pathlib import Path


def read_f32(path: str) -> list[float]:
    data = Path(path).read_bytes()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not a multiple of f32")
    return list(struct.unpack(f"<{len(data) // 4}f", data))


def main():
    golden = read_f32("golden_dst.bin")
    out = read_f32("dst.bin")
    if len(golden) != len(out):
        print(f"[ERROR] shape mismatch, golden={len(golden)}, out={len(out)}")
        sys.exit(2)
    for idx, (expected, actual) in enumerate(zip(golden, out)):
        if not math.isclose(expected, actual, rel_tol=0.0, abs_tol=0.0):
            print(
                f"[ERROR] mismatch at idx={idx}, golden={expected}, out={actual}"
            )
            sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
