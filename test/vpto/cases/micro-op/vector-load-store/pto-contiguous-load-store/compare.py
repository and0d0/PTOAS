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


def read_values(path: str, fmt: str):
    data = Path(path).read_bytes()
    size = struct.calcsize(fmt)
    if len(data) % size != 0:
        raise ValueError(f"{path} size is not a multiple of {fmt}")
    return list(struct.unpack(f"<{len(data) // size}{fmt}", data))


def compare_exact(name: str, golden, out) -> bool:
    if len(golden) != len(out):
        print(f"[ERROR] {name} shape mismatch, golden={len(golden)}, out={len(out)}")
        return False
    for idx, (expected, actual) in enumerate(zip(golden, out)):
        if isinstance(expected, float):
            ok = math.isclose(expected, actual, rel_tol=0.0, abs_tol=0.0)
        else:
            ok = expected == actual
        if not ok:
            print(
                f"[ERROR] {name} mismatch at idx={idx}, golden={expected}, out={actual}"
            )
            return False
    return True


def main():
    ok = True
    ok = compare_exact(
        "dst_f32",
        read_values("golden_dst_f32.bin", "f"),
        read_values("dst_f32.bin", "f"),
    ) and ok
    ok = compare_exact(
        "dst_i32",
        read_values("golden_dst_i32.bin", "i"),
        read_values("dst_i32.bin", "i"),
    ) and ok
    if not ok:
        sys.exit(2)
    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
