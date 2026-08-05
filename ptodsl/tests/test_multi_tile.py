# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptodsl import pto


@pto.jit(target="a5")
def multi_tile_probe(*, BLOCK: pto.const_expr = 128, dim: pto.const_expr = 16):
    tiles = pto.alloc_multi_tile(shape=[BLOCK, dim], dtype=pto.f32, count=2)
    with pto.for_(0, BLOCK, step=1) as iv:
        tile = pto.multi_tile_get(tiles, iv % 2)
        tile.fill(0.0)


def test_multi_tile_is_public_and_compiles():
    assert hasattr(pto, "alloc_multi_tile")
    assert hasattr(pto, "multi_tile_get")

    text = multi_tile_probe.compile().mlir_text()
    assert "pto.alloc_multi_tile" in text
    assert "pto.multi_tile_get" in text
    assert "count=2" in text


def main():
    test_multi_tile_is_public_and_compiles()
    print("ptodsl_multi_tile: PASS")


if __name__ == "__main__":
    main()
