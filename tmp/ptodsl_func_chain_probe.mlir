// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// Generated from tmp/ptodsl_func_chain_probe.py while developing issue #946.
// The __ptodsl_<hash> suffixes are specialization hashes and may differ across runs.
//
// Observed behavior:
// 1. Plain undecorated helper: the internal `if True + range(2)` executes during tracing.
//    The caller contains two expanded `pto.barrier <PIPE_ALL>` ops and no helper func.
// 2. @pto.func dynamic loop: `for _ in range(limit)` is AST-rewritten and the helper body contains `scf.for`.
// 3. @pto.func dynamic if: `if lhs > rhs` is AST-rewritten and the helper body contains `scf.if`.
// 4. @pto.func(ast_rewrite=False): static `if True + range(2)` does not generate scf.
//    It trace-time expands inside the helper body into two `arith.addi` ops.
// 5. Chained calls: `multi_return_helper -> chain_mid -> dyn_loop_helper/dyn_if_helper/no_rewrite_static_helper`.
// 6. Multiple returns: `multi_return_helper` returns `(i32, i32)`.
// 7. Reuse: `dyn_if_helper` is defined once and called twice.
//
// Counts:
// scf.for: 1
// scf.if: 1
// func.func @dyn_loop_helper__ptodsl_: 1
// func.func @dyn_if_helper__ptodsl_: 1
// func.func @no_rewrite_static_helper__ptodsl_: 1
// func.func @chain_mid__ptodsl_: 1
// func.func @multi_return_helper__ptodsl_: 1
// call @dyn_if_helper__ptodsl_: 2
// pto.barrier <PIPE_ALL>: 2

module attributes {pto.target_arch = "a5"} {
  module attributes {pto.backend = "vpto", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"} {
    func.func @func_chain_probe(%arg0: i32) attributes {pto.entry} {
      %c0_i32 = arith.constant 0 : i32
      %0:2 = call @multi_return_helper__ptodsl_406ab2a269(%arg0, %c0_i32) : (i32, i32) -> (i32, i32)
      %1 = call @dyn_if_helper__ptodsl_82f7abb427(%0#0, %0#1) : (i32, i32) -> i32
      pto.barrier <PIPE_ALL>
      pto.barrier <PIPE_ALL>
      return
    }
    func.func @multi_return_helper__ptodsl_406ab2a269(%arg0: i32, %arg1: i32) -> (i32, i32) attributes {pto.ptodsl.callable_kind = "func", pto.ptodsl.logical_name = "multi_return_helper"} {
      %0 = call @chain_mid__ptodsl_be1027dd8d(%arg0, %arg1) : (i32, i32) -> i32
      %c1_i32 = arith.constant 1 : i32
      %1 = arith.addi %0, %c1_i32 : i32
      return %0, %1 : i32, i32
    }
    func.func @chain_mid__ptodsl_be1027dd8d(%arg0: i32, %arg1: i32) -> i32 attributes {pto.ptodsl.callable_kind = "func", pto.ptodsl.logical_name = "chain_mid"} {
      %0 = call @dyn_loop_helper__ptodsl_795afe8017(%arg0, %arg1) : (i32, i32) -> i32
      %1 = call @dyn_if_helper__ptodsl_82f7abb427(%0, %arg1) : (i32, i32) -> i32
      %2 = call @no_rewrite_static_helper__ptodsl_eccc8a4ce9(%1) : (i32) -> i32
      return %2 : i32
    }
    func.func @dyn_loop_helper__ptodsl_795afe8017(%arg0: i32, %arg1: i32) -> i32 attributes {pto.ptodsl.callable_kind = "func", pto.ptodsl.logical_name = "dyn_loop_helper"} {
      %c1_i32 = arith.constant 1 : i32
      %c0 = arith.constant 0 : index
      %0 = arith.index_cast %arg0 : i32 to index
      %c1 = arith.constant 1 : index
      %1 = scf.for %arg2 = %c0 to %0 step %c1 iter_args(%arg3 = %arg1) -> (i32) {
        %2 = arith.addi %arg3, %c1_i32 : i32
        scf.yield %2 : i32
      }
      return %1 : i32
    }
    func.func @dyn_if_helper__ptodsl_82f7abb427(%arg0: i32, %arg1: i32) -> i32 attributes {pto.ptodsl.callable_kind = "func", pto.ptodsl.logical_name = "dyn_if_helper"} {
      %0 = arith.cmpi sgt, %arg0, %arg1 : i32
      %1 = scf.if %0 -> (i32) {
        %2 = arith.subi %arg0, %arg1 : i32
        scf.yield %2 : i32
      } else {
        %2 = arith.subi %arg1, %arg0 : i32
        scf.yield %2 : i32
      }
      return %1 : i32
    }
    func.func @no_rewrite_static_helper__ptodsl_eccc8a4ce9(%arg0: i32) -> i32 attributes {pto.ptodsl.callable_kind = "func", pto.ptodsl.logical_name = "no_rewrite_static_helper"} {
      %c1_i32 = arith.constant 1 : i32
      %0 = arith.addi %arg0, %c1_i32 : i32
      %c1_i32_0 = arith.constant 1 : i32
      %1 = arith.addi %0, %c1_i32_0 : i32
      return %1 : i32
    }
  }
}
