# PTO Op 全量清单（TileOp 测试规划基线）

> 由 ODS 定义自动提取：`include/PTO/IR/{PTOOps,PTOGenericOps,VMIOps,VPTOOps,VPTOUbOps}.td`（全部属于同一 `pto` 方言）。
> 基线 commit `37f5f25e3`（2026-10-21，origin/master）。助记符以 ODS 为准。

**统计总览**（共 **713** 个 op）：

| 家族 | 数量 | 定义文件 |
|---|---:|---|
| **TileOp（`TileOpInterface`，tile 级指令）— 测试重点** | **142** | PTOOps.td |
| Tile 域辅助 op（视图/分配/桥接/量化配置） | 28 | PTOOps.td / VPTOOps.td |
| VPTO 向量/MTE/SIMT/atomic/redux/config/comm/标量/其它 | 351 | VPTOOps.td + PTOOps.td |
| UB 直连 op（`ub.*`） | 28 | VPTOUbOps.td |
| VMI 语义层向量 op（`vmi.*`） | 126 | VMIOps.td |
| Generic 标量 op（执行域无关） | 38 | PTOGenericOps.td |

## 测例位置说明（重要）

各表「测例示例」列给出**实际使用该 op 的测试文件**，路径相对 `test/`；「测试层」列出覆盖该 op 的测试层。
对 `test/` 全树文本文件（`.pto/.mlir/.py/.cpp/.txt/…`）做**内容级匹配**，匹配两种书写形态：
IR 助记符 `pto.<mnemonic>`（`.pto`/`.mlir`/dump 文件）与 Python 绑定类名 `pto.<Class>Op`（样例 `.py`）。注意：

- `tilelang_st`/`dsl-st` 等生成式测试只在 **dump 出的 `.pto`** 中被计入（源 `.py` 的 TileLang 调用不含上述两种字面量）；
- `tilelib-st/a5/<op名>/` 目录按目录名对齐计为该 op 的 ST 测试层（见 `test/tilelib-st/a5/`）；
- lit 层（`test/lit/pto`、`test/lit/vpto` 等）为 IR 检查（emitc/CHECK），不含板端运行；
- `samples` 层为可运行样例（python 构建 IR + golden 比对），最接近「每个 op 每种格式」的运行时验证目标。

> 助记符重名说明：VMI op 与 VPTO op 存在同名（如 `pto.vmi.vadd` 与 `pto.vadd`），完整助记符带各自前缀，互不冲突；
> `bitcast` 同时存在于 PTOOps.td（tile 域 `pto.bitcast`）与 VMIOps.td（`pto.vmi.bitcast`）。

---

## 一、TileOp（TileOpInterface，共 142 个）— 测试重点

定义基类 `PTO_TOp`（`include/PTO/IR/PTOOps.td`），自动挂 `TileOpInterface`。tile-op 文档对照见 `docs/isa/tile-op/`。

### 1.1 DMA 数据搬运（5 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tload` | `TLoadOp` | PTOOps.td:574 | samples·tilelang-st·lit·python | `samples/Or/or.py`<br>`lit/pto/tload_offset_view.pto`<br>`lit/pto/tload_cache_policy.pto`<br>…共 1101 文件 |
| `pto.tstore` | `TStoreOp` | PTOOps.td:794 | samples·tilelang-st·lit | `samples/Ci/ci.py`<br>`lit/pto/tstore_forms_emitc.pto`<br>`lit/pto/tstore_mat_invalid.pto`<br>…共 1142 文件 |
| `pto.tprefetch` | `TPrefetchOp` | PTOOps.td:695 | samples·lit | `samples/TPrefetch/tprefetch.py`<br>`lit/pto/tprefetch_emitc.pto`<br>`lit/pto/tprefetch_tile_native.pto`<br>…共 8 文件 |
| `pto.ttrans` | `TTransOp` | PTOOps.td:924 | samples·tilelib-st·lit·vpto | `samples/Trans/trans.py`<br>`tilelib-st/a5/ttrans/`<br>`lit/pto/ttrans_tmp_placeholder_a5.pto`<br>…共 21 文件 |
| `pto.tmov` | `TMovOp` | PTOOps.td:1385 | samples·tilelang-st·lit·python | `samples/Gemv/gemv.py`<br>`lit/pto/tmov_x2zz_emitc.pto`<br>`lit/pto/tmov_forms_emitc.pto`<br>…共 268 文件 |

### 1.2 MatMul / GEMV（Cube 域）（12 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tmatmul` | `TMatmulOp` | PTOOps.td:1103 | samples·tilelib-st·tilelang-st·lit·vpto·bridge | `samples/MatMul/tmatmulk.py`<br>`tilelib-st/a5/tmatmul/`<br>`lit/pto/tmatmul_accphase_emitc.pto`<br>…共 262 文件 |
| `pto.tmatmul.acc` | `TMatmulAccOp` | PTOOps.td:1139 | samples·tilelang-st·lit | `samples/MatMul/tmatmulk.py`<br>`lit/pto/tmatmul_accphase_emitc.pto`<br>`lit/pto/tmatmul_acc_phase_emitc.pto`<br>…共 166 文件 |
| `pto.tmatmul.bias` | `TMatmulBiasOp` | PTOOps.td:957 | samples·tilelang-st·lit | `samples/MatMul/tmatmulk.py`<br>`lit/pto/tmatmul_accphase_emitc.pto`<br>`lit/pto/tmatmul_low_precision_emitc.pto`<br>…共 11 文件 |
| `pto.tmatmul.mx` | `TMatmulMxOp` | PTOOps.td:996 | samples·tilelang-st·lit | `samples/MatmulMxLowPrecision/matmul_mx_low_precision.py`<br>`lit/pto/tmatmul_mx_emitc.pto`<br>`lit/pto/tmatmul_mx_f8e5m2_emitc.pto`<br>…共 10 文件 |
| `pto.tmatmul.mx.acc` | `TMatmulMxAccOp` | PTOOps.td:1032 | samples·lit | `samples/MatmulMxLowPrecision/matmul_mx_low_precision.py`<br>`lit/pto/tmatmul_mx_accphase_emitc.pto`<br>`lit/pto/tmatmul_mx_emitc.pto` |
| `pto.tmatmul.mx.bias` | `TMatmulMxBiasOp` | PTOOps.td:1069 | samples·tilelang-st·lit | `samples/MatmulMxLowPrecision/matmul_mx_low_precision.py`<br>`lit/pto/tmatmul_mx_emitc.pto`<br>`lit/pto/tmatmul_mx_low_precision_variants_emitc.pto`<br>…共 4 文件 |
| `pto.tgemv` | `TGemvOp` | PTOOps.td:1175 | samples·tilelib-st·tilelang-st·lit | `samples/Gemv/gemv.py`<br>`tilelib-st/a5/tgemv/`<br>`lit/pto/tgemv_accphase_emitc.pto`<br>…共 8 文件 |
| `pto.tgemv.acc` | `TGemvAccOp` | PTOOps.td:1211 | samples·tilelang-st·lit | `samples/Gemv/gemvacc.py`<br>`lit/pto/tgemv_accphase_emitc.pto`<br>`tilelang_st/npu/a5/src/st/testcase/tgemv/tgemv.pto` |
| `pto.tgemv.bias` | `TGemvBiasOp` | PTOOps.td:1247 | samples·tilelang-st·lit | `samples/Gemv/gemvbias.py`<br>`lit/pto/tgemv_accphase_emitc.pto`<br>`tilelang_st/npu/a5/src/st/testcase/tgemv/tgemv.pto` |
| `pto.tgemv.mx` | `TGemvMxOp` | PTOOps.td:1282 | samples·tilelang-st·lit | `samples/Gemvmx/gemvmx.py`<br>`lit/pto/tgemv_mx_emitc.pto`<br>`lit/pto/tgemv_mx_f8e5m2_emitc.pto`<br>…共 8 文件 |
| `pto.tgemv.mx.acc` | `TGemvMxAccOp` | PTOOps.td:1316 | samples·tilelang-st·lit | `samples/Gemvmx/gemvmx.py`<br>`lit/pto/tgemv_mx_accphase_emitc.pto`<br>`lit/pto/tgemv_mx_variants_emitc.pto`<br>…共 4 文件 |
| `pto.tgemv.mx.bias` | `TGemvMxBiasOp` | PTOOps.td:1351 | samples·tilelang-st·lit | `samples/Gemvmx/gemvmx.py`<br>`lit/pto/tgemv_mx_accphase_emitc.pto`<br>`lit/pto/tgemv_mx_variants_emitc.pto`<br>…共 4 文件 |

### 1.3 AIV/AIC tile 生命周期与同步（13 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.talloc_to_aiv` | `TAllocToAivOp` | PTOOps.td:1760 | samples·lit | `samples/TPipe/a3_pipe_roundtrip.pto`<br>`lit/pto/tpush_tpop_globaltensor_frontend_a3.pto`<br>`lit/pto/tpush_tpop_globaltensor_frontend_a5.pto`<br>…共 10 文件 |
| `pto.talloc_to_aic` | `TAllocToAicOp` | PTOOps.td:1777 | samples·lit | `samples/TPipe/a3_pipe_roundtrip.pto`<br>`lit/pto/tpush_tpop_globaltensor_frontend_a5.pto`<br>`lit/pto/tpush_tpop_odd_split_unsupported_invalid.pto`<br>…共 4 文件 |
| `pto.tpush_to_aiv` | `TPushToAivOp` | PTOOps.td:1794 | samples·lit | `samples/TPushTPop/test1/kernel.pto`<br>`lit/pto/tpush_tpop_dir_both.pto`<br>`lit/pto/fixpipe_basic_deqf16_nz2nd.pto`<br>…共 88 文件 |
| `pto.tpush_to_aic` | `TPushToAicOp` | PTOOps.td:1814 | samples·lit | `samples/TPushTPop/test3/kernel.pto`<br>`lit/pto/tpush_tpop_dir_both.pto`<br>`lit/pto/tpush_tpop_odd_split_emitc.pto`<br>…共 22 文件 |
| `pto.tpop_from_aic` | `TPopFromAicOp` | PTOOps.td:1837 | samples·lit | `samples/TPushTPop/test1/kernel.pto`<br>`lit/pto/tpush_tpop_dir_both.pto`<br>`lit/pto/fixpipe_basic_deqf16_nz2nd.pto`<br>…共 99 文件 |
| `pto.tpop_from_aiv` | `TPopFromAivOp` | PTOOps.td:1860 | samples·lit | `samples/TPushTPop/test3/kernel.pto`<br>`lit/pto/tpush_tpop_dir_both.pto`<br>`lit/pto/tpush_tpop_odd_split_emitc.pto`<br>…共 25 文件 |
| `pto.tfree_from_aic` | `TFreeFromAicOp` | PTOOps.td:1879 | samples·lit | `samples/TPushTPop/test1/kernel.pto`<br>`lit/pto/tpush_tpop_dir_both.pto`<br>`lit/pto/fixpipe_basic_deqf16_nz2nd.pto`<br>…共 98 文件 |
| `pto.tfree_from_aiv` | `TFreeFromAivOp` | PTOOps.td:1897 | samples·lit | `samples/TPushTPop/test3/kernel.pto`<br>`lit/pto/tpush_tpop_dir_both.pto`<br>`lit/pto/tpush_tpop_odd_split_emitc.pto`<br>…共 24 文件 |
| `pto.talloc` | `TAllocOp` | PTOOps.td:2503 | lit | `lit/pto/tpush_tpop_globaltensor_frontend_a3.pto`<br>`lit/pto/tpush_tpop_globaltensor_internal_a3.pto`<br>`lit/pto/tpush_tpop_globaltensor_entry_mismatch_invalid.pto` |
| `pto.tpush` | `TPushOp` | PTOOps.td:2436 | lit·vpto·bridge | `lit/pto/tpush_tpop_emitc.pto`<br>`lit/pto/tpush_tpop_split_invalid.pto`<br>`vpto/cases/kernels/cube-tmatmul-bridge/kernel.pto`<br>…共 22 文件 |
| `pto.tpop` | `TPopOp` | PTOOps.td:2763 | lit·vpto | `lit/pto/tpush_tpop_emitc.pto`<br>`lit/pto/issue1251_subview_of_tpop_a3.pto`<br>`vpto/cases/kernels/cube-tmatmul-bridge/kernel.pto`<br>…共 19 文件 |
| `pto.tfree` | `TFreeOp` | PTOOps.td:2842 | lit·vpto | `lit/pto/issue489_interleaved_tpop_tfree_a3.pto`<br>`lit/pto/pipe_tile_native.pto`<br>`vpto/cases/kernels/cube-tmatmul-bridge/kernel.pto`<br>…共 15 文件 |
| `pto.tsync` | `TSyncOp` | PTOOps.td:3168 | lit | `lit/pto/tsync_emitc.pto`<br>`lit/pto/tsync_tile_native.pto`<br>`lit/vpto/vecscope_sync_verify.pto` |

### 1.4 逐元素一元（11 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tabs` | `TAbsOp` | PTOOps.td:3586 | samples·tilelang-st·lit | `samples/Abs/abs.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/tabs.pto`<br>`lit/pto/issue1606_tabs_i16_a3_invalid.pto`<br>…共 39 文件 |
| `pto.tneg` | `TNegOp` | PTOOps.td:5349 | samples·tilelang-st·lit | `samples/Neg/neg.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/tneg.pto`<br>`lit/vpto/expand_tile_op_tilelang_tneg.pto`<br>…共 56 文件 |
| `pto.tnot` | `TNotOp` | PTOOps.td:5380 | samples·tilelang-st·lit | `samples/Not/not.py`<br>`lit/vpto/expand_tile_op_tilelang_tnot.pto`<br>`lit/pto/unary_compute_tile_native.pto`<br>…共 5 文件 |
| `pto.texp` | `TExpOp` | PTOOps.td:4718 | samples·tilelang-st·lit | `samples/Exp/exp.py`<br>`lit/pto/texp_precision_emitc.pto`<br>`lit/vpto/expand_tile_op_tilelang_texp.pto`<br>…共 77 文件 |
| `pto.tlog` | `TLogOp` | PTOOps.td:5076 | samples·tilelang-st·lit | `samples/Log/log.py`<br>`lit/pto/tlog_precision_emitc.pto`<br>`lit/vpto/ub/tile_to_ub/elementwise/tlog.pto`<br>…共 12 文件 |
| `pto.tsqrt` | `TSqrtOp` | PTOOps.td:6805 | samples·tilelang-st·lit | `samples/Sqrt/sqrt.py`<br>`lit/pto/tsqrt_precision_emitc.pto`<br>`lit/vpto/expand_tile_op_tilelang_tsqrt.pto`<br>…共 47 文件 |
| `pto.trsqrt` | `TRsqrtOp` | PTOOps.td:6499 | samples·tilelang-st·lit | `samples/Rsqrt/rsqrt.py`<br>`lit/pto/trsqrt_emitc.pto`<br>`lit/pto/trsqrt_precision_emitc.pto`<br>…共 37 文件 |
| `pto.trecip` | `TRecipOp` | PTOOps.td:5870 | samples·tilelang-st·lit | `samples/Recip/recip.py`<br>`lit/pto/trecip_precision_emitc.pto`<br>`lit/vpto/ub/tile_to_ub/elementwise/trecip.pto`<br>…共 96 文件 |
| `pto.trelu` | `TReluOp` | PTOOps.td:5902 | samples·tilelang-st·lit | `samples/Relu/relu.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/trelu.pto`<br>`lit/vpto/expand_tile_op_trelu_tilelang.pto`<br>…共 9 文件 |
| `pto.tlrelu` | `TLReluOp` | PTOOps.td:5105 | samples·tilelang-st·lit | `samples/Lrelu/lrelu.py`<br>`lit/pto/tlrelu_tile_native.pto`<br>`lit/vpto/expand_tile_op_tlrelu_tilelang.pto`<br>…共 5 文件 |
| `pto.trandom` | `TRandomOp` | PTOOps.td:4489 | tilelib-st·tilelang-st·lit | `tilelib-st/a5/trandom/`<br>`lit/pto/trandom_emitc.pto`<br>`lit/pto/expand_tile_op_trandom_tilelang.pto`<br>…共 5 文件 |

### 1.5 逐元素二元（12 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tadd` | `TAddOp` | PTOOps.td:3618 | samples·tilelib-st·tilelang-st·lit | `samples/SubView/subview.py`<br>`tilelib-st/a5/tadd/`<br>`lit/vpto/ub/tile_to_ub/elementwise/tadd_a2.pto`<br>…共 309 文件 |
| `pto.tsub` | `TSubOp` | PTOOps.td:6834 | samples·tilelang-st·lit | `samples/Sub/sub.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/tsub.pto`<br>`lit/pto/sub_tile_native.pto`<br>…共 91 文件 |
| `pto.tmul` | `TMulOp` | PTOOps.td:5284 | samples·tilelang-st·lit | `samples/Mul/mul.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/tmul.pto`<br>`lit/pto/tpartmul_a5_i8_invalid.pto`<br>…共 186 文件 |
| `pto.tdiv` | `TDivOp` | PTOOps.td:4523 | samples·tilelib-st·tilelang-st·lit | `samples/Div/div.py`<br>`tilelib-st/a5/tdiv/`<br>`lit/pto/tdiv_precision_emitc.pto`<br>…共 58 文件 |
| `pto.tmax` | `TMaxOp` | PTOOps.td:5134 | samples·tilelang-st·lit | `samples/Max/max.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/tmax_multi_row.pto`<br>`lit/vpto/ub/tile_to_ub/elementwise/tmax_single_row.pto`<br>…共 72 文件 |
| `pto.tmin` | `TMinOp` | PTOOps.td:5192 | samples·tilelang-st·lit | `samples/Min/min.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/tmin.pto`<br>`lit/pto/minmax_tile_native.pto`<br>…共 6 文件 |
| `pto.tfmod` | `TFModOp` | PTOOps.td:4585 | samples·tilelib-st·tilelang-st·lit | `samples/Fmod/fmod.py`<br>`tilelib-st/a5/tfmod/`<br>`lit/pto/tfmod_precision_emitc.pto`<br>…共 7 文件 |
| `pto.trem` | `TRemOp` | PTOOps.td:5933 | samples·tilelib-st·tilelang-st·lit | `samples/Rem/rem.py`<br>`tilelib-st/a5/trem/`<br>`lit/pto/trem_emitc.pto`<br>…共 12 文件 |
| `pto.tpow` | `TPowOp` | PTOOps.td:4648 | samples·lit | `samples/Tpow/tpow.py`<br>`lit/pto/tpow_emitc.pto`<br>`lit/pto/tpow_int_no_tmp.pto`<br>…共 12 文件 |
| `pto.tand` | `TAndOp` | PTOOps.td:3818 | samples·tilelang-st·lit | `samples/And/and.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/tand.pto`<br>`lit/vpto/expand_tile_op_tilelang_tand.pto`<br>…共 9 文件 |
| `pto.tor` | `TOrOp` | PTOOps.td:5411 | samples·tilelang-st·lit | `samples/Or/or.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/tor.pto`<br>`lit/vpto/expand_tile_op_tilelang_tor.pto`<br>…共 8 文件 |
| `pto.txor` | `TXorOp` | PTOOps.td:7005 | samples·tilelang-st·lit | `samples/Xor/xor.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/txor.pto`<br>`lit/vpto/ub/planned_addresses/txor_planned_addr.pto`<br>…共 12 文件 |

### 1.6 标量操作数变体（xxs/xxc 后缀）（17 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tadds` | `TAddSOp` | PTOOps.td:3727 | samples·tilelang-st·lit·dsl | `samples/Adds/adds.py`<br>`lit/pto/tadds_validrow_mismatch.pto`<br>`lit/vpto/ub/tile_to_ub/elementwise/tadds.pto`<br>…共 170 文件 |
| `pto.tsubs` | `TSubSOp` | PTOOps.td:6902 | samples·tilelang-st·lit | `samples/Subs/subs.py`<br>`lit/vpto/expand_tile_op_tilelang_tsubs.pto`<br>`lit/pto/sub_tile_native.pto`<br>…共 53 文件 |
| `pto.tmuls` | `TMulSOp` | PTOOps.td:5317 | samples·tilelang-st·lit | `samples/Muls/muls.py`<br>`lit/vpto/ub/tile_to_ub/elementwise/tmuls.pto`<br>`lit/vpto/expand_tile_op_tilelang_tmuls.pto`<br>…共 197 文件 |
| `pto.tdivs` | `TDivSOp` | PTOOps.td:4558 | samples·tilelib-st·tilelang-st·lit | `samples/Divs/divs.py`<br>`tilelib-st/a5/tdivs/`<br>`lit/pto/tdivs_dual_order_emitc.pto`<br>…共 64 文件 |
| `pto.tmaxs` | `TMaxSOp` | PTOOps.td:5163 | samples·tilelang-st·lit | `samples/Maxs/maxs.py`<br>`lit/pto/tmaxs_bf16.pto`<br>`lit/vpto/expand_tile_op_tilelang_tmaxs.pto`<br>…共 49 文件 |
| `pto.tmins` | `TMinSOp` | PTOOps.td:5221 | samples·tilelang-st·lit | `samples/Mins/mins.py`<br>`lit/vpto/expand_tile_op_tilelang_tmins.pto`<br>`lit/pto/minmax_scalar_tile_native.pto`<br>…共 45 文件 |
| `pto.tfmods` | `TFModSOp` | PTOOps.td:4615 | samples·tilelib-st·tilelang-st·lit | `samples/Fmods/fmods.py`<br>`tilelib-st/a5/tfmods/`<br>`lit/vpto/expand_tile_op_tilelang_tfmods.pto`<br>…共 7 文件 |
| `pto.trems` | `TRemSOp` | PTOOps.td:5963 | samples·tilelib-st·tilelang-st·lit | `samples/Rems/rems.py`<br>`tilelib-st/a5/trems/`<br>`lit/pto/trems_emitc.pto`<br>…共 13 文件 |
| `pto.tpows` | `TPowSOp` | PTOOps.td:4685 | samples·lit | `samples/Tpows/tpows.py`<br>`lit/pto/tpows_emitc.pto`<br>`lit/pto/tpows_int_no_tmp.pto`<br>…共 7 文件 |
| `pto.tands` | `TAndSOp` | PTOOps.td:3920 | samples·tilelang-st·lit | `samples/Ands/ands.py`<br>`lit/pto/issue1606_tands_f16_invalid.pto`<br>`lit/vpto/expand_tile_op_tilelang_tands.pto`<br>…共 7 文件 |
| `pto.tors` | `TOrSOp` | PTOOps.td:5443 | samples·tilelang-st·lit | `samples/Ors/ors.py`<br>`lit/vpto/expand_tile_op_tilelang_tors.pto`<br>`lit/pto/logic_scalar_tile_native.pto`<br>…共 6 文件 |
| `pto.txors` | `TXorSOp` | PTOOps.td:6972 | samples·tilelang-st·lit | `samples/Xors/xors.py`<br>`lit/pto/txors_tmp_type_a3.pto`<br>`lit/pto/txors_tmp_layout_a3.pto`<br>…共 11 文件 |
| `pto.taddsc` | `TAddSCOp` | PTOOps.td:3789 | samples·lit | `samples/Addsc/addsc.py`<br>`lit/pto/add_carry_tile_native.pto` |
| `pto.tsubsc` | `TSubSCOp` | PTOOps.td:6936 | samples·lit | `samples/Subsc/subsc.py`<br>`lit/pto/sub_tile_native.pto` |
| `pto.taddc` | `TAddCOp` | PTOOps.td:3693 | samples·lit | `samples/Addc/addc.py`<br>`lit/pto/add_carry_tile_native.pto`<br>`samples/Addc/addc.pto` |
| `pto.tsubc` | `TSubCOp` | PTOOps.td:6866 | samples·lit | `samples/Subc/subc.py`<br>`lit/pto/sub_tile_native.pto` |
| `pto.taxpy` | `TAxpyOp` | PTOOps.td:3760 | samples·lit | `samples/TAxpy/taxpy.pto`<br>`lit/pto/taxpy_emitc.pto`<br>`lit/pto/axpy_dequant_tile_native.pto`<br>…共 4 文件 |

### 1.7 融合激活（2 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.taddrelu` | `TAddReluOp` | PTOOps.td:3655 | lit | `lit/pto/taddrelu_a5_invalid.pto`<br>`lit/vpto/ub/tile_to_ub/elementwise/taddrelu.pto` |
| `pto.tprelu` | `TPReluOp` | PTOOps.td:5677 | samples·tilelib-st·tilelang-st·lit | `samples/Prelu/prelu.py`<br>`tilelib-st/a5/tprelu/`<br>`lit/pto/tprelu_tmp_contract_a3_invalid.pto`<br>…共 13 文件 |

### 1.8 移位（4 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tshl` | `TShlOp` | PTOOps.td:6656 | samples·tilelang-st·lit | `samples/Shl/shl.py`<br>`lit/pto/tshl_dtype_mismatch.pto`<br>`lit/vpto/expand_tile_op_tilelang_tshl.pto`<br>…共 6 文件 |
| `pto.tshr` | `TShrOp` | PTOOps.td:6688 | samples·tilelang-st·lit | `samples/Shr/shr.py`<br>`lit/pto/tshr_dtype_mismatch.pto`<br>`lit/vpto/expand_tile_op_tilelang_tshr.pto`<br>…共 6 文件 |
| `pto.tshls` | `TShlSOp` | PTOOps.td:6720 | samples·tilelang-st·lit | `samples/Shls/shls.py`<br>`lit/vpto/expand_tile_op_tilelang_tshls.pto`<br>`lit/pto/shift_tile_native.pto`<br>…共 6 文件 |
| `pto.tshrs` | `TShrSOp` | PTOOps.td:6744 | samples·tilelang-st·lit | `samples/Shrs/shrs.py`<br>`lit/vpto/expand_tile_op_tilelang_tshrs.pto`<br>`lit/pto/shift_tile_native.pto`<br>…共 6 文件 |

### 1.9 行/列归约与统计（13 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.trowsum` | `TRowSumOp` | PTOOps.td:6371 | samples·tilelang-st·lit | `samples/Rowsum/rowsum.py`<br>`lit/vpto/expand_tile_op_tilelang_trowsum.pto`<br>`lit/tile_fusion/prefusion_analysis_tadd_tmul_trowsum.pto`<br>…共 113 文件 |
| `pto.trowprod` | `TRowProdOp` | PTOOps.td:6471 | samples·tilelang-st·lit | `samples/Rowprod/rowprod.py`<br>`lit/vpto/expand_tile_op_tilelang_trowprod.pto`<br>`lit/pto/reduction_tile_native.pto`<br>…共 7 文件 |
| `pto.trowmax` | `TRowMaxOp` | PTOOps.td:6265 | samples·tilelang-st·lit | `samples/Rowmax/rowmax.py`<br>`lit/vpto/expand_tile_op_tilelang_trowmax.pto`<br>`lit/pto/trow_extreme_i8_a5.pto`<br>…共 77 文件 |
| `pto.trowmin` | `TRowMinOp` | PTOOps.td:6318 | samples·tilelang-st·lit | `samples/Rowmin/rowmin.py`<br>`lit/vpto/expand_tile_op_tilelang_trowmin.pto`<br>`lit/pto/trow_extreme_i8_a5.pto`<br>…共 9 文件 |
| `pto.trowargmax` | `TRowArgMaxOp` | PTOOps.td:6290 | samples·tilelang-st·lit | `samples/TRowArgMax/trowargmax.pto`<br>`lit/pto/trowargmax_emitc.pto`<br>`lit/pto/issue558_trowargmax_vec_overflow_a5.pto`<br>…共 10 文件 |
| `pto.trowargmin` | `TRowArgMinOp` | PTOOps.td:6343 | samples·tilelang-st·lit | `samples/TRowArgMin/trowargmin.pto`<br>`lit/pto/trowargmin_emitc.pto`<br>`lit/pto/issue558_trowargmin_vec_overflow_a5.pto`<br>…共 8 文件 |
| `pto.tcolsum` | `TColSumOp` | PTOOps.td:4406 | samples·tilelib-st·tilelang-st·lit | `samples/Colsum/colsum.py`<br>`tilelib-st/a5/tcolsum/`<br>`lit/vpto/tcolsum.pto`<br>…共 10 文件 |
| `pto.tcolprod` | `TColProdOp` | PTOOps.td:4436 | samples·tilelang-st·lit | `samples/Colprod/colprod.py`<br>`lit/vpto/tcolprod.pto`<br>`lit/pto/reduction_tile_native.pto`<br>…共 5 文件 |
| `pto.tcolmax` | `TColMaxOp` | PTOOps.td:4300 | samples·tilelang-st·lit | `samples/Colmax/colmax.py`<br>`lit/vpto/tcolmax.pto`<br>`lit/pto/col_minmax_tile_native.pto`<br>…共 8 文件 |
| `pto.tcolmin` | `TColMinOp` | PTOOps.td:4353 | samples·tilelang-st·lit | `samples/Colmin/colmin.py`<br>`lit/vpto/tcolmin.pto`<br>`lit/pto/col_minmax_tile_native.pto`<br>…共 5 文件 |
| `pto.tcolargmax` | `TColArgMaxOp` | PTOOps.td:4328 | samples·tilelang-st·lit | `samples/TColArgMax/tcolargmax.pto`<br>`lit/vpto/tcolargmax.pto`<br>`lit/pto/tcolargmax_emitc.pto`<br>…共 12 文件 |
| `pto.tcolargmin` | `TColArgMinOp` | PTOOps.td:4381 | samples·tilelang-st·lit | `samples/TColArgMin/tcolargmin.pto`<br>`lit/vpto/tcolargmin.pto`<br>`lit/pto/tcolargmin_emitc.pto`<br>…共 9 文件 |
| `pto.thistogram` | `THistogramOp` | PTOOps.td:3465 | tilelib-st·lit | `tilelib-st/a5/thistogram/`<br>`lit/pto/thistogram_emitc.pto`<br>`lit/pto/thistogram_emitc_u32.pto`<br>…共 11 文件 |

### 1.10 部分元素（partial elementwise）（6 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tpartadd` | `TPartAddOp` | PTOOps.td:5475 | samples·tilelang-st·lit | `samples/Partadd/partadd.py`<br>`lit/vpto/expand_tile_op_tilelang_tpartadd.pto`<br>`lit/pto/part_arithmetic_tile_native.pto`<br>…共 9 文件 |
| `pto.tpartmul` | `TPartMulOp` | PTOOps.td:5645 | samples·tilelang-st·lit | `samples/Partmul/partmul.py`<br>`lit/pto/tpartmul_emitc.pto`<br>`lit/pto/tpartmul_a5_i8_invalid.pto`<br>…共 9 文件 |
| `pto.tpartmax` | `TPartMaxOp` | PTOOps.td:5507 | samples·tilelang-st·lit | `samples/Partmax/partmax.py`<br>`lit/vpto/expand_tile_op_tilelang_tpartmax.pto`<br>`lit/pto/part_arithmetic_tile_native.pto`<br>…共 5 文件 |
| `pto.tpartmin` | `TPartMinOp` | PTOOps.td:5539 | samples·tilelang-st·lit | `samples/Partmin/partmin.py`<br>`lit/vpto/expand_tile_op_tilelang_tpartmin.pto`<br>`lit/pto/part_arithmetic_tile_native.pto`<br>…共 5 文件 |
| `pto.tpartargmax` | `TPartArgMaxOp` | PTOOps.td:5568 | samples·lit | `samples/Partarg/partarg.py`<br>`lit/pto/tpartarg_emitc.pto`<br>`lit/pto/arg_reduction_tile_native.pto` |
| `pto.tpartargmin` | `TPartArgMinOp` | PTOOps.td:5605 | samples·lit | `samples/Partarg/partarg.py`<br>`lit/pto/tpartarg_emitc.pto`<br>`lit/pto/arg_reduction_tile_native.pto` |

### 1.11 广播 / 扩展（expand）（17 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.texpands` | `TExpandsOp` | PTOOps.td:4747 | samples·tilelang-st·lit | `samples/Expands/expand.py`<br>`lit/pto/texpands_bf16.pto`<br>`lit/pto/expands_tile_native.pto`<br>…共 232 文件 |
| `pto.trowexpand` | `TRowExpandOp` | PTOOps.td:6021 | samples·tilelib-st·tilelang-st·lit | `samples/Rowexpand/rowexpand.py`<br>`tilelib-st/a5/trowexpand/`<br>`lit/vpto/trowexpand_tile_op_expand.pto`<br>…共 11 文件 |
| `pto.trowexpandadd` | `TRowExpandAddOp` | PTOOps.td:6146 | samples·tilelang-st·lit | `samples/Rowexpandadd/rowexpandadd.py`<br>`lit/pto/trowexpandadd_tmp_emitc.pto`<br>`lit/vpto/trowexpandadd_tile_op_expand.pto`<br>…共 13 文件 |
| `pto.trowexpandsub` | `TRowExpandSubOp` | PTOOps.td:6117 | samples·tilelang-st·lit | `samples/Rowexpandsub/rowexpandsub.py`<br>`lit/vpto/trowexpandsub_tile_op_expand.pto`<br>`lit/vpto/trowexpandsub_src0expand_tile_op_expand.pto`<br>…共 31 文件 |
| `pto.trowexpandmul` | `TRowExpandMulOp` | PTOOps.td:6085 | samples·tilelang-st·lit | `samples/Rowexpandmul/rowexpandmul.py`<br>`lit/vpto/trowexpandmul_tile_op_expand.pto`<br>`lit/pto/issue660_trowexpandmul_set_validshape_preserves_alloc_valid.pto`<br>…共 166 文件 |
| `pto.trowexpanddiv` | `TRowExpandDivOp` | PTOOps.td:6052 | samples·tilelang-st·lit | `samples/Rowexpanddiv/rowexpanddiv.py`<br>`lit/pto/trowexpanddiv_precision_emitc.pto`<br>`lit/vpto/trowexpanddiv_tile_op_expand.pto`<br>…共 37 文件 |
| `pto.trowexpandexpdif` | `TRowExpandExpdifOp` | PTOOps.td:6175 | samples·tilelang-st·lit | `samples/Rowexpandexpdif/rowexpandexpdif.py`<br>`lit/vpto/trowexpandexpdif_tile_op_expand.pto`<br>`lit/pto/rowexpand_tile_native.pto`<br>…共 7 文件 |
| `pto.trowexpandmax` | `TRowExpandMaxOp` | PTOOps.td:6204 | samples·tilelang-st·lit | `samples/Rowexpandmax/rowexpandmax.py`<br>`lit/vpto/trowexpandmax_tile_op_expand.pto`<br>`lit/pto/rowexpand_tile_native.pto`<br>…共 11 文件 |
| `pto.trowexpandmin` | `TRowExpandMinOp` | PTOOps.td:6233 | samples·tilelang-st·lit | `samples/Rowexpandmin/rowexpandmin.py`<br>`lit/vpto/trowexpandmin_tile_op_expand.pto`<br>`lit/pto/rowexpand_tile_native.pto`<br>…共 6 文件 |
| `pto.tcolexpand` | `TColExpandOp` | PTOOps.td:4068 | samples·tilelib-st·tilelang-st·lit | `samples/Colexpand/colexpand.py`<br>`tilelib-st/a5/tcolexpand/`<br>`lit/vpto/tcolexpand_tilelang.pto`<br>…共 24 文件 |
| `pto.tcolexpandadd` | `TColExpandAddOp` | PTOOps.td:4096 | samples·tilelang-st·lit | `samples/Colexpandadd/colexpandadd.py`<br>`lit/vpto/tcolexpandadd_tilelang.pto`<br>`lit/pto/colexpand_tile_native.pto`<br>…共 8 文件 |
| `pto.tcolexpandsub` | `TColExpandSubOp` | PTOOps.td:4184 | samples·tilelang-st·lit | `samples/Colexpandsub/colexpandsub.py`<br>`lit/vpto/tcolexpandsub_tilelang.pto`<br>`lit/vpto/tcolexpandsub_src0expand_tilelang.pto`<br>…共 7 文件 |
| `pto.tcolexpandmul` | `TColExpandMulOp` | PTOOps.td:4125 | samples·tilelang-st·lit | `samples/Colexpandmul/colexpandmul.py`<br>`lit/vpto/tcolexpandmul_tilelang.pto`<br>`lit/pto/colexpand_tile_native.pto`<br>…共 144 文件 |
| `pto.tcolexpanddiv` | `TColExpandDivOp` | PTOOps.td:4154 | samples·tilelib-st·tilelang-st·lit | `samples/Colexpanddiv/colexpanddiv.py`<br>`tilelib-st/a5/tcolexpanddiv/`<br>`lit/vpto/tcolexpanddiv_tilelang.pto`<br>…共 8 文件 |
| `pto.tcolexpandexpdif` | `TColExpandExpdifOp` | PTOOps.td:4213 | samples·tilelang-st·lit | `samples/Colexpandexpdif/colexpandexpdif.py`<br>`lit/vpto/tcolexpandexpdif_tilelang.pto`<br>`lit/pto/colexpand_tile_native.pto`<br>…共 8 文件 |
| `pto.tcolexpandmax` | `TColExpandMaxOp` | PTOOps.td:4242 | samples·tilelang-st·lit | `samples/Colexpandmax/colexpandmax.py`<br>`lit/vpto/tcolexpandmax_tilelang.pto`<br>`lit/pto/colexpand_tile_native.pto`<br>…共 5 文件 |
| `pto.tcolexpandmin` | `TColExpandMinOp` | PTOOps.td:4271 | samples·tilelang-st·lit | `samples/Colexpandmin/colexpandmin.py`<br>`lit/vpto/tcolexpandmin_tilelang.pto`<br>`lit/pto/colexpand_tile_native.pto`<br>…共 5 文件 |

### 1.12 比较 / 选择（6 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tcmp` | `TCmpOp` | PTOOps.td:4002 | samples·tilelang-st·lit·dsl-st·npu-val | `samples/Cmp/cmp.py`<br>`lit/vpto/expand_tile_op_tilelang_tcmp.pto`<br>`lit/pto/compare_tile_native.pto`<br>…共 30 文件 |
| `pto.tcmps` | `TCmpSOp` | PTOOps.td:4037 | samples·tilelang-st·lit·dsl-st·npu-val | `samples/Cmps/cmps.py`<br>`lit/vpto/expand_tile_op_tilelang_tcmps.pto`<br>`lit/pto/issue1606_tcmps_bf16_a3_invalid.pto`<br>…共 32 文件 |
| `pto.tsel` | `TSelOp` | PTOOps.td:6590 | samples·tilelang-st·lit | `samples/Sel/sel.py`<br>`lit/pto/tsel_bf16.pto`<br>`lit/pto/tsel_tmp_contract_a3_invalid.pto`<br>…共 13 文件 |
| `pto.tsels` | `TSelSOp` | PTOOps.td:6623 | samples·tilelang-st·lit | `samples/Sels/sels.py`<br>`lit/vpto/expand_tile_op_tsels_tilelang.pto`<br>`lit/pto/select_tile_native.pto`<br>…共 9 文件 |
| `pto.tci` | `TCIOp` | PTOOps.td:3944 | samples·tilelib-st·lit | `samples/Ci/ci.py`<br>`tilelib-st/a5/tci/`<br>`lit/pto/tci_i16_emitc.pto`<br>…共 78 文件 |
| `pto.ttri` | `TTriOp` | PTOOps.td:3973 | samples·tilelib-st·lit | `samples/TTri/ttri.py`<br>`tilelib-st/a5/ttri/`<br>`lit/pto/ttri_emitc.pto` |

### 1.13 类型转换 / 量化（5 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tcvt` | `TCvtOp` | PTOOps.td:4464 | samples·tilelib-st·tilelang-st·lit | `samples/Cvt/tcvt.py`<br>`tilelib-st/a5/tcvt/`<br>`lit/pto/tcvt_e2e.pto`<br>…共 310 文件 |
| `pto.tquant` | `TQuantOp` | PTOOps.td:5710 | samples·lit | `samples/Quant/quant.py`<br>`lit/pto/tquant.pto`<br>`lit/pto/tquant_e2e.pto`<br>…共 15 文件 |
| `pto.tquant.mx` | `TQuantMxOp` | PTOOps.td:5767 | samples·tilelang-st·lit | `samples/TquantMx/tquant_mx.pto`<br>`lit/pto/tquant_mxfp8_a5.pto`<br>`lit/pto/tquant_mx_a5_emitc.pto`<br>…共 11 文件 |
| `pto.tdequant` | `TDequantOp` | PTOOps.td:5822 | samples·tilelib-st·lit | `samples/Dequant/dequant.py`<br>`tilelib-st/a5/tdequant/`<br>`lit/pto/tdequant.pto`<br>…共 5 文件 |
| `pto.tget_scale_addr` | `TGetScaleAddrOp` | PTOOps.td:3495 | samples·lit | `samples/Gemvmx/gemvmx-pto.pto`<br>`lit/pto/tget_scale_addr_emitc.pto`<br>`lit/pto/tget_scale_addr_verify_invalid.pto`<br>…共 11 文件 |

### 1.14 填充 / 补齐（1 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tfillpad` | `TFillPadOp` | PTOOps.td:4955 | samples·tilelang-st·lit | `samples/Fillpad/fillpad.py`<br>`lit/pto/tfillpad_a5_mat_pipe_selection.pto`<br>`lit/pto/tfillpad_plan_memory_inference.pto`<br>…共 49 文件 |

### 1.15 数据重排 / 索引访存（14 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tgather` | `TGatherOp` | PTOOps.td:5003 | samples·tilelib-st·lit | `samples/Gather/gather.py`<br>`tilelib-st/a5/tgather/`<br>`lit/pto/tgather_a2a3_index_emitc.pto`<br>…共 83 文件 |
| `pto.tgatherb` | `TGatherBOp` | PTOOps.td:5047 | samples·tilelib-st·lit | `samples/Gatherb/gatherb.py`<br>`tilelib-st/a5/tgatherb/`<br>`lit/pto/tgatherb_generic_sync.pto`<br>…共 10 文件 |
| `pto.tscatter` | `TScatterOp` | PTOOps.td:6527 | samples·tilelib-st·lit | `samples/Scatter/scatter.py`<br>`tilelib-st/a5/tscatter/`<br>`lit/pto/tscatter_indexed_a5_emitc.pto`<br>…共 10 文件 |
| `pto.mgather` | `MGatherOp` | PTOOps.td:3314 | samples·lit | `samples/Mgather/mgather.py`<br>`lit/pto/mgather_gm2l1_emitc.pto`<br>`lit/pto/mgather_gm2l1_sync_a5.pto`<br>…共 25 文件 |
| `pto.mscatter` | `MScatterOp` | PTOOps.td:3526 | samples·lit | `samples/Mscatter/mscatter.py`<br>`lit/pto/mscatter_missing_coalesce_verify_invalid.pto`<br>`lit/pto/mgather_mscatter_tile_native.pto`<br>…共 11 文件 |
| `pto.tconcat` | `TConcatOp` | PTOOps.td:3841 | samples·tilelib-st·lit | `samples/Concat/concat.py`<br>`tilelib-st/a5/tconcat/`<br>`lit/pto/tconcatidx.pto`<br>…共 15 文件 |
| `pto.tconcatidx` | `TConcatidxOp` | PTOOps.td:3877 | samples·lit | `samples/TConcatIdx/tconcatidx.pto`<br>`lit/pto/tconcatidx.pto`<br>`lit/pto/tconcatidx_candidates_preserved.pto`<br>…共 5 文件 |
| `pto.tinterleave` | `TInterleaveOp` | PTOOps.td:6396 | samples | `samples/Interleave/tinterleave_runtime.py`<br>`samples/Interleave/tinterleave-pto.pto` |
| `pto.tdeinterleave` | `TDeInterleaveOp` | PTOOps.td:6434 | samples | `samples/DeInterleave/tdeinterleave_two_runtime.py`<br>`samples/DeInterleave/tdeinterleave-pto.pto`<br>`samples/DeInterleave/tdeinterleave_single_runtime.py` |
| `pto.tmrgsort` | `TMrgSortOp` | PTOOps.td:5250 | samples·tilelang-st·lit | `samples/Mrgsort/mrgsort.py`<br>`lit/pto/tmrgsort_variants_emitc.pto`<br>`lit/pto/tmrgsort_dst_wider_implicit_tmp.pto`<br>…共 16 文件 |
| `pto.tsort32` | `TSort32Op` | PTOOps.td:6771 | samples·tilelang-st·lit | `samples/Sort32/sort32.py`<br>`lit/pto/tsort32_emitc.pto`<br>`lit/pto/tsort32_dynamic_width_implicit_tmp.pto`<br>…共 11 文件 |
| `pto.textract` | `TExtractOp` | PTOOps.td:4775 | samples·tilelang-st·lit | `samples/Gemv/gemv.py`<br>`lit/pto/textract_forms_emitc.pto`<br>`lit/pto/textract_a5_fp8_verify.pto`<br>…共 193 文件 |
| `pto.tinsert` | `TInsertOp` | PTOOps.td:4867 | samples·tilelang-st·lit | `samples/TInsert/tinsert.py`<br>`lit/pto/tinsert_forms_emitc.pto`<br>`lit/pto/tinsert_a5_modes_emitc.pto`<br>…共 33 文件 |
| `pto.treshape` | `TReshapeOp` | PTOOps.td:5992 | samples·tilelang-st·lit | `samples/Reshape/reshape.py`<br>`lit/pto/treshape_tile_native_arg.pto`<br>`lit/pto/treshape_static_valid_shape_emitc.pto`<br>…共 15 文件 |

### 1.16 杂项 / 调试（4 个）

| 助记符 | C++ 类 | 定义位置 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|
| `pto.tsetval` | `TSetValOp` | PTOOps.td:3406 | samples·lit | `samples/TileSetGetValue/tileSetGetValue.py`<br>`lit/pto/tile_scalar_access_tile_native.pto`<br>`samples/DeepseekV4DecodeA3/kernels/aiv/route_hash.pto`<br>…共 29 文件 |
| `pto.tgetval` | `TGetValOp` | PTOOps.td:3438 | samples·lit | `samples/TileSetGetValue/tileSetGetValue.py`<br>`lit/pto/tgetval_level3_index_cast_regression.pto`<br>`lit/pto/issue1478_scalar_addf_emitc.pto`<br>…共 21 文件 |
| `pto.tprint` | `TPrintOp` | PTOOps.td:7037 | samples·lit·vpto·bridge | `samples/Print/print.py`<br>`lit/pto/tprint_tmp_invalid.pto`<br>`lit/vpto/bridge/tprint_invalid.pto`<br>…共 24 文件 |
| `pto.tdump` | `TensorDumpOp` | PTOOps.td:858 | lit·vpto | `lit/vpto/tdump_5d.pto`<br>`lit/pto/tdump_invalid.pto`<br>`vpto/cases/kernels/tdump-vec/main.cpp`<br>…共 34 文件 |

### ⚠️ 仅有 lit（IR 检查）、无运行时层测例的 TileOp

运行时层 = `samples` / `tilelib-st` / `tilelang-st`(dump) / `vpto`。以下 op 目前只被 lit 覆盖：

- `pto.taddrelu` — Fused elementwise add + ReLU of two tiles
- `pto.talloc` — Bind a declared GlobalTensor descriptor to the next producer FIFO slot
- `pto.tsync` — Direct TSYNC mapping (variadic operands)

> 说明：`talloc/tpush/tpop/tfree` 是 A2/A3 风格 tile 生命周期伪 op（A5 对应 `*_to_aiv/_from_aic` 系列，已有 `TPushTPop`/`TPipe` 样例）；
> 这些 op 若在当前 target 由前端展开，lit 层即为其正确测试位置。

---

## 二、Tile 域辅助 op（非 TileOpInterface，共 28 个）

不挂 `TileOpInterface`，但属于 tile 编程面：GM 视图/分区、tile 分配、tile↔UB 地址桥接、量化配置、事件/局部数组。

| 助记符 | C++ 类 | 定义位置 | 说明 | 测试层 | 测例示例（相对 test/） |
|---|---|---|---|---|---|
| `pto.addptr` | `AddPtrOp` | PTOOps.td:160 | Add an element offset to a !pto.ptr | samples·lit·vpto·dsl-st·dsl·comm | `samples/AddPtr/addptr.py`<br>`lit/pto/ptr_scalar_addptr_emitc.pto`<br>`lit/pto/insert_sync_addptr_alias.pto`<br>…共 103 文件 |
| `pto.castptr` | `CastPtrOp` | PTOOps.td:185 | Cast between integer and !pto.ptr, or between !pto.ptr types | tilelib-st·tilelang-st·lit·vpto·dsl-st·kernel-test·dsl | `lit/pto/castptr_verify.pto`<br>`lit/pto/castptr_emitc_tile_and_int.pto`<br>`tilelang_st/npu/a5/src/st/testcase/tgemv_mx/tgemv_mx.pto`<br>…共 524 文件 |
| `pto.bitcast` | `BitcastOp` | PTOOps.td:537 | Bitwise reinterpretation of a value or tile buffer view | samples·lit | `samples/Reshape/bitcast_dtype_alias.py`<br>`lit/pto/bitcast_tile_native.pto`<br>`lit/pto/mx_noalias_plan_memory.pto`<br>…共 10 文件 |
| `pto.make_tensor_view` | `MakeTensorViewOp` | PTOOps.td:210 | Wrap a pointer as a tensor_view descriptor (no allocation, no copy). | samples·tilelib-st·tilelang-st·lit·vpto·dsl-st·dsl | `samples/Ci/ci.py`<br>`lit/pto/inttoptr_make_tensor_view_invalid.pto`<br>`lit/pto/tcvt_e2e.pto`<br>…共 1293 文件 |
| `pto.partition_view` | `PartitionViewOp` | PTOOps.td:230 | Partition a tensor view into a smaller logical view (logical slicing). | samples·tilelang-st·lit·vpto·dsl-st·dsl | `samples/Ci/ci.py`<br>`lit/pto/issue995_cmo_partition_view_emitc.pto`<br>`lit/pto/mgather_gm2l1_partition_view_emitc.pto`<br>…共 1236 文件 |
| `pto.get_tensor_view_dim` | `GetTensorViewDimOp` | PTOOps.td:261 | Get the size of a dimension of a tensor_view. | lit | `lit/pto/gm_tensor_view_direct_backend.pto`<br>`lit/pto/gm_tensor_view_native_through_sync.pto`<br>`lit/pto/emitc_memref_cast_from_dynamic_subview.pto` |
| `pto.get_tensor_view_stride` | `GetTensorViewStrideOp` | PTOOps.td:283 | Get the stride of a dimension of a tensor_view. | lit | `lit/pto/gm_tensor_view_direct_backend.pto`<br>`lit/pto/gm_tensor_view_native_through_sync.pto`<br>`lit/pto/issue1575_fp4_partition_view_dma_stride_emitc.pto` |
| `pto.alloc_tile` | `AllocTileOp` | PTOOps.td:301 | Allocates a tile buffer (logical buffer). | samples·tilelib-st·tilelang-st·lit·vpto·dsl-st·dsl | `samples/Ci/ci.py`<br>`lit/pto/alloc_tile_addr_level12_reject.pto`<br>`lit/pto/alloc_tile_low_precision_valid.pto`<br>…共 1914 文件 |
| `pto.alloc_multi_tile` | `AllocMultiTileOp` | PTOOps.td:344 | Allocate an N-slot multi-buffer tile | lit | `lit/pto/alloc_multi_tile_addr_alignment_invalid.pto`<br>`lit/pto/sync_event_rotation.pto`<br>`lit/pto/multi_tile_n4_planmem_e2e.pto`<br>…共 25 文件 |
| `pto.multi_tile_get` | `MultiTileGetOp` | PTOOps.td:387 | Pick one physical slot of a multi_tile_buf | lit | `lit/pto/multi_tile_get_verify_slot.pto`<br>`lit/pto/multi_tile_get_dyn_slot_lowering.pto`<br>`lit/pto/sync_event_rotation.pto`<br>…共 20 文件 |
| `pto.subview` | `SubViewOp` | PTOOps.td:427 | Create a tile subview from a parent tile (logical size + valid dims). | samples·lit | `samples/Or/or.py`<br>`lit/pto/subview_validshape_guard.pto`<br>`lit/pto/subview_explicit_result_type.pto`<br>…共 192 文件 |
| `pto.set_validshape` | `SetValidShapeOp` | PTOOps.td:474 | Update valid row/col metadata in-place on a dynamic tile | samples·lit | `samples/SetValidShape/set_validshape.py`<br>`lit/pto/set_validshape_if.pto`<br>`lit/pto/set_validshape_local_lowering.pto`<br>…共 74 文件 |
| `pto.get_validshape` | `GetValidShapeOp` | PTOOps.td:502 | Read runtime valid row/col metadata from a tile | lit | `lit/pto/get_validshape_emitc.pto`<br>`lit/pto/validshape_tile_native.pto`<br>`lit/pto/ptodsl_tileop_dynamic_valid_shape.pto`<br>…共 5 文件 |
| `pto.declare_tile` | `DeclareTileOp` | PTOOps.td:2483 | Declare a tile value whose address will be assigned later | lit·vpto | `lit/pto/declare_tile_tile_native.pto`<br>`lit/pto/tpush_tpop_emitc.pto`<br>`vpto/cases/kernels/cube-tmatmul-bridge/kernel.pto`<br>…共 22 文件 |
| `pto.declare_global` | `DeclareGlobalOp` | PTOOps.td:2493 | Declare a GlobalTensor descriptor whose address is assigned later | lit | `lit/pto/tpush_tpop_globaltensor_frontend_a3.pto`<br>`lit/pto/tpush_tpop_globaltensor_internal_a3.pto`<br>`lit/pto/tpush_tpop_globaltensor_entry_mismatch_invalid.pto` |
| `pto.tassign` | `TAssignOp` | PTOOps.td:2528 | Rebind a declared tile handle to a runtime address | lit | `lit/pto/tassign_reject_memref.pto`<br>`lit/pto/tassign_tile_native_arg.pto`<br>`lit/pto/tassign_level3_loop_rebind.pto`<br>…共 5 文件 |
| `pto.set_quant_scalar` | `SetQuantScalarOp` | PTOOps.td:1919 | Set scalar quantization parameters for fixpipe producer | lit | `lit/pto/fixpipe_frontend_verify_missing_set_quant_scalar_a5.pto`<br>`lit/pto/fixpipe_basic_deqf16_nz2nd.pto`<br>`lit/pto/fixpipe_frontend_emitc_qs322bf16_a5.pto`<br>…共 20 文件 |
| `pto.set_quant_vector` | `SetQuantVectorOp` | PTOOps.td:1946 | Set vector quantization parameters for fixpipe producer | samples·lit | `samples/Movfp/movfp_fixpipe_reuse-pto.pto`<br>`lit/pto/fixpipe_frontend_verify_missing_set_quant_vector_a5.pto`<br>`lit/pto/fixpipe_frontend_verify_set_quant_vector_float_payload_a5.pto`<br>…共 20 文件 |
| `pto.make_prefetch_async_context` | `MakePrefetchAsyncContextOp` | PTOOps.td:731 | Build a PrefetchAsyncContext from GM workspace. | samples·lit | `samples/TPrefetchAsync/tprefetch_async_binding.py`<br>`lit/pto/tprefetch_async_emitc.pto`<br>`lit/pto/tprefetch_async_invalid_non_1d.pto`<br>…共 4 文件 |
| `pto.get_prefetch_async_session` | `GetPrefetchAsyncSessionOp` | PTOOps.td:754 | Project session from a PrefetchAsyncContext. | samples·lit | `samples/TPrefetchAsync/tprefetch_async_binding.py`<br>`lit/pto/tprefetch_async_emitc.pto`<br>`samples/TPrefetchAsync/tprefetch_async_binding.pto` |
| `pto.tprefetch_async` | `TPrefetchAsyncOp` | PTOOps.td:770 | Asynchronously prefetch global data into L2 cache via SDMA CMO. | samples·lit | `samples/TPrefetchAsync/tprefetch_async_binding.py`<br>`lit/pto/tprefetch_async_emitc.pto`<br>`lit/pto/tprefetch_async_invalid_non_1d.pto`<br>…共 4 文件 |
| `pto.declare_eventid_array` | `DeclareEventIdArrayOp` | PTOOps.td:2551 | Declare a manual-only local event-id array | lit | `lit/pto/eventid_array_no_cse.pto`<br>`lit/pto/eventid_array_dyn_sync.pto`<br>`lit/pto/eventid_array_get_set_get.pto` |
| `pto.eventid_array_get` | `EventIdArrayGetOp` | PTOOps.td:2566 | Read one element from a manual event-id array | lit | `lit/pto/eventid_array_get_set_get.pto`<br>`lit/pto/eventid_array_no_cse.pto`<br>`lit/pto/eventid_array_dyn_sync.pto` |
| `pto.eventid_array_set` | `EventIdArraySetOp` | PTOOps.td:2580 | Write one element into a manual event-id array | lit | `lit/pto/eventid_array_no_cse.pto`<br>`lit/pto/eventid_array_dyn_sync.pto`<br>`lit/pto/eventid_array_get_set_get.pto` |
| `pto.declare_local_array` | `DeclareLocalArrayOp` | PTOOps.td:2604 | Declare a stack-local statically-shaped array | samples·lit | `samples/DeepseekV4DecodeA5/kernels/aiv/dispatch_meta.pto`<br>`lit/pto/local_array_1d_emitc.pto`<br>`lit/pto/local_array_2d_emitc.pto`<br>…共 11 文件 |
| `pto.local_array_get` | `LocalArrayGetOp` | PTOOps.td:2621 | Read one element from a stack-local array | samples·lit | `samples/DeepseekV4DecodeA5/kernels/aiv/dispatch_meta.pto`<br>`lit/pto/local_array_get_rvalue_emitc.pto`<br>`lit/pto/issue713_local_array_get_snapshot.pto`<br>…共 11 文件 |
| `pto.timg2col` | `TImg2colOp` | PTOOps.td:2021 | Image-to-column transform from ConvTile to TileBuf | samples | `samples/Img2col/timg2col_runtime.py`<br>`samples/Img2col/timg2col-pto.pto`<br>`samples/ConvPipeline/conv_pipeline-pto.pto`<br>…共 4 文件 |
| `pto.tile_buf_addr` | `TileBufAddrOp` | VPTOOps.td:371 | Extract address from a tile_buf. | tilelang-st·lit·vpto·vpto_tilelang_inline_soft_divmod_fastpath.pto | `lit/pto/tile_buf_addr_tile_native_arg.pto`<br>`lit/vpto/vscatter_tile_buf_addr_view_to_memref.pto`<br>`tilelang_st/npu/a5/src/st/testcase/tgemv/tgemv.pto`<br>…共 42 文件 |

---

## 三、其余 op 家族（共 543 个）

### 3.1 VPTO 向量微指令（vecscope 内 vreg/mask 计算）（121 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.vlds` | `PTO_VldsOp` | VPTOOps.td:1617 | 348 | `lit/vpto/vlds_post_update_verify_invalid.pto`<br>…共 348 文件 |
| `pto.vldsx2` | `PTO_Vldsx2Op` | VPTOOps.td:1638 | 32 | `lit/vpto/vldsx2_vsldb_post_update_verify_invalid.pto`<br>…共 32 文件 |
| `pto.vldas` | `PTO_VldasOp` | VPTOOps.td:1659 | 17 | `lit/vpto/address_analysis.pto`<br>…共 17 文件 |
| `pto.init_align` | `PTO_InitAlignOp` | VPTOOps.td:1675 | 35 | `lit/vpto/soft_postupdate_spr_vstas.pto`<br>…共 35 文件 |
| `pto.sprclr` | `PTO_SprclrOp` | VPTOOps.td:1687 | 7 | `lit/vpto/a5_spr_store_vpto_llvm.pto`<br>…共 7 文件 |
| `pto.sprsti` | `PTO_SprstiOp` | VPTOOps.td:1699 | 9 | `lit/vpto/a5_spr_store_vpto_llvm.pto`<br>…共 9 文件 |
| `pto.sprsts` | `PTO_SprstsOp` | VPTOOps.td:1719 | 8 | `lit/vpto/a5_spr_store_vpto_llvm.pto`<br>…共 8 文件 |
| `pto.vldus` | `PTO_VldusOp` | VPTOOps.td:1739 | 24 | `lit/vpto/vldus_vstus_post_update_verify_invalid.pto`<br>…共 24 文件 |
| `pto.uvld` | `PTO_UvldOp` | VPTOOps.td:1764 | 1 | `lit/vpto/vreg_low_precision_memory_vpto_llvm.pto` |
| `pto.vbr` | `PTO_VbrOp` | VPTOOps.td:1781 | 31 | `lit/vpto/vpto_optimize_vcvt.pto`<br>…共 31 文件 |
| `pto.vdup` | `PTO_VdupOp` | VPTOOps.td:1792 | 86 | `lit/vpto/auto_vecscope_infer_revert_cse_vdup.pto`<br>…共 86 文件 |
| `pto.pset_b8` | `PTO_PsetB8Op` | VPTOOps.td:1807 | 41 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 41 文件 |
| `pto.pset_b16` | `PTO_PsetB16Op` | VPTOOps.td:1818 | 89 | `lit/vpto/vcadd_types.pto`<br>…共 89 文件 |
| `pto.pset_b32` | `PTO_PsetB32Op` | VPTOOps.td:1833 | 173 | `lit/vpto/vcadd_types.pto`<br>…共 173 文件 |
| `pto.pge_b8` | `PTO_PgeB8Op` | VPTOOps.td:1844 | 2 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 2 文件 |
| `pto.pge_b16` | `PTO_PgeB16Op` | VPTOOps.td:1855 | 4 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 4 文件 |
| `pto.pge_b32` | `PTO_PgeB32Op` | VPTOOps.td:1866 | 4 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 4 文件 |
| `pto.plt_b8` | `PTO_PltB8Op` | VPTOOps.td:1877 | 16 | `lit/vpto/tcvt_4to1_store_mask.pto`<br>…共 16 文件 |
| `pto.plt_b16` | `PTO_PltB16Op` | VPTOOps.td:1888 | 23 | `lit/vmi_new/vmi_to_vpto_abs.pto`<br>…共 23 文件 |
| `pto.plt_b32` | `PTO_PltB32Op` | VPTOOps.td:1899 | 106 | `lit/vpto/vpto_scheduler_cli.pto`<br>…共 106 文件 |
| `pto.pltm_b8` | `PTO_PltmB8Op` | VPTOOps.td:1910 | 1 | `lit/vpto/a5_extra_predicate_vpto_llvm.pto` |
| `pto.pltm_b16` | `PTO_PltmB16Op` | VPTOOps.td:1921 | 1 | `lit/vpto/a5_extra_predicate_vpto_llvm.pto` |
| `pto.pltm_b32` | `PTO_PltmB32Op` | VPTOOps.td:1932 | 1 | `lit/vpto/a5_extra_predicate_vpto_llvm.pto` |
| `pto.ppack` | `PTO_PpackOp` | VPTOOps.td:1954 | 7 | `lit/vmi_new/vmi_to_vpto_reduce_addf_store.pto`<br>…共 7 文件 |
| `pto.punpack` | `PTO_PunpackOp` | VPTOOps.td:1961 | 6 | `lit/vpto/vmi_fastmath_prefix_bf16_store.pto`<br>…共 6 文件 |
| `pto.pbitcast` | `PTO_PbitcastOp` | VPTOOps.td:1968 | 7 | `lit/vmi_new/vmi_group_sparse_compare.pto`<br>…共 7 文件 |
| `pto.pnot` | `PTO_PnotOp` | VPTOOps.td:1970 | 8 | `lit/vpto/vpto_mask_simplify_pand.pto`<br>…共 8 文件 |
| `pto.psel` | `PTO_PselOp` | VPTOOps.td:1984 | 1 | `dsl-st/legacy_micro_op_materialization_predicate_pdintlv_b16.py` |
| `pto.pand` | `PTO_PandOp` | VPTOOps.td:1999 | 16 | `lit/vpto/vpto_mask_simplify_pand.pto`<br>…共 16 文件 |
| `pto.por` | `PTO_PorOp` | VPTOOps.td:2014 | 6 | `lit/vpto/vpto_mask_simplify_pand.pto`<br>…共 6 文件 |
| `pto.pxor` | `PTO_PxorOp` | VPTOOps.td:2029 | 6 | `lit/vpto/vpto_mask_simplify_pand.pto`<br>…共 6 文件 |
| `pto.plds` | `PTO_PldsOp` | VPTOOps.td:2044 | 15 | `lit/vpto/address_analysis.pto`<br>…共 15 文件 |
| `pto.pldi` | `PTO_PldiOp` | VPTOOps.td:2064 | 12 | `lit/vpto/address_analysis.pto`<br>…共 12 文件 |
| `pto.psti` | `PTO_PstiOp` | VPTOOps.td:2084 | 14 | `lit/vpto/address_semantics_contract.pto`<br>…共 14 文件 |
| `pto.vabs` | `PTO_VabsOp` | VPTOOps.td:2104 | 10 | `lit/vmi_new/vmi_to_vpto_abs.pto`<br>…共 10 文件 |
| `pto.vexp` | `PTO_VexpOp` | VPTOOps.td:2132 | 13 | `lit/vpto/vexp_low_precision_verify_invalid.pto`<br>…共 13 文件 |
| `pto.vln` | `PTO_VlnOp` | VPTOOps.td:2133 | 4 | `lit/vmi_new/vmi_to_vpto_unary_math.pto`<br>…共 4 文件 |
| `pto.vsqrt` | `PTO_VsqrtOp` | VPTOOps.td:2134 | 4 | `lit/vmi_new/vmi_to_vpto_unary_math.pto`<br>…共 4 文件 |
| `pto.vneg` | `PTO_VnegOp` | VPTOOps.td:2135 | 12 | `lit/vpto/vneg_bf16_a5_expand.pto`<br>…共 12 文件 |
| `pto.vrelu` | `PTO_VreluOp` | VPTOOps.td:2136 | 6 | `lit/vpto/vrelu_verify_invalid.pto`<br>…共 6 文件 |
| `pto.vnot` | `PTO_VnotOp` | VPTOOps.td:2137 | 4 | `lit/vmi_new/vmi_to_vpto_bitwise.pto`<br>…共 4 文件 |
| `pto.vcadd` | `PTO_VcaddOp` | VPTOOps.td:2138 | 40 | `lit/vpto/vcadd_types.pto`<br>…共 40 文件 |
| `pto.vcmax` | `PTO_VcmaxOp` | VPTOOps.td:2566 | 21 | `lit/vpto/vpto_combine_reductions.pto`<br>…共 21 文件 |
| `pto.vcmin` | `PTO_VcminOp` | VPTOOps.td:2574 | 10 | `lit/vpto/vpto_combine_reductions.pto`<br>…共 10 文件 |
| `pto.vcgadd` | `PTO_VcgaddOp` | VPTOOps.td:2582 | 37 | `lit/vmi_new/vmi_to_vpto_group_reduce_vcgadd.pto`<br>…共 37 文件 |
| `pto.vcgmax` | `PTO_VcgmaxOp` | VPTOOps.td:2583 | 10 | `lit/vpto/vpto_pdintlv_allf_fold.pto`<br>…共 10 文件 |
| `pto.vcgmin` | `PTO_VcgminOp` | VPTOOps.td:2584 | 5 | `lit/vpto/vpto_combine_reductions.pto`<br>…共 5 文件 |
| `pto.vcpadd` | `PTO_VcpaddOp` | VPTOOps.td:2585 | 1 | `dsl-st/legacy_micro_op_reduction_vcadd.py` |
| `pto.chistv2` | `PTO_Chistv2Op` | VPTOOps.td:2603 | 7 | `lit/vmi_new/vmi_to_vpto_vchist.pto`<br>…共 7 文件 |
| `pto.dhistv2` | `PTO_Dhistv2Op` | VPTOOps.td:2604 | 11 | `lit/vmi_new/vmi_to_vpto_vdhist.pto`<br>…共 11 文件 |
| `pto.vadd` | `PTO_VaddOp` | VPTOOps.td:2642 | 80 | `lit/vpto/vadd_low_precision_verify_invalid.pto`<br>…共 80 文件 |
| `pto.vsub` | `PTO_VsubOp` | VPTOOps.td:2643 | 34 | `lit/vmi_new/vmi_to_vpto_iota.pto`<br>…共 34 文件 |
| `pto.vmul` | `PTO_VmulOp` | VPTOOps.td:2644 | 49 | `lit/vmi_new/vmi_to_vpto_vmuls.pto`<br>…共 49 文件 |
| `pto.vdiv` | `PTO_VdivOp` | VPTOOps.td:2645 | 34 | `lit/vpto/issue_1607_vdiv_unsigned.pto`<br>…共 34 文件 |
| `pto.vmax` | `PTO_VmaxOp` | VPTOOps.td:2654 | 28 | `lit/vmi_new/vmi_lower_vmaxs_vmins_integer.pto`<br>…共 28 文件 |
| `pto.vmin` | `PTO_VminOp` | VPTOOps.td:2655 | 15 | `lit/vmi_new/vmi_lower_vmaxs_vmins_integer.pto`<br>…共 15 文件 |
| `pto.vand` | `PTO_VandOp` | VPTOOps.td:2656 | 14 | `lit/vmi_new/vmi_to_vpto_abs.pto`<br>…共 14 文件 |
| `pto.vor` | `PTO_VorOp` | VPTOOps.td:2657 | 59 | `lit/vmi_new/vmi_to_vpto_truncf.pto`<br>…共 59 文件 |
| `pto.vxor` | `PTO_VxorOp` | VPTOOps.td:2658 | 7 | `lit/vmi_new/vmi_to_vpto_bitwise.pto`<br>…共 7 文件 |
| `pto.vmadd` | `PTO_VmaddOp` | VPTOOps.td:2676 | 2 | `lit/vpto/a5_extra_arith_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.vaddc` | `PTO_VaddcOp` | VPTOOps.td:2678 | 2 | `lit/vmi_new/vmi_to_vpto_carry.pto`<br>…共 2 文件 |
| `pto.vsubc` | `PTO_VsubcOp` | VPTOOps.td:2696 | 3 | `lit/vmi_new/vmi_to_vpto_carry.pto`<br>…共 3 文件 |
| `pto.vaddcs` | `PTO_VaddcsOp` | VPTOOps.td:2714 | 2 | `lit/vmi_new/vmi_to_vpto_carry.pto`<br>…共 2 文件 |
| `pto.vsubcs` | `PTO_VsubcsOp` | VPTOOps.td:2733 | 3 | `lit/vmi_new/vmi_to_vpto_carry.pto`<br>…共 3 文件 |
| `pto.vshl` | `PTO_VshlOp` | VPTOOps.td:2752 | 7 | `lit/vmi_new/vmi_to_vpto_shli.pto`<br>…共 7 文件 |
| `pto.vshr` | `PTO_VshrOp` | VPTOOps.td:2753 | 9 | `lit/vmi_new/vmi_lower_vshrs_signedness.pto`<br>…共 9 文件 |
| `pto.vsel` | `PTO_VselOp` | VPTOOps.td:2755 | 73 | `lit/vpto/vsel_low_precision_verify_invalid.pto`<br>…共 73 文件 |
| `pto.vcmp` | `PTO_VcmpOp` | VPTOOps.td:2770 | 29 | `lit/vmi_new/vmi_to_vpto_small_vl_vsel_vcmps.pto`<br>…共 29 文件 |
| `pto.vcmps` | `PTO_VcmpsOp` | VPTOOps.td:2786 | 17 | `lit/vpto/expand_tile_op_tilelang_trem.pto`<br>…共 17 文件 |
| `pto.pdintlv_b8` | `PTO_PdintlvB8Op` | VPTOOps.td:2820 | 5 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 5 文件 |
| `pto.pdintlv_b16` | `PTO_PdintlvB16Op` | VPTOOps.td:2822 | 5 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 5 文件 |
| `pto.pdintlv_b32` | `PTO_PdintlvB32Op` | VPTOOps.td:2824 | 8 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 8 文件 |
| `pto.pintlv_b8` | `PTO_PintlvB8Op` | VPTOOps.td:2827 | 3 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 3 文件 |
| `pto.pintlv_b16` | `PTO_PintlvB16Op` | VPTOOps.td:2829 | 6 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 6 文件 |
| `pto.pintlv_b32` | `PTO_PintlvB32Op` | VPTOOps.td:2831 | 7 | `lit/vpto/vpto_mask_simplify.pto`<br>…共 7 文件 |
| `pto.vtrc` | `PTO_VtrcOp` | VPTOOps.td:2865 | 7 | `lit/vpto/expand_tile_op_tilelang_trem.pto`<br>…共 7 文件 |
| `pto.vcvt` | `PTO_VcvtOp` | VPTOOps.td:2877 | 147 | `lit/vpto/vcvt_part_modes_vpto_llvm.pto`<br>…共 147 文件 |
| `pto.vmulscvt` | `PTO_VmulscvtOp` | VPTOOps.td:2891 | 4 | `lit/vpto/vmulscvt_vpto_llvm.pto`<br>…共 4 文件 |
| `pto.vbitcast` | `PTO_VbitcastOp` | VPTOOps.td:2908 | 65 | `lit/vpto/vbitcast_vpto_llvm.pto`<br>…共 65 文件 |
| `pto.vci` | `PTO_VciOp` | VPTOOps.td:2921 | 28 | `lit/vpto/vci_bf16_verify_invalid.pto`<br>…共 28 文件 |
| `pto.vbitsort` | `PTO_VbitsortOp` | VPTOOps.td:2935 | 2 | `lit/vpto/expand_tile_op_tilelang_tsort32.pto`<br>…共 2 文件 |
| `pto.vmrgsort4` | `PTO_Vmrgsort4Op` | VPTOOps.td:2955 | 2 | `lit/vpto/expand_tile_op_tilelang_tmrgsort.pto`<br>…共 2 文件 |
| `pto.vgather2` | `PTO_Vgather2Op` | VPTOOps.td:2977 | 24 | `lit/vpto/vgather2_u16_vpto_llvm.pto`<br>…共 24 文件 |
| `pto.vgatherb` | `PTO_VgatherbOp` | VPTOOps.td:3000 | 2 | `tilelib-st/a5/tgatherb/case.py`<br>…共 2 文件 |
| `pto.vgather2_bc` | `PTO_Vgather2BcOp` | VPTOOps.td:3019 | 10 | `lit/vmi_new/vmi_to_vpto_gather.pto`<br>…共 10 文件 |
| `pto.vmuls` | `PTO_VmulsOp` | VPTOOps.td:3036 | 19 | `lit/vpto/vmuls_low_precision_verify_invalid.pto`<br>…共 19 文件 |
| `pto.vadds` | `PTO_VaddsOp` | VPTOOps.td:3037 | 31 | `lit/vmi_new/vmi_to_vpto_iota.pto`<br>…共 31 文件 |
| `pto.vmaxs` | `PTO_VmaxsOp` | VPTOOps.td:3038 | 4 | `lit/vmi_new/vmi_lower_vmaxs_vmins_integer.pto`<br>…共 4 文件 |
| `pto.vmins` | `PTO_VminsOp` | VPTOOps.td:3039 | 4 | `lit/vmi_new/vmi_lower_vmaxs_vmins_integer.pto`<br>…共 4 文件 |
| `pto.vlrelu` | `PTO_VlreluOp` | VPTOOps.td:3040 | 3 | `lit/vpto/expand_tile_op_tlrelu_tilelang.pto`<br>…共 3 文件 |
| `pto.vshls` | `PTO_VshlsOp` | VPTOOps.td:3041 | 7 | `lit/vpto/expand_tile_op_tilelang_tshls.pto`<br>…共 7 文件 |
| `pto.vshrs` | `PTO_VshrsOp` | VPTOOps.td:3042 | 13 | `lit/vmi_new/vmi_lower_vshrs_signedness.pto`<br>…共 13 文件 |
| `pto.vsts` | `PTO_VstsOp` | VPTOOps.td:3044 | 409 | `lit/vpto/vsts_1pt_mask_vpto_llvm.pto`<br>…共 409 文件 |
| `pto.vscatter` | `PTO_VscatterOp` | VPTOOps.td:3066 | 12 | `lit/vpto/vscatter_verify_invalid.pto`<br>…共 12 文件 |
| `pto.psts` | `PTO_PstsOp` | VPTOOps.td:3085 | 27 | `lit/vpto/a5_extra_arith_vpto_llvm.pto`<br>…共 27 文件 |
| `pto.vselr` | `PTO_VselrOp` | VPTOOps.td:3652 | 25 | `lit/vmi_new/vmi_to_vpto_vselr.pto`<br>…共 25 文件 |
| `pto.vsqz` | `PTO_VsqzOp` | VPTOOps.td:3666 | 9 | `lit/vmi_new/vmi_to_vpto_compress.pto`<br>…共 9 文件 |
| `pto.vusqz` | `PTO_VusqzOp` | VPTOOps.td:3668 | 6 | `lit/vmi_new/vmi_to_vpto_active_prefix_index.pto`<br>…共 6 文件 |
| `pto.vpack` | `PTO_VpackOp` | VPTOOps.td:3682 | 26 | `lit/vmi_new/vmi_group_execution_paths.pto`<br>…共 26 文件 |
| `pto.vsunpack` | `PTO_VsunpackOp` | VPTOOps.td:3696 | 2 | `lit/vmi_new/vmi_to_vpto_group_slot_integer_unpack.pto`<br>…共 2 文件 |
| `pto.vzunpack` | `PTO_VzunpackOp` | VPTOOps.td:3710 | 15 | `lit/vpto/vpto_optimize_vcvt.pto`<br>…共 15 文件 |
| `pto.vselrv2` | `PTO_Vselrv2Op` | VPTOOps.td:3724 | 0 |  |
| `pto.vintlv` | `PTO_VintlvOp` | VPTOOps.td:3738 | 48 | `lit/vmi_new/vmi_to_vpto_vintlv.pto`<br>…共 48 文件 |
| `pto.vdintlv` | `PTO_VdintlvOp` | VPTOOps.td:3752 | 49 | `lit/vmi_new/vmi_to_vpto_vdintlv.pto`<br>…共 49 文件 |
| `pto.vintlvv2` | `PTO_Vintlvv2Op` | VPTOOps.td:3766 | 0 |  |
| `pto.vdintlvv2` | `PTO_Vdintlvv2Op` | VPTOOps.td:3781 | 0 |  |
| `pto.vmull` | `PTO_VmullOp` | VPTOOps.td:3796 | 5 | `lit/vmi_new/vmi_layout_assignment_vmull.pto`<br>…共 5 文件 |
| `pto.vmula` | `PTO_VmulaOp` | VPTOOps.td:3811 | 8 | `lit/vpto/vmula_i8_invalid.pto`<br>…共 8 文件 |
| `pto.vprelu` | `PTO_VpreluOp` | VPTOOps.td:3858 | 4 | `lit/vpto/expand_tile_op_tprelu_tilelang.pto`<br>…共 4 文件 |
| `pto.vexpdif` | `PTO_VexpdifOp` | VPTOOps.td:3859 | 18 | `lit/vpto/vexpdif_part_invalid.pto`<br>…共 18 文件 |
| `pto.vaxpy` | `PTO_VaxpyOp` | VPTOOps.td:3885 | 2 | `lit/vmi_new/vmi_to_vpto_unified_masked_compute.pto`<br>…共 2 文件 |
| `pto.vaddreluconv` | `PTO_VaddreluconvOp` | VPTOOps.td:3901 | 0 |  |
| `pto.vmulconv` | `PTO_VmulconvOp` | VPTOOps.td:3915 | 0 |  |
| `pto.vstsx2` | `PTO_Vstsx2Op` | VPTOOps.td:3929 | 22 | `lit/vmi_new/vmi_layout_fold_store.pto`<br>…共 22 文件 |
| `pto.vsldb` | `PTO_VsldbOp` | VPTOOps.td:3949 | 41 | `lit/vpto/vsldb_typed_callee_vpto_llvm.pto`<br>…共 41 文件 |
| `pto.vsstb` | `PTO_VsstbOp` | VPTOOps.td:3970 | 27 | `lit/vpto/vsstb_typed_callee_vpto_llvm.pto`<br>…共 27 文件 |
| `pto.vstas` | `PTO_VstasOp` | VPTOOps.td:3991 | 28 | `lit/vpto/soft_postupdate_spr_vstas.pto`<br>…共 28 文件 |
| `pto.vstar` | `PTO_VstarOp` | VPTOOps.td:4010 | 7 | `lit/vpto/soft_postupdate_vldus_vstus.pto`<br>…共 7 文件 |
| `pto.pstu` | `PTO_PstuOp` | VPTOOps.td:4031 | 1 | `dsl-st/legacy_micro_op_predicate_load_store_pstu.py` |
| `pto.vstus` | `PTO_VstusOp` | VPTOOps.td:4048 | 41 | `lit/vpto/vstus_mixed_types_vpto_llvm.pto`<br>…共 41 文件 |
| `pto.vstur` | `PTO_VsturOp` | VPTOOps.td:4069 | 6 | `lit/vmi_new/vmi_to_vpto_compress_store.pto`<br>…共 6 文件 |

### 3.2 VPTO MTE 数据搬运微指令（35 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.copy_gm_to_ubuf` | `PTO_CopyGmToUbufOp` | VPTOOps.td:1107 | 13 | `lit/vpto/ub/a5/copy_gm_ub.pto`<br>…共 13 文件 |
| `pto.mte_gm_ub` | `PTO_MteGmUbOp` | VPTOOps.td:1137 | 284 | `lit/vpto/ub/a5/mte_gm_ub.pto`<br>…共 284 文件 |
| `pto.copy_ubuf_to_ubuf` | `PTO_CopyUbufToUbufOp` | VPTOOps.td:1186 | 1 | `lit/vpto/vpto_scheduler_dependencies.pto` |
| `pto.copy_cbuf_to_ubuf` | `PTO_CopyCbufToUbufOp` | VPTOOps.td:1208 | 3 | `lit/vpto/cube_store_dma_copy.pto`<br>…共 3 文件 |
| `pto.copy_ubuf_to_cbuf` | `PTO_CopyUbufToCbufOp` | VPTOOps.td:1233 | 6 | `lit/vpto/cube_store_dma_copy.pto`<br>…共 6 文件 |
| `pto.mte_ub_ub` | `PTO_MteUbUbOp` | VPTOOps.td:1258 | 1 | `dsl-st/legacy_micro_op_vector_load_store_dma_copy_rearrange.py` |
| `pto.mte_ub_l1` | `PTO_MteUbL1Op` | VPTOOps.td:1283 | 2 | `lit/vpto/cube_store_dma_copy.pto`<br>…共 2 文件 |
| `pto.copy_gm_to_cbuf` | `PTO_CopyGmToCbufOp` | VPTOOps.td:1309 | 1 | `dsl-st/legacy_micro_op_cube_matmul_cube_load_frac_layouts.py` |
| `pto.copy_gm_to_cbuf_multi_nd2nz` | `PTO_CopyGmToCbufMultiNd2NzOp` | VPTOOps.td:1327 | 5 | `lit/vpto/cube_load_frac.pto`<br>…共 5 文件 |
| `pto.copy_gm_to_cbuf_multi_dn2nz` | `PTO_CopyGmToCbufMultiDn2NzOp` | VPTOOps.td:1351 | 3 | `lit/vpto/cube_load_frac.pto`<br>…共 3 文件 |
| `pto.copy_cbuf_to_bt` | `PTO_CopyCbufToBtOp` | VPTOOps.td:1375 | 1 | `lit/vpto/cube/cube_bridge_load_buffer_like.pto` |
| `pto.copy_cbuf_to_fbuf` | `PTO_CopyCbufToFbufOp` | VPTOOps.td:1395 | 2 | `lit/vpto/copy_cbuf_to_fbuf_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.load_cbuf_to_ca` | `PTO_LoadCbufToCaOp` | VPTOOps.td:1414 | 14 | `lit/vpto/load_cbuf_to_ca_vpto_llvm.pto`<br>…共 14 文件 |
| `pto.load_cbuf_to_ca_s4` | `PTO_LoadCbufToCaS4Op` | VPTOOps.td:1439 | 6 | `lit/pto/mte_l1_l0_fp4_s4_expand_a5.pto`<br>…共 6 文件 |
| `pto.load_cbuf_to_cb` | `PTO_LoadCbufToCbOp` | VPTOOps.td:1464 | 11 | `lit/vpto/load_cbuf_to_cb_vpto_llvm.pto`<br>…共 11 文件 |
| `pto.load_cbuf_to_cb_s4` | `PTO_LoadCbufToCbS4Op` | VPTOOps.td:1489 | 6 | `lit/pto/mte_l1_l0_fp4_s4_expand_a5.pto`<br>…共 6 文件 |
| `pto.load_cbuf_to_ca_mx` | `PTO_LoadCbufToCaMxOp` | VPTOOps.td:1514 | 2 | `lit/vpto/cube/cube_bridge_load_buffer_like.pto`<br>…共 2 文件 |
| `pto.load_cbuf_to_cb_mx` | `PTO_LoadCbufToCbMxOp` | VPTOOps.td:1539 | 2 | `lit/vpto/cube/cube_bridge_load_buffer_like.pto`<br>…共 2 文件 |
| `pto.copy_matrix_cc_to_gm` | `PTO_CopyMatrixCcToGmOp` | VPTOOps.td:1564 | 7 | `lit/vpto/acc_store_gm.pto`<br>…共 7 文件 |
| `pto.copy_matrix_cc_to_cbuf` | `PTO_CopyMatrixCcToCbufOp` | VPTOOps.td:1581 | 6 | `lit/vpto/acc_store.pto`<br>…共 6 文件 |
| `pto.copy_matrix_cc_to_ub` | `PTO_CopyMatrixCcToUbOp` | VPTOOps.td:1599 | 9 | `lit/vpto/copy_matrix_cc_to_ub_vpto_llvm.pto`<br>…共 9 文件 |
| `pto.copy_ubuf_to_gm` | `PTO_CopyUbufToGmOp` | VPTOOps.td:3106 | 14 | `lit/vpto/async_sdma_gm_gm.pto`<br>…共 14 文件 |
| `pto.mte_ub_gm` | `PTO_MteUbGmOp` | VPTOOps.td:3133 | 317 | `lit/vpto/mte_ub_gm_l2_cache_ctl.pto`<br>…共 317 文件 |
| `pto.mte_gm_l1` | `PTO_MteGmL1Op` | VPTOOps.td:3178 | 29 | `lit/vpto/mte_gm_l1_unaligned_without_pad_warns.pto`<br>…共 29 文件 |
| `pto.mte_l1_ub` | `PTO_MteL1UbOp` | VPTOOps.td:3218 | 12 | `lit/vpto/cube_store_dma_copy.pto`<br>…共 12 文件 |
| `pto.mte_l1_bt` | `PTO_MteL1BtOp` | VPTOOps.td:3259 | 9 | `lit/vpto/bias_load_vpto_llvm.pto`<br>…共 9 文件 |
| `pto.mte_l1_fb` | `PTO_MteL1FbOp` | VPTOOps.td:3286 | 7 | `lit/vpto/fp_load.pto`<br>…共 7 文件 |
| `pto.mte_gm_l1_frac` | `PTO_MteGmL1FracOp` | VPTOOps.td:3313 | 47 | `lit/vpto/cube_load_frac.pto`<br>…共 47 文件 |
| `pto.mte_l1_l0a` | `PTO_MteL1L0aOp` | VPTOOps.td:3444 | 58 | `lit/pto/mte_l1_l0_fp4_s4_expand_a5.pto`<br>…共 58 文件 |
| `pto.mte_l1_l0b` | `PTO_MteL1L0bOp` | VPTOOps.td:3470 | 53 | `lit/pto/mte_l1_l0_fp4_s4_expand_a5.pto`<br>…共 53 文件 |
| `pto.mte_l1_l0a_mx` | `PTO_MteL1L0aMxOp` | VPTOOps.td:3496 | 13 | `lit/vpto/cube/cube_bridge_load_buffer_like.pto`<br>…共 13 文件 |
| `pto.mte_l1_l0b_mx` | `PTO_MteL1L0bMxOp` | VPTOOps.td:3521 | 14 | `lit/vpto/cube/cube_bridge_load_buffer_like.pto`<br>…共 14 文件 |
| `pto.mte_l0c_l1` | `PTO_MteL0cL1Op` | VPTOOps.td:3546 | 18 | `lit/vpto/acc_store.pto`<br>…共 18 文件 |
| `pto.mte_l0c_gm` | `PTO_MteL0cGmOp` | VPTOOps.td:3580 | 63 | `lit/vpto/acc_store_gm.pto`<br>…共 63 文件 |
| `pto.mte_l0c_ub` | `PTO_MteL0cUbOp` | VPTOOps.td:3616 | 32 | `lit/vpto/acc_store_ub.pto`<br>…共 32 文件 |

### 3.3 VMI 语义层向量 op（`pto.vmi.*`）（126 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.vmi.constant` | `VMIConstantOp` | VMIOps.td:48 | 11 | `lit/vmi_new/vmi_to_vpto_constant.pto`<br>…共 11 文件 |
| `pto.vmi.broadcast` | `VMIBroadcastOp` | VMIOps.td:55 | 14 | `lit/vmi_new/vmi_layout_assignment_broadcast_remat.pto`<br>…共 14 文件 |
| `pto.vmi.iota` | `VMIIotaOp` | VMIOps.td:63 | 4 | `lit/vmi_new/vmi_iota_bf16_rejected.pto`<br>…共 4 文件 |
| `pto.vmi.group_iota` | `VMIGroupIotaOp` | VMIOps.td:82 | 2 | `lit/vmi_new/vmi_group_iota_bf16_rejected.pto`<br>…共 2 文件 |
| `pto.vmi.create_mask` | `VMICreateMaskOp` | VMIOps.td:100 | 160 | `lit/vmi_new/vmi_to_vpto_create_mask.pto`<br>…共 160 文件 |
| `pto.vmi.create_group_mask` | `VMICreateGroupMaskOp` | VMIOps.td:108 | 18 | `lit/vmi_new/vmi_create_group_mask_invalid.pto`<br>…共 18 文件 |
| `pto.vmi.constant_mask` | `VMIConstantMaskOp` | VMIOps.td:124 | 15 | `lit/vmi_new/vmi_to_vpto_constant_mask.pto`<br>…共 15 文件 |
| `pto.vmi.mask_and` | `VMIMaskAndOp` | VMIOps.td:131 | 3 | `lit/vmi_new/vmi_bitwise_interface_split.pto`<br>…共 3 文件 |
| `pto.vmi.mask_or` | `VMIMaskOrOp` | VMIOps.td:139 | 1 | `lit/vmi_new/vmi_layout_sink_materialization_mask.pto` |
| `pto.vmi.mask_xor` | `VMIMaskXOrOp` | VMIOps.td:147 | 0 |  |
| `pto.vmi.mask_not` | `VMIMaskNotOp` | VMIOps.td:155 | 2 | `lit/vmi_new/vmi_layout_sink_materialization_mask.pto`<br>…共 2 文件 |
| `pto.vmi.cmpf` | `VMICmpFOp` | VMIOps.td:163 | 7 | `lit/vmi_new/vmi_layout_assignment_widen.pto`<br>…共 7 文件 |
| `pto.vmi.cmpi` | `VMICmpIOp` | VMIOps.td:171 | 2 | `lit/vmi_new/vmi_layout_sink_materialization_binary.pto`<br>…共 2 文件 |
| `pto.vmi.select` | `VMISelectOp` | VMIOps.td:179 | 13 | `lit/vmi_new/vmi_layout_assignment_mask_select_store.pto`<br>…共 13 文件 |
| `pto.vmi.active_prefix_index` | `VMIActivePrefixIndexOp` | VMIOps.td:188 | 6 | `lit/vmi_new/vmi_to_vpto_active_prefix_index.pto`<br>…共 6 文件 |
| `pto.vmi.compress` | `VMICompressOp` | VMIOps.td:196 | 6 | `lit/vmi_new/vmi_to_vpto_compress.pto`<br>…共 6 文件 |
| `pto.vmi.compress_store` | `VMICompressStoreOp` | VMIOps.td:204 | 4 | `lit/vmi_new/vmi_to_vpto_compress_store.pto`<br>…共 4 文件 |
| `pto.vmi.reduce_addi` | `VMIReduceAddIOp` | VMIOps.td:213 | 2 | `lit/vmi_new/vmi_to_vpto_reduce_addi_i16_invalid.pto`<br>…共 2 文件 |
| `pto.vmi.reduce_addf` | `VMIReduceAddFOp` | VMIOps.td:222 | 2 | `lit/vmi_new/vmi_to_vpto_reduce_shape_invalid.pto`<br>…共 2 文件 |
| `pto.vmi.reduce_maxf` | `VMIReduceMaxFOp` | VMIOps.td:232 | 2 | `lit/vmi_new/vmi_to_vpto_reduce_maxf_tail_invalid.pto`<br>…共 2 文件 |
| `pto.vmi.reduce_minf` | `VMIReduceMinFOp` | VMIOps.td:241 | 1 | `lit/vmi_new/vmi_to_vpto_reduce_shape_invalid.pto` |
| `pto.vmi.reduce_maxi` | `VMIReduceMaxIOp` | VMIOps.td:250 | 0 |  |
| `pto.vmi.reduce_mini` | `VMIReduceMinIOp` | VMIOps.td:259 | 0 |  |
| `pto.vmi.group_reduce_addf` | `VMIGroupReduceAddFOp` | VMIOps.td:268 | 48 | `lit/vmi_new/vmi_layout_assignment_vexpdif.pto`<br>…共 48 文件 |
| `pto.vmi.group_reduce_maxf` | `VMIGroupReduceMaxFOp` | VMIOps.td:279 | 9 | `lit/vmi_new/vmi_layout_assignment_group_reduce_maxf_quant.pto`<br>…共 9 文件 |
| `pto.vmi.group_reduce_minf` | `VMIGroupReduceMinFOp` | VMIOps.td:289 | 3 | `lit/vmi_new/vmi_to_vpto_continuous_reduce_store.pto`<br>…共 3 文件 |
| `pto.vmi.group_reduce_addi` | `VMIGroupReduceAddIOp` | VMIOps.td:299 | 8 | `lit/vmi_new/vmi_native_group_sum_producer.pto`<br>…共 8 文件 |
| `pto.vmi.group_reduce_maxi` | `VMIGroupReduceMaxIOp` | VMIOps.td:309 | 1 | `lit/vmi_new/vmi_native_group_sum_producer.pto` |
| `pto.vmi.group_reduce_mini` | `VMIGroupReduceMinIOp` | VMIOps.td:319 | 1 | `lit/vmi_new/vmi_native_group_sum_producer.pto` |
| `pto.vmi.group_broadcast` | `VMIGroupBroadcastOp` | VMIOps.td:329 | 15 | `lit/vmi_new/vmi_to_vpto_group_broadcast_slots1_four_block.pto`<br>…共 15 文件 |
| `pto.vmi.vdhist` | `VMIVdhistOp` | VMIOps.td:355 | 15 | `lit/vmi_new/vdhist_verifier_reject_signed.pto`<br>…共 15 文件 |
| `pto.vmi.vchist` | `VMIVchistOp` | VMIOps.td:358 | 9 | `lit/vmi_new/vmi_to_vpto_vchist.pto`<br>…共 9 文件 |
| `pto.vmi.extf` | `VMIExtFOp` | VMIOps.td:361 | 32 | `lit/vmi_new/vmi_layout_gate_extf_support_invalid.pto`<br>…共 32 文件 |
| `pto.vmi.truncf` | `VMITruncFOp` | VMIOps.td:369 | 39 | `lit/vmi_new/vmi_layout_assignment_load_truncf.pto`<br>…共 39 文件 |
| `pto.vmi.fptosi` | `VMIFPToSIOp` | VMIOps.td:379 | 7 | `lit/vmi_new/vmi_vcvt_fptosi_lower_to_legacy_new.pto`<br>…共 7 文件 |
| `pto.vmi.fptoui` | `VMIFPToUIOp` | VMIOps.td:389 | 6 | `lit/vmi_new/vmi_fptoui_verifier.pto`<br>…共 6 文件 |
| `pto.vmi.sitofp` | `VMISIToFPOp` | VMIOps.td:399 | 2 | `lit/vpto/vmi_sitofp.pto`<br>…共 2 文件 |
| `pto.vmi.extsi` | `VMIExtSIOp` | VMIOps.td:407 | 11 | `lit/vmi_new/vmi_vcg_integer_native_layout.pto`<br>…共 11 文件 |
| `pto.vmi.extui` | `VMIExtUIOp` | VMIOps.td:415 | 14 | `lit/vmi_new/vmi_zero_gap_extui_load.pto`<br>…共 14 文件 |
| `pto.vmi.trunci` | `VMITruncIOp` | VMIOps.td:423 | 16 | `lit/vmi_new/vmi_trunci_saturate_bad_value_invalid.pto`<br>…共 16 文件 |
| `pto.vmi.bitcast` | `VMIBitcastOp` | VMIOps.td:432 | 15 | `lit/vmi_new/vmi_bitcast_width_changing_boundary.pto`<br>…共 15 文件 |
| `pto.vmi.load` | `VMILoadOp` | VMIOps.td:440 | 49 | `lit/vmi_new/vmi_layout_fold_load.pto`<br>…共 49 文件 |
| `pto.vmi.deinterleave_load` | `VMIDeinterleaveLoadOp` | VMIOps.td:448 | 2 | `lit/vmi_new/vmi_deinterleave_load_layout_propagation.pto`<br>…共 2 文件 |
| `pto.vmi.group_load` | `VMIGroupLoadOp` | VMIOps.td:456 | 15 | `lit/vmi_new/vmi_layout_assignment_group_load.pto`<br>…共 15 文件 |
| `pto.vmi.group_slot_load` | `VMIGroupSlotLoadOp` | VMIOps.td:465 | 11 | `lit/vmi_new/vmi_to_vpto_group_slot_load_support.pto`<br>…共 11 文件 |
| `pto.vmi.group_broadcast_load` | `VMIGroupBroadcastLoadOp` | VMIOps.td:474 | 5 | `lit/vmi_new/vmi_to_vpto_group_broadcast_load_e2b_b16_stride_invalid.pto`<br>…共 5 文件 |
| `pto.vmi.stride_load` | `VMIStrideLoadOp` | VMIOps.td:483 | 5 | `lit/vmi_new/vmi_to_vpto_stride_load.pto`<br>…共 5 文件 |
| `pto.vmi.masked_load` | `VMIMaskedLoadOp` | VMIOps.td:495 | 2 | `lit/vmi_new/vmi_to_vpto_memory_space_invalid.pto`<br>…共 2 文件 |
| `pto.vmi.gather` | `VMIGatherOp` | VMIOps.td:505 | 4 | `lit/vmi_new/vmi_layout_assignment_gather.pto`<br>…共 4 文件 |
| `pto.vmi.expand_load` | `VMIExpandLoadOp` | VMIOps.td:523 | 7 | `lit/vmi_new/vmi_layout_assignment_expand_load.pto`<br>…共 7 文件 |
| `pto.vmi.store` | `VMIStoreOp` | VMIOps.td:533 | 40 | `lit/vmi_new/vmi_layout_fold_store.pto`<br>…共 40 文件 |
| `pto.vmi.interleave_store` | `VMIInterleaveStoreOp` | VMIOps.td:541 | 1 | `lit/vmi_new/vmi_interleaved_memory_ops.pto` |
| `pto.vmi.group_store` | `VMIGroupStoreOp` | VMIOps.td:550 | 44 | `lit/vmi_new/vmi_to_vpto_group_store_vsstb.pto`<br>…共 44 文件 |
| `pto.vmi.masked_store` | `VMIMaskedStoreOp` | VMIOps.td:559 | 23 | `lit/vmi_new/vmi_lane_stride_masked_store.pto`<br>…共 23 文件 |
| `pto.vmi.stride_store` | `VMIStrideStoreOp` | VMIOps.td:568 | 2 | `lit/vmi_new/vmi_repeat_stride_removed_invalid.pto`<br>…共 2 文件 |
| `pto.vmi.scatter` | `VMIScatterOp` | VMIOps.td:581 | 4 | `lit/vmi_new/vmi_layout_assignment_scatter.pto`<br>…共 4 文件 |
| `pto.vmi.shuffle` | `VMIShuffleOp` | VMIOps.td:592 | 5 | `lit/vmi_new/vmi_shuffle_indices_invalid.pto`<br>…共 5 文件 |
| `pto.vmi.channel_split` | `VMIChannelSplitOp` | VMIOps.td:600 | 14 | `lit/vmi_new/vmi_to_vpto_channel_split_merge.pto`<br>…共 14 文件 |
| `pto.vmi.channel_merge` | `VMIChannelMergeOp` | VMIOps.td:607 | 14 | `lit/vmi_new/vmi_to_vpto_channel_merge4_contiguous.pto`<br>…共 14 文件 |
| `pto.vmi.vunzip` | `VMIVUnzipOp` | VMIOps.td:614 | 8 | `lit/vmi_new/vmi_to_vpto_vunzip_vzip.pto`<br>…共 8 文件 |
| `pto.vmi.vzip` | `VMIVZipOp` | VMIOps.td:666 | 6 | `lit/vmi_new/vmi_to_vpto_vunzip_vzip.pto`<br>…共 6 文件 |
| `pto.vmi.ensure_layout` | `VMIEnsureLayoutOp` | VMIOps.td:694 | 77 | `lit/vmi_new/vmi_to_vpto_ensure_layout_deint4.pto`<br>…共 77 文件 |
| `pto.vmi.ensure_mask_layout` | `VMIEnsureMaskLayoutOp` | VMIOps.td:702 | 30 | `lit/vmi_new/vmi_to_vpto_ensure_mask_layout.pto`<br>…共 30 文件 |
| `pto.vmi.ensure_mask_granularity` | `VMIEnsureMaskGranularityOp` | VMIOps.td:710 | 22 | `lit/vmi_new/vmi_to_vpto_ensure_mask_granularity_direct.pto`<br>…共 22 文件 |
| `pto.vmi.unpack` | `VMIUnpackOp` | VMIOps.td:718 | 132 | `lit/vmi_new/vmi_to_vpto_pack_unpack.pto`<br>…共 132 文件 |
| `pto.vmi.pack` | `VMIPackOp` | VMIOps.td:725 | 4 | `lit/vmi_new/vmi_pack_arity_invalid.pto`<br>…共 4 文件 |
| `pto.vmi.vbrc` | `VMIVbrcOp` | VMIOps.td:734 | 131 | `lit/vmi_new/vmi_layout_fold_store.pto`<br>…共 131 文件 |
| `pto.vmi.vci` | `VMIVciOp` | VMIOps.td:742 | 29 | `lit/vmi_new/vmi_vci_bf16_rejected.pto`<br>…共 29 文件 |
| `pto.vmi.pset` | `VMIPsetOp` | VMIOps.td:764 | 19 | `lit/vmi_new/vmi_to_vpto_scf_if.pto`<br>…共 19 文件 |
| `pto.vmi.pge` | `VMIPgeOp` | VMIOps.td:785 | 0 |  |
| `pto.vmi.plt` | `VMIPltOp` | VMIOps.td:808 | 0 |  |
| `pto.vmi.vadd` | `VMIVaddOp` | VMIOps.td:837 | 78 | `lit/vmi_new/vmi_to_vpto_add.pto`<br>…共 78 文件 |
| `pto.vmi.vaddc` | `VMIVaddcOp` | VMIOps.td:847 | 2 | `lit/vmi_new/vmi_to_vpto_carry.pto`<br>…共 2 文件 |
| `pto.vmi.vsubc` | `VMIVsubcOp` | VMIOps.td:867 | 2 | `lit/vmi_new/vmi_to_vpto_carry.pto`<br>…共 2 文件 |
| `pto.vmi.vaddcs` | `VMIVaddcsOp` | VMIOps.td:890 | 2 | `lit/vmi_new/vmi_to_vpto_carry.pto`<br>…共 2 文件 |
| `pto.vmi.vsubcs` | `VMIVsubcsOp` | VMIOps.td:916 | 2 | `lit/vmi_new/vmi_to_vpto_carry.pto`<br>…共 2 文件 |
| `pto.vmi.vsub` | `VMIVsubOp` | VMIOps.td:942 | 17 | `lit/vmi_new/vmi_to_vpto_sub_mul.pto`<br>…共 17 文件 |
| `pto.vmi.vmul` | `VMIVmulOp` | VMIOps.td:952 | 83 | `lit/vmi_new/vmi_to_vpto_sub_mul.pto`<br>…共 83 文件 |
| `pto.vmi.vdiv` | `VMIVdivOp` | VMIOps.td:962 | 25 | `lit/vmi_new/vmi_to_vpto_divf.pto`<br>…共 25 文件 |
| `pto.vmi.vmin` | `VMIVminOp` | VMIOps.td:972 | 15 | `lit/vmi_new/vmi_lower_vmaxs_vmins_integer.pto`<br>…共 15 文件 |
| `pto.vmi.vmax` | `VMIVmaxOp` | VMIOps.td:982 | 19 | `lit/vmi_new/vmi_lower_vmaxs_vmins_integer.pto`<br>…共 19 文件 |
| `pto.vmi.vneg` | `VMIVnegOp` | VMIOps.td:992 | 8 | `lit/vmi_new/vmi_ptoas_cli_integer_vneg_layout.pto`<br>…共 8 文件 |
| `pto.vmi.vabs` | `VMIVabsOp` | VMIOps.td:1002 | 37 | `lit/vmi_new/vmi_to_vpto_abs.pto`<br>…共 37 文件 |
| `pto.vmi.vsqrt` | `VMIVsqrtOp` | VMIOps.td:1027 | 3 | `lit/vmi_new/vmi_to_vpto_unary_math.pto`<br>…共 3 文件 |
| `pto.vmi.vexp` | `VMIVexpOp` | VMIOps.td:1037 | 7 | `lit/vmi_new/vmi_to_vpto_unary_math.pto`<br>…共 7 文件 |
| `pto.vmi.vln` | `VMIVlnOp` | VMIOps.td:1047 | 3 | `lit/vmi_new/vmi_to_vpto_unary_math.pto`<br>…共 3 文件 |
| `pto.vmi.vrelu` | `VMIVreluOp` | VMIOps.td:1057 | 4 | `lit/vmi_new/vmi_to_vpto_unary_math.pto`<br>…共 4 文件 |
| `pto.vmi.vand` | `VMIVandOp` | VMIOps.td:1067 | 22 | `lit/vmi_new/vmi_to_vpto_bitwise.pto`<br>…共 22 文件 |
| `pto.vmi.vor` | `VMIVorOp` | VMIOps.td:1077 | 5 | `lit/vmi_new/vmi_to_vpto_bitwise.pto`<br>…共 5 文件 |
| `pto.vmi.vxor` | `VMIVxorOp` | VMIOps.td:1087 | 4 | `lit/vmi_new/vmi_to_vpto_bitwise.pto`<br>…共 4 文件 |
| `pto.vmi.vshl` | `VMIVshlOp` | VMIOps.td:1097 | 17 | `lit/vmi_new/vmi_to_vpto_shli.pto`<br>…共 17 文件 |
| `pto.vmi.vshr` | `VMIVshrOp` | VMIOps.td:1107 | 17 | `lit/vmi_new/vmi_lower_vshrs_signedness.pto`<br>…共 17 文件 |
| `pto.vmi.vnot` | `VMIVnotOp` | VMIOps.td:1117 | 7 | `lit/vmi_new/vmi_to_vpto_bitwise.pto`<br>…共 7 文件 |
| `pto.vmi.andi` | `VMIAndIOp` | VMIOps.td:1137 | 2 | `lit/vmi_new/vmi_bitwise_interface_split.pto`<br>…共 2 文件 |
| `pto.vmi.ori` | `VMIOrIOp` | VMIOps.td:1147 | 0 |  |
| `pto.vmi.xori` | `VMIXOrIOp` | VMIOps.td:1157 | 0 |  |
| `pto.vmi.not` | `VMINotOp` | VMIOps.td:1167 | 2 | `lit/vmi_new/vmi_bitwise_interface_split.pto`<br>…共 2 文件 |
| `pto.vmi.vcmp` | `VMIVcmpOp` | VMIOps.td:1177 | 25 | `lit/vmi_new/vmi_to_vpto_small_vl_vsel_vcmps.pto`<br>…共 25 文件 |
| `pto.vmi.vcmps` | `VMIVcmpsOp` | VMIOps.td:1211 | 9 | `lit/vmi_new/vmi_to_vpto_small_vl_vsel_vcmps.pto`<br>…共 9 文件 |
| `pto.vmi.vadds` | `VMIAddSOp` | VMIOps.td:1270 | 10 | `lit/vmi_new/vmi_to_vpto_iota_group2.pto`<br>…共 10 文件 |
| `pto.vmi.vmuls` | `VMIMulSOp` | VMIOps.td:1271 | 5 | `lit/vmi_new/vmi_to_vpto_vmuls.pto`<br>…共 5 文件 |
| `pto.vmi.vmaxs` | `VMIMaxSOp` | VMIOps.td:1272 | 2 | `lit/vmi_new/vmi_lower_vmaxs_vmins_integer.pto`<br>…共 2 文件 |
| `pto.vmi.vmins` | `VMIMinSOp` | VMIOps.td:1273 | 2 | `lit/vmi_new/vmi_lower_vmaxs_vmins_integer.pto`<br>…共 2 文件 |
| `pto.vmi.vshls` | `VMIShlSOp` | VMIOps.td:1274 | 3 | `lit/vmi_new/vmi_to_vpto_vector_scalar_ops.pto`<br>…共 3 文件 |
| `pto.vmi.vshrs` | `VMIShrSOp` | VMIOps.td:1275 | 4 | `lit/vmi_new/vmi_lower_vshrs_signedness.pto`<br>…共 4 文件 |
| `pto.vmi.vsel` | `VMIvSelOp` | VMIOps.td:1277 | 47 | `lit/vmi_new/vmi_to_vpto_small_vl_vsel_vcmps.pto`<br>…共 47 文件 |
| `pto.vmi.vcadd` | `VMIvcaddOp` | VMIOps.td:1301 | 130 | `lit/vmi_new/vmi_vcadd_element_type_invalid.pto`<br>…共 130 文件 |
| `pto.vmi.vcmax` | `VMIvcmaxOp` | VMIOps.td:1332 | 54 | `lit/vmi_new/vmi_pmode_merge_invalid.pto`<br>…共 54 文件 |
| `pto.vmi.vcmin` | `VMIvcminOp` | VMIOps.td:1358 | 14 | `lit/vmi_new/vmi_pmode_merge_invalid.pto`<br>…共 14 文件 |
| `pto.vmi.vcvt` | `VMICvtOp` | VMIOps.td:1384 | 243 | `lit/vmi_new/issue_585_vcvt_round_z.pto`<br>…共 243 文件 |
| `pto.vmi.vload` | `VMIvLoadOp` | VMIOps.td:1427 | 281 | `lit/vmi_new/vmi_ptoas_cli_licm.pto`<br>…共 281 文件 |
| `pto.vmi.vstore` | `VMIvStoreOp` | VMIOps.td:1460 | 321 | `lit/vmi_new/vmi_ptoas_cli_licm.pto`<br>…共 321 文件 |
| `pto.vmi.vsstb` | `VMIVsstbOp` | VMIOps.td:1496 | 2 | `lit/vmi_new/vmi_vsstb.pto`<br>…共 2 文件 |
| `pto.vmi.vinterpret_cast` | `VMIVinterpretCastOp` | VMIOps.td:1511 | 58 | `lit/vmi_new/vmi_to_vpto_bitcast.pto`<br>…共 58 文件 |
| `pto.vmi.vselr` | `VMIVselrOp` | VMIOps.td:1519 | 4 | `lit/vmi_new/vmi_to_vpto_vselr.pto`<br>…共 4 文件 |
| `pto.vmi.vintlv` | `VMIVintlvOp` | VMIOps.td:1546 | 15 | `lit/vmi_new/vmi_to_vpto_vintlv.pto`<br>…共 15 文件 |
| `pto.vmi.vdintlv` | `VMIVdintlvOp` | VMIOps.td:1564 | 17 | `lit/vmi_new/vmi_to_vpto_vdintlv.pto`<br>…共 17 文件 |
| `pto.vmi.vgather` | `VMIVgatherOp` | VMIOps.td:1582 | 32 | `lit/vmi_new/vmi_to_vpto_gather.pto`<br>…共 32 文件 |
| `pto.vmi.vgatherb` | `VMIVgatherbOp` | VMIOps.td:1612 | 2 | `lit/vmi_new/vmi_gather_scatter_address_space_invalid.pto`<br>…共 2 文件 |
| `pto.vmi.vscatter` | `VMIVscatterOp` | VMIOps.td:1635 | 10 | `lit/vmi_new/vmi_to_vpto_scatter.pto`<br>…共 10 文件 |
| `pto.vmi.vexpdif` | `VMIVexpdifOp` | VMIOps.td:1663 | 6 | `lit/vmi_new/vmi_to_vpto_vexpdif_f16.pto`<br>…共 6 文件 |
| `pto.vmi.vaxpy` | `VMIVaxpyOp` | VMIOps.td:1686 | 1 | `lit/vmi_new/vmi_to_vpto_unified_masked_compute.pto` |
| `pto.vmi.vlrelu` | `VMIVlreluOp` | VMIOps.td:1708 | 1 | `lit/vmi_new/vmi_to_vpto_unified_masked_compute.pto` |
| `pto.vmi.vprelu` | `VMIVpreluOp` | VMIOps.td:1730 | 1 | `lit/vmi_new/vmi_to_vpto_unified_masked_compute.pto` |
| `pto.vmi.vmull` | `VMIVmullOp` | VMIOps.td:1753 | 8 | `lit/vmi_new/vmi_vmull_verifier.pto`<br>…共 8 文件 |
| `pto.vmi.vmula` | `VMIVmulaOp` | VMIOps.td:1782 | 10 | `lit/vmi_new/vmi_vmula_integer.pto`<br>…共 10 文件 |

### 3.4 UB 直连 op（`pto.ub.*`）（28 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.ub.vadd` | `PTO_UBVaddOp` | VPTOUbOps.td:60 | 19 | `lit/vpto/ub/a5/ub_vadd_noop.pto`<br>…共 19 文件 |
| `pto.ub.vsub` | `PTO_UBVsubOp` | VPTOUbOps.td:61 | 2 | `lit/vpto/ub/tile_to_ub/elementwise/tsub.pto`<br>…共 2 文件 |
| `pto.ub.vmul` | `PTO_UBVmulOp` | VPTOUbOps.td:62 | 2 | `lit/vpto/ub/tile_to_ub/elementwise/tmul.pto`<br>…共 2 文件 |
| `pto.ub.vdiv` | `PTO_UBVdivOp` | VPTOUbOps.td:63 | 5 | `lit/vpto/ub/tile_to_ub/elementwise/tdiv.pto`<br>…共 5 文件 |
| `pto.ub.vmax` | `PTO_UBVmaxOp` | VPTOUbOps.td:64 | 5 | `lit/vpto/ub/round_trip/binary_ops.pto`<br>…共 5 文件 |
| `pto.ub.vmin` | `PTO_UBVminOp` | VPTOUbOps.td:65 | 3 | `lit/vpto/ub/round_trip/binary_ops.pto`<br>…共 3 文件 |
| `pto.ub.vand` | `PTO_UBVandOp` | VPTOUbOps.td:66 | 4 | `lit/vpto/ub/round_trip/binary_ops.pto`<br>…共 4 文件 |
| `pto.ub.vor` | `PTO_UBVorOp` | VPTOUbOps.td:67 | 4 | `lit/vpto/ub/round_trip/binary_ops.pto`<br>…共 4 文件 |
| `pto.ub.vaddrelu` | `PTO_UBVaddReluOp` | VPTOUbOps.td:68 | 2 | `lit/vpto/ub/tile_to_ub/elementwise/taddrelu.pto`<br>…共 2 文件 |
| `pto.ub.vnot` | `PTO_UBVnotOp` | VPTOUbOps.td:110 | 1 | `lit/vpto/ub/tile_to_ub/elementwise/txor.pto` |
| `pto.ub.vabs` | `PTO_UBVabsOp` | VPTOUbOps.td:111 | 1 | `lit/vpto/ub/tile_to_ub/elementwise/tabs.pto` |
| `pto.ub.vrelu` | `PTO_UBVreluOp` | VPTOUbOps.td:112 | 1 | `lit/vpto/ub/tile_to_ub/elementwise/trelu.pto` |
| `pto.ub.vexp` | `PTO_UBVexpOp` | VPTOUbOps.td:113 | 0 |  |
| `pto.ub.vln` | `PTO_UBVlnOp` | VPTOUbOps.td:114 | 2 | `lit/vpto/ub/ub_to_llvm/elementwise/vln.pto`<br>…共 2 文件 |
| `pto.ub.vsqrt` | `PTO_UBVsqrtOp` | VPTOUbOps.td:115 | 0 |  |
| `pto.ub.vrsqrt` | `PTO_UBVrsqrtOp` | VPTOUbOps.td:116 | 0 |  |
| `pto.ub.vshl` | `PTO_UBVshlOp` | VPTOUbOps.td:159 | 3 | `lit/vpto/ub/round_trip/binary_ops.pto`<br>…共 3 文件 |
| `pto.ub.vshr` | `PTO_UBVshrOp` | VPTOUbOps.td:160 | 3 | `lit/vpto/ub/round_trip/binary_ops.pto`<br>…共 3 文件 |
| `pto.ub.vmuls` | `PTO_UBVmulSOp` | VPTOUbOps.td:161 | 2 | `lit/vpto/ub/tile_to_ub/elementwise/tneg.pto`<br>…共 2 文件 |
| `pto.ub.vadds` | `PTO_UBVaddSOp` | VPTOUbOps.td:162 | 1 | `lit/vpto/ub/tile_to_ub/elementwise/tadds.pto` |
| `pto.ub.vmaxs` | `PTO_UBVmaxSOp` | VPTOUbOps.td:163 | 0 |  |
| `pto.ub.vmins` | `PTO_UBVminSOp` | VPTOUbOps.td:164 | 0 |  |
| `pto.ub.vdup` | `PTO_UBVdupOp` | VPTOUbOps.td:166 | 2 | `lit/vpto/ub/ub_to_llvm/elementwise/vdup.pto`<br>…共 2 文件 |
| `pto.ub.vgatherb` | `PTO_UBVgatherbOp` | VPTOUbOps.td:195 | 5 | `lit/vpto/ub/ub_vgatherb_parse.pto`<br>…共 5 文件 |
| `pto.ub.vgather` | `PTO_UBVgatherOp` | VPTOUbOps.td:232 | 4 | `lit/vpto/ub/ub_vgather_parse.pto`<br>…共 4 文件 |
| `pto.ub.set_mask` | `PTO_UBSetMaskOp` | VPTOUbOps.td:262 | 13 | `lit/vpto/ub/a5/set_mask_noop.pto`<br>…共 13 文件 |
| `pto.ub.set_mask_count` | `PTO_UBSetMaskCountOp` | VPTOUbOps.td:285 | 13 | `lit/vpto/ub/tile_to_ub/elementwise/tshl.pto`<br>…共 13 文件 |
| `pto.ub.set_mask_norm` | `PTO_UBSetMaskNormOp` | VPTOUbOps.td:308 | 12 | `lit/vpto/ub/tile_to_ub/elementwise/tshl.pto`<br>…共 12 文件 |

### 3.5 SIMT/SPMD op（38 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.ldg` | `PTOLdgOp` | VPTOOps.td:220 | 9 | `lit/vpto/simt_ldg_stg_scope_invalid.pto`<br>…共 9 文件 |
| `pto.stg` | `PTOStgOp` | VPTOOps.td:257 | 11 | `lit/vpto/simt_ldg_stg_scope_invalid.pto`<br>…共 11 文件 |
| `pto.store_vfsimt_info` | `PTO_StoreVfSimtInfoOp` | VPTOOps.td:658 | 15 | `lit/vpto/simt_launch_sugar.pto`<br>…共 15 文件 |
| `pto.get_tid_x` | `PTO_GetTidXOp` | VPTOOps.td:704 | 25 | `lit/vpto/simt_launch_sugar.pto`<br>…共 25 文件 |
| `pto.get_tid_y` | `PTO_GetTidYOp` | VPTOOps.td:705 | 3 | `lit/vpto/vpto_scheduler_dependencies.pto`<br>…共 3 文件 |
| `pto.get_tid_z` | `PTO_GetTidZOp` | VPTOOps.td:706 | 1 | `dsl-st/legacy_micro_op_simt_simt_store_tid.py` |
| `pto.get_block_dim_x` | `PTO_GetBlockDimXOp` | VPTOOps.td:707 | 3 | `lit/pto/issue1493_tid_pow2_range.pto`<br>…共 3 文件 |
| `pto.get_block_dim_y` | `PTO_GetBlockDimYOp` | VPTOOps.td:708 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.get_block_dim_z` | `PTO_GetBlockDimZOp` | VPTOOps.td:709 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.get_grid_dim_x` | `PTO_GetGridDimXOp` | VPTOOps.td:710 | 2 | `lit/pto/issue1493_tid_pow2_range.pto`<br>…共 2 文件 |
| `pto.get_grid_dim_y` | `PTO_GetGridDimYOp` | VPTOOps.td:711 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.get_grid_dim_z` | `PTO_GetGridDimZOp` | VPTOOps.td:712 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.get_block_idx_x` | `PTO_GetBlockIdxXOp` | VPTOOps.td:713 | 3 | `lit/pto/issue1493_tid_pow2_range.pto`<br>…共 3 文件 |
| `pto.get_block_idx_y` | `PTO_GetBlockIdxYOp` | VPTOOps.td:714 | 1 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto` |
| `pto.get_block_idx_z` | `PTO_GetBlockIdxZOp` | VPTOOps.td:715 | 1 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto` |
| `pto.get_veccoreid` | `PTO_GetVecCoreIdOp` | VPTOOps.td:716 | 2 | `lit/pto/issue1493_tid_pow2_range.pto`<br>…共 2 文件 |
| `pto.get_laneid` | `PTO_GetLaneIdOp` | VPTOOps.td:717 | 7 | `lit/pto/issue1493_tid_pow2_range.pto`<br>…共 7 文件 |
| `pto.get_clock32` | `PTO_GetClock32Op` | VPTOOps.td:718 | 3 | `lit/vpto/vpto_guarded_licm.pto`<br>…共 3 文件 |
| `pto.get_clock64` | `PTO_GetClock64Op` | VPTOOps.td:719 | 1 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto` |
| `pto.get_lanemask_eq` | `PTO_GetLaneMaskEqOp` | VPTOOps.td:720 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.get_lanemask_le` | `PTO_GetLaneMaskLeOp` | VPTOOps.td:721 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.get_lanemask_lt` | `PTO_GetLaneMaskLtOp` | VPTOOps.td:722 | 3 | `lit/pto/issue1493_tid_pow2_range.pto`<br>…共 3 文件 |
| `pto.get_lanemask_ge` | `PTO_GetLaneMaskGeOp` | VPTOOps.td:723 | 1 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto` |
| `pto.get_lanemask_gt` | `PTO_GetLaneMaskGtOp` | VPTOOps.td:724 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.vote_all` | `PTO_VoteAllOp` | VPTOOps.td:726 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.vote_any` | `PTO_VoteAnyOp` | VPTOOps.td:735 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.vote_uni` | `PTO_VoteUniOp` | VPTOOps.td:744 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.vote_ballot` | `PTO_VoteBallotOp` | VPTOOps.td:753 | 3 | `lit/vpto/vpto_guarded_licm.pto`<br>…共 3 文件 |
| `pto.shuffle_idx` | `PTO_ShuffleIdxOp` | VPTOOps.td:762 | 7 | `lit/vpto/vpto_guarded_licm.pto`<br>…共 7 文件 |
| `pto.shuffle_up` | `PTO_ShuffleUpOp` | VPTOOps.td:775 | 3 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 3 文件 |
| `pto.shuffle_down` | `PTO_ShuffleDownOp` | VPTOOps.td:788 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.shuffle_bfly` | `PTO_ShuffleBflyOp` | VPTOOps.td:801 | 3 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 3 文件 |
| `pto.prmt` | `PTO_PrmtOp` | VPTOOps.td:913 | 3 | `lit/pto/ptodsl_tileop_reject_contract_edges.pto`<br>…共 3 文件 |
| `pto.mulhi` | `PTO_MulhiOp` | VPTOOps.td:922 | 2 | `lit/vpto/simt_lowlevel_scalar_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.mul_i32toi64` | `PTO_MulI32ToI64Op` | VPTOOps.td:941 | 2 | `lit/vpto/simt_lowlevel_scalar_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.syncthreads` | `PTO_SyncthreadsOp` | VPTOOps.td:1048 | 3 | `lit/pto/section_simt_syncthreads_roundtrip.pto`<br>…共 3 文件 |
| `pto.keep` | `PTO_KeepOp` | VPTOOps.td:1085 | 12 | `lit/vpto/simt_keep_resume_i64_odd_slot_invalid.pto`<br>…共 12 文件 |
| `pto.resume` | `PTO_ResumeOp` | VPTOOps.td:1096 | 11 | `lit/vpto/simt_keep_resume_direct_slots_vpto_llvm.pto`<br>…共 11 文件 |

### 3.6 配置类 op（set_*/流水线配置）（31 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.set_loop2_stride_outtoub` | `PTO_SetLoop2StrideOutToUbOp` | VPTOOps.td:590 | 10 | `dsl-st/legacy_micro_op_compare_select_vsel.py`<br>…共 10 文件 |
| `pto.set_loop1_stride_outtoub` | `PTO_SetLoop1StrideOutToUbOp` | VPTOOps.td:591 | 11 | `dsl-st/legacy_micro_op_compare_select_vsel.py`<br>…共 11 文件 |
| `pto.set_loop_size_outtoub` | `PTO_SetLoopSizeOutToUbOp` | VPTOOps.td:592 | 5 | `dsl/abs.py`<br>…共 5 文件 |
| `pto.set_loop2_stride_outtol1` | `PTO_SetLoop2StrideOutToL1Op` | VPTOOps.td:593 | 0 |  |
| `pto.set_loop1_stride_outtol1` | `PTO_SetLoop1StrideOutToL1Op` | VPTOOps.td:594 | 0 |  |
| `pto.set_loop_size_outtol1` | `PTO_SetLoopSizeOutToL1Op` | VPTOOps.td:595 | 0 |  |
| `pto.set_loop2_stride_ubtoout` | `PTO_SetLoop2StrideUbToOutOp` | VPTOOps.td:596 | 12 | `dsl-st/legacy_micro_op_compare_select_vsel.py`<br>…共 12 文件 |
| `pto.set_loop1_stride_ubtoout` | `PTO_SetLoop1StrideUbToOutOp` | VPTOOps.td:597 | 16 | `dsl-st/legacy_micro_op_compare_select_vsel.py`<br>…共 16 文件 |
| `pto.set_loop_size_ubtoout` | `PTO_SetLoopSizeUbToOutOp` | VPTOOps.td:598 | 4 | `dsl/abs.py`<br>…共 4 文件 |
| `pto.set_loop3_para` | `PTO_SetLoop3ParaOp` | VPTOOps.td:599 | 8 | `lit/vpto/acc_store.pto`<br>…共 8 文件 |
| `pto.set_channel_para` | `PTO_SetChannelParaOp` | VPTOOps.td:600 | 8 | `lit/vpto/acc_store.pto`<br>…共 8 文件 |
| `pto.set_mte2_nz_para` | `PTO_SetMte2NzParaOp` | VPTOOps.td:601 | 2 | `lit/vpto/cube_load_frac.pto`<br>…共 2 文件 |
| `pto.set_pad_val_outtol1` | `PTO_SetPadValOutToL1Op` | VPTOOps.td:602 | 0 |  |
| `pto.set_ctrl` | `PTO_SetCtrlOp` | VPTOOps.td:621 | 31 | `lit/vpto/ctrl_ops.pto`<br>…共 31 文件 |
| `pto.set_store_atomic_cfg` | `PTO_SetStoreAtomicCfgOp` | VPTOOps.td:657 | 0 |  |
| `pto.sbitset0` | `PTO_Sbitset0Op` | VPTOOps.td:688 | 8 | `lit/vpto/ctrl_ops.pto`<br>…共 8 文件 |
| `pto.sbitset1` | `PTO_Sbitset1Op` | VPTOOps.td:689 | 3 | `lit/vpto/ctrl_ops.pto`<br>…共 3 文件 |
| `pto.set_quant_pre` | `PTO_SetQuantPreOp` | VPTOOps.td:690 | 8 | `lit/vpto/acc_store.pto`<br>…共 8 文件 |
| `pto.set_relu_alpha` | `PTO_SetReluAlphaOp` | VPTOOps.td:691 | 2 | `lit/vpto/acc_store.pto`<br>…共 2 文件 |
| `pto.set_fix_clip_relu` | `PTO_SetFixClipReluOp` | VPTOOps.td:692 | 3 | `lit/vpto/acc_store_ub_clip.pto`<br>…共 3 文件 |
| `pto.set_fpc` | `PTO_SetFpcOp` | VPTOOps.td:693 | 6 | `lit/vpto/acc_store_gm.pto`<br>…共 6 文件 |
| `pto.set_atomic_add` | `PTO_SetAtomicAddOp` | VPTOOps.td:694 | 1 | `lit/vpto/store_atomic_ctrl_expand.pto` |
| `pto.set_atomic_max` | `PTO_SetAtomicMaxOp` | VPTOOps.td:695 | 1 | `lit/vpto/store_atomic_ctrl_expand.pto` |
| `pto.set_atomic_min` | `PTO_SetAtomicMinOp` | VPTOOps.td:696 | 1 | `lit/vpto/store_atomic_ctrl_expand.pto` |
| `pto.set_atomic_none` | `PTO_SetAtomicNoneOp` | VPTOOps.td:697 | 2 | `lit/vpto/store_atomic_ctrl_expand.pto`<br>…共 2 文件 |
| `pto.set_atomic_f32` | `PTO_SetAtomicF32Op` | VPTOOps.td:698 | 1 | `lit/vpto/store_atomic_ctrl_expand.pto` |
| `pto.set_atomic_f16` | `PTO_SetAtomicF16Op` | VPTOOps.td:699 | 1 | `lit/vpto/store_atomic_ctrl_expand.pto` |
| `pto.set_atomic_bf16` | `PTO_SetAtomicBF16Op` | VPTOOps.td:700 | 1 | `lit/vpto/store_atomic_ctrl_expand.pto` |
| `pto.set_atomic_s32` | `PTO_SetAtomicS32Op` | VPTOOps.td:701 | 2 | `lit/vpto/store_atomic_ctrl_expand.pto`<br>…共 2 文件 |
| `pto.set_atomic_s16` | `PTO_SetAtomicS16Op` | VPTOOps.td:702 | 1 | `lit/vpto/store_atomic_ctrl_expand.pto` |
| `pto.set_atomic_s8` | `PTO_SetAtomicS8Op` | VPTOOps.td:703 | 1 | `lit/vpto/store_atomic_ctrl_expand.pto` |

### 3.7 原子类 op（9 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.atomic_cas` | `PTO_AtomicCasOp` | VPTOOps.td:903 | 4 | `lit/vpto/simt_lowlevel_atomic_vpto_llvm.pto`<br>…共 4 文件 |
| `pto.atomic_exch` | `PTO_AtomicExchOp` | VPTOOps.td:904 | 3 | `lit/vpto/simt_lowlevel_atomic_vpto_llvm.pto`<br>…共 3 文件 |
| `pto.atomic_add` | `PTO_AtomicAddOp` | VPTOOps.td:905 | 8 | `lit/vpto/simt_lowlevel_atomic_vpto_llvm.pto`<br>…共 8 文件 |
| `pto.atomic_sub` | `PTO_AtomicSubOp` | VPTOOps.td:906 | 1 | `lit/vpto/simt_lowlevel_atomic_vpto_llvm.pto` |
| `pto.atomic_min` | `PTO_AtomicMinOp` | VPTOOps.td:907 | 4 | `lit/vpto/simt_lowlevel_atomic_vpto_llvm.pto`<br>…共 4 文件 |
| `pto.atomic_max` | `PTO_AtomicMaxOp` | VPTOOps.td:908 | 3 | `lit/vpto/simt_lowlevel_atomic_vpto_llvm.pto`<br>…共 3 文件 |
| `pto.atomic_and` | `PTO_AtomicAndOp` | VPTOOps.td:909 | 2 | `lit/vpto/simt_lowlevel_atomic_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.atomic_or` | `PTO_AtomicOrOp` | VPTOOps.td:910 | 0 |  |
| `pto.atomic_xor` | `PTO_AtomicXorOp` | VPTOOps.td:911 | 0 |  |

### 3.8 Redux 归约 op（6 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.redux_addi` | `PTO_ReduxAddIOp` | VPTOOps.td:845 | 3 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 3 文件 |
| `pto.redux_addf` | `PTO_ReduxAddFOp` | VPTOOps.td:846 | 3 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 3 文件 |
| `pto.redux_maxi` | `PTO_ReduxMaxIOp` | VPTOOps.td:847 | 4 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 4 文件 |
| `pto.redux_maxf` | `PTO_ReduxMaxFOp` | VPTOOps.td:848 | 1 | `dsl-st/legacy_micro_op_simt_simt_control_typed_core.py` |
| `pto.redux_mini` | `PTO_ReduxMinIOp` | VPTOOps.td:849 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.redux_minf` | `PTO_ReduxMinFOp` | VPTOOps.td:850 | 2 | `lit/vpto/simt_lowlevel_intrinsics_vpto_llvm.pto`<br>…共 2 文件 |

### 3.9 通信 op（`pto.comm.*`）（14 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.comm.build_async_session` | `BuildAsyncSessionOp` | PTOOps.td:2103 | 7 | `samples/AsyncComm/async_comm.py`<br>…共 7 文件 |
| `pto.comm.tput_async` | `TPutAsyncOp` | PTOOps.td:2127 | 5 | `samples/AsyncComm/async_comm.py`<br>…共 5 文件 |
| `pto.comm.tget_async` | `TGetAsyncOp` | PTOOps.td:2148 | 3 | `samples/AsyncComm/async_comm.py`<br>…共 3 文件 |
| `pto.comm.wait_async_event` | `WaitAsyncEventOp` | PTOOps.td:2169 | 7 | `samples/AsyncComm/async_comm.py`<br>…共 7 文件 |
| `pto.comm.test_async_event` | `TestAsyncEventOp` | PTOOps.td:2188 | 2 | `samples/AsyncComm/async_comm.py`<br>…共 2 文件 |
| `pto.comm.tput` | `TPutOp` | PTOOps.td:2207 | 20 | `samples/CommSync/comm_p2p.py`<br>…共 20 文件 |
| `pto.comm.tget` | `TGetOp` | PTOOps.td:2227 | 5 | `samples/CommSync/comm_p2p.py`<br>…共 5 文件 |
| `pto.comm.tnotify` | `TNotifyOp` | PTOOps.td:2246 | 24 | `samples/CommSync/comm_p2p.py`<br>…共 24 文件 |
| `pto.comm.twait` | `TWaitOp` | PTOOps.td:2263 | 24 | `samples/CommSync/comm_p2p.py`<br>…共 24 文件 |
| `pto.comm.ttest` | `TTestOp` | PTOOps.td:2280 | 3 | `samples/CommSync/comm_p2p.py`<br>…共 3 文件 |
| `pto.comm.tbroadcast` | `TBroadcastOp` | PTOOps.td:2297 | 10 | `samples/CommSync/comm_collective.py`<br>…共 10 文件 |
| `pto.comm.tgather` | `CommTGatherOp` | PTOOps.td:2314 | 6 | `samples/CommSync/comm_collective.py`<br>…共 6 文件 |
| `pto.comm.tscatter` | `CommTScatterOp` | PTOOps.td:2331 | 6 | `samples/CommSync/comm_collective.py`<br>…共 6 文件 |
| `pto.comm.treduce` | `TReduceOp` | PTOOps.td:2348 | 7 | `samples/CommSync/comm_collective.py`<br>…共 7 文件 |

### 3.10 Generic 标量 op（执行域无关，对应 arith 语义）（38 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.constant` | `PTO_ConstantOp` | PTOGenericOps.td:63 | 4 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 4 文件 |
| `pto.addi` | `PTO_AddIOp` | PTOGenericOps.td:114 | 4 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 4 文件 |
| `pto.addf` | `PTO_AddFOp` | PTOGenericOps.td:116 | 4 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 4 文件 |
| `pto.subi` | `PTO_SubIOp` | PTOGenericOps.td:118 | 2 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 2 文件 |
| `pto.subf` | `PTO_SubFOp` | PTOGenericOps.td:120 | 1 | `lit/vpto/scalar_frontend_to_standard.pto` |
| `pto.muli` | `PTO_MulIOp` | PTOGenericOps.td:122 | 3 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 3 文件 |
| `pto.mulf` | `PTO_MulFOp` | PTOGenericOps.td:124 | 1 | `lit/vpto/scalar_frontend_to_standard.pto` |
| `pto.addui_extended` | `PTO_AddUIExtendedOp` | PTOGenericOps.td:127 | 1 | `lit/vpto/scalar_arith_attributes.pto` |
| `pto.mul_extended` | `PTO_MulExtendedOp` | PTOGenericOps.td:150 | 2 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 2 文件 |
| `pto.negi` | `PTO_NegIOp` | PTOGenericOps.td:171 | 1 | `lit/vpto/scalar_frontend_to_standard.pto` |
| `pto.negf` | `PTO_NegFOp` | PTOGenericOps.td:188 | 2 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 2 文件 |
| `pto.divf` | `PTO_DivFOp` | PTOGenericOps.td:205 | 4 | `lit/vpto/scalar_div_precision.pto`<br>…共 4 文件 |
| `pto.divi` | `PTO_DivIOp` | PTOGenericOps.td:260 | 3 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 3 文件 |
| `pto.floordiv` | `PTO_FloorDivOp` | PTOGenericOps.td:262 | 2 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 2 文件 |
| `pto.ceildiv` | `PTO_CeilDivOp` | PTOGenericOps.td:264 | 2 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 2 文件 |
| `pto.remi` | `PTO_RemIOp` | PTOGenericOps.td:266 | 3 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 3 文件 |
| `pto.remf` | `PTO_RemFOp` | PTOGenericOps.td:268 | 2 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 2 文件 |
| `pto.and` | `PTO_AndOp` | PTOGenericOps.td:285 | 1 | `lit/vpto/scalar_frontend_to_standard.pto` |
| `pto.or` | `PTO_OrOp` | PTOGenericOps.td:287 | 1 | `lit/vpto/scalar_frontend_to_standard.pto` |
| `pto.xor` | `PTO_XorOp` | PTOGenericOps.td:289 | 1 | `lit/vpto/scalar_frontend_to_standard.pto` |
| `pto.shl` | `PTO_ShlOp` | PTOGenericOps.td:308 | 2 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 2 文件 |
| `pto.shr` | `PTO_ShrOp` | PTOGenericOps.td:325 | 1 | `lit/vpto/scalar_frontend_to_standard.pto` |
| `pto.cmpi` | `PTO_CmpIOp` | PTOGenericOps.td:351 | 4 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 4 文件 |
| `pto.cmpf` | `PTO_CmpFOp` | PTOGenericOps.td:367 | 3 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 3 文件 |
| `pto.maxi` | `PTO_MaxIOp` | PTOGenericOps.td:403 | 2 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 2 文件 |
| `pto.mini` | `PTO_MinIOp` | PTOGenericOps.td:405 | 2 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 2 文件 |
| `pto.maximum` | `PTO_MaximumOp` | PTOGenericOps.td:428 | 1 | `lit/vpto/scalar_arith_attributes.pto` |
| `pto.minimum` | `PTO_MinimumOp` | PTOGenericOps.td:430 | 1 | `lit/vpto/scalar_arith_attributes.pto` |
| `pto.maxf` | `PTO_MaxFOp` | PTOGenericOps.td:433 | 7 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 7 文件 |
| `pto.minf` | `PTO_MinFOp` | PTOGenericOps.td:435 | 6 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 6 文件 |
| `pto.absi` | `PTO_AbsIOp` | PTOGenericOps.td:438 | 2 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 2 文件 |
| `pto.absf` | `PTO_AbsFOp` | PTOGenericOps.td:454 | 5 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 5 文件 |
| `pto.exti` | `PTO_ExtIOp` | PTOGenericOps.td:476 | 2 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 2 文件 |
| `pto.trunci` | `PTO_TruncIOp` | PTOGenericOps.td:486 | 2 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 2 文件 |
| `pto.ftof` | `PTO_FToFOp` | PTOGenericOps.td:498 | 10 | `lit/vpto/scalar_arith_attributes.pto`<br>…共 10 文件 |
| `pto.ftoi` | `PTO_FToIOp` | PTOGenericOps.td:515 | 7 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 7 文件 |
| `pto.itof` | `PTO_IToFOp` | PTOGenericOps.td:531 | 7 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 7 文件 |
| `pto.select` | `PTO_SelectOp` | PTOGenericOps.td:571 | 3 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 3 文件 |

### 3.11 标量专用 op（13 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.load` | `PTOLoadOp` | VPTOOps.td:171 | 258 | `samples/ScalarPtr/scalar_ptr.py`<br>…共 258 文件 |
| `pto.store` | `PTOStoreOp` | VPTOOps.td:195 | 149 | `samples/ScalarPtr/scalar_ptr.py`<br>…共 149 文件 |
| `pto.sqrt` | `PTO_SqrtOp` | VPTOOps.td:957 | 6 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 6 文件 |
| `pto.exp` | `PTO_ExpOp` | VPTOOps.td:1010 | 5 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 5 文件 |
| `pto.log` | `PTO_LogOp` | VPTOOps.td:1011 | 5 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 5 文件 |
| `pto.sin` | `PTO_SinOp` | VPTOOps.td:1012 | 2 | `lit/vpto/issue_993_sincos_softlib.pto`<br>…共 2 文件 |
| `pto.cos` | `PTO_CosOp` | VPTOOps.td:1013 | 2 | `lit/vpto/issue_993_sincos_softlib.pto`<br>…共 2 文件 |
| `pto.ceil` | `PTO_CeilOp` | VPTOOps.td:1014 | 2 | `lit/vpto/simt_lowlevel_float_math_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.floor` | `PTO_FloorOp` | VPTOOps.td:1015 | 2 | `lit/vpto/simt_lowlevel_float_math_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.rint` | `PTO_RintOp` | VPTOOps.td:1016 | 2 | `lit/vpto/simt_lowlevel_float_math_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.round` | `PTO_RoundOp` | VPTOOps.td:1017 | 2 | `lit/vpto/simt_lowlevel_float_math_vpto_llvm.pto`<br>…共 2 文件 |
| `pto.pow` | `PTO_PowOp` | VPTOOps.td:1032 | 5 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 5 文件 |
| `pto.fma` | `PTO_FmaOp` | VPTOOps.td:1034 | 6 | `lit/vpto/scalar_frontend_to_standard.pto`<br>…共 6 文件 |

### 3.12 其它（结构/同步/事件/数组/打印等）（84 个）

| 助记符 | C++ 类 | 定义位置 | 测例数 | 测例示例（相对 test/） |
|---|---|---|---:|---|
| `pto.bridge.instance` | `BridgeInstanceOp` | PTOOps.td:99 | 8 | `lit/vpto/bridge/instances.pto`<br>…共 8 文件 |
| `pto.bridge.call` | `BridgeCallOp` | PTOOps.td:117 | 15 | `lit/vpto/bridge/pipe_abi.pto`<br>…共 15 文件 |
| `pto.bridge.object.create` | `BridgeObjectCreateOp` | PTOOps.td:134 | 6 | `lit/vpto/bridge/instances.pto`<br>…共 6 文件 |
| `pto.get_block_idx` | `GetBlockIdxOp` | PTOOps.td:1488 | 19 | `samples/Complex/mix_kernel.py`<br>…共 19 文件 |
| `pto.get_subblock_idx` | `GetSubBlockIdxOp` | PTOOps.td:1506 | 28 | `samples/Complex/mix_kernel.py`<br>…共 28 文件 |
| `pto.get_block_num` | `GetBlockNumOp` | PTOOps.td:1524 | 6 | `samples/Sync/add_double_dynamic.py`<br>…共 6 文件 |
| `pto.get_subblock_num` | `GetSubBlockNumOp` | PTOOps.td:1541 | 4 | `samples/Complex/mix_kernel.py`<br>…共 4 文件 |
| `pto.get_l2_cache_offset` | `GetL2CacheOffsetOp` | PTOOps.td:1558 | 1 | `lit/pto/get_l2_cache_offset_emitc.pto` |
| `pto.record_event` | `RecordEventOp` | PTOOps.td:1580 | 6 | `samples/MatMul/tmatmulk.py`<br>…共 6 文件 |
| `pto.wait_event` | `WaitEventOp` | PTOOps.td:1597 | 5 | `samples/MatMul/tmatmulk.py`<br>…共 5 文件 |
| `pto.barrier_sync` | `BarrierSyncOp` | PTOOps.td:1615 | 2 | `samples/Sync/test_barrier_sync.py`<br>…共 2 文件 |
| `pto.section.cube` | `SectionCubeOp` | PTOOps.td:1648 | 55 | `samples/Complex/cv_region.py`<br>…共 55 文件 |
| `pto.section.vector` | `SectionVectorOp` | PTOOps.td:1649 | 61 | `samples/Complex/cv_region.py`<br>…共 61 文件 |
| `pto.section.simt` | `SectionSimtOp` | PTOOps.td:1651 | 50 | `lit/pto/section_simt_roundtrip.pto`<br>…共 50 文件 |
| `pto.fusion_region` | `FusionRegionOp` | PTOOps.td:1675 | 29 | `lit/tile_fusion/fusion_region_basic.pto`<br>…共 29 文件 |
| `pto.yield` | `YieldOp` | PTOOps.td:1692 | 25 | `lit/tile_fusion/fusion_region_verify_missing_yield.pto`<br>…共 25 文件 |
| `pto.aic_initialize_pipe` | `AicInitializePipeOp` | PTOOps.td:1714 | 120 | `samples/TPushTPop/test1/kernel.pto`<br>…共 120 文件 |
| `pto.aiv_initialize_pipe` | `AivInitializePipeOp` | PTOOps.td:1737 | 109 | `samples/TPushTPop/test1/kernel.pto`<br>…共 109 文件 |
| `pto.set_fmatrix` | `SetFmatrixOp` | PTOOps.td:1982 | 4 | `samples/Img2col/timg2col_runtime.py`<br>…共 4 文件 |
| `pto.set_img2col_rpt` | `SetImg2colRptOp` | PTOOps.td:1995 | 4 | `samples/Img2col/timg2col_runtime.py`<br>…共 4 文件 |
| `pto.set_img2col_padding` | `SetImg2colPaddingOp` | PTOOps.td:2008 | 4 | `samples/Img2col/timg2col_runtime.py`<br>…共 4 文件 |
| `pto.reserve_buffer` | `ReserveBufferOp` | PTOOps.td:2057 | 117 | `samples/TPushTPop/ptodsl/local_c2v/kernel.py`<br>…共 117 文件 |
| `pto.import_reserved_buffer` | `ImportReservedBufferOp` | PTOOps.td:2081 | 112 | `samples/TPushTPop/ptodsl/local_c2v/kernel.py`<br>…共 112 文件 |
| `pto.initialize_l2g2l_pipe` | `InitializeL2G2LPipeOp` | PTOOps.td:2367 | 14 | `lit/pto/pipe_tile_native.pto`<br>…共 14 文件 |
| `pto.initialize_l2l_pipe` | `InitializeL2LPipeOp` | PTOOps.td:2404 | 23 | `lit/pto/tpush_tpop_emitc.pto`<br>…共 23 文件 |
| `pto.local_array_set` | `LocalArraySetOp` | PTOOps.td:2643 | 11 | `samples/DeepseekV4DecodeA5/kernels/aiv/dispatch_meta.pto`<br>…共 11 文件 |
| `pto.declare_struct` | `DeclareStructOp` | PTOOps.td:2677 | 16 | `lit/vpto/async_sdma_gm_gm.pto`<br>…共 16 文件 |
| `pto.struct_get` | `StructGetOp` | PTOOps.td:2709 | 8 | `lit/pto/struct_get_aggregate_invalid.pto`<br>…共 8 文件 |
| `pto.struct_set` | `StructSetOp` | PTOOps.td:2736 | 9 | `lit/vpto/struct_vpto_llvm.pto`<br>…共 9 文件 |
| `pto.set_flag` | `SetFlagOp` | PTOOps.td:2867 | 412 | `samples/Sync/sync.py`<br>…共 412 文件 |
| `pto.wait_flag` | `WaitFlagOp` | PTOOps.td:2882 | 406 | `samples/Sync/sync.py`<br>…共 406 文件 |
| `pto.set_flag_dyn` | `SetFlagDynOp` | PTOOps.td:2897 | 15 | `lit/vpto/async_sdma_gm_gm.pto`<br>…共 15 文件 |
| `pto.wait_flag_dyn` | `WaitFlagDynOp` | PTOOps.td:2910 | 14 | `lit/vpto/async_sdma_gm_gm.pto`<br>…共 14 文件 |
| `pto.get_buf` | `GetBufOp` | PTOOps.td:2930 | 11 | `samples/Sync/test_a5_buf_sync.py`<br>…共 11 文件 |
| `pto.get_buf_dyn` | `GetBufDynOp` | PTOOps.td:2962 | 2 | `lit/vpto/vecscope_sync_verify.pto`<br>…共 2 文件 |
| `pto.rls_buf` | `RlsBufOp` | PTOOps.td:2983 | 9 | `samples/Sync/test_a5_buf_sync.py`<br>…共 9 文件 |
| `pto.rls_buf_dyn` | `RlsBufDynOp` | PTOOps.td:3009 | 2 | `lit/vpto/vecscope_sync_verify.pto`<br>…共 2 文件 |
| `pto.sync.set` | `SyncSetOp` | PTOOps.td:3028 | 11 | `samples/runop.sh`<br>…共 11 文件 |
| `pto.sync.wait` | `SyncWaitOp` | PTOOps.td:3048 | 7 | `lit/vpto/vecscope_sync_verify.pto`<br>…共 7 文件 |
| `pto.set_cross_block` | `SetCrossBlockOp` | PTOOps.td:3074 | 4 | `lit/pto/named_sync_a5_emitc.pto`<br>…共 4 文件 |
| `pto.wait_cross_block` | `WaitCrossBlockOp` | PTOOps.td:3086 | 4 | `lit/pto/named_sync_a5_emitc.pto`<br>…共 4 文件 |
| `pto.set_intra_block` | `SetIntraBlockOp` | PTOOps.td:3095 | 24 | `samples/Sync/test_intercore_sync_a5.py`<br>…共 24 文件 |
| `pto.wait_intra_block` | `WaitIntraBlockOp` | PTOOps.td:3107 | 24 | `samples/Sync/test_intercore_sync_a5.py`<br>…共 24 文件 |
| `pto.barrier` | `BarrierOp` | PTOOps.td:3119 | 391 | `samples/CommSync/tgather_root_binding.py`<br>…共 391 文件 |
| `pto.cmo.cacheinvalid` | `CmoCacheInvalidOp` | PTOOps.td:3125 | 41 | `samples/DeepseekV4DecodeA5/kernels/aiv/combine.pto`<br>…共 41 文件 |
| `pto.fence.barrier_all` | `FenceBarrierAllOp` | PTOOps.td:3152 | 29 | `samples/CommSync/comm_p2p.py`<br>…共 29 文件 |
| `pto.syncall` | `SyncAllOp` | PTOOps.td:3183 | 7 | `samples/SyncAll/syncall_binding.py`<br>…共 7 文件 |
| `pto.tile_valid_rows` | `TileValidRowsOp` | PTOOps.td:3218 | 4 | `lit/vpto/fold_tile_buf_intrinsics.pto`<br>…共 4 文件 |
| `pto.tile_valid_cols` | `TileValidColsOp` | PTOOps.td:3235 | 4 | `lit/vpto/fold_tile_buf_intrinsics.pto`<br>…共 4 文件 |
| `pto.set_ffts` | `SetFFTsOp` | PTOOps.td:3256 | 10 | `samples/Sync/test_intercore_sync_a3.py`<br>…共 10 文件 |
| `pto.print` | `PrintOp` | PTOOps.td:3267 | 13 | `samples/Print/print_scalar.py`<br>…共 13 文件 |
| `pto.trap` | `TrapOp` | PTOOps.td:3281 | 2 | `samples/Trap/trap.py`<br>…共 2 文件 |
| `pto.assert` | `AssertOp` | PTOOps.td:3294 | 4 | `lit/pto/assert_emitc.pto`<br>…共 4 文件 |
| `pto.ld_dev` | `PTOLdDevOp` | VPTOOps.td:286 | 7 | `lit/vpto/async_sdma_gm_gm.pto`<br>…共 7 文件 |
| `pto.st_dev` | `PTOStDevOp` | VPTOOps.td:316 | 9 | `lit/vpto/aicore_ld_st_dev_invalid.pto`<br>…共 9 文件 |
| `pto.tensor_view_addr` | `TensorViewAddrOp` | VPTOOps.td:347 | 1 | `lit/vmi_new/vmi_vpto_memref_compat.pto` |
| `pto.vecscope` | `VecScopeOp` | VPTOOps.td:395 | 508 | `lit/vpto/vecscope_membar_all.pto`<br>…共 508 文件 |
| `pto.strict_vecscope` | `StrictVecScopeOp` | VPTOOps.td:413 | 6 | `lit/vpto/vecscope_sync_verify.pto`<br>…共 6 文件 |
| `pto.mem_bar` | `PTO_MemBarOp` | VPTOOps.td:432 | 40 | `lit/vpto/vecscope_membar_all.pto`<br>…共 40 文件 |
| `pto.dsb` | `DsbOp` | VPTOOps.td:445 | 7 | `lit/vpto/dcci_dsb_vpto_llvm.pto`<br>…共 7 文件 |
| `pto.dcci` | `DcciOp` | VPTOOps.td:460 | 6 | `lit/vpto/dcci_dsb_vpto_llvm.pto`<br>…共 6 文件 |
| `pto.set_mov_pad_val` | `PTO_SetMovPadValOp` | VPTOOps.td:579 | 1 | `lit/vpto/vpto_scheduler_coverage.pto` |
| `pto.get_ctrl` | `PTO_GetCtrlOp` | VPTOOps.td:605 | 29 | `lit/vpto/ctrl_ops.pto`<br>…共 29 文件 |
| `pto.get_vms4_sr` | `PTO_GetVms4SrOp` | VPTOOps.td:613 | 3 | `lit/vpto/get_vms4_sr_vpto.pto`<br>…共 3 文件 |
| `pto.ctrl_state_guard` | `PTO_CtrlStateGuardOp` | VPTOOps.td:628 | 2 | `lit/vpto/cube/mad_semantic_to_raw.pto`<br>…共 2 文件 |
| `pto.simt_launch` | `PTO_SimtLaunchOp` | VPTOOps.td:666 | 20 | `lit/vpto/simt_launch_sugar.pto`<br>…共 20 文件 |
| `pto.threadfence` | `PTO_ThreadfenceOp` | VPTOOps.td:1059 | 2 | `lit/vpto/vecscope_sync_verify.pto`<br>…共 2 文件 |
| `pto.threadfence_block` | `PTO_ThreadfenceBlockOp` | VPTOOps.td:1072 | 2 | `lit/vpto/vecscope_sync_verify.pto`<br>…共 2 文件 |
| `pto.mad` | `PTO_MadOp` | VPTOOps.td:2148 | 46 | `lit/vpto/ctrl/mad_ctrl_shared.pto`<br>…共 46 文件 |
| `pto.mad_acc` | `PTO_MadAccOp` | VPTOOps.td:2199 | 4 | `lit/vpto/cube/mad_runtime_flags.pto`<br>…共 4 文件 |
| `pto.mad_bias` | `PTO_MadBiasOp` | VPTOOps.td:2250 | 5 | `lit/vpto/cube/mad_runtime_flags.pto`<br>…共 5 文件 |
| `pto.mad_mx` | `PTO_MadMxOp` | VPTOOps.td:2301 | 8 | `lit/vpto/cube/mad_mx_semantic_to_raw.pto`<br>…共 8 文件 |
| `pto.mad_mx_acc` | `PTO_MadMxAccOp` | VPTOOps.td:2352 | 4 | `lit/vpto/cube/mad_mx_semantic_to_raw.pto`<br>…共 4 文件 |
| `pto.mad_mx_bias` | `PTO_MadMxBiasOp` | VPTOOps.td:2403 | 3 | `lit/vpto/cube/mad_mx_semantic_to_raw.pto`<br>…共 3 文件 |
| `pto.mad_raw` | `PTO_MadRawOp` | VPTOOps.td:2454 | 13 | `lit/vpto/ctrl/mad_ctrl_shared.pto`<br>…共 13 文件 |
| `pto.mad_bias_raw` | `PTO_MadBiasRawOp` | VPTOOps.td:2482 | 3 | `lit/vpto/cube/mad_runtime_flags.pto`<br>…共 3 文件 |
| `pto.mad_mx_raw` | `PTO_MadMxRawOp` | VPTOOps.td:2510 | 1 | `lit/vpto/cube/mad_mx_semantic_to_raw.pto` |
| `pto.mad_mx_bias_raw` | `PTO_MadMxBiasRawOp` | VPTOOps.td:2538 | 1 | `lit/vpto/cube/mad_mx_semantic_to_raw.pto` |
| `pto.vcbmax` | `PTO_VcbmaxOp` | VPTOOps.td:2624 | 1 | `lit/vpto/a5_extra_arith_vpto_llvm.pto` |
| `pto.vcbmin` | `PTO_VcbminOp` | VPTOOps.td:2625 | 1 | `lit/vpto/a5_extra_arith_vpto_llvm.pto` |
| `pto.raw_fill_l1` | `PTO_RawFillL1Op` | VPTOOps.td:3350 | 10 | `lit/vpto/raw_fill_l1_vpto_llvm.pto`<br>…共 10 文件 |
| `pto.create_cbuf_matrix` | `PTO_CreateCbufMatrixOp` | VPTOOps.td:3405 | 6 | `lit/vpto/raw_fill_l1_tilelang_views.pto`<br>…共 6 文件 |
| `pto.session_init` | `SessionInitOp` | VPTOOps.td:4101 | 7 | `lit/vpto/async_session_init.pto`<br>…共 7 文件 |
| `pto.sdma_gm_gm` | `SdmaGmGmOp` | VPTOOps.td:4139 | 7 | `lit/vpto/async_sdma_gm_gm.pto`<br>…共 7 文件 |

---

## 四、文档与 ODS 的已知差异

- `docs/isa/tile-op/03` 中的 `pto.subset` 在 ODS 中实际为 `pto.subview`（`SubViewOp`）。
- `docs/isa/tile-op/03` 提到的 `pto.tensor_view_addr` 在 ODS 中无对应定义。
- 文档第 2–12 章未覆盖：`tmatmul*`/`tgemv*` 全家族、`timg2col`、`talloc/tpush/tpop/tfree` 生命周期族、
  `mgather/mscatter/thistogram/trandom/tsetval/tgetval/tprint/tdump` 等。

## 五、测试口径（已锁定）与执行计划

**口径**：每个 tile op 都有 **kernel 级、数值正确性验证**的测例（tilelib-st 运行时 ST + golden 比对），
且每个合法格式/参数组合都有对应验证用例。lit 层仅承担格式合法性的正反向 IR 检查，不计入本口径。

**执行计划见 [`test/tilelib-st/TILEOP_TEST_PLAN.md`](../tilelib-st/TILEOP_TEST_PLAN.md)**，要点：

- 载体：`test/tilelib-st/a5/<op>/case.py`（PTODSL 编写 → 编译执行 → golden 比对）；
- 142 个 tile op 分三类：**110 个**可直接经 `pto.tile.*` 编写；**19 个**需先扩 ptodsl 绑定
  （`tmatmul.bias`/`tquant`/`tpow` 等，列表见计划 §4.2）；**13 个**生命周期/同步 op 不适用 per-op
  数值测例（auto 模式隐式覆盖，行为验证归 lit）；
- 现状基线：tilelib-st 仅 28 个 op 目录 / 36 个运行时用例，格式覆盖单一；
- 格式轴（dtype / loc / blayout+slayout / valid region / 操作数形态 / op 特有参数）的合法取值
  以 verifier + 展开模板 + ptodsl 绑定三者交集为真值，Phase 1 逐 op 提取格式矩阵；
- 粗估总规模 1000–1500 个用例，分 P0（每 op 1 用例基线）→ P1（格式矩阵）→ P2（格式铺开）→
  P3（扩面统一载体）四批推进。

