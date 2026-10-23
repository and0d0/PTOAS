# TileOp 全格式数值验证测试计划

> 目标（已锁定口径）：**每个 tile op（142 个）都有 kernel 级、验证数值正确性的测例；
> 且每个合法格式/参数组合都有对应的运行时验证用例。**
> 配套 op 清单与现有测例位置见 `test/samples/OP_INVENTORY.md`（基线 `37f5f25e3`，2026-10-21）。

## 1. 测例定义（什么算一个测例）

一个测例 = `test/tilelib-st/a5/<op>/case.py` 中的一个 `golden_output_case`：

- 用 PTODSL（`pto.tile.*`）编写单 op kernel，`@pto.jit` 编译；
- 经 PTOAS 编译为 kernel，在 torch_npu / 仿真器运行时**真实执行**；
- 输入由确定性种子生成，输出与 numpy/torch golden **数值比对**（rtol/atol 显式声明）；
- 一个「格式」= 一组 (dtype, shape, layout, valid region, 操作数形态, op 特有参数) 的具体取值，
  每种合法组合至少一个 case（用例名编码格式，如 `tdiv_i32_soft_1x64_reciprocal`）。

运行方式：

```bash
python3 test/tilelib-st/a5/__main__.py            # 全量
python3 test/tilelib-st/a5/__main__.py <op>       # 单 op（按现有 __main__ 约定）
```

## 2. 格式/参数轴与真值来源

| 轴 | 取值来源（真值） |
|---|---|
| dtype | C++ verifier（`lib/PTO/IR/PTOTile/*.cpp`）+ **展开/lowering 模板**（文档明确 dtype legality 是 target-defined，verifier 之外的合法性在展开期决定） |
| loc（vec/l0a/l0b/l0c/bias/scaling） | 各 op verifier + tile 类型约束 |
| blayout/slayout/fractal/pad | `PTOTile/PTOMoveLayoutVerification.cpp`、`PTOBufferSyncAndMatrixOps.cpp` 等 |
| valid region（静态部分有效 / 动态 `?`） | `PTOTile/PTOExtentCompatibility.cpp` |
| 操作数形态（tile×tile / tile×scalar / 两种次序） | ODS assemblyFormat + 文档（如 `tdivs` 双操作数次序） |
| op 特有参数（precision、acc_phase、tmp 等） | ODS 属性 + ptodsl 绑定签名（`ptodsl/ptodsl/_ops_tile.py`） |

> 注意：verifier 通过 ≠ 展开模板支持。**Phase 1 首个任务就是逐 op 抽取
> "verifier ∩ 展开模板 ∩ ptodsl 绑定"的合法格式矩阵**，输出为本计划的附表（后续补充）。

## 3. 现状基线（commit `37f5f25e3`）

- 现有 28 个 op 目录 / 36 个运行时用例（部分经循环参数化展开为更多 shape）：
  `tadd tci tcolexpand tcolexpandbinop tcolexpanddiv tcolsum tconcat tcvt tdequant tdiv tdivs
  tfmod tfmods tgather tgatherb tgemv thistogram tload_store tmatmul tprelu trandom trem trems
  trowexpand tscatter tsincos ttrans ttri`
  （`tsincos`、`tcolexpandbinop` 为组合用例目录，非清单内独立 tile op；`tload_store` 同时覆盖 tload/tstore）
- 格式覆盖极浅：绝大多数 op 仅 1 种 dtype×shape（f32/f16/i32 为主），未覆盖
  bf16/i8/i64/fp8、col_major、部分有效域、动态 valid 等变体。

## 4. 差距分析（142 个 tile op 分三类）

### 4.1 可直接编写（110 个）— 主战场

`pto.tile.*` 命名空间（`ptodsl/ptodsl/_tile_namespace.py`）已暴露 110 个 tile op
（加减乘除/xxs 变体、exp/log/sqrt 等 SFU、行列归约 13 个、row/col expand 16 个、
比较选择、位运算移位、gather/scatter、matmul/gemv 11 个、tload/tstore、mgather/mscatter 等）。
**动作：每 op 建 `a5/<op>/` 目录 + 按 Phase 1 格式矩阵铺 case。**

### 4.2 ptodsl 面未暴露（19 个）— 前置扩面任务

`taddc, tsubc, taddsc, tsubsc, taxpy, tpow, tpows, tinterleave, tdeinterleave,
tconcatidx, tpartargmax, tpartargmin, tquant, tquant.mx, tget_scale_addr,
tsetval, tgetval, tprefetch, tmatmul.bias`

- 建议路径：在 `_ops_tile.py` + `_tile_namespace.py` 补绑定（`tmatmul.bias` 尤其反常——
  `tmatmul.mx.bias` 与 `tgemv.bias` 都已暴露，仅它缺席）；
- 临时替代载体：samples 式裸 Python 绑定（`_pto.TPowOp` 直构，`samples/Tpow/` 已有先例），
  但长期应统一到 tilelib-st。

### 4.3 生命周期/同步类（13 个）— 不适用 per-op 数值测例

`talloc, tpush, tpop, tfree, tsync`（A2/A3 伪 op）与
`talloc_to_aiv/aic, tpush_to_aiv/aic, tpop_from_aic/aiv, tfree_from_aic/aiv`（A5 互操作）。

ptodsl auto 模式下 tile 分配/同步由 PTOAS 自动插入，**每个 tilelib-st 用例都在隐式执行
这些 op**；它们没有独立的数值语义（搬运/管理类），per-op 数值测例不适用。
其行为验证归 lit 层（`lit/pto/tpush_tpop_*`、`TPipe`/`TPushTPop` 样例已覆盖）。
142 − 13 = **129 个 op 需要 per-op 数值测例**（110 直接 + 19 扩面后）。

## 5. 规模估算与分批

格式矩阵未提取前只能粗估：典型 op 合法 dtype 4–6 种 × layout 1–2 种 × valid 变体 2 种
≈ 每 op 8–12 个格式用例 → 129 op ≈ **1000–1500 个用例**（Phase 1 完成后按矩阵修正）。

| 批次 | 内容 | 产出 |
|---|---|---|
| **P0 基线补齐** | 129 op 每个先落 1 个最小数值用例（f32 或 op 最典型 dtype） | 覆盖率 0→100% |
| **P1 格式矩阵提取** | 逐 op 读 verifier+展开模板+绑定，产出"每 op 合法格式表"（附于本文件） | 格式真值清单 |
| **P2 格式铺开** | 按 P1 矩阵逐 op 补齐 dtype/layout/valid/参数变体 | 每格式 ≥1 用例 |
| **P3 扩面** | 19 个未暴露 op 补 ptodsl 绑定（可与 P0 并行，先走裸绑定保底） | 129 op 全部统一载体 |

## 6. 验收标准

1. 129 个适用 op（§4.1+§4.2）每个在 `test/tilelib-st/a5/<op>/` 有 `case.py`，
   `CASES` 非空，`__main__.py` 全绿；
2. 每 op 的用例集合覆盖 P1 格式矩阵的全部合法组合（非法组合由 lit 负向用例保证拒绝，另行跟踪）；
3. `OP_INVENTORY.md` 的覆盖列同步更新，保持单一事实来源。

## 7. 当前缺口速查（P0 待建目录清单）

129 − 27（现有目录对齐的 tile op 数；`tcolexpandbinop`/`tsincos` 为组合用例目录，不计入）= **102 个 op 无 tilelib-st 目录**。其中 83 个可直接编写
（110 − 27），19 个待扩面（§4.2 全部）。明细按功能分组见 `test/samples/OP_INVENTORY.md` 第一节。
