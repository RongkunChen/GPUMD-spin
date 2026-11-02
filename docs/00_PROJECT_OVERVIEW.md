# 00 — 项目总览（GPUMD-spin V3）

**版本**：`v3.0.0-r17`（上游 GPUMD r17 + 自旋扩展）　**状态**：RESEARCH_PREVIEW
**适用范围**：磁性体系的经验势（NEP4-spin）训练、自旋动力学、自旋–晶格耦合、重启与复现。

## 1. 这个项目是什么

在 GPUMD（GPU 分子动力学，r17）之上加了一条**磁自由度**支路：

- 局域能量模型 $E(\mathbf R,\mathbf S)=\sum_i e_i(\mathbf R,\mathbf S)$，其中 $\mathbf S_i$ 是原子 $i$ 的磁矩矢量（单位 $\mu_B$，由 `spin_length * initial_magmoms` 给出）。
- 自旋描述符把 $\mathbf S_i,\mathbf S_j$ 与键向 $\hat{\mathbf r}_{ij}$ 耦合进 NEP 的局域环境，因此能量、力 $\mathbf F=-\partial E/\partial\mathbf R$、磁梯度 $\mathbf g=-\partial E/\partial\mathbf S$ 来自**同一个**标量函数。
- 在此能量面上提供：LLG 自旋动力学（有限温度、含 FDT 噪声）、保守自旋–晶格分裂积分、完整重启、REMD 副本、以及静态/动力学分析工具。

## 2. 与上游 GPUMD 的真实差别

| 项 | 上游 GPUMD r17 | 本项目 |
|---|---|---|
| `model.xyz` | species/pos/vel | 额外 `spin:R:3`（磁矩， $\mu_B$） |
| `nep.txt` | 无自旋 | `spin 1` + `spin_layout` + `spin_blocks` 掩码 |
| `nep.exe` | 无 `lambda_m` | 增加磁梯度损失 $\lambda_m$，输出 `spin_train.out` |
| `gpumd.exe` | 无自旋 | `spin_llg`、`spin_split`、`spin_substeps`、`dump_restart` 写 `spin_restart.bin` |
| 重启 | `restart.xyz`（R,v） | 追加 `spin_restart.bin`：RNG 状态 + 模长锚点 + 身份校验 |
| `gpumd_replica` | 独立构建 | 同步了 `spin_block_mask` 解析，可与主树一起编译 |

## 3. 分层能力（本版本实测）

| 层 | 内容 | 状态 | 证据 |
|---|---|---|---|
| I | 编译工程（`gpumd`/`nep`/`gpumd_replica`） | **PASS** | `tests/smoke.py` T1–T6 全过；`build.log` |
| II | Hamiltonian 静态导数/对称性 | **PARTIAL** | $F=-{\partial E}/{\partial R}$、 $g=-{\partial E}/{\partial S}$、混合导数互易：相对差 $3.3\times10^{-4}$、 $2.1\times10^{-3}$、 $6.4\times10^{-4}$；**逐原子 trainer↔MD 门禁 G6 仍 FAIL**（1.19e-6 > 1.05e-6 eV） |
| III | 随机/耦合动力学 | **PARTIAL** | 重启逐位一致 PASS；耦合能量漂移 1.067e-5 > 1e-5 FAIL |
| IV | 训练集外物理预测 | **NOT_VALIDATED** | 只有 2k/8k 帧的工程候选，未达发布级精度门槛 |

> 结论只按层给。禁止把 I 层通过当成“物理可用”。

## 4. 能力边界（务必先读）

1. **没有 SOC 数据**：mask17（`Sex|Smag`）是无 SOC 模型；mask31（五块全开）只是“SOC-like 表示”，不证明数据含 SOC，也不能据此报 DMI/K1/MAE 数值。
2. **`Sia` 不能表示 bcc 的四阶立方各向异性**：它只到 $l=2$ 的四极矩，完美立方环境恒零。
3. **`Sdm` 是 P-odd 描述符**：单个描述符的反演奇性不等于总能量满足宇称；只有联合旋转下不变。
4. **`Smag` 在 $|\mathbf S|=0$ 不可微**：实现给 $|\mathbf S|>0$ 的精确导数，原点写 0 是约定。
5. **固定模长 vs 可变模长**：SLLG 固定每个原子的 $|\mathbf S|$；与允许 $|\mathbf S|$ 涨落的 MC 不可直接比较同一统计。
6. **耦合守恒未过**：`COUPLED_NOT_VALIDATED`，因此热导率/耦合输运一律 BLOCKED。

## 5. 版本与支持范围

- 上游：GPUMD r17（许可证见 `LICENSES/`，第三方归属保留）。
- 本机验证环境：Windows 11 + MSVC 2022 + CUDA 12.8 + RTX 4060 Ti (sm_89)。
- Linux：构建脚本已给（`build.sh`），**未在本机验证**，标 `UNTESTED`。
- 复现入口：`INSTALL.ps1` → `tests/smoke.py` → `REPRODUCIBILITY.md`。

## 6. 文档地图

理论：`04`（Hamiltonian/描述符）、`05`（力/磁梯度/应力）、`06`（LLG/FDT/MC）、`07`（耦合积分/重启）、`08`（热流/GK）、`09`（声子/磁子/寿命）、`10`（Berry/拓扑）。
工程：`01`/`02`（安装）、`03`（快速上手）、`12`（开发者架构）。
验收：`11`（训练与验证）；顶层 `VALIDATION_REPORT.md`（验证结果）、`KNOWN_LIMITATIONS.md`（已知限制）、
`REPRODUCIBILITY.md`（复现命令）。
