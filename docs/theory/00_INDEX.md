# 理论与方法文档索引

本目录按统一规范组织：每篇给出
**数学定义 → 物理假设 → 数值算法 → 代码入口 → 单位 → 验证方法与实测结果 → 已知限制**。
公式编号篇内独立；代码入口写成 `文件:函数`，行号随版本漂移，所以只给函数名。

版本：**r18**（2026-11-01）。相较 r17 的变化集中在 F26/F27 缺陷修复与文档补强，
物理公式与积分格式未改。

| 文件 | 内容 | 关键结论 |
|---|---|---|
| `01_HAMILTONIAN.md` | 总 Hamiltonian、局域能量分解、单位制度 | 能量分解与 dump 一致（<1 ulp） |
| `02_DESCRIPTORS.md` | Sex/San/Sdm/Sia/Smag、掩码、镜像枚举 | 描述符与 float64 参考差 1.2e-8 |
| `03_SYMMETRY_PARITY_CHIRALITY.md` | 空间/时间反演、旋转协变、手性 | 无单离子各向异性（结构限制） |
| `04_ANN_AND_TRAINING.md` | ANN 结构、损失、进化算法、可复现性 | 同 seed 同数据 → 同模型 |
| `05_FORCES_GRADIENTS_VIRIAL.md` | $\mathbf F$、$\mathbf g$、virial、Maxwell 互易 | 梯度 PASS 2.6e-5；混合 Hessian 1.3e-3 |
| `06_LLG_FDT_INTEGRATORS.md` | LL/Gilbert、FDT、Stratonovich、三种 scheme | 二聚体 midpoint 偏差 3% |
| `07_CONSERVATIVE_SLD.md` | S–V–S 分裂、等时输出、**时间步政策** | 0.05 fs 参考 / 0.1 平衡 / 0.2 不建议 |
| `08_MC_AND_FINITE_SIZE.md` | Metropolis 测度、固定模长、Binder | MC 可用；有限尺寸标度未实现 |
| `09_PHONON_MAGNON_HESSIAN.md` | $\Phi$、$K$、$C$、动力学矩阵、ASR | 静态 Hessian 可用；谱 BLOCKED |
| `10_MAGNETOELASTIC_COUPLING.md` | 耦合矩阵元、反交叉、模态投影 | 矩阵元 PASS；模态分解未实现 |
| `11_ENERGY_CONTINUITY_HEAT_FLUX.md` | 局域连续性、热流、**PBC 面项** | 连续性判据在 PBC 下不成立 |
| `12_GREEN_KUBO.md` | GK、单位、统计收敛 | 估计器 PASS；5 ps 不收敛 |
| `13_SPECTRA_LIFETIME_TOPOLOGY.md` | SED、线宽、Berry/Wilson | 未实现（接口留白） |
| `14_NUMERICAL_PRECISION.md` | float32/double、误差预算、长时间漂移 | 4.6e-9 eV/atom/ps → 2.2 ns |
| `15_DEFECTS_AND_FIXES.md` | F19–F27 缺陷日志 | F26/F27 已修并三重验证 |

## 推荐阅读路径

1. **想跑起来**：`../docs/03_QUICKSTART_AND_EXAMPLES.md` → `07` §7.5（时间步）→
   `../USER_MANUAL.md`
2. **想理解模型**：`01` → `02` → `05` → `04`
3. **想改代码**：`05` → `14` → `../DEVELOPER_GUIDE.md` → `15`
4. **想做磁性物理**：`02` → `06` → `08` → `09`
5. **想做热输运**：`11` → `12`（先读"为什么不报数"）

## 状态约定

每篇末尾的表格使用统一词表：

| 词 | 含义 |
|---|---|
| PASS | 有可复现的定量证据，且达到声明的判据 |
| FAIL | 有可复现的定量证据，未达到判据（保留原始数据，不放宽阈值） |
| PARTIAL | 部分算例通过、部分未通过，或只在小体系验证 |
| BLOCKED | 前置条件未满足，不允许给出结论 |
| NOT_IMPLEMENTED | 代码路径不存在 |
| NOT_VALIDATED | 代码存在，但判据/归一化未得到独立确认 |
| NOT_VALIDATED_BY_DESIGN | 判据本身不成立（例如 PBC 下的能量矩判据） |
| ENGINEERING_ONLY | 能跑、数值稳定，但不能当定量物理结论使用 |

所有判定的汇总以 `../VALIDATION_REPORT.md` 与交付包里的
`FINAL_TEST_MATRIX.csv` 为准，本文档不重复给总表。
