# GPUMD-spin V3（`v3.1.0-r18`）

磁自旋扩展的 NEP 势 + GPUMD 分子动力学：训练含自旋标签的势函数，跑有限温
自旋动力学（SLLG / 保守自旋–晶格）、重启续算、副本交换，并给出可核对的
验证矩阵与已知限制。

**这是研究预览版。** 部分功能通过验证、部分阻断、部分未实现。
逐项状态见 `VALIDATION_REPORT.md`、`KNOWN_LIMITATIONS.md` 与
`docs/16_RELEASE_STATUS_AND_CAPABILITIES.md`。**不要**把"能跑"当成"验证过"。

## 三条命令上手

```bat
:: 1) 解压后进入本目录
:: 2) 编译（自动找 nvcc / MSVC / CUDA arch）
INSTALL.cmd -NoSmoke
:: 3) 冒烟测试（7 项，含重启逐位续跑）
SMOKE_TEST.cmd
```

`INSTALL.cmd` / `BUILD.cmd` / `SMOKE_TEST.cmd` 只是给对应 `.ps1` 加
`-ExecutionPolicy Bypass` 的包装器，**只对当前进程生效**，不改机器策略。

想更严格：

```bat
python tests\acceptance.py --bin src
python tests\physics_checks.py --bin src --model models\fe_mask17_2desc_r18.nep.txt --quick
```

## 目录

```
src/            完整源码（main_nep / main_gpumd / main_replica / force / spin /
                measure / model / integrate / utilities，384 个 .cu/.cuh）
docs/           理论文档（docs/theory/ 15 篇）+ 安装 / 快速上手 / 验证 / 已知问题
examples/       可直接跑的输入（synthetic / spin_llg / restart / replica / training /
                fe_bcc_ground_state / fe_curie_temperature / fe_nve_stability）
models/         Fe 模型（mask17 2 描述符、mask31 5 描述符、r17 基线）+ 训练输入
tests/          smoke.py（7 项）、acceptance.py（5 项）、physics_checks.py（5 项物理）
tools/          physics_validation/（长时间稳定性、热流审计）、numerical_accuracy/
scripts/        批量运行脚本
```

## 能力一览

| 功能 | 状态 |
|---|---|
| `gpumd.exe` 单点 / NVE / NVT-Langevin MD | **READY** |
| `nep.exe` 训练与预测（NEP4-spin，含 spin_forces） | **READY**（精度见 `VALIDATION_REPORT.md`） |
| 固定模长 SLLG（geometric / 随机隐式中点） | **READY** |
| 保守自旋–晶格（S–V–S，`alpha=0 T=0`） | **READY**（0.05 / 0.1 fs） |
| 重启续算（含 RNG 状态，逐位一致） | **READY** |
| 副本交换（`gpumd_replica.exe`） | **READY** |
| `dump_magnetization`（M、χ、Msat） | READY（本版用它算 $T_c$） |
| `add_bfield`（含 FMR 正弦驱动） | 存在，**未验证** |
| `compute_sed`（声子 / 磁子 SED）、`compute_dos` | 存在，**未验证** |
| `compute_hac` / `dump_heat_flux` | 存在，**在自旋体系上不可用** |
| 声子/磁子色散、Berry/Wilson 拓扑 | **未实现** |
| 外磁场磁化曲线、矫顽力、$K_1$ | **不可用**（模型无单离子各向异性） |

## 本版（r18）改了什么

1. **修 F26/F27**：NVE 下 runaway guard 的限温恒为 300 K（`Integrate::temperature`
   未初始化，读到 denormal）。修复后 $T\ge300$ K 的 NVE 可以正常跑；
   同一输入/模型下输出与修复前**逐位相同**。
2. **重训两个 Fe 模型**（50k 代，`spin_mag_scale` 数据驱动，$\lambda_e=\lambda_m=5$）：
   DEV 10.38 / 8.75 meV/atom；$a_0=2.8258/2.8273$ Å；$|S|_0=2.413/2.432\,\mu_B$；
   $T_c\approx950/850$ K。
3. **时间步政策**：0.05 fs 参考 / 0.1 fs 平衡 / 0.2 fs 不建议；适用范围写死。
4. **文档与示例补全**：`docs/theory/` 15 篇 + 缺陷日志 F19–F27 + 三个可跑的物理示例。

## 三条必须知道的限制

* **不输出任何热导率**：热流绝对归一化未验证（三套数字差 $10^7$），
  连续性判据在 PBC 下不成立。见 `docs/theory/11`、`12`。
* **没有单离子各向异性**：固定晶格整体旋转自旋能量不变，$K_1$ 不可表示。
* **同一 `run.in` 里 NVT→NVE 切换在 $N\ge432$ 会挂死**（未定位），拆成两次调用。

## 许可

源代码沿用 GPUMD 的 GPL-3.0（见 `LICENSES/`）；本项目的验证脚本与文档同许可。
