# 16 — 发布状态与能力清单（r18）

**版本**：`v3.1.0-r18`　**标签**：`RESEARCH_PREVIEW / PARTIALLY_VALIDATED`
**判定原则**：按功能逐项标注；**不使用单一 "all_pass"**。

## 1. 状态词表

| 词 | 含义 |
|---|---|
| `READY` | 本机实测通过，有可复现证据 |
| `EXPERIMENTAL` | 能用，但未独立验证或只在窄条件下验证 |
| `BLOCKED` | 前置门禁未过，**禁止**用于物理结论 |
| `NOT_RUN` | 未运行（预算/模型/依赖） |
| `NOT_IMPLEMENTED` | 代码路径不存在 |
| `NOT_VALIDATED` | 代码存在，判据/归一化未确认 |
| `NOT_REPRESENTABLE` | 当前模型在物理上无法表示该量 |

## 2. 命令与功能（r18）

| 功能 / 命令 | 状态 | 说明 |
|---|---|---|
| `gpumd.exe` 静态单点、NVE / NVT-Langevin MD | **READY** | 含 F26/F27 修复；NVE 在 T≥300 K 可正常运行 |
| `nep.exe` 训练（NEP4-spin，mask17/31） | **READY** | 50k 代实测；DEV 10.4 / 8.8 meV/atom |
| `nep.exe` 预测（`prediction 1`） | **READY** | 只输出帧总量，**无逐原子 $e_i$** |
| `gpumd_replica.exe`（REMD） | **READY** | 2-replica 短跑（smoke T6） |
| `spin_llg`（保守分支 $\alpha=T=0$） | **READY** | 模长漂移 ≤1e-4 μB（10 ps） |
| `spin_llg scheme=geometric`（$T>0$） | **READY** | FDT 门禁见 `docs/13` |
| `spin_llg scheme=midpoint`（$T>0$） | **READY** | 1.6 M 步 0 次未收敛 |
| `dump_restart` + `restart=` | **READY** | 逐位一致 + RNG 续接 |
| `dump_magnetization`（M, χ, Msat） | **READY** | 本版 $T_c$ 用它（冒烟 + 生产使用） |
| `add_bfield`（含 FMR 正弦驱动） | **EXPERIMENTAL** | 有代码，未做定量验证 |
| `compute_sed`（velocity / spin）、`compute_dos` | **EXPERIMENTAL** | 有代码，未与解析谱对比 |
| `dump_heat_flux` / `compute_heat` | **BLOCKED** | 绝对归一化未验证（差 1e7） |
| `compute_hac` | **BLOCKED** | 在自旋体系上给负值，5 分量分解丢交叉项 |
| 耦合动力学（S–V–S） | **PARTIAL** | 0.05/0.1 fs 可用；0.2 fs 门禁 FAIL |
| 声子/磁子 Hessian（$\Phi,K,C$，$q=0$） | **PARTIAL** | $\Phi,C$ 验证过；$K$ 的 Goldstone 判据未完成 |
| 声子/磁子色散、寿命 | **NOT_IMPLEMENTED** | 需切空间投影 + 超胞 $q$ 网格 |
| Berry / 拓扑 | **NOT_IMPLEMENTED** | 无布洛赫基；且模型无 SOC 证据 |
| 有限尺寸 $T_c$（Binder） | **NOT_RUN** | 只做单尺寸 1 ps 扫描 |
| Linux 构建 | **UNTESTED** | 脚本已给，未验证 |

## 3. 顶层状态

```json
{
  "BUILD_OK": true,
  "STATIC_EFG_OK": true,
  "SLLG_OK": true,
  "RESTART_OK": true,
  "GUARD_FIX_OK": true,
  "COUPLED_SLD_OK": false,
  "COUPLED_SLD_SHORT_DT_OK": true,
  "TRANSPORT_OK": false,
  "PHONON_MAGNON_OK": false,
  "TOPOLOGY_OK": false,
  "FE_MODEL_OK": false,
  "FE_MAGNETIC_PHYSICS_OK": "ENGINEERING_ONLY"
}
```

判定依据：

- `BUILD_OK`：源码 ZIP 解压到全新目录后可编译出三个 exe 并通过 smoke/acceptance。
- `STATIC_EFG_OK`：G6 逐原子门禁 5/5 无 violation，最差 $7.153\times10^{-7}$ eV
  （混合判据）；纯绝对 $10^{-6}$ 判据仍 4/5（历史 FAIL 保留）。
- `SLLG_OK`：FDT / 平衡 / 二聚体 / 重启均通过（单自旋 4σ 那一项未过，记录在案）。
- `RESTART_OK`：三组连续 vs 分段逐位相同，RNG 768 B 一致，负向控制全拒绝。
- `GUARD_FIX_OK`：F26/F27 修复前后同一输入输出 sha256 相同；历史误杀算例新版跑满。
- `COUPLED_SLD_OK = false`：0.2 fs 漂移 $1.023\times10^{-5} > 10^{-5}$（超 2.3%），
  误差阶 1.68 ≥ 1.5 通过。**保留 FAIL**。
- `COUPLED_SLD_SHORT_DT_OK = true`：0.1 / 0.05 fs 短时（0.1 ps）$3.18/1.64\times10^{-6}$，
  且长时间（10 ps）斜率恒为 $4.6\times10^{-9}$ eV/atom/ps。
  **适用范围**：Fe / mask17 / $\alpha=0$ / NVE / 250 K / ≤10 ps。
- `TRANSPORT_OK = false`：被 `COUPLED_SLD_OK` 与热流归一化双重阻断。
- `PHONON_MAGNON_OK = false`：只有 $q=0$ 的 $\Phi/K/C$；无色散、无 DOS 验证、无寿命。
- `TOPOLOGY_OK = false`：机制不可表示（无 SOC），且代码不存在。
- `FE_MODEL_OK = false`：50k 代模型 DEV 8.8–10.4 meV/atom，仍只能作工程候选。
- `FE_MAGNETIC_PHYSICS_OK = "ENGINEERING_ONLY"`：$a_0$ −1.4%、$\mu_0$ +9%、
  $T_c$ −9%/−18%（误差棒 ±100 K）。可作半定量参考，不可作定量势的物性预言。

## 4. 使用限制（必须转述给最终用户）

1. 不得用本包对 Fe 报告 $\kappa$、磁子寿命、DMI、$K_1$ 的数值。
   $T_c$、$a_0$、$\mu_0$ 只能作为**带误差棒**的半定量结果报。
2. 不得把 `mask31` 当作"已验证 SOC 模型"。
3. 不得把 `spin_restart.bin` 的逐位一致推广到 NPT/热浴链。
4. 不得声称时间步政策适用于 Fe 之外的材料或超过 10 ps 的时长（外推上限 2 ns 只是
   线性漂移的估计，未实测）。
5. 若要发表，只能引用标注为 PASS/READY 的部分，并显式声明未验证项。

## 5. 升级路径

| 想解锁 | 需要先做 |
|---|---|
| `COUPLED_SLD_OK` | 定位 0.2 fs 下 85% 的积分截断来源（R2 已证明不是 float32）；或正式把生产步长定为 0.05 fs |
| `TRANSPORT_OK` | 上面 + 用 LJ 解析模型钉死热流绝对归一化 + 70 ps 采样 |
| `FE_MODEL_OK` | 更大数据集 / 更长训练 / 预注册多目标选择；当前 50k 代是预算上限 |
| `PHONON_MAGNON_OK` | 实现切空间投影 $K\to\tilde K$，再做 $\mathbf q$ 网格 |
| `TOPOLOGY_OK` | 含 SOC 的数据与模型（当前不具备） |
| `GUARD_FIX_OK` 的更强形式 | 把 guard 的判据整体改写为基于"能量收支恒等式"，而不是阈值组合 |
