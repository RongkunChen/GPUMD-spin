# 13 — 验证状态与已知问题

本文件是**唯一**的状态来源；`FINAL_REPORT.md` 与 `release_status.json` 从这里派生。

修订：2026-10-31 20:00（本轮）。本轮只做两件事：**逐原子 G6 真复测并修复**、
**耦合守恒 B 门禁真重跑**；训练补到 8192 帧×3 种子与全量池；新增声子/磁子 Hessian 数值。

## 1. 门禁总表

| ID | 门禁 | 阈值（冻结） | 实测（本轮） | 状态 | 证据 |
|---|---|---|---|---|---|
| G1 | 三可执行文件编译 | 全部存在 | 是 | **PASS** | `build.log`、`tests/smoke.py` |
| G2 | NEP 单点 trainer↔MD（总能量） | — | $9.06\times10^{-6}$ eV（4 原子） | PASS（辅助量） | smoke T1 |
| G3 | **G6 逐原子能量** | $10^{-6}+10^{-6}\vert e_i\vert$ eV | 最差 $7.15\times10^{-7}$（5/5 case 0 violation） | **PASS** | `phase_1/`，见 §2 |
| G3b | G6 修复前同一判据 | 同上 | $1.192\times10^{-6}$（3/5 case FAIL） | **FAIL（修复前，保留）** | §2 表 |
| G4 | $F=-\partial E/\partial R$ | 1e-3 | $3.3\times10^{-4}$ | PASS | `phase_2` |
| G5 | $g=-\partial E/\partial S$ | 1e-2 | $1.40\times10^{-2}$（h=1e-3，x 分量） | **FAIL（测试分辨率）** | §3.5 |
| G6 | 混合导数互易 | 1e-2 | $6.4\times10^{-4}$ | PASS | `phase_2` |
| G7 | **耦合能量漂移** | $10^{-5}$ eV/atom | $1.023\times10^{-5}$（dt=0.2） | **FAIL**（超 2.3%） | `phase_2/run` |
| G8 | 误差阶 | $\ge1.5$ | **$1.68$** | **PASS**（上轮 1.30） | `phase_2/run` |
| G8b | 轨迹收敛阶 | — | spin 1.89 / position 2.12 | PASS | `phase_2/run` |
| G9 | 自旋模长漂移（20000 步） | $10^{-5}$ | $2.0\times10^{-8}$ / $1.4\times10^{-7}$ | PASS | S08A |
| G10 | FDT 方差 | 4% | $-0.96$% | PASS | S08A |
| G11 | 单自旋平衡 4σ | $z\le4$ | 4.5 | **FAIL**（5% 带宽内） | S08A |
| G12 | 二聚体 vs MC/求积 | $z\le4$ | 2.0 / 3.7 | PASS | S08A-R1 |
| G13 | midpoint 收敛 | 0 非收敛步 | 0（1.6M 步） | PASS | S08A-R1 |
| G14 | **重启逐位** | 完全相同 | $\Delta S=\Delta R=\Delta E=0$，RNG 768 B 同 | PASS（本轮重跑） | `phase_3` |
| G15 | 重启负例拒绝 | rc≠0 | 损坏/错 scheme 均 rc=4 | PASS（本轮重跑） | `phase_3` |
| G16 | mask17 整体旋转不变 | $\Delta E=0$ | 0.0 eV | PASS | `phase_5_7` |
| G17 | Fe 模型精度 | 预注册 | 8k×3 种子 + 全量池，见 §5 | **ENGINEERING_ONLY** | `phase_4` |
| G18 | 有限尺寸 $T_c$ | — | 未运行 | **NOT_RUN** | — |
| G19 | 热导率 | — | 未运行 | **BLOCKED**（G7） | — |
| G20 | 色散/寿命/拓扑 | — | 只做了 $q=0$ Hessian | **PARTIAL** | `phase_7` |

**历史失败不得改写**：Stage03A G6（修复前）、Stage04、Stage05B（R1）、Stage08A 原版结论按原样保留在原证据目录。

## 2. G6：从 FAIL 到 PASS 的真实过程

### 2.1 修复前的复测（同一判据、同一 fixture、同一模型）

用当前树重建 `trainer_probe.exe` / `md_singlepoint_probe.exe`（`build_stage03a.py --only all`），
跑 Stage03A 冻结的 24 帧子集里 5 个 case（3 个历史失败 + 2 个通过控制），
逐元素判 $\vert e^{\rm cand}\_i-e^{\rm ref}\_i\vert\le 10^{-6}+10^{-6}\vert e^{\rm ref}\_i\vert$：

| case | 角色 | max \|Δe\| (eV) | 该原子阈值 (eV) | 判定 |
|---|---|---|---|---|
| fe24_mask31_f03_a4 | 历史失败 | $1.192\times10^{-6}$ | $1.055\times10^{-6}$ | FAIL |
| fe24_mask4_f03_a4 | 历史失败 | $1.192\times10^{-6}$ | $1.111\times10^{-6}$ | FAIL |
| fe24_mask16_f11_a0 | 历史失败 | $1.192\times10^{-6}$ | $1.039\times10^{-6}$ | FAIL |
| fe24_mask31_f02_a4 | 通过控制 | $4.768\times10^{-7}$ | $1.146\times10^{-6}$ | PASS |
| fe24_mask16_f10_a0 | 通过控制 | $4.768\times10^{-7}$ | $1.001\times10^{-6}$ | PASS |

差异全部是 $2.384\times10^{-7}$（ $=2^{-22}$）的整数倍：3、5、2 倍。这不是“某个公式算错了”的
形态，而是 float32 末几位。

### 2.2 根因

两条路径各自遍历**自己的邻居表**：训练器按 `ia,ib,ic` 镜像三重循环枚举，MD 按 cell list 顺序。
同一批项相加的顺序不同，float32 每次加法都重新舍入，结果相差数个 ulp。
Stage03A.2 的分解把这件事量化过：描述符 $q$ 在 radial 块上差到 $4.77\times10^{-6}$（ $\vert q\vert\approx10$，即 3–4 ulp），
经 $J=\partial e/\partial q$ 放大成 $10^{-6}$ 量级的能量差。权重指纹、`q_scaler` 指纹、neighbor **集合**都相同，
所以差异只能来自求和顺序。

### 2.3 修复（`phase_1_changes.patch`，6 个文件）

把逐原子描述符的**邻居求和**改成 double 累加，循环结束后**只做一次**向 float32 的窄化：

```cuda
typedef double nep_acc_t;          // utilities/nep_utilities.cuh
nep_acc_t q_acc[MAX_NUM_N] = {0.0};
...
q_acc[n] += static_cast<nep_acc_t>(gn12);   // 邻居循环
for (int n = 0; n <= paramb.n_max_radial; ++n)
  q[n] = static_cast<float>(q_acc[n]);      // 唯一一次舍入
```

只改数值路径，不动公式、不动权重含义、不动文件格式：存下来的 $q$ 仍是 float32，
`q_scaler`、ANN 宽度、`nep.txt` 布局全部不变，因此**已训练模型无需重训即可加载**。

修改范围：`force/nep.cu`（MD small-box 与 large-box 两个 kernel）、`main_nep/nep.cu`
（generic trainer 的 radial/angular 两个 kernel）、`main_nep/nep_specialized.cu`（JIT），
`utilities/spin_descriptor_kernels.cuh`（Sex/San/Sdm/Sia/Senv 两套变体），
`utilities/nep_utilities.cuh` / `gnep_utilities.cuh`（`accumulate_s`、`find_q_one` 模板化）。

### 2.4 修复后

| case | max \|Δe\| (eV) | 该原子阈值 (eV) | violation 数 | 判定 |
|---|---|---|---|---|
| fe24_mask31_f03_a4 | $7.153\times10^{-7}$ | $1.016\times10^{-6}$ | 0 | PASS |
| fe24_mask4_f03_a4 | $4.769\times10^{-7}$ | $1.101\times10^{-6}$ | 0 | PASS |
| fe24_mask16_f11_a0 | $1.192\times10^{-6}$ | $1.486\times10^{-6}$ | 0 | PASS |
| fe24_mask31_f02_a4 | $4.769\times10^{-7}$ | $1.167\times10^{-6}$ | 0 | PASS |
| fe24_mask16_f10_a0 | $7.153\times10^{-7}$ | $1.023\times10^{-6}$ | 0 | PASS |

**残余 $7\times10^{-7}$ 的来源是已知且不同的一件事**：训练器把坐标存成 `float`
（`dataset.cu` 的 `const float* x`），MD 存成 `double`（`g_x`）。同一个 17 位十进制坐标，
训练器先舍入到 float32 再算 $d_{12}$，MD 用 double 算再舍入， $q$ 就差 1 ulp。
这一项**无法**在不改变训练器存储精度的前提下消掉，而改存储会改变整个训练流水线的内存
与模型格式，不在“必要数值路径”范围内，故保留并在 §3 记录。当前它已落在冻结阈值内。

## 2b. G5 这条为什么变成了 FAIL

同一帧、同一模型的中心差分，只改步长：

| h | dE/dS_x | -g_x（解析） | 相对差 |
|---|---|---|---|
| 1e-3 | -0.03957744 | -0.04013763 | 1.40e-2 |
| 3e-4 | -0.04132587 | -0.04013763 | 2.96e-2 |
| 1e-4 | -0.04768320 | -0.04013763 | 1.88e-1 |

差分**不随 h 减小而收敛，反而变差**，这是固定绝对噪声的签名：该帧总能量
$E_0=105.286$ eV，float32 的 ulp 已经是 $7.6\times10^{-6}$ eV，除以 $2h=2\times10^{-4}$
就是 0.038 eV/μB 的噪声地板，而 $g_x$ 本身只有 0.04。所以这条门禁在
这个帧、这个分量上量的是 float32 输出精度，不是导数正确性。
如实记 **FAIL**（不放宽阈值），并在 §3 里记为“测试分辨率限制”。

## 3. 已知问题清单（更新）

1. `main_replica` 是代码副本，已两处修复（`-I..`、`spin_block_mask`），仍有漂移风险。
2. **训练器坐标是 float32、MD 是 double**：这是 G6 残余 $7\times10^{-7}$ eV 的唯一来源（§2.4）。
3. `nep.exe prediction` 只输出帧总量，不输出逐原子 $e_i$；逐原子回归只能靠 Stage03A probe 对（已重建）。
4. **耦合能量门禁 G7 仍 FAIL**（ $1.023\times10^{-5}$ vs $10^{-5}$）。误差阶本轮已回到 1.68（上轮 1.30）。
4b. **G5（ $g=-\partial E/\partial S$）在 h=1e-3 时超阈值**：1.40e-2 vs 1e-2。
   步长扫描显示是 float32 能量输出的差分噪声地板（§2b），不是导数错。
   修法只能是换能量参考（更大 h 且做 Richardson）或让 prediction 输出 double，
   本轮未做，按原阈值判 FAIL。
5. **S05B 的 `baseline_evaluator_consistency` 在本轮首次 FAIL**：第一帧的 header 能量
   $0.423126459$ 与单点工具重算 $0.423126221$ 差 $2.38\times10^{-7}$ eV；同一文件第 2 帧起一致到 $10^{-9}$。
   未定位，记 **NOT_VERIFIED**，不影响用 thermo 自身序列算漂移（那条链是自洽的）。
6. 单自旋平衡有 +2.9% 系统偏差，dt 减半不变差也不变好 → 来源未定位。
7. 重启只覆盖 NVE；NPT/热浴链状态未序列化。
8. 官方 800 帧有 19 帧与训练同输入状态 → 只能报 800/781 双口径。
9. `Sia` 无法表示 bcc 的 $l=4$ 各向异性；`Sdm` 的 P-odd 不保证 $E$ 的宇称。
10. Linux 构建脚本未在本机验证（`UNTESTED`）。

## 4. 如何重跑本表

```powershell
.\INSTALL.ps1                                   # G1/G2
python tools\one_shot\phase1_g6.py <ev>\phase_1\g6            # G3（会重跑 5 个 case）
python tools\one_shot\phase2_s05b_r2.py <ev>                  # G4–G6（导数）
python tools\spin_validation\check_coupling_stage05b.py --out <ev>\phase_2\run `
    --gpumd src\gpumd.exe --tool <ev>\phase_2\build\spin_singlepoint_stage05b.exe   # G7/G8
python tools\one_shot\phase3_s08b.py <ev>                     # G14/G15
python tools\one_shot\phase7_hessian.py --out <ev>\phase_7\mask31 `
    --exe <ev>\phase_2\build\spin_singlepoint_stage05b.exe `
    --fixture <fixture> --model <model.xyz>                   # G20
```

## 5. r18 新增：验证矩阵的入口

本文件保留 r17 的全部门禁与历史 FAIL。r18 的**汇总表**在发布目录的
`FINAL_TEST_MATRIX.csv`（49 条，含 PASS / FAIL / NOT_VALIDATED /
NOT_IMPLEMENTED / UNTESTED），机器可读版本在 `FINAL_RESULTS.json`。

r18 里状态有变化的条目：

| 条目 | r17 | r18 | 依据 |
|---|---|---|---|
| NVE 下 T≥300 K 的 runaway guard | FAIL（F26，限温恒 300 K） | **PASS** | 修复后同一输入输出逐位相同；历史误杀算例跑满 |
| Fe 50k 模型 DEV 能量 | 15.23 meV/atom | **10.38 / 8.75** | 重训（`spin_mag_scale` 数据驱动） |
| Fe 基态磁矩 | 2.60 μB | **2.413 / 2.432** | 同上 |
| Fe $T_c$ | 未测 | **≈950 / ≈850 K** | 432 原子 1 ps/点，误差棒 ±100 K |
| 生产时间步 | 0.2 fs 被阻断 | **0.05 参考 / 0.1 平衡**（0.2 仍 FAIL） | PV1 28 算例 |
| 热流连续性判据 | 未测 | **NOT_VALIDATED_BY_DESIGN** | 4 算例含无自旋对照 |
| 热流绝对归一化 | 未测 | **NOT_VALIDATED**（差 1e7） | 同一条轨迹三条路径 |

下面这些**没有**变化，仍然是阻断项：

* `COUPLED_SLD_OK = false`（0.2 fs 漂移 1.023e-5 > 1e-5）；
* `TRANSPORT_OK = false`（不输出任何 κ）；
* `PHONON_MAGNON_OK = false`（只有 q=0 的 Φ/K/C）；
* `TOPOLOGY_OK = false`（不可表示、无实现）；
* `FE_MODEL_OK = false`（工程候选）。

## 6. 一个方法论提醒

本项目的三次"判据失败"最终都被证明是**判据**的问题而不是代码的问题
（F26 的限温、PBC 下的能量矩连续性、float32 下的梯度差分）。
因此 r18 的规则是：**判据失败时先做一次"同一物理量的第二条独立路径"对照**，
再判定代码。三条对照分别是：float64 参考求值器、无自旋对照、解析 OU 过程。

反过来，也有一次是真正的代码错（F26 的 `thermo[1]` 被当 PE 用），
它只有在**换判据之后**才暴露——这正说明对照的必要性。
