# CHANGELOG

## v3.1.0-r19 (2026-11-01) — 修复 F-25：单文件多段 `run` 的第二段不产数据

**只有一处行为改变，不动物理**：`run.in` 里写两个 `run` 块（典型用法：NVT 平衡 ->
NVE 生产）现在能正常工作，并且与"拆成两次 `gpumd.exe` 调用"给出**逐位相同**的结果。

### Fixed — F-25：NVT→NVE 在同一 `run.in` 里，$N\ge432$ 表现为挂死
- 症状（历史记录）：先 `ensemble nvt_lan` 再 `ensemble nve`，$N=432$ 与 $N=1024$ 上
  "进程存活、CPU 占用、26 分钟无任何输出"；拆成两次调用正常；$N=128$ 不复现。
- **根因**：`Measure::post_run()` 末尾无条件执行 `actions_.clear()`。第一个 `run` 块结束
  时所有 measure 被销毁，第二个块只做积分：`thermo.out`、`traj.xyz`、`restart.xyz`
  都不再更新，而 stdout 仍在打进度（重定向时被块缓冲，肉眼更看不到）。于是"跑得好好的
  但文件不动"被当成了挂死；$N$ 越大，等到的耐心越短，所以"阈值"看起来在 $N\approx432$。
- **修复**：`post_run()` 不再清空，而是把 dump 类 action（`dump_thermo`、`dump_xyz`、
  `dump_exyz`、`dump_netcdf`、`dump_restart`）保留下来；它们本来就是无状态追加写，
  `pre_run()` 会以 `"a"` 重新打开文件。`dump_thermo` 增加 `header_written_`，同一次进程里
  只写一次表头，避免第二段在数据中间插一段表头。时间平均类（`compute_*`）仍在段末
  总结并丢弃——它们的累加器按段定义，静默续用会把两个 ensemble 平均到一起。
- **验收**（`validation/BCC_Fe_full_physics/raw_data/nvt_nve_switch/`）：
  | 检查 | 结果 |
  |---|---|
  | $N$=250 / 432 / 1024，NVT 3000 步 + NVE 12000 步，`dump_thermo 100` | 150/150 行（旧版只有 31 行） |
  | 同上 + `dump_xyz 500` | 18/18 帧，第二段继续增长 |
  | 单块运行（NVT 或 NVE 各 2000 步）新旧二进制对比 | `thermo.out` **逐字节相同** |
  | 单文件双段 vs 两次调用（NVT 3000 → 重启 → NVE 6000） | 末态 T / PE / KE **完全相同** |
  | 第二段 60000 步长跑（旧版"26 分钟无输出"的形态） | 105 s 正常结束，无挂死 |
- **回归风险**：单块路径未改（`actions_` 仍然只有一份，`post_run` 的调用顺序不变），
  逐字节回归已证；多块路径新增了"dump 动作被复用"，对 `compute_*` 的行为没有影响。

## v3.1.0-r18 (2026-11-01) — 可用性发布

本版**不改变任何物理公式与积分格式**，修的是让程序在正常条件下无法工作的缺陷，
并把验证矩阵、时间步政策与文档补齐。

### Fixed — F26 / F27：NVE 下 guard 的限温恒为 300 K
- 症状：任何 `ensemble nve` 且 T >= 300 K 的运行，在一定步数后被
  `RUNAWAY GUARD STOP: lattice temperature excursion (limit 300.000 K)` 停机。
- 根因：`Integrate::temperature` 在 NVE 下从未赋值；发布版读到 denormal
  `6.9514519871473713e-310`（> 0 为真），使 `if (!(t > 0))` 回退分支永不执行，
  `max(1.5t, t+300)` 塌成常数 300 K。
- 修法：`integrate.cuh` 三个温度成员初始化为 0；`run.cu` 的回退链改用 `> 1.0`
  判据并逐级回退（primed 参考 → 实测温度 → 300 K）。
- 同源修正：PE 取 `thermo[2]`（原取 `thermo[1]`，那是动能）；总能量改为
  `PE + KE`（原式 `thermo[1] + 1.5 N kB T` 把动能算了两遍）。
- 验证：同一输入/模型，修复前后 `thermo.out` 与 `traj.xyz` 的 sha256 **完全相同**；
  历史误杀算例旧版 rc=2 停 1000/5000 步、新版 rc=0 跑满并给出 limit 609.3 K；
  `tests/physics_checks.py::P5` 为常驻回归。

### Changed — 时间步政策
- 0.05 fs（参考）/ 0.1 fs（平衡）/ 0.2 fs（不建议）。历史 0.2 fs 的 S05B FAIL 保留，
  但不再作为所有后续研究的统一阻断条件。适用范围写死为 Fe / mask17 / alpha=0 /
  NVE / 250 K / <=10 ps（外推至 ~2 ns）。

### Changed — 训练
- `nep.in` 不再写死 `spin_mag_scale`，由训练器按 1/max|S| 数据驱动（Fe: 0.234148785）。
- 新增两个 50000 代 Fe 模型：`models/fe_mask17_2desc_r18.nep.txt`（Sex Smag）与
  `models/fe_mask31_5desc_r18.nep.txt`（全 5 块），lambda_e = lambda_m = 5、lambda_f = 1。
  DEV：10.38 / 8.75 meV/atom（r17 基线 15.23）。
  基态 a0 = 2.8258 / 2.8273 A；|S|0 = 2.413 / 2.432 mu_B；Tc ~ 950 / 850 K。

### Added
- `tests/physics_checks.py`：晶格常数、基态磁矩、Tc、NVE 漂移、guard 回归五项物理检查。
- `examples/fe_bcc_ground_state/`、`examples/fe_curie_temperature/`、
  `examples/fe_nve_stability/`：可直接运行的复现示例（均已实测）。
- `models/`：三个 Fe 模型 + 训练输入 + DEV 指标。
- `tools/physics_validation/`：长时间稳定性与热流审计脚本。
- `docs/theory/15_DEFECTS_AND_FIXES.md`：F19–F27 缺陷日志。
- `docs/theory/` 全篇改写：单位表、验证状态词表、PBC 面项、GK 自检、精度决策表。

### Kept as-is (deliberately)
- FDT 系数、S05A 保守分支、所有冻结阈值、全部历史 FAIL。
- 0.2 fs 的 COUPLED_ENERGY_GATE = FAIL。

### Known gaps（未修）
- F-25：同一 `run.in` 里 NVT -> NVE 切换在 N >= 432 挂死（有绕法）。
- 热流绝对归一化未验证、连续性判据在 PBC 下不成立 -> 不输出任何 kappa。
- 无单离子各向异性、无长程偶极、无拓扑/谱实现。


## 2026-11-01（晚）— 50 万代训练与全物理验证结果入库

新增 `results/`（259 个文件，7.0 MB），收录当天最新的训练与测试产物：

- `results/training/C_5desc_500k/` — 50 万代训练（5 描述符，$\lambda_e=5$、
  $\lambda_f=1$、$\lambda_m=1$、batch 1000）的执行记录 `TRAINING_STATUS.md`、
  `loss.out`、训练输入 `nep.in`、最终权重 `nep.txt`、5 个检查点权重，以及
  `series/SERIES_REPORT.md` 与逐检查点 `gen_{110k,200k,300k,400k,500k}.json`。
- `results/validation/` — `validation_summary.md`（BCC Fe 全物理 33 条：
  22 PASS / 3 FAIL / 2 PARTIAL / 2 NOT_SUPPORTED / 1 BLOCKED / 3 REPORTED）、
  `validation_results.json`、`benchmark_comparison.csv`、27 张 `figures/` 图、
  复现命令 `reproduce_commands.ps1` 与 `scripts/`。
- `results/tests/` — r18 三套测试（smoke 7 项 / acceptance 5 项 / physics 5 项）
  的原始输出（已排除 `traj.xyz` 与 `nep.restart`）。
- `models/C_5desc_500k_model.nep.txt` + `models/nep_C_5desc.in` — 50 万代最终权重与训练输入；
  `models/DEV_METRICS.json` 增加 C 条目。

**结论（`results/validation/validation_summary.md` §14.7–14.8）**：

| 量 | 110k → 500k | 判断 |
|---|---|---|
| DEV E MAE (meV/atom) | 6.53 → 6.36 | 20 万代后饱和 |
| DEV F MAE (eV/Å) | 0.0872 → 0.0843 | 单调缓降 |
| Heisenberg 拟合残差 (meV/atom) | 0.82 → 0.41 | **−50%**，磁子谱的 LSWT 描述更可信 |
| D（meV·Å²，§15 更正后） | 303/310 → 292/296 | **五个检查点全部落在实验 280–330 内**，与 DFT 外推 298 差 2% |
| $m_{eq}$ (μB) / $a_0$ (Å) / $T_c$ (K) | +11% / −1.1% / 912（−13%） | 全部卡住不动 |
| AFM-A 最软支 (THz) | −0.003 → −1.648 | **200k 起出现真软模并饱和** |

→ **可用于 FM α-Fe 的磁–声耦合与热输运研究；不可用于 AFM-A / AFM-G 态的声子。**

**2026-11-01 更**：上一版这里写的"D 偏硬 36–56%，FAIL"**已撤销**。ħω 换算误写成
$2S\Delta E/N$（正确为 $2\Delta E/(NS)$），D 闭式误写成 $(S/6)\Sigma Jzr^2$（正确为
$\frac{1}{6S}\Sigma Jzr^2$），同一个错同时进了冻结螺旋与 LSWT 两条路线，所以比值永远
对得上、一直没被发现。更正后 D 从 436–476 降到 **292–316**，全部落在实验区间内；
"D 偏硬源于磁矩偏大（$D\propto m^{2.35}$）"的解释同样作废——正确的
$d\ln D/d\ln m = 0.02$–$0.35$，即 **D 基本与所用磁矩无关**。详见
`results/validation/validation_summary.md` §15。

仍然没变的是：磁矩 +11%、$a_0$ −1.1%、$T_c$ 912 K 卡住，AFM-A 从 20 万代起的
−1.6…−1.8 THz 软模——这些是配方层面的系统偏差，继续加训练步数不会解决。

## v3.0.0-r17 (2026-10-31) — research preview

### Fixed (2026-10-31, later revision) — per-atom trainer<->MD consistency
- Descriptor neighbour sums are accumulated in `double` (`nep_acc_t`) and narrowed
  to float32 exactly once per atom, in the MD kernels (`force/nep.cu`, both box
  paths), the generic trainer (`main_nep/nep.cu`), the JIT kernels
  (`main_nep/nep_specialized.cu`) and the spin blocks
  (`utilities/spin_descriptor_kernels.cuh`).  Two kernels that walk different
  neighbour lists used to add the same terms in a different order and end up
  3-4 float32 ulp apart in `q`, which exceeded the frozen per-atom energy gate.
- `accumulate_s` / `find_q_one` are now templates on the accumulator type;
  existing float32 call sites are unchanged.
- Effect: Stage03A G6 (per-atom trainer vs MD) goes from 3/5 cases FAIL
  (max 1.192e-6 eV vs a 1.055e-6 eV gate) to 5/5 PASS (max 7.15e-7, 0 element
  violations).  Model file format, `q_scaler` and ANN width are unchanged, so
  already-trained `nep.txt` files load and mean the same thing.
- Side effect measured: the coupled energy-conservation error order at
  dt = 0.2/0.1/0.05 fs moves from 1.30 to 1.68 (gate 1.5, now passing) and the
  drift from 1.067e-5 to 1.023e-5 eV/atom (gate 1e-5, still failing by 2.3%).


### Added
- `spin_llg ... restart=<file>`: load the magnetic part of a checkpoint explicitly
  (never auto-picked from the working directory).
- `dump_restart` now also writes `spin_restart.bin`: magic, schema version, natoms,
  `nep.txt` FNV-1a hash, gamma/alpha/T/dt/scheme, SLLG and physics step counters,
  per-atom |S| anchor, per-atom curandState, payload checksum. Atomic write (tmp+rename).
- `spin_restart.bin` validation: atom count, model hash, physics parameters, scheme,
  payload checksum; any mismatch exits with code 4.
- `tests/smoke.py`: six numeric smoke tests (single point, conservative SLLG, geometric,
  midpoint, dimer, restart bit-exact, replica).
- `BUILD.ps1` / `INSTALL.ps1` / `SMOKE_TEST.ps1` / `build.sh` / `test.sh`.
- `docs/00..16`: theory, install, quickstart, validation, release status.

### Fixed
- `main_replica/makefile`: added `-I..` so the shared `utilities/spin_descriptor*.cuh`
  headers are found (the replica build failed at the first `#include`).
- `main_replica/force/nep.cuh` + `nep.cu`: added `ParaMB::spin_block_mask` and the
  `spin_blocks` parsing (same semantics as the main tree, default 31). Without it the
  replica build failed on every `paramb.spin_block_mask` use.

### Kept as-is (deliberately)
- FDT coefficient `2 a kB T / (gamma dt (1+a^2))` (Stage04 frozen).
- S05A conservative branch (alpha = T = 0).
- All frozen tolerances; historical FAILs (Stage03A G6, Stage04, Stage05B, Stage08A original).

### Known gaps
See `docs/13_VALIDATION_AND_KNOWN_ISSUES.md`.
