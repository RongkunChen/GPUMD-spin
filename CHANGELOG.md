# CHANGELOG

## Post-r22 revision（2026-11-04）— F-10：让 `compute_phonon` / `compute_elastic` 真正带上自旋

### Fixed: clamped-spin force/energy in the non-MD drivers
- `src/force/force.cuh`, `src/force/force.cu`, `src/main_gpumd/run.cu`
- `compute_phonon` (`Hessian::find_H12` -> `Force::compute`), `compute_cohesive`
  / `compute_elastic` (`Cohesive` -> `Minimizer_SD` -> `Force::compute`) and
  `minimize ... spin=off` all go through the **short 7-argument**
  `Force::compute`, which never called `Potential::set_spin()` and never called
  `SpinExternal::apply()`.  For a `nep4_spin` model `NEP::spin_` therefore
  stayed null: the NON-MAGNETIC Hamiltonian was evaluated silently, and
  `magfield` / `magelec` contributed zero energy, force, torque and virial.
  The reported phonon spectrum and elastic tensor belonged to a different
  (non-magnetic) system even though `model.xyz` carried a spin column.
- Fix: `Force::set_static_spin()` registers the device spin/torque buffers,
  and `Run` binds them through a scoped RAII guard **only for the three
  `compute_*` commands** (`compute_phonon`, `compute_cohesive`,
  `compute_elastic`); the binding is released when the command returns.  The
  short overload now binds them on every potential, suppresses spin-flux
  output, and applies the external Zeeman / magneto-electric terms at the same
  point as the long MD overload.  A null binding (run without a spin column)
  keeps the pre-V4 behaviour.
- Semantics: these drivers are **clamped-spin** (frozen magnetic order).  They
  now include the magnetic energy and its position derivative in the Hessian
  and elastic tensor, but do not relax or propagate the moments and do not
  provide magnon-phonon coupling.  A one-line status message is emitted for
  spin runs only; non-spin stdout is unchanged.
- `minimize ... spin=off` is deliberately **not** covered: its lattice-only
  behaviour is documented by the existing R22 warning and is pinned by the
  r21 regression suite (`tests/v4_regression.py` compares its stdout line by line
  with r19).  Use `spin=on` / `spin=auto` there when the field should act.
- Regression scope: no change to the long MD overload, `compute_heat.cu`,
  `Force::spin_flux_A`, the Green-Kubo spin heat current, or the F-11 BLOCKED
  status.  `force.cu` is no longer byte-identical to r21 (the short overload
  had to be fixed), so the R22 read-only F-11 audit reports `CHANGED` for that
  file; `compute_heat.cu` remains byte-identical.  The R22 C2
  magneto-electric switch is picked up automatically by the short overload
  through `SpinExternal::apply()`.

## v4.3.2-r22-nonheat（Stage R22-NonHeat，2026-11-04）

### Fixed: magneto-electric hard-cutoff discontinuity (R22-ISSUE-01, opt-in)
- `src/spin/spin_external.cu`, `src/spin/spin_external.cuh`, `src/main_gpumd/run.cu`
- The r21 Hamiltonian `H_ij = kappa (E x r_ij).(S_i x S_j)` used a hard step at
  `r_c`, so energy, force, spin effective field and all nine virial components
  are DISCONTINUOUS there.  R22 adds a versioned C2 quintic switch
  `f_c(x) = 1 - 10x^3 + 15x^4 - 6x^5`, `x = (r-r_on)/(r_c-r_on)`, with
  `df_c/dr` included in the force and hence in every virial component, and
  `f_c` multiplying the spin effective field.  Ordered-pair bookkeeping and
  periodic-image handling are unchanged.
- Selected explicitly with `magelec <rc> <kappa> Ex Ey Ez cutoff=smooth
  [smooth_width=<frac in (0,1]> | r_on=<A>]`.  `cutoff=legacy` reproduces r21.
- **DEFAULT IS STILL r21 LEGACY** so that no existing input or NEP model
  silently changes its predictions.  Legacy mode prints an explicit warning
  that it is not reliable for long conservative dynamics.

### Fixed: runaway-guard thermo slot-index error (R22-ISSUE-03)
- `src/main_gpumd/run.cu` (R19 spin runaway guard).
- `Thermo::compute` writes `[T, PE, s_xx, ...]` (see
  `src/integrate/thermo.cu` and `src/measure/dump_thermo.cu`), and KE is
  reconstructed as `1.5*N*kB*T`.  The r21 "F26" change incorrectly treated
  `thermo[2]` (a stress component) as PE and `thermo[1]` (PE) as KE, so the
  guard monitored stress instead of total energy and stopped conservative NVE
  runs with a spurious "energy-conservation drift" (measured: stop at step
  8000, dt=0.05 fs, reported dE=0.0205 eV/atom while the true PE drift was
  ~1e-5 eV/atom).  The guard now uses `PE = thermo[1]` and
  `KE = 1.5*N_moving*kB*T`, exactly as `dump_thermo.out` does.  No heat-flux
  code, no residual/non-convergence fix and no F-01..F-12 behaviour changed.

### Fixed: NVE test observed potential energy instead of total energy (R22-ISSUE-04)
- `tests/r22_core.py::R22.ME.NVE_STABILITY.*` (test harness only; no production
  source change).  An NVE run conserves E = K + U, not U.  The earlier test read
  thermo column 2 (U) and required |dU| < 1 eV; during thermalisation K and U
  exchange several eV at constant E, which produced two spurious FAILs even
  after the R22-ISSUE-03 guard fix.  The test now asserts on |d(K+U)| < 0.1 eV
  over 10000 steps / 216 atoms (= 4.6e-4 eV/atom, 43x tighter than the R19
  guard's 0.02 eV/atom), keeps |dU| as a diagnostic, and compares smooth vs
  legacy total-energy drift.  Measured on the final binary: smooth
  dE_tot=4.43e-6 eV, legacy dE_tot=2.61e-2 eV, ratio 1.7e-4; the smooth
  cutoff removes the hard-cutoff energy jump.  The original failing logs are
  preserved in R22_BUILD_AND_RUN_LOGS.zip.

### Added: versioned external-field energy/work ledger (R22-ISSUE-02)
- `spin_energy_ledger.out`, format_version 1, written in addition to the
  unchanged `spin_work.out`.  Columns: step, time_fs, E_total, E_potential,
  E_zeeman, E_me, E_spin_related(NA), W_drive, W_damping_proxy, W_noise,
  W_ext, W_spin_internal, W_bath(NA), balance_residual, ledger_completeness.
- `W_drive = sum_steps -sum_i S_i(t_n).[B(t_{n+1/2}) - B(t_{n-1/2})]` is the
  exact explicit-time work of the discrete midpoint-sampled drive.
- `W_rot + W_noise + W_ext = dU_spin` is the r21 F16 identity, UNCHANGED.
- Quantities that cannot be strictly defined are written as literal `NA`,
  never as a fabricated zero.  `balance_residual` is emitted only when
  `ledger_completeness=COMPLETE` (NVE, T=0, no `add_bfield`, strang split).
  SLLG / thermostatted runs are explicitly labelled INCOMPLETE.

### Added: independent reference tools and validation drivers
- `tools/r22/magnon_spectra.py` (DSF, analytic FM/AFM/anisotropy-gap
  dispersions, Lorentzian linewidth with HWHM/FWHM and NOT_RESOLVED guards)
- `tools/r22/statmech.py` (Boltzmann/Langevin, autocorrelation/ESS/CI,
  Binder crossing, KS/chi-square, analytic LLG references)
- `tools/r22/symmetry_analysis.py` (Heisenberg/MAE/DMI/chirality references,
  constrained mixed derivatives, domain-wall relaxation, descriptor invariance
  table, Berry-curvature feasibility checklist)
- `tests/r22_magnon.py`, `tests/r22_statmech.py`, `tests/r22_symmetry.py`,
  `tests/r22_core.py`, `tests/r22_gpu_physics.py`, `tests/r22_performance.py`

### NOT changed / explicitly excluded
- `src/measure/compute_heat.cu`, `hac.cu`, `dump_heat_flux.cu`:
  general-NEP spin total-energy flux remains **BLOCKED**.  No formula, warning
  or status was modified.
- NEP model file format, SIA/SDM descriptor layout, training weights,
  `spin_work.out` column layout, `add_bfield`/`magfield` semantics,
  restart schema, r21 F-01..F-12 fixes.
- No dipole-dipole interaction, no new SOC/DMI descriptor, no quantitative
  Berry curvature / Chern number / topological magnon implementation.

### Known limitations newly documented
- `src/utilities/compact_nep.cu::make_compact_file()` cannot parse the
  `spin`/`spin_layout` header lines of a `nep4_spin` model when only a subset
  of the model species is active (single-species subset).  Pre-existing r21
  behaviour; R22 validation drivers keep both Fe and Ge active.  Not modified
  because it concerns the NEP model loader, which is outside this stage.
- At `dt = 0.05 fs` the shipped conservative spin-lattice Strang splitting
  drifts 0.0205 eV/atom in 8000 steps and the r19 runaway guard stops the run
  (threshold 0.02 eV/atom).  Use a smaller dt for long conservative runs.

## v4.3.1-r21（实现纠错，2026-11-04）

### Fixed
- `src/spin/spin.cu`: complete 3-component implicit-midpoint residual, explicit NaN/Inf/overflow flag, hard failure (exit 4) on non-convergence, fixed |S| anchor with pre-projection error, strict `spin_llg`/`spin_midpoint` parsing, v2 restart with external-field clock, machine-readable residual/iteration diagnostics.
- `src/spin/spin_external.cu`: ME virial ordered-pair double counting; `magelec_pair` CPU/GPU dirty-flag synchronisation, negative/zero/symmetric overrides; both diagnostic energy accumulators cleared before every evaluation; unit/convention comments corrected.
- `src/main_gpumd/run.cu`: `spin_midpoint` keyword, midpoint sampling of the time-dependent Zeeman envelope, restart time restoration, end-of-run |S| check.
- `src/measure/compute_heat.cu`, `hac.cu`, `dump_heat_flux.cu`: explicit BLOCKED warnings for spin-resolved heat flux, tilted/non-periodic binning now rejected; HAC cross-correlation confirmed already present (no change).
- `tests/v4_regression.py`: virial reference no longer copies the implementation's double-counting bug; independent affine energy-strain FD for all 9 components.

### Changed / compatibility
- `gamma` must now be > 0; r20's `gamma=0` static probe is rejected. Use a tiny positive gamma with a tiny dt, or a physical gamma.
- restart schema v2; v1 checkpoints remain readable.
- `E` in `magelec` is documented as dimensionless (kappa carries the units).

### Added
- Source tree: `tests/unit/midpoint_residual_test.cu`, `tests/r21_validation.py`,
  `tests/r21_physics_extra.py`, `PHYSICS_DERIVATION.md`.
- Delivered **alongside** the source tarball (not inside it):
  `FIXES.patch`, `AUDIT_AND_FIX_REPORT.md`, `TEST_RESULTS.json`,
  `TEST_MATRIX.csv`, `PERFORMANCE.csv`, `BUILD_AND_RUN_LOGS.zip`,
  `MANIFEST.sha256`.

### BLOCKED (unchanged, not fixed)
- General-NEP spin heat current is NOT a validated total energy flux. `heat_flux.out`/`hac.out` from a spin run are diagnostic only.

> 以下 `## V4` 段落是 r19→r20 的**历史记录**，其中「g=2 硬编码」等描述
> 不代表 r21 行为；r21 以本节与 `README.md` 为准。

## V4（在 v3.1.0-r19 之上）

### Added

#### 磁矩参与能量最小化
- `src/minimize/minimizer_spin.cuh` / `.cu`：阻尼自旋弛豫（LAMMPS `min_style spin`
  的测地线实现）+ 横向力矩归约 + 能量自适应信任域。
- `minimize` 新增尾随选项：`spin=on|off|auto`、`spin_tol=`、`alpha_damp=`、
  `discrete_factor=`。位置参数与语义不变。
- SD / FIRE / FIRE-box 三条路径都改为「力 + 横向力矩」双判据、同循环推进自旋。
- 自旋模式下改用 11 参数 `Force::compute`，NEP 磁学分块与力矩才进入势能与判据
  （旧路径 `spin=nullptr` 时磁能整块被跳过，FeGe 216 原子上差 11.10 eV）。

#### 外磁场（Zeeman）
- `src/spin/spin_external.cuh` / `.cu`：`magfield <H> <nx> <ny> <nz>`，
  `unit=tesla|ev`、`wave=static|sin|cos`、`freq=THz`、`phase=rad`。
  `g=2`、`B[eV/μB] = 1.157676e-4 × H[T]`，与 LAMMPS `fix precession/spin zeeman` 一致。
  场能写入 `potential_per_atom`，力矩写入 `atom.torque`。

#### 磁电耦合
- 同一模块：`magelec <cutoff> <kappa> <Ex> <Ey> <Ez>` + `magelec_pair <ti> <tj> <kappa>`，
  `H = Σ κ(E×r_ij)·(S_i×S_j)`，与 LAMMPS `pair_style spin/magelec` 同构；
  能量/力矩/力/维里全部写入对应数组。

#### 接线
- `Force::set_spin_external()` + `Force::compute` 末尾 `apply()`：MD 与所有最小化器
  自动获得外场与磁电项。
- `Run` 新增 `SpinExternal` 成员、三个关键字的解析、每步推进场的时间包络、
  把实际施加的场交给磁子热流模块。

### Changed
- `Minimize::parse_minimize(..., Spin*)`：新增带默认值的尾参。
- FIRE 收敛打印条件改为「周期打印 + 双判据满足时打印」；非自旋路径的打印
  与 r19 逐字节一致。
- `src/main_replica/makefile`：无改动（replica 使用自带的 minimize/force 副本）。

### Verified
- `compat/`：已有 `spin_llg` MD 与纯晶格 `minimize` 的 stdout/输出与 r19 逐字节相同。
- `probe2/`：Zeeman 与磁电项的能量/力矩/力对 float64 独立实现，全部落在 NEP 自身
  float32 噪声底（2e-7…6e-7）之内；有限差分自洽 2.5e-13 eV/Å、7e-11 eV/μB。
- `min_energy/`：单点能差分与解析式一致到 1e-10（Zeeman）与 6e-11（磁电）。
- `zeeman/`：M(B) 饱和 cos(M,B)=0.9976（50 T），饱和区 ΔE = −M·B 相对误差 1e-4。
- `magelec/`：κ ≤ 2e-4 时 dχ/dκ = 4.4~4.8 且对 κ 为奇函数；κ ≥ 1e-3 破坏共线序。

### Known limitations
见 `docs/V4_FIELD_MINIMIZATION_REPORT.md` §6（V4-L1 … V4-L7）。
最需要提醒的两条：磁电项的维里未独立标定（变胞最小化叠加 `magelec` 不可信）；
FeGe 势本身 rmse_E = 1.77 meV/atom（测试集），物理结论是定性/半定量。

---

## V4.1 — 缺陷修正、精度/性能、回归与文献比对

### Fixed
- **磁电项维里写错索引**（`virial[n+k]` → `virial[n+k*N]`）。`virial_per_atom`
  是分量主序；原写法把九个分量散到九个相邻原子上，应力张量只有 xx 非零、
  trace 与能量差 6 倍。
- **成对能量非原子读改写**：`potential[n2] += 0.5e` 跨 block 竞争丢更新，
  使磁电能量偏低 4–20%（力/力矩不受影响，因此极难通过力判断）。
  重写为「一原子一线程、只写自身槽位」，全程无 `atomicAdd`。
- **小盒子重复计数**：5×5×5 胞扫描在 `num_bins < 5` 时同一像落入两个胞 →
  `num_bins ≥ 5` 才用胞表，否则回退 O(N²) 核。
- **`add_bfield` + `magfield` 双向互斥**（原来只单向检查）。
- **`minimize spin=off` + 外场静默失效**：现在打印明确 WARNING。
- **信任域被晶格能量污染**：只在力已收敛时允许自旋步长回涨。
- 删除死代码 `Minimizer::relax_spins`。

### Changed (precision / performance)
- 力矩改为 double 寄存器累加、最后一次舍入到 float（原来 `atomicAdd(float*)`）。
- 每个力评估的主机同步从 2 次降到 0 次（能量只在 `report()` 回读）。
- 每对原子的原子操作从 26 次降到 0 次。
- 新增 `SpinExternal::report(tag)`，在 `minimize` 之后与每个 `run` 块结束时
  输出 Zeeman / 磁电能量。

### Verified (new numbers, all machine precision)
- Zeeman 逐原子能量 vs `-S·B`：6.8e-18 eV（原 4.8e-7）。
- 磁电逐原子能量 vs float64：6.9e-18 eV（原 5.6e-7）。
- 磁电应力 vs 2×Σ_pairs r⊗F_j：2e-15 eV。
- 力矩极限 1.6e-8 eV/μB = float32 存储量子，非实现误差。
- **可复现性前提**：带自旋且无速度列的结构必须写 `velocity <T> seed <n>`，
  否则初速度按时间播种、逐位比较失效。

### Added
- `tests/v4_regression.py`：5 组 33 项检查，14 s，全通过。
- `tests/data/`：FeGe 势函数与 B20 结构。
- `docs/FEGE_SKYRMION_VS_LITERATURE.md`：与文献的 10 项比对表、误差范围与偏差说明。
- `validation_v4/fege_build.py`、`skyrmion.py`、`helix/scan_helix.py`：
  构型生成、斯格明子弛豫、E(q) 扫描。

### Known (unchanged)
- 势函数的长度尺度不对：螺旋周期 25 Å（实验 700 Å），DMI 小约 150 倍，
  磁矩 2.0 μB（实验 1.0–1.4 μB）。后果是斯格明子直径 2.4 nm（实验 20–70 nm）。
  这是训练数据问题，不是实现问题；详见 `docs/FEGE_SKYRMION_VS_LITERATURE.md`。

---

## V4.2 — 修正 FeGe 比对结论（无 DMI → 有 DMI，长度尺度小 30 倍）

### Fixed（文档与测试的物理结论，不是代码）
- **`docs/FEGE_SKYRMION_VS_LITERATURE.md` 重写**。V4.1 的结论「模型无 DMI、
  |D| < 5.4e-8 eV·Å、p > 55 μm、算不出斯格明子」来自三个测量缺陷：
  1. **±q 构造退化**：`e2 = q̂ × e1` 随 q 翻号，`sin(φ)·e2` 不变，+q 与 −q
     生成的是同一个构型 —— 测到的 E(+q)=E(−q) 是恒等式；
  2. **只测 Bloch 极化**，而该模型的手性耦合是 Néel 型；
  3. **用 Bloch 初态去测 Néel 型 DMI**，两者不耦合，任何 D 都"坍缩"。
- 修正后的实测：模型**破缺手性对称**，L/R 螺旋在 λ=60 Å 差 4.09e-3 eV/216 原子，
  长波极限 D ≈ (0.8–1.0)e-5 eV·Å/atom。
- 修正后的斯格明子：**不注入任何 DMI**，0.2 T 下 Q = −1.000、R = 8.5–15.7 Å
  （随初态螺旋度），1 T 时 9.46 Å，半径随场减小。螺旋度扫描表明模型选出的是
  **旋转方向**（γ ∈ {0°,180°,225°,270°} 存活，{45°,90°,135°} 解绕），
  而 Néel/Bloch 之分的能量差只有 0.57 meV/atom —— 选不出类型。
- **rmse 读数修正**：`1.770e-03 eV/atom` 曾被读成 17.7 meV/atom，实为
  **1.77 meV/atom**；力 51.6 meV/Å、维里 31.6 meV/atom、自旋 21.1 meV/μB。
- **数据集描述修正**：训练集是 **SOC 非共线**构型（每帧 |⟨S⟩| ≈ 0.15，
  近邻 ⟨cos θ⟩ ≈ −0.26），不是"近共线、无手性"。真正的限制是胞只有 16–24 原子，
  装不下 p ≈ 70 nm 的长波螺旋。

### Changed
- `tests/v4_regression.py`：
  - 删除错误的检查项「skyrmion collapses without DMI」；
  - 换成一对：**γ=0 存活（Q=−1）/ γ=+90° 解绕（Q=0）**；
  - 新增 `chirality` 组：E(R) > E(L)、螺旋能量低于铁磁。
  - 全部 36 项通过（5 组 → 6 组）。
- `validation_v4/dmi_scan.py`：新增 `assert`，直接拦住 ±q 退化；
  同时测 Bloch 与 Néel 两种极化。
- `validation_v4/skyrmion.py`：新增 `helicity` 参数（默认改为 Néel）。

### Added
- `48/b20/_chir/run_chirality.sh`：216 原子 B20 上 L/R 螺旋的完整扫描。

### 注意
- `48/b20/nep.txt`（md5 `43c72906…`）与 `48/fege/run01/nep.txt`（`eba27743…`）
  **不是同一个势函数**；混用会让同一构型能量差 18 eV 以上。本项目所有结论
  均基于 `fege/run01`。

---

## V4.3 — 公式审计（修正 g 因子）+ FeGe 斯格明子全尺寸复现

### Fixed（真实公式错误）
- **Zeeman 的 g 因子被算了两次**。积分器的 `gamma = 3.0385`（fs⁻¹ per eV/μB）
  本身即 g=2 的电子旋磁比（1.7588e11 / 1.7276e4 × 1e-15），而 Tesla→eV/μB
  的换算里又乘了一个 g=2，使 Zeeman 能量翻倍、等效场减半。
  改为 `g = 1`（自旋列本身是 μB），保留 `g=` 关键字供角动量约定使用。
  **影响**：此前所有特斯拉数字的等效场只有标称的一半。
- 回归新增判据：`magfield 5 … tesla` 与 `magfield 2.894e-4 … unit=ev` 的
  Zeeman 能量必须逐原子相同（实测 0.00e+00 eV），g 再被乘进去会立刻失败。

### Verified（逐式核对，均有数值判据）
- 磁电项与 LAMMPS `pair_style spin/magelec` 的 ω、F 解析式逐项一致；
  `F = −dH/dr` 有限差分 1.4e-13 eV/Å，维里 vs `2Σ r⊗F_j` 2.2e-15 eV。
- 测地线旋转保模长（3.7e-6）、能量单调下降、收敛判据自洽。

### Changed（测试）
- 删除 `topology` 组里基于 12×12×2 盒子的「斯格明子存活/坍缩」两条判据：
  Lz = 9.6 Å < 2×6 Å 截断，属非法盒子，结论是几何假象。
- `external` 组新增 1 条 g 因子判据。全部 **35 项通过**（原 36 项）。
- 注意：旧输入里 `magfield 5` 现在等效于原来的 2.5 T；要复现旧结果请写 `g=2`。

### Added
- `docs/FEGE_FULL_SYSTEM_AND_FORMULA_AUDIT.md`：公式审计 + 12 万原子全尺寸复现 + 文献对比。
- `validation_v4/full_skyrmion.py`：全尺寸建胞/运行/分析驱动（含元胞平均磁化、
  Berg–Lüscher 拓扑荷、结构因子）。

### Result
- 12 万原子薄膜（24 nm × 24 nm × 3.45 nm）：铁磁态均匀稳定（⟨m_z⟩=0.773）；
  斯格明子初态在块体合法盒子里**坍缩**（Q=0），在薄膜里只剩填满周期盒的
  Q=−1 纹理，无有限半径（R/L 恒为 0.54）。
- 与文献（p = 70 nm，Yu 2010）对比：螺旋周期小 28 倍、DMI 小 150 倍、
  斯格明子晶格不存在。基态类型、单一手性、Ge 反平行三条一致。

---

## v3.1.0-r19 (2026-11-01) — 修复 F-25：单文件多段 `run` 的第二段不产数据

**只有一处行为改变，不动物理**：`run.in` 里写两个 `run` 块（典型用法：NVT 平衡 ->
NVE 生产）现在能正常工作，并且与"拆成两次 `gpumd.exe` 调用"给出**逐位相同**的结果。

### Fixed — F-25：NVT→NVE 在同一 `run.in` 里， $N\ge432$ 表现为挂死
- 症状（历史记录）：先 `ensemble nvt_lan` 再 `ensemble nve`， $N=432$ 与 $N=1024$ 上
  "进程存活、CPU 占用、26 分钟无任何输出"；拆成两次调用正常； $N=128$ 不复现。
- **根因**：`Measure::post_run()` 末尾无条件执行 `actions_.clear()`。第一个 `run` 块结束
  时所有 measure 被销毁，第二个块只做积分：`thermo.out`、`traj.xyz`、`restart.xyz`
  都不再更新，而 stdout 仍在打进度（重定向时被块缓冲，肉眼更看不到）。于是"跑得好好的
  但文件不动"被当成了挂死； $N$ 越大，等到的耐心越短，所以"阈值"看起来在 $N\approx432$。
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
- 缺陷日志 F19–F27：见 `KNOWN_LIMITATIONS.md` 与 `CHANGELOG.md`。
- 理论推导文档全篇改写：单位表、验证状态词表、PBC 面项、GK 自检、精度决策表。

### Kept as-is (deliberately)
- FDT 系数、S05A 保守分支、所有冻结阈值、全部历史 FAIL。
- 0.2 fs 的 COUPLED_ENERGY_GATE = FAIL。

### Known gaps（未修）
- F-25：同一 `run.in` 里 NVT -> NVE 切换在 N >= 432 挂死（有绕法）。
- 热流绝对归一化未验证、连续性判据在 PBC 下不成立 -> 不输出任何 kappa。
- 无单离子各向异性、无长程偶极、无拓扑/谱实现。


## 2026-11-01（晚）— 50 万代训练与全物理验证结果入库

新增 `results/`（259 个文件，7.0 MB），收录当天最新的训练与测试产物：

- `results/training/C_5desc_500k/` — 50 万代训练（5 描述符， $\lambda_e=5$、
  $\lambda_f=1$、 $\lambda_m=1$、batch 1000）的执行记录 `TRAINING_STATUS.md`、
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
"D 偏硬源于磁矩偏大（ $D\propto m^{2.35}$）"的解释同样作废——正确的
$d\ln D/d\ln m = 0.02$– $0.35$，即 **D 基本与所用磁矩无关**。详见
`results/validation/validation_summary.md` §15。

仍然没变的是：磁矩 +11%、 $a_0$ −1.1%、 $T_c$ 912 K 卡住，AFM-A 从 20 万代起的
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
See `VALIDATION_REPORT.md` and `KNOWN_LIMITATIONS.md`.
