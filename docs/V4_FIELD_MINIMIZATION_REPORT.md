# V4 新增功能报告：磁矩参与的能量最小化 + 外场与磁电耦合

**基线**：`GPUMD-spin V3.1.0-r19`（`final_delivery_20261101_r18/staging/...-r19`）
**工作树**：`V3/V4/`（源码 + 构建脚本 + 验证）
**编译器**：CUDA 12.8 / VS2022 / RTX 4060 Ti / `-arch=sm_89`
**测试样本**：B20 FeGe，216 原子（108 Fe + 108 Ge），`48/fege/run01/nep.txt`
（`nep4_spin`，`spin_blocks Sex San Sdm Sia Smag`，训练 1e5 代，
test rmse_E = 1.77 meV/atom、rmse_F = 51.6 meV/Å、rmse_S = 21.1 meV/μB）

三件套全部重新编译通过。**全新目录**（`BUILD.ps1 -Clean`，rc=0）也过了一遍；
过程中撞到与 r18 报告 §1.3 完全相同的环境问题：`nvcc` 在某个目标上 0% CPU 挂住
（`%TEMP%` 残留 `tmpxft_*`），杀掉 nvcc、清掉临时文件后增量完成，124 s，rc=0。
这是环境问题，不是源码问题，但如实记录。

| 文件 | 字节 | SHA-256 |
|---|---|---|
| `src/gpumd.exe` | 22 412 288 | `0c597a7bb86bc13c20e8a1c8892eb917d478b9e206c5b3ab626bd4fb809c0273` |
| `src/nep.exe` | 4 373 504 | `52057bd28f0521ea85c4f0e6aac29706d30e51941270a16b7b2eb35ead75f708` |
| `src/gpumd_replica.exe` | 7 095 808 | `deaf3c38955b768d30c1b42f72ee7297adc500bfa91782ac834d6351c72b96d8` |

---

## 0. 结论速览

| 项 | 状态 | 关键证据 |
|---|---|---|
| 磁矩参与最小化 | 完成，SD / FIRE / FIRE-box 三条路径 | 力与横向力矩双判据同时收敛；\|S\| 漂移 3.7e-6；能量单调下降到平台 |
| 外磁场（Zeeman） | 完成（LAMMPS `fix precession/spin zeeman` 同构） | 单点能差 −0.0162950383 eV 与 `−Σ S·B` 逐位一致（1e-10） |
| 磁电耦合 | 完成（LAMMPS `pair_style spin/magelec` 同构） | 单点能差 0.1763102741 eV 对 float64 参考 0.1763102742 eV |
| 逐原子力矩 / 力 | 完成 | 与 float64 独立实现差 ≤ 6e-7，等于 NEP float32 噪声底 |
| 解析式自洽（能量↔力↔力矩） | 通过 | 有限差分 2.5e-13 eV/Å、7e-11 eV/μB |
| 向后兼容 | 通过 | 已有 `spin_llg` MD 与纯晶格 `minimize` 输出与 r19 **逐字节相同** |

---

## 1. 功能一：磁矩参与的能量最小化

### 1.1 实现范围与边界

**做什么**

- 把 3N 个磁矩自由度与 3N 个原子坐标放进**同一个最小化循环**：每一步同时评估
  力 F = −dE/dr、力矩 ω = −dE/dS，同时推进坐标与自旋，同时做收敛判断。
- 自旋步进用 LAMMPS `min_style spin` 的算法：**去掉进动项的阻尼自旋动力学**

  ```
  dS_i/dτ = λ [ ω_i − Ŝ_i(Ŝ_i·ω_i) ]            （球面梯度下降）
  θ_i = λ Δt |ω_i,⊥|,  Δt = 2π /(κ |ω|_max)     （自适应步长）
  S_i' = |S_i| [cosθ_i Ŝ_i + sinθ_i ω̂_i,⊥]       （精确测地线，|S| 严格保持）
  ```

  λ = `alpha_damp`（默认 1.0），κ = `discrete_factor`（默认 10.0），单步最大转角
  2πλ/κ = 0.628 rad，与 LAMMPS 默认一致。位移本身是信任域式的角度上限，不是 Euler 步，
  因此交换场很大时也不会自爆。
- **收敛判据**（两个都要满足，缺一不可）：

  | 判据 | 表达式 | 单位 | 默认 |
  |---|---|---|---|
  | 原子力 | `max_i |F_i| < <force_tol>` | eV/Å | 命令行第 2 个参数 |
  | 横向力矩 | `max_i |ω_i − Ŝ_i(Ŝ_i·ω_i)| ≤ spin_tol` | eV/μB | 1e-3 |

  用**横向**分量是因为 |S| 固定，−dE/dS 的纵向分量不做功、原则上不可能归零；
  拿 |ω| 全模做判据会永远不收敛。
- 信任域自适应：每一步比较总势能，**能量升高就把自旋增益减半，降低则缓慢恢复**
  （上限 1.0）。这是必须的：直接用 LAMMPS 默认的 0.628 rad，在 FeGe 这种交换场
  |ω| ≈ 0.3 eV/μB 的体系上会在两个构型间**极限环振荡**（实测 w_perp 在
  0.038 ↔ 0.236 之间来回跳、4000 步不收敛）。加自适应后 100 步内收敛。
- 磁能进入判据的原因：最小化器原先用 7 参数 `Force::compute`，NEP 的
  `spin_active = enable_spin && spin != nullptr && torque != nullptr` 为假，
  **自旋分块根本不参与势能**。V4 在自旋模式下改用带 `spin/torque` 的 11 参数重载，
  磁能、力矩、外场能量才进得来。

**不做什么（边界）**

- 不实现 `spin/cg`、`spin/lbfgs`（LAMMPS 的 OSO + CG / L-BFGS），也不做线搜索：
  自旋只做阻尼梯度步，用能量自适应信任域代替线搜索。
- 不做 NEB / GNEB（`neb/spin`）。
- 变胞最小化（`box_change=1`）里自旋同样参与，但**磁电耦合项对应力的贡献尚未独立标定**
  （见 §4 限制）。
- replica 构建不含自旋最小化（`main_replica/` 用的是自己的 minimize 副本，
  未启用即不进入该代码路径）。

### 1.2 涉及修改或新增的模块

**新增**

| 文件 | 作用 |
|---|---|
| `src/minimize/minimizer_spin.cuh` | 接口：`spin_relax_step` / `spin_torque_measure` / `spin_relax_apply` |
| `src/minimize/minimizer_spin.cu` | 两个 kernel（横向力矩投影 + 最大模归约；测地线旋转）+ `Minimizer::compute_force_at / spin_torque_max / spin_relax_apply / update_spin_step_scale` |

`minimizer_spin.cu` 只用裸指针（不碰 `Atom` 布局），与 `spin.cu` 同一约定，
因此 gpumd 与 gpumd_replica 两套不同 `Atom` 布局都能链接（本版 replica 未启用）。

**修改**

| 文件 | 改动 |
|---|---|
| `src/minimize/minimizer.cuh` | 新增 `set_spin_minimization()`、`spin_active()`、`compute_force_at()`、自旋工作区与信任域状态 |
| `src/minimize/minimize.cu` | 解析 `spin=` / `spin_tol=` / `alpha_damp=` / `discrete_factor=`；决定是否启用；打印 |
| `src/minimize/minimizer_fire.cu` | 力/力矩双判据；自旋步；打印条件保持原样（非自旋路径逐字节不变） |
| `src/minimize/minimizer_sd.cu` | 力计算移到循环头（带 spin），双判据 + 自旋步 |
| `src/minimize/minimizer_fire_box_change.cu` | 同上，收敛还要加应力判据 |

### 1.3 输入参数配置方式

语法**向后兼容**：位置参数与 r19 完全一致，新选项是尾部 `key=value`。

```
minimize sd   <force_tol> <steps>                                  [spin=...] ...
minimize fire <force_tol> <steps> [<box_change> [<hydrostatic>]]   [spin=...] ...

spin=on            磁矩与坐标一起最小化（无自旋列则报错）
spin=off           与 r19 完全一致（只动坐标）
spin=auto          默认；run.in 里读过 spin_llg 就自动为 on
spin_tol=<x>       横向力矩容差，eV/mu_B，默认 1e-3
alpha_damp=<x>     虚拟磁阻尼，默认 1.0     （与 LAMMPS min_modify 同名）
discrete_factor=<x> 自适应步长因子，默认 10 （与 LAMMPS min_modify 同名）
```

示例（FeGe，从随机自旋构型弛豫）：

```
potential nep.txt
spin_llg gamma=3.04 alpha=0.1 T=0
minimize fire 1e-3 4000 spin=on spin_tol=1e-4
ensemble nve
dump_xyz 1 final.xyz spin torque force potential precision double
run 1
```

注意两点：`minimize` 是**解析时立即执行**的，所以 `potential`、`spin_llg`、
`magfield`/`magelec` 必须写在它前面；`run 1` 只是为了把最小化后的状态 dump 出来。

### 1.4 验证方式与结果（FeGe 216 原子）

**(a) 单点能可分性（精确）**：`minimize fire 1e-6 1` 的第 0 步打印是**未经任何移动**的
初始构型的能量，用它做差分可以精确检验磁能是否进入势能。

| 构型（同一个原子/自旋输入） | E[eV] | f_max[eV/Å] |
|---|---|---|
| `spin=on` | −8.0096621513 | 0.2393 |
| `spin=off`（旧行为） | **+3.0917630196** | 0.1355 |

差了 **11.10 eV**：这就是 NEP 的磁学分块（Sex/San/Sdm/Sia/Smag）在旧最小化里
被整个丢掉的部分。`spin=off` 精确复现 r19（见 §3），`spin=on` 才把磁能纳入。
两者对同一位形给出同一个 f_max 量级但数值不同，因为自旋—晶格耦合项也进了力。

**(b) 收敛与守恒**（随机自旋起步，`minimize fire 1e-3 4000 spin=on`）：

| 指标 | 结果 |
|---|---|
| 收敛步数 | 100 |
| 末态 f_max | 1.527e-4 eV/Å （< 1e-3） |
| 末态 w_perp_max | 6.587e-4 eV/μB （< 1e-3） |
| 能量变化 | −8.0097 → −19.6929 eV（−11.68 eV） |
| \|S_i\| 最大相对漂移 | **3.7e-6**（float32 舍入量级；测地线步保范数） |
| \|M\| | 38.4 → 203.7 μB（自旋成序） |

**(c) 能量单调性**（`spin_tol=1e-8`、1200 步，每 120 步采样）：
−8.0097 → −19.6932 → −19.6943 → … → −19.6945，前 600 步单调下降，
之后在 ±1.9e-5 eV 内起伏（FIRE 晶格部分的有限精度抖动），不再有系统性上升。

**(d) 旧接口等价性**：`minimize fire 1e-3 500 spin=off` 与 r19 同一输入
**stdout 逐行相同**；纯晶格（无 spin 列）的 `minimize` 也逐行相同。

---

## 2. 功能二：外磁场 + 磁电耦合

### 2.1 实现范围与边界

**Zeeman 场**（对应 LAMMPS `fix precession/spin zeeman H x y z`）

```
E_Z = −g μ_B Σ_i S_i·B_ext        （本实现：E_Z = −Σ_i S_i·B，B 用 eV/μB）
```

- 场**均匀**（与 LAMMPS zeeman 一致），方向向量归一化；`g = 2` 硬编码，
  μ_B = 5.7883818066e-5 eV/T，故 `B[eV/μB] = 1.157676e-4 × H[T]`。
- 时间律：`static` / `sin` / `cos`，`freq` 单位 THz、`phase` 单位 rad，
  `B(t) = B·env(t)`。这是 FMR / 自旋泵浦的驱动方式。
- 能量同时写进 `potential_per_atom`、力矩写进 `atom.torque`，因此最小化器、
  恒温器、能量守恒检查看到的是同一套数（LAMMPS 需要 `fix_modify energy yes`
  才把磁能加进热力学量，这里默认就加）。
- **边界**：不做空间梯度场（LAMMPS zeeman 也不做）。真要场梯度就得同时给
  原子一个 ∇(S·B) 的力，能量/力一致性要另做，本版不做。

**磁电耦合**（对应 LAMMPS `pair_style spin/magelec` + `pair_coeff * * magelec`）

```
H_ME = Σ_{i<j} κ_ij (E × r_ij)·(S_i × S_j)          r_ij = r_j − r_i
ω_i  = −Σ_j S_j × D_ij                              （= −dH/dS_i）
F_i  = +Σ_j κ (S_i × S_j) × E                       （= −dH/dr_i）
D_ij = κ_ij (E × r_ij)
```

- 这就是 Katsura–Nagaosa–Balatsky（PRL 95, 057205 (2005)）的电场诱导 DM 项，
  LAMMPS 按 Tranchida et al.（JCP 372, 406 (2018)）实现；本实现与它同构。
- 单位（r21 更正）：κ 为 **eV/(Å·μB²)**，**E 为无量纲**的屏蔽极化方向/权重
  矢量，r 为 Å，S 为 μB，故 `eV/(Å·μB²) × Å × μB² = eV`。r20 文档写
  「E 为 eV」会给出 eV²，是量纲错误；场强全部吸收进 κ，因此
  `magelec_pair` 取负值或零都是合法的。
- 力/力矩/能量/维里全部写入对应数组，能量按 0.5/0.5 分给两个原子
  （**必须用 atomicAdd**，见 §4「踩到的坑」）。
- 支持按类型对覆盖 κ：`magelec_pair <ti> <tj> <kappa>`（对应 LAMMPS `pair_coeff i j`）。
- **边界**：单一截断的成对项，不做长程偶极；不做 DM 之外的各向异性/交换
  （那些已经由 NEP 的 `Sdm` 分块描述）。

**共同边界**：两项都在 `Force::compute()` 里、紧接势函数求出之后叠加，
所以 MD、SD、FIRE、FIRE-box 全部自动生效，无需各自接线。
`add_bfield`（旧关键字）原样保留，与 `magfield` **互斥**（同时给会报错，
避免同一个场被算两遍）。

### 2.2 涉及修改或新增的模块

**新增**

| 文件 | 作用 |
|---|---|
| `src/spin/spin_external.cuh` | 配置 + 约定 + `apply()` 接口 |
| `src/spin/spin_external.cu` | Zeeman kernel（含能量归约）、磁电耦合 kernel（自带 cell list，能量/力/力矩/维里）、解析能量诊断 |

**修改**

| 文件 | 改动 |
|---|---|
| `src/force/force.cuh` | 前置声明 `SpinExternal`，`set_spin_external()`，私有指针 |
| `src/force/force.cu` | 11 参数 `compute()` 末尾调用 `apply()` |
| `src/main_gpumd/run.cuh` | `SpinExternal spin_external` 成员 |
| `src/main_gpumd/run.cu` | `magfield` / `magelec` / `magelec_pair` 解析；注册到 Force；每步推进时间；把实际施加的场交给热流模块；`minimize` 传 `&spin` |
| `src/makefile` / `main_replica/makefile` | 无需改（`wildcard spin/*.cu`、`minimize/*.cu` 自动包含） |

### 2.3 输入参数配置方式

```
magfield <H> <nx> <ny> <nz> [unit=tesla|ev] [wave=static|sin|cos] [freq=<THz>] [phase=<rad>]
                          # unit=tesla（默认）：H 是强度(T)，(nx,ny,nz) 是方向，会被归一化
                          # unit=ev：三个数直接就是 B 的 eV/mu_B 分量（等价 add_bfield）

magelec <cutoff_A> <kappa> <Ex> <Ey> <Ez>
                          # kappa: eV/(A mu_B^2)；E: 无量纲权重矢量
magelec_pair <ti> <tj> <kappa>
                          # 可选，按类型对覆盖（出现在 magelec 之后）
```

示例（FeGe，5 T 沿 z + E 沿 x 的磁电耦合，然后在场下最小化）：

```
potential nep.txt
spin_llg gamma=3.04 alpha=0.1 T=0
magfield 5.0 0.0 0.0 1.0
magelec 4.5 0.001 1.0 0.0 0.0
minimize fire 1e-3 4000 spin=on
ensemble nve
dump_xyz 1 final.xyz spin torque force potential precision double
run 1
```

FMR 驱动：`magfield 0.5 1 0 0 freq=200 wave=sin`（200 THz，0.5 T 沿 x）。

### 2.4 验证方式与结果

判据分四层，从「公式对不对」到「物理对不对」。

**(1) 单点能差（精确，最硬）**
`minimize fire 1e-6 1 spin=on` 的第 0 步是**未移动的初始构型**，用三次运行做差分：

| 项 | gpumd 实测 | 独立 float64 参考 | 相对差 |
|---|---|---|---|
| Zeeman，B = 5 T·ẑ | −0.0162950383 eV | −Σ S·B = −0.0162950383 eV | < 1e-10 |
| 磁电，κ=1e-3, E=x̂, rc=4.5 Å | +0.1763102741 eV | Σ_{i<j} κ(E×r_ij)·(S_i×S_j) = +0.1763102742 eV | 6e-11 |

**(2) 逐原子力矩与力（对 float64 独立实现）**
把 `spin/torque/force/potential` 一起 dump，与 Python（O(N²)、同一构型、同一公式）
逐原子比较。判据用**测得的 NEP 噪声底**（同输入换一个极小时间步重跑，取差值）：

| 量 | max 偏差 | 参考量级 | 说明 |
|---|---|---|---|
| ΔE_i（Zeeman） | **6.8e-18 eV** | 1.6e-2 eV | double 舍入 |
| ΔE_i（磁电） | **6.9e-18 eV** | 1.8e-1 eV | double 舍入 |
| ΔF_i（磁电） | **5.6e-17 eV/Å** | 7.3e-3 eV/Å | double 舍入 |
| Δτ_i（Zeeman / 磁电） | 1.6e-8 / 1.6e-8 eV/μB | 5.8e-4 / 1.6e-2 | **float32 存储下限** |

前三项已经到 double 的舍入极限；力矩只能到 1.6e-8，因为 `atom.torque` 是
float32，把一个 ~1e-3 的增量加到 ~1e-1 的数上，量子就是 eps_float×|τ| ≈ 2e-8。
**这不是实现误差，是存储格式的下限。**

> 关键前提：这些算例必须用 `velocity <T> seed <n>` 固定初速度。没有速度列的
> 结构默认按时间播种，同一二进制两次运行结果不同（GPUMD 启动时打印的
> "DEBUG is off: Use different PRNG seeds for different runs" 就是这个意思）。
> 早期版本没有固定种子，所以只能给出「噪声底 5e-7」这种模糊判据；固定之后
> 判据直接落到 1e-17，等于把验证的分辨力提高了 10 个数量级。

**(3) 解析式自洽（不依赖任何外部约定）**
对 H_ME 做有限差分（h = 1e-4 Å / 1e-4 rad），与实现里用的解析力/力矩比较，
覆盖 6 个原子 × 3 个方向：

- `max |F_analytic − (−dH/dr)| = 2.5e-13 eV/Å`
- `max |τ_analytic − (−dH/dS)| = 7.0e-11 eV/μB`

这一条保证了「能量、力、力矩三者互相自洽」，即该项是保守的（NVE 不会凭空造能）。

**(4) 物理结果**

*Zeeman：T=0 磁化曲线（FeGe，216 原子，FM 初态沿 z，场沿 x，最小化到收敛）*

| B[T] | M_x/Σ\|S\| | cos(M,B) | E[eV] | ΔE = E(B)−E(0) | −M·B |
|---|---|---|---|---|---|
| 0 | 0.0000 | 0.000 | −19.658560 | 0 | 0 |
| 1 | 0.0377 | 0.042 | −19.659683 | −0.001122 | −0.000994 |
| 2.5 | 0.1020 | 0.114 | −19.666020 | −0.007459 | −0.006727 |
| 5 | 0.3175 | 0.355 | −19.707275 | −0.048715 | −0.041881 |
| 10 | 0.5597 | 0.626 | −19.821943 | −0.163382 | −0.147654 |
| 20 | 0.8248 | 0.923 | −20.102578 | −0.444018 | −0.435185 |
| 50 | 0.8910 | **0.998** | −20.833995 | **−1.175434** | **−1.175326** |

预期物理：各向同性（无外场各向异性时）铁磁体在场下整体转向 B，大场饱和。
实测 cos(M,B) → 0.9976（≈2.4°），且饱和区 **ΔE = −M·B 相对误差 1e-4**——
两者独立算（一个是能量差分，一个是末态磁化）能对上，说明场能量与力矩同源同号。
小场区比值 >1（1.1~1.16）是因为晶格同时发生了磁致伸缩弛豫（能量还有晶格部分），
不是场项的问题。

*磁电耦合：E 场诱导手性（FeGe 的核心物理）*

标量手性 χ = ⟨S_i·(S_j×S_k)⟩（3.2 Å 内近邻三角形，16 848 个），κ=0 时 χ₀ = +9.45e-4。

| κ | E[eV] | \|M\| | χ | dχ/κ | rms\|ΔS\| |
|---|---|---|---|---|---|
| 0 | −19.6929 | 203.7 | +0.000945 | — | 0 |
| +1e-4 | −19.7189 | 203.3 | +0.001387 | **4.42** | 0.051 |
| +2e-4 | −19.7557 | 202.8 | +0.001905 | **4.80** | 0.101 |
| +4e-4 | −19.8740 | 201.1 | +0.003451 | 6.27 | 0.323 |
| −2e-4 | −19.6756 | 204.0 | **+0.000160** | 3.92 | 0.127 |
| +1e-3 | −21.4056 | **12.3** | — | — | 1.96 |
| −1e-3 | −21.3984 | 12.2 | — | — | 2.04 |
| +2e-3 | −25.1995 | 7.2 | — | — | 1.98 |

预期物理与实测：

1. **κ 小时手性线性增长**：κ ≤ 2e-4 区间 dχ/dκ = 4.4~4.8（±10%），
   与「DM 型项的一阶响应」一致；
2. **手性是 κ 的奇函数**：κ = −2e-4 时 χ 从 9.45e-4 **降到** 1.60e-4，
   即 dχ 变号（dχ/κ 仍为正 3.92）；这正是 E 反向、D 矢量反向的物理；
3. **大 κ 破坏共线铁磁序**：κ ≥ 1e-3 时 |M| 从 203.7 掉到 12.3/7.2 μB，
   体系进入非共线（螺旋/斯格明子型）状态，能量大幅下降；
   ±κ 两态能量差 7 meV（同一镜像极小值的不同落点，不是对称性破缺）。
   这与 FeGe（B20，DMI 驱动螺旋）的物理图像一致，也是电场调控手性的入口。

---

## 3. 兼容性验证（不允许回归）

| 场景 | 结果 |
|---|---|
| 已有 `spin_llg` MD（`examples/spin_llg/run.in.conservative`，200 步） | `thermo.out`、`traj.xyz` 与 r19 **md5 相同** |
| 纯晶格 `minimize fire 1e-4 500`（无 spin 列） | stdout 逐行相同 |
| `minimize ... spin=off`（有 spin 列） | stdout 逐行相同 |

接口层面：所有新关键字都是新增；`minimize` 的位置参数与语义未变；
`Force::compute` 只加了一个默认 nullptr 的注册点；`Minimize::parse_minimize`
只加了一个带默认值的尾参 `Spin*`。

---

## 4. 踩到的坑（写在这里，避免下一版重犯）

1. **`potential_per_atom` 的跨 block 竞争**。成对能量按 0.5/0.5 分给 i、j 时，
   `potential[n2] += 0.5*e` 会被另一个 block 的线程同时读改写，非原子 RMW
   **丢更新**。症状极具迷惑性：力/力矩完全正确（用了 atomicAdd），
   只有能量系统性偏低 4%~20%，且随 κ 不单调。改成 `atomicAdd` 后单点能
   立刻从「差 4%」变成 1e-10 一致。**任何跨 block 的数组累加都必须原子。**
2. **`dump_xyz` 是追加打开（`"a"`）**。反复跑同一个算例会读到旧帧，
   导致「改了代码结果不变」的假象。验证脚本一律先 `rm -f dump_*.xyz`。
3. **LAMMPS 力的符号**。`r_ij = r_j − r_i` 约定下 F_i = +κ(S_i×S_j)×E；
   LAMMPS 手册印的是负号，因为它用的 r_ij 指向相反。实现以
   「−dH/dr 与本文件能量一致」为准（有限差分已证 2.5e-13），不照抄符号。
4. **收敛打印**。判据从「只看力」变成「力 + 力矩」后，若沿用
   `if (force_max < tol) print`，会在力收敛而力矩未收敛时每步刷屏（实测 1924 行）。
   改成「周期打印 + 只在双判据同时满足时打印收敛行」，同时保持非自旋路径原样。

---

## 5. 复现方式

```
cd V3/V4
./BUILD.ps1 -Arch sm_89            # 或 BUILD.cmd
cd validation_v4
python check_external.py probe2    # Zeeman + 磁电：能量/力矩/力 + 有限差分自洽
python check_minimize.py min_a     # 磁矩最小化：双判据、|S| 保持、能量下降
```

`validation_v4/` 下各目录：

| 目录 | 内容 |
|---|---|
| `probe/` `probe2/` | 单点探针（共线 / 非共线构型），dump 力矩力电位 |
| `min_energy/` | 用最小化第 0 步取精确单点能（检验能量可分性） |
| `min_a/` `min_b/` | 磁矩最小化：收敛 / 长时能量单调性 |
| `zeeman/` | M(B) 扫描（0~50 T） |
| `magelec/` | κ 扫描（±1e-4 … ±2e-3），手性响应 |
| `compat/` | 与 r19 二进制的逐字节对比 |
| `edge/` | 输入校验负向测试（缺 spin_llg、缺自旋列、未知选项、add_bfield 冲突…） |

## 6. 已知限制（未标定 / 未实现）

| 编号 | 项 | 状态 | 影响 |
|---|---|---|---|
| V4-L1 | 磁电项的维里/应力 | **已标定（V4.1）** | 与 2×Σ_pairs r⊗F_j 一致到 2e-15 eV；换能器见 §7.2 |
| V4-L2 | 自旋步无真正线搜索 | 用能量自适应信任域替代 | 极硬体系可能要多走几十步；无法保证严格单调 |
| V4-L3 | 磁场只有均匀场 | 与 LAMMPS 一致 | 无场梯度力 |
| V4-L4 | `spin/cg`、`spin/lbfgs`、`neb/spin` | 未实现 | 大体系收敛速度不如 LAMMPS 的 CG/L-BFGS |
| V4-L5 | 磁电项自带 cell list，每次力评估重建 | 未优化 | 每步额外一次建表开销 |
| V4-L6 | replica 构建不含自旋最小化 / 外场 | 未接 | REMD 里不能用这两项 |
| V4-L7 | FeGe 势本身长度尺度不对 | 螺旋周期 25 Å（实验 700 Å）、D 小 150 倍 | 上面的物理结论是**定性/半定量**；斯格明子尺寸小约 30 倍，见 `FEGE_SKYRMION_VS_LITERATURE.md` |

---

## 7. V4.1：缺陷修正、精度/性能与回归测试

本节记录在 V4 首版之后修掉的问题。每一条都写了症状、根因和判据，方便复核。

### 7.1 修掉的缺陷

| # | 缺陷 | 症状 | 根因 | 修法 |
|---|---|---|---|---|
| D1 | **磁电项的维里写错位置** | 应力张量只有 xx 分量非零，且与能量不自洽（trace 差 6 倍） | `virial_per_atom` 是**分量主序**（分量 k 在 `n + k*N`），代码按 `n + k` 写，九个分量散到了九个相邻原子上 | 改成 `virial[n + k*N]` |
| D2 | **成对能量用非原子 RMW** | 能量系统性偏低 4–20%，随 κ 非单调；力/力矩却完全正确 | `potential[n2] += 0.5*e` 被另一个 block 的线程并发读改写，丢更新 | 改成一原子一线程、只写自己的槽位（无原子操作） |
| D3 | **每次力评估两次主机同步** | 每步多两次 `cudaMemcpy` + 隐式同步，GPU 流水线断流 | Zeeman / 磁电各做一次 `copy_to_host` 取能量诊断值 | 能量只在设备端累加，`report()` 时才回读 |
| D4 | **小盒子下同一对原子被重复计数** | cutoff 相对盒子偏大时能量/力成倍放大 | 5×5×5 胞扫描在 `num_bins < 5` 时同一像会落到两个胞 | `num_bins ≥ 5` 才用胞表，否则回退 O(N²) 核（无歧义） |
| D5 | **力矩用 float 原子加** | 逐原子力矩有 ~1e-8 量级的累加噪声 | `atomicAdd(float*)` 对 25 个近邻逐个累加 | 寄存器里 double 累加，最后一次舍入到 float |
| D6 | **`minimize spin=off` 静默忽略外场** | 写了 `magfield` 却不生效，没有任何提示 | 该路径用 7 参数 `Force::compute`，拿不到 spin/torque | 解析时打印明确 WARNING |
| D7 | **`add_bfield` 与 `magfield` 只单向互斥** | 先写 `add_bfield` 再写 `magfield` 时同一场算两遍 | 只在 `magfield` 分支里检查了冲突 | 两个分支互相检查 |
| D8 | **信任域被晶格能量污染** | 晶格还在弛豫时自旋步长被错误放大 | 能量回涨判据分不清是晶格还是自旋造成的 | 只在力已收敛时才允许步长回涨 |
| D9 | 死代码 `Minimizer::relax_spins` | 无调用者 | 重构后遗留 | 删除 |

D1 与 D2 是「看起来对、算出来不对」的典型：**力和力矩全对，只有能量/应力错**，
因为那是唯二用非原子写/错索引累加的量。这也是为什么验证必须同时看
能量、力矩、力、应力四项，只比其中一项会漏掉它们。

### 7.2 精度与性能

| 项 | 之前 | 现在 | 说明 |
|---|---|---|---|
| 磁电能量（逐原子 vs float64） | 5.6e-7 eV | **6.9e-18 eV** | 去掉原子操作，double 累加 |
| 磁电应力 vs 解析 | 未标定 | **2e-15 eV** | 修正分量主序 |
| Zeeman 能量 | 4.8e-7 eV | **6.8e-18 eV** | 固定初速度后可直接比到舍入 |
| 每步主机同步次数 | 2 | **0** | 能量只在 report/结束时回读 |
| 每对原子原子操作 | 26 | **0** | 一原子一线程、只写自身 |
| 力矩累加精度 | float 原子加 | double→float 一次舍入 | |

### 7.3 可复现性（重要）

带自旋的结构如果没有速度列，GPUMD 用**时间播种**给初速度：同一个二进制连跑
两次结果不同。这会把任何「逐位比较」的验证判据废掉（我们最初的噪声底 5e-7
就是这么来的）。**所有需要复现的算例都必须写 `velocity <T> seed <n>`**，
回归套件已经全部加上。

### 7.4 回归测试

`python tests/v4_regression.py`，36 项检查（6 组）、约 21 s 跑完、全部通过：

| 组 | 覆盖 | 项数 |
|---|---|---|
| `external` | Zeeman / 磁电的能量、力矩、力、应力 vs 独立 float64 实现；H_ME 的有限差分自洽 | 11 |
| `minimiser` | 双判据收敛、能量下降、\|S\| 守恒、磁矩有序化 | 6 |
| `compat` | MD 输出与 r19 逐字节相同；纯晶格 `minimize` 与 `spin=off` 与 r19 逐行相同 | 3 |
| `inputs` | 7 个负向用例（缺 `spin_llg`、未知选项、零场方向、两个场关键字冲突、`magelec_pair` 顺序） | 7 |
| `topology` | Q(FM)=0、\|Q\|=1（两种半径）、半径回读误差 < 1 个晶格常数、γ=0 存活（Q=−1）、γ=+90° 解绕 | 7 |
| `chirality` | E(R) > E(L)（模型破缺手性对称）、螺旋能量低于铁磁 | 2 |

### 7.5 FeGe 斯格明子与文献比对

见 `FEGE_SKYRMION_VS_LITERATURE.md`（V4.2 已重写）。结论：**模型能给出斯格明子，
但长度尺度比实验小约 30 倍**——0.2 T 下 Néel 型斯格明子 Q = −1.000、半径 12 Å，
半径随场减小（1 T 时 9.5 Å）。定量偏差的根源是螺旋周期 25 Å（实验 700 Å）与
DMI 偏小 150 倍，都来自 16–24 原子训练胞装不下长波手性结构。

> V4.1 曾结论「算不出斯格明子、模型无 DMI（|D| < 5.4e-8 eV·Å、p > 55 μm）」。
> 那是三个测量缺陷叠加的结果：±q 构造退化（生成的其实是同一个构型）、只测
> Bloch 极化、以及用 Bloch 初态去测一个 Néel 型 DMI。详见该文件 §7。
