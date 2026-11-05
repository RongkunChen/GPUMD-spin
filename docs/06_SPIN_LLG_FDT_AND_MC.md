# 06 — LL/Gilbert 方程、FDT、随机积分与 MC

## 1. 方程与约定

本实现积分的是 **Landau–Lifshitz (LL)** 形式：

$$
\frac{d\mathbf S_i}{dt}=-\ \gamma\ \mathbf S_i\times\mathbf B_i
-\frac{\gamma\alpha}{|\mathbf S_i|}\ \mathbf S_i\times\big(\mathbf S_i\times\mathbf B_i\big)
+\ \mathbf b_i(t),
$$

- $\gamma$：代码里的 `gamma`，单位 $\mu_B/(\mathrm{eV\ fs})$，物理值 $g\mu_B/\hbar\approx3.0385$（ $g=2$）。
- $\alpha$：无量纲 Gilbert 阻尼。
- $\mathbf B_i=-\partial E/\partial\mathbf S_i=\mathbf g_i$（eV/ $\mu_B$）。
- 与 Gilbert 形式的换算：LL 的 $\gamma_{\rm LL}=\gamma_G/(1+\alpha^2)$。**代码里 `gamma` 是 LL 的 $\gamma_{\rm LL}$**，因此噪声振幅必须带 $1/(1+\alpha^2)$，否则双重修正（阶段04 冻结结论）。

## 2. 涨落–耗散（FDT）

对上述 LL 方程，热场方差为

$$
\big\langle b_{i,a}(t)b_{j,b}(t')\big\rangle
=\frac{2\alpha k_BT}{\gamma\ |\mathbf S_i|\ (1+\alpha^2)}\ \delta_{ij}\delta_{ab}\ \delta(t-t').
$$

离散化后（步长 $\Delta t$）：

$$
\sigma^2=\frac{2\alpha k_BT}{\gamma\ \Delta t\ (1+\alpha^2)\ |\mathbf S_i|}.
$$

- 代码：`src/spin/spin.cu` 的 `gpu_fill_thermal_field`，`prefactor` 即上式分子，除 $\Delta t$ 与 $|\mathbf S_i|$ 在 kernel 内完成。
- 单位： $k_B=8.617333262\times10^{-5}$ eV/K； $T$ 用 K； $\Delta t$ 用 fs。
- **该系数自 Stage04 冻结，任何改动都必须重跑 FDT 门禁**。

## 3. 离散格式

### 3.1 几何格式（`scheme=geometric`，默认）

在步初磁场 $\mathbf B_n$ 上做 Strang 分裂：
1. 绕 $\hat{\mathbf B}_n$ 精确 Rodrigues 旋转 $\theta=\gamma|\mathbf B_n|\Delta t$；
2. 阻尼解析收缩 $\tan(\theta/2)\mapsto\tan(\theta/2)e^{-\gamma\alpha|\mathbf B_n|\Delta t}$；
3. 再旋转半步。

两者都是等距映射 → $|\mathbf S|$ 在精确算术下守恒；对常场**精确**，唯一离散误差来自噪声分段常数（Wong–Zakai → Stratonovich）。

### 3.2 随机隐式中点（`scheme=midpoint`）

$$
\mathbf S_{n+1}=\mathbf S_n-g\Delta t\ \mathbf m\times\mathbf H
-\frac{g\Delta t\ \alpha}{|\mathbf m|}\ \mathbf m\times(\mathbf m\times\mathbf H),
\quad \mathbf m=\frac{\mathbf S_n+\mathbf S_{n+1}}{2},
$$

$\mathbf H=\mathbf B_{\rm det}(\mathbf m)+\mathbf b_n+\mathbf B_{\rm ext}$。
**噪声每步只抽一次并在整个不动点迭代期间保持不变**（否则不是同一个随机格式）。
精确算术下 $\mathbf S_{n+1}-\mathbf S_n\perp\mathbf m$ 且 $\mathbf S_{n+1}+\mathbf S_n=2\mathbf m$ → $|\mathbf S|$ 严格守恒。

迭代终止： $\max_i|\mathbf S^{(k+1)}-\mathbf S^{(k)}|\le$ `midpoint_tolerance`（默认 $10^{-6}\mu_B$），
上限 `midpoint_max_iterations`（默认 50）。不收敛时**保留最后迭代值**并计数，绝不回退到冻结场。

### 3.3 保模长与约束投影

两个随机方案每步末尾都把 $|\mathbf S_i|$ 投影回每原子锚点（`project_onto_norm_anchor`）。
投影前记录**原始映射**的相对误差（`sllg_pre_projection_norm_error`）——那才是积分器守恒性的证据；
投影只清除 float32 每步 $\sim10^{-7}$ 的累积随机游走。

## 4. MC 与一致性检验

参考 MC（`tools/spin_validation/check_sllg_stage08a.py::mc_dimer`）：单位球面切平面随机旋转提案 + Metropolis。
两个体系的测度必须一致：**SLLG 固定 $|\mathbf S|$，MC 也必须在同一球面上提案**；若 MC 允许 $|\mathbf S|$ 涨落，
则不能与固定模长 SLLG 比较同一统计量。

验收（S08A-R1，实测）：

| 量 | 精确 | MC | SLLG geometric | SLLG midpoint |
|---|---|---|---|---|
| $\langle\mathbf S_1 \cdot \mathbf S_2\rangle$ | 0.2625 | 0.2615 | 0.2583 ($z=2.0$) | 0.2709 ($z=3.7$) |

$z_{\max}=4$（冻结）。单自旋： $\langle\cos\theta\rangle=\coth x-1/x$，实测 $+2.9$%（ $x=1$，64 replica），
dt 减半后 $+3.4$% → **偏差不随 $\Delta t$ 减小**，属采样/离散化之外的系统效应，已记录为已知问题。

## 5. 统计与误差

- 样本间隔取 $2\tau_G$， $\tau_G=1/(\gamma\alpha k_BT)$；仍用**分块标准误**与积分自相关时间双口径。
- 不得把相关帧当独立样本： $\tau_{\rm int}$ 用初始正序列截断的积分自相关（FFT 计算）。
- 误差条报告：replica 间标准误（保守）+ 自相关修正后的有效样本数标准误。

## 6. 代码入口

| 功能 | 文件 | 关键符号 |
|---|---|---|
| 参数解析 | `src/spin/spin.cu` | `Spin::parse`（`gamma= alpha= T= seed= scheme= restart=`） |
| 随机格式 | 同上 | `propagate_stochastic_midpoint` |
| 几何格式 | 同上 | `gpu_llg_step` |
| 保守分支 | 同上 | `propagate_conservative`（ $\alpha=T=0$） |
| 噪声 | 同上 | `gpu_fill_thermal_field` |
| 投影 | 同上 | `project_onto_norm_anchor` |

## 7. 本版本状态

| 项 | 状态 |
|---|---|
| FDT 方差（6 个 $\alpha$ 用例） | PASS（var 相对偏差 $-9.6\times10^{-3}$，mean/cov $z<1$） |
| 单自旋平衡在 5% 预注册带宽内 | PASS（+2.9%） |
| 4 SE 判据 | **FAIL**（ $z=4.5$）——已记录，未放宽 |
| 二聚体 vs MC/精确求积 | PASS（R1） |
| $\lVert \mathbf S\rVert$ 漂移 < 1e-5（20000 步） | PASS（ $2.0\times10^{-8}$ / $1.4\times10^{-7}$） |
