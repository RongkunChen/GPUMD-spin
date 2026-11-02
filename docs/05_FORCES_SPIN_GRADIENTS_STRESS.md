# 05 — 力、磁梯度、应力：定义、单位与互易性

## 1. 定义（全部来自同一个 $E$）

$$
\mathbf F_i=-\frac{\partial E}{\partial \mathbf r_i}\ [\mathrm{eV/\text{\AA}}],\qquad
\mathbf g_i=-\frac{\partial E}{\partial \mathbf S_i}\ [\mathrm{eV/\mu_B}],\qquad
\boldsymbol\tau_i=\mathbf S_i\times\mathbf g_i\ [\mathrm{eV}].
$$

- $\mathbf g$ 是**梯度**（training 标签 `spin_forces_vert` 按约定取 $-\partial E/\partial\mathbf S$，即 $\mathbf g$ 本身；字段来源状态 `ASSUMED_LABEL_CONTRACT`，不是 DFT 已核验量）。
- $\boldsymbol\tau$ 是力矩（能量量纲）。**不要把 $\mathbf g$ 和 $\boldsymbol\tau$ 混用**：前者沿磁矩方向有分量，后者严格垂直于 $\mathbf S$。
- 固定模长动力学只用 $\boldsymbol\tau$ 的切向部分： $\dot{\mathbf S}=-\gamma\ \mathbf S\times\mathbf B_{\rm eff}$，纵向分量不进入运动方程，但在训练/可变模长理论里有意义。

## 2. 变分一致性与混合导数

$E$ 二阶连续可微时

$$
\frac{\partial F_{i\alpha}}{\partial S_{j\beta}}
=-\frac{\partial^2 E}{\partial r_{i\alpha}\partial S_{j\beta}}
=\frac{\partial g_{j\beta}}{\partial r_{i\alpha}} .
$$

数值检验（生产 NEP，16 Fe 原子，中心差分 $h=10^{-3}$ Å / $\mu_B$）：

| 量 | 解析 | 差分 | 相对差 |
|---|---|---|---|
| $F_x=-\partial E/\partial r_x$ | $-3.172968$ | $-3.171921$ | $3.3\times10^{-4}$ |
| $g_x=-\partial E/\partial S_x$ | $-0.040138$ | $-0.040054$ | $2.1\times10^{-3}$ |
| $\partial F_x/\partial S_x$ vs $\partial g_x/\partial r_x$ | $+0.143528$ | $+0.143619$ | $6.4\times10^{-4}$ |

脚本：`tools/one_shot/phase2_s05b_r2.py`；证据：`evidence/final_one_shot_*/phase_2_results.json`。
差分误差随 $h$ 的收敛已验证（ $h=10^{-6}$ 落入 float32 噪声地板， $h=10^{-3}$ 最优点）。

> 该检查只证明**导数互相自洽**，不证明积分器守恒。守恒是 `07` 的门禁。

## 3. Virial 与应力

MD 侧每原子 9 分量，顺序 `xx yy zz xy xz yz yx zx zy`（`src/force/nep.cu` 的 `g_virial` 写入）。
trainer/特殊化侧是 6 分量 `xx yy zz xy yz zx`。**比较时必须按名字映射**，不能按数组前缀比较（`xy\neq yx` 在数值上）。

$$
\Pi_{\alpha\beta}=\sum_i \mathbf r_{i\alpha}F_{i\beta},\qquad
\sigma_{\alpha\beta}=-\frac{\Pi_{\alpha\beta}}{V}\ \text{(符号依约定)}.
$$

单位：eV/Å³；`dump_xyz` 头部的 `stress=` 与 `virial=` 由同一 `gpu_total_virial` 派生。

## 4. 数值算法

1. 力/梯度在同一 kernel：先算 $q$，再 $\partial q/\partial\mathbf r$、 $\partial q/\partial\mathbf S$，最后乘 ANN 的 $\partial e/\partial q$。
2. 累加顺序：float32 逐邻居累加（**这是 G6 差异的根源**，见 `VALIDATION_REPORT.md`）。
3. `spin_work` 记账： $dU=-B_{\rm det} \cdot d\mathbf S$ 拆成 rotation / noise / external 三项（`src/spin/spin.cu` F16）。

## 5. 常见错误（验收反例）

| 错误 | 表现 | 判别 |
|---|---|---|
| 把 $\mathbf g$ 当 $\boldsymbol\tau$ 用 | 磁矩出现纵向增长 | $\mathbf S\cdot\mathbf g\neq0$ 但 $\mathbf S\cdot\boldsymbol\tau=0$ |
| virial 分量顺序按 6 分量截断 | $xy\ne yx$ 的体系应力不对称 | 逐分量按名映射后比较 |
| 有限差分步长过小 | $h=10^{-6}$ 时 $\Delta E$ 落入 float32 噪声 | 做 $h$ 扫描看收敛平台 |
| 单位混用 | 力矩大 $10^{3}$ | $\gamma$ 用 $\mu_B/(\mathrm{eV\ fs})$， $\mathbf g$ 用 $\mathrm{eV}/\mu_B$ |

## 6. 本版本状态

| 项 | 状态 |
|---|---|
| $F=-{\partial E}/{\partial R}$ | PASS（ $3.3\times10^{-4}$） |
| $g=-{\partial E}/{\partial S}$ | PASS（ $2.1\times10^{-3}$） |
| 混合导数互易 | PASS（ $6.4\times10^{-4}$） |
| **逐原子** trainer↔MD 一致（G6） | **FAIL**（1.19e-6 > 1.05e-6 eV，历史；本轮未重测 e_i） |
| virial 符号/顺序 | 已按名映射；未做独立解析验证（NOT_VALIDATED） |
