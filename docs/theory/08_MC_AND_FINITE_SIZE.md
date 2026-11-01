# 08 — Monte Carlo、统计力学与有限尺寸标度

## 8.1 Metropolis（固定模长）

状态 $\{\mathbf S_i\}$，$|\mathbf S_i|=s_i$ 固定。单自旋提议：在球面上均匀取新方向，接受概率
$p=\min(1,e^{-\beta\Delta E})$。细致平衡由对称提议 + Metropolis 判据保证。
一个 sweep = $N$ 次提议。热化 $N_{\rm burn}$ sweeps，采样 $N_{\rm sample}$ sweeps，按块平均给误差。

与 SLLG 比较的前提：
1. 同一个 $E(\mathbf R,\mathbf S)$（本项目用同一份 `nep.txt` + 同一个 `Force::compute`）；
2. 同样的 $|\mathbf S_i|$（不能一边固定一边让纵向涨落）；
3. 同样的 $T$、同样的边界条件。

## 8.2 观测量

$$
m=\frac{1}{N}\Big|\sum_i \mathbf S_i\Big|,\quad
\chi=\frac{N}{k_BT}\big(\langle m^2\rangle-\langle m\rangle^2\big),\quad
U_4=1-\frac{\langle m^4\rangle}{3\langle m^2\rangle^2},
$$

$$
C_V=\frac{\langle E^2\rangle-\langle E\rangle^2}{k_BT^2}.
$$

$m$ 是**序参量**；对反铁磁要换成交错磁化，否则 $m\approx0$ 会被误读成“无序”。

## 8.3 有限尺寸标度

临界点由 Binder cumulant 的交点定：不同 $L$ 的 $U_4(T)$ 在 $T_c$ 相交。
交点附近的标度形式

$$
U_4(L,T)=f\big((T-T_c)L^{1/\nu}\big),\qquad
\chi_{\max}\propto L^{\gamma/\nu},\qquad
\big|\partial_T U_4\big|_{\max}\propto L^{1/\nu}.
$$

只有 (a) 每个尺寸都充分热化、(b) 至少 3 个尺寸、(c) 每个温度点有独立种子与误差条，
才允许报 $T_c$。**本项目没有做**（模型未过资格门禁），记 `NOT_RUN`，不给 $T_c$。

## 8.4 自相关与有效样本

$$
\tau_{\rm int}=\frac12+\sum_{t=1}^{W}\rho(t),\qquad
N_{\rm eff}=\frac{N_{\rm sample}}{2\tau_{\rm int}} .
$$

本项目用 FFT 做 Wiener–Khinchin 求 $\rho(t)$（逐 lag 的 $O(n^2)$ 太慢）。
误差条一律按块平均（块长 $\gg\tau_{\rm int}$），不按原子分量当独立样本。

## 8.5 代码入口

* `mc/mc.cu`、`mc/mc_ensemble_canonical.cu`（GPUMD 自带 MC，用于晶格自由度）
* 自旋 MC：本项目**没有**内置实现，用 `tools/spin_validation/check_sllg_stage08a.py`
  里的独立 Python Metropolis 做小体系基准
* 统计：`tools/spin_validation/check_sllg_stage08a.py::integrated_tau`

## 8.6 已验证 / 未验证

* 已验证：二聚体（2 自旋）SLLG vs Metropolis vs 四维求积，$z\le4$（S08A-R1）。
* 未验证：Fe 的有限尺寸扫描、$T_c$、临界指数。状态 `NOT_RUN_MODEL_GATE`。

## 8.7 限制

* Python MC 每步要调一次生产求值器（~0.2 s），只适合 ≤10 原子的小体系。
* 固定模长的 MC 不能用来研究纵向磁矩涨落（需要另一套测度）。
