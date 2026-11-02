# 12 — Green–Kubo、单位换算与统计收敛

## 12.1 公式

广延热流 $\mathbf J$（eV·Å/ps）时

$$
\kappa_{\alpha\beta}=\frac{1}{Vk_BT^2}\int_0^\infty
\big\langle J_\alpha(t)J_\beta(0)\big\rangle\ dt .
\tag{12.1}
$$

用密度热流 $\mathbf j=\mathbf J/V$（eV/(Å²·ps)）时前因子换成 $V/(k_BT^2)$。
离散实现：

$$
\kappa_{\alpha\beta}\approx\frac{\Delta t}{Vk_BT^2}
\sum_{m=0}^{M}\big\langle J_\alpha(m\Delta t)J_\beta(0)\big\rangle\ w_m ,
\tag{12.2}
$$

$w_m$ 是梯形权重， $M$ 是积分窗（必须看到相关函数衰减到零）。

**约定陷阱**： $V$ 用 Å³、 $k_B=8.617333262\times10^{-5}$ eV/K、 $t$ 用 ps，
则 (12.1) 给出的 $\kappa$ 单位是 eV/(Å·ps·K)，再乘 §12.2 的因子才是 W/(m·K)。
在 128 原子胞上曾把 $V$ 误用成 nm³，结果差 $10^{3}$——这类错误不会报错，
只会给出一个"看起来合理"的数，所以必须用解析模型自检（§12.4）。

## 12.2 单位换算（可重算）

$$
1\ \frac{\rm eV}{\rm \AA\cdot ps}
=1.602176634\times10^{-19}\cdot\frac{1}{10^{-10}\cdot10^{-12}}\ \frac{\rm W}{\rm m}
=1.602176634\times10^{3}\ \frac{\rm W}{\rm m}.
\tag{12.3}
$$

若热流用 eV/(Å²·ps)（通量密度），则

$$
1\ \frac{\rm eV}{\rm \AA^2\cdot ps}=1.602176634\times10^{-14}\ {\rm W/m^2},
\tag{12.4}
$$

乘 $1/(Vk_BT^2)$ 时 $V$ 必须换成 m³（ $1\ {\rm \AA^3}=10^{-30}\ {\rm m^3}$）。
本项目 `tools/physics_validation/analyze_heat.py` 里把这两个常数写成
`EV_A_PER_PS_TO_W = 1.602176634e-14` 与 `KB_J = 1.380649e-23`，
并在同一函数里与 MD 数据和自检数据共用，避免两条路径各写一套。

## 12.3 分解

$\mathbf J=\mathbf J\_{\rm L}+\mathbf J\_{\rm S}$ 时

$$
\kappa=\kappa_{\rm LL}+\kappa_{\rm SS}+\kappa_{\rm LS}+\kappa_{\rm SL},
\tag{12.5}
$$

$\kappa\_{\rm LS}+\kappa\_{\rm SL}$ 即 cross 项。三项都依赖 11.3 的规范选择；
只有总和是可观测量。**报分解时必须同时报规范。**

## 12.4 估计器自检（必须做，且已做）

把已知解析解的 Ornstein–Uhlenbeck 过程
$\langle J(0)J(t)\rangle=\sigma^2e^{-t/\tau}$ 喂进同一个积分器，
解析结果 $\kappa=\sigma^2\tau/(Vk_BT^2)$。60 次独立实现（ $n=40000$，
$\Delta t=5\times10^{-4}$ ps， $\tau=0.05$ ps）：

| 量 | 实测 | 判据 |
|---|---|---|
| 估计/解析 | **0.978 ± 0.123** | $1\pm0.10$ → PASS |
| 报告的 1σ 误差 | 9.3% | — |
| 实际散布 | 12.3% | 误差被低估 **1.32×** |
| 1σ 覆盖率 | 0.52 | $\ge0.5$ → PASS |

结论：**积分器与单位换算是对的**（否则比值会是 $10^{-7}$ 那种量级），
但 8 段分块的误差棒偏小 1.3 倍，报数时要乘 2 保守使用。

这条自检是本篇唯一"可用"的正面结论。它也直接否掉了" $\kappa$ 差 7 个数量级
是积分器写错了"这一假设——问题在输入的热流本身（11 篇 §11.6）。

## 12.5 统计收敛：5 ps 远远不够

`gk_N1024`：1024 原子、NVE、 $\Delta t=0.05$ fs、5 ps、每 10 步采样，
共 10000 个样本。

| 量 | 值 |
|---|---|
| $\tau$（ $C(t)$ 降到 $0.1C_0$ 的积分时间） | **0.176 ps** |
| 独立样本数 $t_{\rm tot}/2\tau$ | **14.2** |
| $C(t)/C(0)$ at 0.005 / 0.05 / 0.25 / 0.5 ps | 0.982 / 0.600 / 0.275 / 0.196 |
| $\langle J\_x\rangle,\langle J\_y\rangle,\langle J\_z\rangle$ | −4.2, −20.6, −10.5 eV·Å/ps（ $\sigma$=34, 28, 25） |
| $\langle J\_x^2\rangle/\langle J\_y^2\rangle$ | 1.45 |

**平台不存在**： $5\tau$– $12\tau$ 窗口内的积分值在 0 附近来回，分段误差发散。
5 ps 只有 14 个独立样本，距 10% 精度所需的 ~200 个样本（ $\approx70$ ps）差一个数量级。

可用的定量结论只有一条：*1024 原子 Fe、250 K 下热流自相关的衰减时间是
0.1–0.2 ps 量级，要 10% 的 $\kappa$ 至少需要 70 ps 的 NVE 采样，且需多组独立初态。*

## 12.6 误差与收敛的检查清单

* **块平均**：切成 $n_b$ 块，块长 $\gg\tau_{\rm int}$，误差 $=\mathrm{std}(\kappa\_b)/\sqrt{n\_b}$。
  注意块数太少会低估误差（本项目的自检给出 1.32× 低估， $n_b=8$）。
* **积分平台**： $\kappa(M)$ 必须出现平台；没有平台说明窗口不够或统计不足。
* **尺寸收敛**： $L\to2L$ 时 $\kappa$ 变化应在误差内。
* **各向异性**：立方体系 $\kappa\_{xx}=\kappa\_{yy}=\kappa\_{zz}$；实测 1.45 说明远未收敛。
* **多初态**：至少 4 个独立微观态，报标准误而不是单次值。

## 12.7 与 NEMD 互证（未做）

一端注入、一端抽取能量，测稳态 $\nabla T$， $\kappa=-J/\nabla T$。
两者在同一 Hamiltonian 下应在统计误差内一致。**本项目未做**，
记为 `NOT_RUN`。

## 12.8 状态

`BLOCKED`。理由链：耦合守恒门禁未过（07）→ 漂移会被积进 GK 积分；
热流绝对值未验证（11.6）→ 即使收敛也只是一个错误的数乘上正确的积分器。

代码路径（`compute_heat.cu`、`dump_heat_flux.cu`、`hac.cu`）存在且能跑，
`compute_hac` 在本体系上给出负值，已判定不可用（11.6）。

**本项目不输出任何 $\kappa$ 数值。**
