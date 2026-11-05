# PHYSICS_DERIVATION — GPUMD-spin v4.3.1-r21

本文件给出 r21 修复所用到的**独立推导**。所有公式从代码实际采用的
Hamiltonian、数组布局和积分格式出发，而不是从另一份实现抄过来。
`AUDIT_AND_FIX_REPORT.md` 引用本文件的编号；`tests/r21_validation.py`
与 `tests/r21_physics_extra.py` 给出对应的 float64 独立参考实现。

---

## 0. 记号与单位

| 量 | 记号 | 单位 / 约定 |
|---|---|---|
| 原子位置 | $\mathbf r_i$ | Å |
| 键矢 | $\mathbf r_{ij}=\mathbf r_j-\mathbf r_i$ | Å，最小镜像 |
| 自旋 / 磁矩 | $\mathbf S_i$ | μB；`model.xyz` 的 `spin:R:3` 列是**磁矩**，不是角动量 |
| 总有效场 | $\mathbf B_i=-\partial U/\partial \mathbf S_i$ | eV/μB |
| ME 耦合 | $\kappa_{ij}$ | eV/(Å·μB²) |
| ME 极化矢量 | $\mathbf E$ | **无量纲**权重/方向矢量（r21 更正） |
| LL 旋磁比 | $\gamma_{\rm LL}$ | μB/(eV·fs)，物理值 3.0385 |
| Gilbert 旋磁比 | $\gamma_G=\gamma_{\rm LL}(1+\alpha^2)$ | 同上 |
| 时间 | `run.in` 的 `time_step` 为 fs；内部 `time_step /= TIME_UNIT_CONVERSION` | 1 自然时间单位 = 10.1805 fs |
| 能量 | eV | 1 eV = 1.602176634e-19 J |
| Boltzmann | $k_B=8.617333262\times10^{-5}$ eV/K | |

`gamma` 是 **Landau–Lifshitz 约定** $\gamma_{\rm LL}$，不是 Gilbert 约定。
确定性方程

$$
\frac{d\mathbf S}{dt}=-\gamma_{\rm LL}\,\mathbf S\times\mathbf B
-\frac{\gamma_{\rm LL}\alpha}{|\mathbf S|}\,\mathbf S\times(\mathbf S\times\mathbf B),
$$

其噪声关联子（Gilbert 形式）为
$2\alpha k_BT/(\gamma_G|\mathbf S|)=2\alpha k_BT/[\gamma_{\rm LL}(1+\alpha^2)|\mathbf S|]$。
代码 `gpu_fill_thermal_field` 的 `prefactor` 正是后者，**保留** $1/(1+\alpha^2)$；
删掉它会把体系恒温在 $T(1+\alpha^2)$。见 `CPU.FDT.ALPHA_*`。

---

## 1. 磁电 Hamiltonian、力、磁梯度、virial

### 1.1 Hamiltonian 与约定

代码 `gpu_add_magelec_list/full` 采用有序对遍历 $(i,j)$ 与 $(j,i)$，
每个 visit 计算

$$
H_{ij}=\kappa_{ij}\,(\mathbf E\times\mathbf r_{ij})\cdot
(\mathbf S_i\times\mathbf S_j),\qquad
\mathbf r_{ij}=\mathbf r_j-\mathbf r_i .
$$

每个 visit 用 $\tfrac12 H_{ij}$ 计入能量（所以两次 visit 恰好是无序对的一次
$H_{ij}$）。用标量三重积

$$
(\mathbf E\times\mathbf r)\cdot\mathbf c
=\mathbf r\cdot(\mathbf c\times\mathbf E)=\mathbf c\cdot(\mathbf E\times\mathbf r),
\qquad \mathbf c=\mathbf S_i\times\mathbf S_j,
$$

得到 $H_{ij}=\kappa_{ij}\,\mathbf r_{ij}\cdot(\mathbf c\times\mathbf E)=\mathbf D_{ij}\cdot\mathbf c$，其中 $\mathbf D_{ij}=\kappa_{ij}(\mathbf E\times\mathbf r_{ij})$。

**量纲**： $\kappa$ [eV/(Å·μB²)] × $(\mathbf E\times\mathbf r)$ [Å， $\mathbf E$ 无量纲]
× $(\mathbf S_i\times\mathbf S_j)$ [μB²] = **eV**。
r20 文档称 $\mathbf E$ 为 eV 会给出 eV²，量纲不自洽；r21 统一为无量纲，
所有场强都吸收进 $\kappa$。这也解释了为什么 $\kappa_{ij}<0$ 与 $=0$ 必须合法。

### 1.2 力与磁梯度

$$
\frac{\partial H_{ij}}{\partial \mathbf r_{ij}}=\kappa_{ij}(\mathbf c\times\mathbf E)
\equiv \mathbf F_i^{(ij)},\qquad
\mathbf F_i=-\frac{\partial H}{\partial\mathbf r_i}=+\mathbf F_i^{(ij)},\quad
\mathbf F_j=-\mathbf F_i^{(ij)} .
$$

$$
\mathbf B_i=-\frac{\partial H}{\partial\mathbf S_i}
=-(\mathbf S_j\times\mathbf D_{ij}),\qquad
\mathbf B_j=-(\mathbf S_i\times(-\mathbf D_{ij})).
$$

`tests/r21_validation.py::cpu_me_virial_fd` 用纯能量有限差分验证：
最佳步长 $h=10^{-2}$ 时 $\max|\Delta F|=1.93\times10^{-15}$ eV/Å，
$\max|\Delta\tau|=1.24\times10^{-15}$ eV/μB；作用反作用
$\max|\sum_i\mathbf F_i|=7.8\times10^{-18}$ eV/Å。

### 1.3 九分量 virial 与有序对计数

GPUMD 的 per-atom virial 约定（见 `force/lj.cu`）是

$$
W_{ab}=-\sum_{i\lt j} r_{ij,a}\ F^{(ij)}_{i,b},
$$

即每个有序 visit 累加 $-\tfrac12 r_{ij,a}F_{i,b}$，两个 visit 合计
$-r_{ij,a}F_{i,b}$。r20 的 `me_accumulate` 在每个 visit 累加**完整**
$-r_{ij,a}F_{i,b}$，能量却用 $\tfrac12 H_{ij}$，于是总 virial 是正确值的
**两倍**；r20 的 `tests/v4_regression.py` 又把参考值也乘了 2，把错误
"测试通过"。r21 两处都修正：实现加 $\tfrac12$，参考改为无序对求和，
并加入独立的能量—应变有限差分。

因为 $H_{ij}$ 对 $\mathbf r_{ij}$ 是齐次一次，

$$
\operatorname{tr}W=-\sum_{i\lt j}\mathbf r_{ij}\cdot\mathbf F_i
=-\sum_{i\lt j} H_{ij}=-E_{\rm ME},
$$

（**不是** $-2E_{\rm ME}$）。实测 `CPU.ME.TRACE_VIRIAL_EQUALS_MINUS_ENERGY`
给出比值 $1.000000000$。

### 1.4 能量—应变有限差分（两种变形约定，差一个转置）

令变形梯度扰动 $\varepsilon$，位置与晶胞一起仿射变形：
$\mathbf r' = M\mathbf r$， $h'=hM$（行向量写法）。分数坐标
$\mathbf s=\mathbf r h^{-1}$ 在仿射映射下不变，因此**周期最小镜像必须在
分数坐标里取，再用变形后的 $h'$ 映回**；只变形原子而保持旧盒子会在
跨边界键矢上产生约 0.67 eV 的假误差。

两种常见约定：

* $M=I+\varepsilon$（行向量 $\mathbf p'=\mathbf p M$）：
  $\partial r_a/\partial\varepsilon_{ij}=r_i\delta_{aj}$，
  $\partial H/\partial\varepsilon_{ij}=F_{j}r_i$，
  $-\partial H/\partial\varepsilon=W^{T}$ … 具体地
  $-\partial H/\partial\varepsilon_{ij}=-r_i F_j=W_{ij}$ 当 $W_{ij}=-r_iF_j$。
  即 **$-dH/d\varepsilon = W$**。
* $M=I+\varepsilon^T$（GPUMD `change_box` 的列向量形式
  $\mathbf R'=(I+\varepsilon)\mathbf R$）：
  $\partial H/\partial\varepsilon_{ij}=F_i r_j$，
  **$-dH/d\varepsilon = W^{T}$**。

对齐次二次型（中心力） $\mathbf F_i\parallel\mathbf r_{ij}$， $W$ 对称，两者
一致；对 ME（ $\mathbf F_i=\kappa(\mathbf c\times\mathbf E)$ 一般**不平行**于
$\mathbf r_{ij}$） $W$ 非对称，约定差异不可忽略。r21 的独立参考对两种约定
**都**给出解析期望并分别比对：

* `CPU.ME.VIRIAL_9COMP_ENERGY_STRAIN_FD`：九分量最大误差
  $4.76\times10^{-12}$ eV（两种约定各一次）， $|W-W^T|_{\max}=0.199$ eV
  （证明测试对转置确实敏感）。
* `v4_regression.py`（216 原子周期 FeGe）用分数坐标 MIC + 步长扫描
  $10^{-3}\ldots10^{-7}$ 且校验邻居对集合不变：
  `M=I+eps` 误差 $6.7\times10^{-14}$ eV，
  `M=I+eps^T` 误差 $6.7\times10^{-14}$ eV；
  $\min||r|-r_c|=0.0552$ Å，无截断跨越。

倾斜/非正交盒的 per-atom virial 与分箱体积归一化仍是 GPUMD 的既有近似；
r21 不声称修复变胞应力，只明确拒绝不支持的分箱组合。

### 1.5 截断连续性

硬截断 $r_c$ 使 $E$ 分段光滑： $E$ 本身在 $r_c$ 连续（成对能量在该处为有限值
但被直接丢弃，故 $|dE|=0$ 对 $|r_c\pm\delta|$ 的两个"开/关"状态会跳变，
本参考测得 $|dE|=0$ 是因为两侧都在截断外/内时的对称取点），
而力与磁梯度在 $r_c$ 有有限跳变（实测 $|dF|=9.2\times10^{-3}$ eV/Å，
$|d\tau|=1.472\times10^{-2}$ eV/μB）。r21 **不引入平滑开关**：那会改变
Hamiltonian，并要求重新推导力、磁梯度、virial 和能流的附加项。
远离切换区的行为完全不变；此限制写入 `KNOWN_LIMITATIONS.md`。

### 1.6 周期像与 cell-list 路径

`me_reference(..., images=...)` 显式枚举 27 个周期像，允许 $i$ 与自身周期像
在 $|r|<r_c$ 时成为合法邻居（**不能**把所有自周期像一律判为错误），只去除
重复的 $(i,j,\text{image})$ visit。小盒/非周期方向由 `gpu_add_magelec_full`
全遍历路径接管（`list_is_safe` 判据 `num_bins[d]>=5`），大盒走 cell-list，
两条路径的判据与已有 NEP 邻居表一致。最小镜像并不适用于任意 $r_c$：
当 $r_c$ 超过半盒长时必须显式枚举多像，代码通过 `list_is_safe` 拒绝不安全
的 cell-list。

---

## 2. 隐式中点积分与数值健壮性

### 2.1 布局与残差

自旋数组是 component-major 的 $3N$：`[x(0..N-1), y(0..N-1), z(0..N-1)]`。
r20 的 `gpu_max_abs_diff` 以 `grid=ceil(N/128)` 启动、却传入 `n=3N`，
每个线程只读 `a[i]`，因此只覆盖前 `min(3N, grid*128)` 个元素。
当 $N$ 恰为 128 的倍数时覆盖范围正好是 x 块，**y/z 完全未检查**；
小 N 或非整块 N 时部分 y（甚至 z）会被扫到。这个"覆盖模型"是解析可算的：

$$
\text{covered}(N)=\min\bigl(3N,\ \lceil N/128\rceil\cdot128\bigr).
$$

`tests/unit/midpoint_residual_test.cu` 与 `CPU.RESIDUAL.N*` 都按该模型给出
r20 复现内核的解析期望（不是从输出反推）。实测 N=128、1024 时 r20
报 0（假收敛），r21 在所有 N=16/43/128/129/432/1024 上报
$r_{\rm abs}=0.0124999881$ μB。

r21 定义（每个线程检查自己原子的三个分量）：

$$
d_{i,c}=S^{\rm new}_{cN+i}-S^{\rm prev}_{cN+i},\quad
r_{\rm abs}=\max_{i,c}|d_{i,c}|,\quad
s=\max_{i,c}\max(|S^{\rm new}|,|S^{\rm prev}|),\quad
r_{\rm rel}=r_{\rm abs}/\max(s,10^{-30}).
$$

收敛当且仅当 $r_{\rm abs}\le\mathrm{atol}$ **且** $r_{\rm rel}\le\mathrm{rtol}$
**且** `bad==0`；默认 atol=rtol= $10^{-6}$，最多 50 次迭代。atol 单位 μB，
rtol 无量纲。NaN/Inf 与 $|S|>10^{20}$ 进入显式 `bad` 计数而非最大值归约
（IEEE 比较遇 NaN 为假，r20 会把 NaN 残差悄悄丢掉并报 0）。

### 2.2 解析反例： $x$ 不变、 $y/z$ 仍在动

$\mathbf S_0=(0,s,0)$， $\mathbf B=(B,0,0)$， $\theta=\gamma B\Delta t=0.1$，
中点格式的不动点（float64 闭式解）

$$
S_y=s\frac{1-\theta^2/4}{1+\theta^2/4},\qquad
S_z=s\frac{\theta}{1+\theta^2/4},\qquad S_y^2+S_z^2=s^2 .
$$

$x$ 分量在每次迭代都恒为 0（只看 x 会一次就判收敛），而完整首步残差
$=0.0997506$；不动点迭代 5 次降到 $6.26\times10^{-7}$，
迭代到机器精度后 $|S|$ 误差为 0。`GPU.UNIT.midpoint_x_unchanged_yz_changing_N128`
与 `CPU.MIDPOINT.X_FIXED_YZ_MOVING` 分别用 GPU shipped kernel 和 CPU 解析
闭式解验证。

### 2.3 未收敛与模长锚点

非收敛默认 `exit(4)`，不使用最后一次未收敛 iterate，也不做模长投影
（投影无法修复一个没有解的不动点方程）。 $|\mathbf S|$ 锚点只在初始化、
重启恢复或合法状态重建时写入；正常积分中锚点固定。
`gpu_project_anchor` **先**记录相对投影前误差
$\epsilon=||S|-\text{anchor}|/\text{anchor}$，再投影；
$\epsilon>$ 容差或非有限值计数到 `bad` 并在 run 结束时硬失败。
单元测试中把某个原子 $|S|$ 改成 1.5，测得投影前误差 0.5、违规计数 1、
锚点仍为 1.0、投影后模长 1.0。

### 2.4 FDT 与单自旋平衡分布

固定 $|\mathbf S|=s$ 的自旋在静场 $\mathbf B=B\hat z$ 中的解析平衡分布

$$
p(u)=\frac{a}{2\sinh a}e^{au},\quad u=\cos\theta=\frac{S_z}{s},\quad
a=\frac{Bs}{k_BT},
$$

$$
\langle u\rangle=\coth a-\frac1a,\qquad
\operatorname{Var}(u)=\frac{1}{a^2}-\frac{1}{\sinh^2a}.
$$

`EXTRA.SLLG.*` 用两原子远距体系（无邻居、固定晶格以关闭 R19 晶格守护）、
$T=300$ K、 $\alpha=1$、 $\gamma=3.04$、 $\Delta t=1$ fs，烧入 25%，
用初正序列估计积分自相关时间 $\tau_{\rm int}$，统计误差
$\sigma_{\bar u}=\sqrt{\operatorname{Var}(u)\ 2\tau_{\rm int}/N}$，
并给出直方图与解析密度的偏差（以二项标准差为单位）。
两个场强 $B=0.03$ 与 $0.01$ eV/μB 分别给出 $z=0.78$ 与 $1.45$，
在 $3\sigma$ 内一致； $\tau_{\rm int}\approx1.2$– $1.4$ 个采样间隔。

---

## 3. 外磁场、时钟与能量平衡

### 3.1 Tesla 换算与 g

结构文件的自旋列是**磁矩** μB；积分器 $\gamma=3.0385$ 本身已含 $g=2$。
故 $1\ \mathrm T=\mu_B=5.7883818066\times10^{-5}$ eV/μB，默认 $g=1$，
不得再乘一次 $g$。`magfield` 的 `unit=tesla` 用
$\mathbf B=g\mu_B H\hat n$；`unit=ev` 直接取 eV/μB。
`CPU.UNITS.TESLA_TO_EV_PER_MUB`、`CPU.UNITS.DOUBLE_G_EXCLUDED` 与
`v4_regression.py` 的 Zeeman 能量/力矩检查共同排除 double-g。

### 3.2 时间采样与重启

含时包络（`spin_external.cu`）

$$
\mathbf B(t)=\mathbf B_{\rm peak}\,\times
\begin{cases}1,&\text{static}\\ \sin(2\pi f t_{\rm ps}+\phi),&\text{sin}\\
\cos(2\pi f t_{\rm ps}+\phi),&\text{cos}\end{cases}
$$

$f$ 单位 THz， $t_{\rm ps}=t_{\rm fs}\times10^{-3}$。r21 在每步**中点**
$t=(n-\tfrac12)\Delta t$ 采样（`run.cu` 的 `set_time_fs`），与二阶分裂一致。
restart schema v2 保存 `add_bfield`/`magfield` 配置、`global_time` 与
`spin_external.time_fs`；v1 仍可读。`GPU.RESTART.static/sin/cos` 测得
连续 8 步与 4+4 分段重启的最终自旋最大差为 **0 μB**。

### 3.3 含时场的功与能量平衡

含时 Hamiltonian 的显含时功

$$
W_{\rm drive}=\int \frac{\partial H}{\partial t}\,dt
=-\int \frac{d\mathbf B}{dt}\cdot\sum_i\mathbf S_i\,dt .
$$

`EXTRA.WORK.SIN` 用轨迹中点自旋和上面的解析包络独立计算这一积分，
与 `thermo.out` 的裸势能变化对比：逐步最大残差 $<10^{-6}$ eV，总量
$\Delta E_{\rm bare}-W_{\rm drive}$ 在 $10^{-4}$ eV 内。
静态场对应 $W_{\rm drive}=0$，裸总能量守恒（`EXTRA.WORK.STATIC`）。
**结论**：含时外场下不得要求裸总能量恒定，必须把外场功计入。

> 注意 F16 的 `spin_work.out` 中 `W_ext` 只统计 `add_bfield` 通道的
> $\mathbf b_{\rm ext}\cdot d\mathbf S$；`magfield` 的 Zeeman 场通过总场
> 进入 `W_rot`，而显含时的 $\partial H/\partial t$ 不在该预算内。
> 因此 `spin_work.out` **不是**含时外场的完整能量账本；r21 用上面的
> 独立积分作能量平衡验证，并把该限制写入报告。

---

## 4. 热流：为什么保持 BLOCKED

### 4.1 局域能量连续性方程

总能流 $\mathbf J$ 必须满足局域能量守恒

$$
\frac{\partial \varepsilon_i}{\partial t}+\nabla\cdot\mathbf j_i
=\dot q_i^{\rm bath}+\dot q_i^{\rm drive}.
$$

对一般非线性 NEP 势 $U_i(\lbrace \mathbf r\rbrace ,\lbrace \mathbf S\rbrace )$，
势能传递项需从**完整的** $U_i$ 对键矢与自旋的导数推出，并统一原子归属、
周期键矢和符号；此外还要总能量对流项 $\varepsilon_i\mathbf v_i$ 和自旋
传递项。r20/r21 的 `compute_heat.cu` 采用

$$
\mathbf J^{\rm spin}\propto -\tfrac12\,\mathbf S_i\cdot\mathbf B_i
\quad(\text{加上速度/virial 项}),
$$

该关系只对**二次齐次 Heisenberg** $U=-\sum_{i\lt j}J_{ij}\mathbf S_i\cdot\mathbf S_j$
（Euler 齐次性 $-\tfrac12\sum_i\mathbf S_i\cdot\mathbf B_i=U$）成立：

* `CPU.LOCAL_ENERGY.HEISENBERG`： $U=0.310977561763$ eV，
  $-\tfrac12\sum_i\mathbf S_i\cdot\mathbf B_i=0.310977561763$ eV，差 $1.7\times10^{-16}$。
* `CPU.LOCAL_ENERGY.BIQUADRATIC_COUNTEREXAMPLE`：加入 $\lambda q^2$ 后差
  $4.02$ eV；梯度有限差分 err $4.4\times10^{-10}$ 证明参考场正确。
* `CPU.LOCAL_ENERGY.NEP_LIKE_NONLINEAR`：含能量零点偏置与四次项时差 $12.8$ eV。

因此 $-\tfrac12\mathbf S_i\cdot\mathbf B_i$ **不能**当作一般 NEP 的精确局域
自旋能量。DFT 能量零点大不是删项的理由。

### 4.2 尚未闭合的项

* 一般 NEP 的 $\partial U_i/\partial\mathbf r_{ij}$ 与
  $\partial U_i/\partial\mathbf S_j$ 在键矢/自旋上的**局域**导数与
  `force.spin_flux_A` 生成路径的一一对应尚未逐项推导并数值验证；
* 外加 ME / Zeeman 项进入能量与运动方程后，其对应的能量传递贡献
  （键能流与 $\partial H/\partial t$）未纳入 `spin_flux_A`；
* 含阻尼、随机热浴、时间依赖驱动的动力学必须处理源汇项
  $\dot q_i^{\rm bath}$、 $\dot q_i^{\rm drive}$，不能把不完整的确定性
  $d\mathbf S/dt$ 无条件称为实际总能流；
* 固体中若采用等价的能流规范（省略对流项），需给出适用条件与证明；
  r21 未做此证明。

因此 `heat_flux.out` / `hac.out` 在自旋体系中**仅为诊断量**，
r21 在 `compute_heat.cu`、`hac.cu`、`dump_heat_flux.cu` 加入
`[BLOCKED]` 警告，禁止据此报告定量热导或 $\kappa_{pm}$。

### 4.3 Green–Kubo 前因子（推导，不是拟合）

代码里热流的自然单位为 eV/(Å·自然时间·K)。
$1\ \mathrm{eV/(Å\,fs\,K)}=1.602176634\times10^{-19}\,\mathrm J/(10^{-10}\,\mathrm m\times10^{-15}\,\mathrm s\times1\,\mathrm K)=1.602176634\times10^{6}\,\mathrm{W/(m\,K)}$。
除以 10.1805 fs/自然时间单位得
$1.573769\times10^{5}$ W/(m·K)，与
`KAPPA_UNIT_CONVERSION=1.573769e5` 比值 0.99999971。
`CPU.UNITS.GREEN_KUBO_PREFACTOR` 独立复算；**没有**按「除以约 98 后
数值像 Fe」作经验调整。完整单位链与 SI 换算见该测试输出。

### 4.4 已排除的旧疑点

* **HAC 交叉项**：当前 `gpu_find_hac` 同时累加
  $J_{\rm in}(t)J_{\rm out}(t+\tau)$ 与 $J_{\rm out}(t)J_{\rm in}(t+\tau)$
  （x 与 y 通道，含 yi/yo）。`CPU.HAC.CROSS_CORRELATION_PRESENT` 静态
  核对该源码；旧文档「缺交叉项」不是当前版本缺陷。
* **Onsager–Casimir**：有限外场下 $\kappa_{pm}=\kappa_{mp}$ 需要
  $\mathbf B\to-\mathbf B$ 交换条件与通道定义的规范不变性，
  不能仅由两通道都是时间反演奇就无条件写出。r21 不给出该等式。
* **分箱**：`dump_heat_flux bin=` 在倾斜/非正交盒或分箱轴非周期时
  `PRINT_INPUT_ERROR` 明确拒绝，避免静默的错误体积归一化。

---

## 5. 与实现修改无关的物理解释（不当作 bug）

已有 NEP 势对实验的偏差、SOC 训练数据覆盖范围、固定模长约束、
`Sdm` 对称性、`Sia` 表达能力属于**模型/数据层面**，与本次实现纠错分开报告。
r21 没有重训模型、没有删除描述符、没有改模型参数，也没有改变 Hamiltonian
来掩盖实现错误。所有 `nep4_spin` 模型文件与 `spin_layout` 描述符布局保持兼容。
