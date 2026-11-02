# 06 — LL/Gilbert、FDT 与随机积分器

## 6.1 运动方程

代码求解的是固定模长的 **Landau–Lifshitz (LL)** 形式，**不是** Gilbert 形式
（与 `docs/06_SPIN_LLG_FDT_AND_MC.md` §1 一致）：

$$
\frac{d\mathbf S_i}{dt}=-\ \gamma\ \mathbf S_i\times\mathbf B_i
-\frac{\gamma\alpha}{|\mathbf S_i|}\ \mathbf S_i\times\big(\mathbf S_i\times\mathbf B_i\big),
\qquad
\mathbf B_i=-\frac{\partial E}{\partial\mathbf S_i}=\mathbf g_i .
\tag{6.1}
$$

* $\gamma$：代码里的 `gamma`，单位 $\mu_B$/(eV·fs)，物理值 $g\mu_B/\hbar\approx3.0385$
  （ $g=2$），`spin_llg gamma=` 可改。
* $\alpha$：无量纲 Gilbert 阻尼。
* **约定（最容易写错的一处）**：这里的 `gamma` 是 LL 的 $\gamma_{\rm LL}$，与 Gilbert
  形式的关系为 $\gamma\_{\rm LL}=\gamma\_G/(1+\alpha^2)$。若把方程改写成 Gilbert 形式
  （两个系数都带 $1/(1+\alpha^2)$），数学上等价，但**不能再**给噪声振幅额外乘一次
  $1/(1+\alpha^2)$——那正是 (6.4)(6.5) 中 $1/(1+\alpha^2)$ 的来源（Stage04 冻结结论）。
* 单位： $\mathbf B$ 用 $\mathrm{eV}/\mu_B$，时间步长用 **fs**，故 $d\mathbf S/dt$ 为 $\mu_B$/fs。
  `spin_llg` 用的 $\Delta t$ 就是 MD 时间步长，代码内部已是 fs（`src/spin/spin.cuh`）。

**结构性质**（由 (6.1) 直接可验）：

$$
\mathbf S_i\cdot\dot{\mathbf S}_i=0
\quad\Longrightarrow\quad
\frac{d}{dt}|\mathbf S_i|^2=0,
\tag{6.2}
$$

$$
\dot E=\sum_i\frac{\partial E}{\partial\mathbf S_i}\cdot\dot{\mathbf S}_i
=-\sum_i\mathbf g_i\cdot\dot{\mathbf S}_i
=-\sum_i\frac{\gamma\alpha}{|\mathbf S_i|}\big|\mathbf S_i\times\mathbf g_i\big|^2\le0 .
\tag{6.3}
$$

（因 $\mathbf g\_i=-\partial E/\partial\mathbf S\_i$，故 $\dot E=-\sum\mathbf g\_i\cdot\dot{\mathbf S}\_i$；
阻尼项的 $1/|\mathbf S_i|$ 正是 (6.1) 的 LL 系数，不能像全局常数那样提出求和号。）

进动项不做功，零温阻尼必然降低能量。这是 `spin_work.out` 里阻尼功符号的
来源，也是判断"符号写反"的第一道检查。

## 6.2 涨落耗散关系

要求平衡分布 $\propto\exp(-\beta E)$，FDT 给出

$$
\langle b_{i,a}(t)b_{j,b}(t')\rangle
=\frac{2\alpha k_BT}{\gamma\ |\mathbf S_i|\ (1+\alpha^2)}
\delta_{ij}\delta^{ab}\ \delta(t-t').
\tag{6.4}
$$

离散后（每原子、每笛卡尔分量各抽一次，故 $\mathrm{Var}(b_k)=\sigma^2$ 与 $k$ 无关）

$$
\sigma^2=\frac{2\alpha k_BT}{\gamma\ \Delta t\ (1+\alpha^2)\ |\mathbf S_i|},
\qquad
\mathbf b_i=\sigma\ \boldsymbol\xi_i,\quad \xi\sim\mathcal N(0,1).
\tag{6.5}
$$

分母的两个因子都不能漏：

* $1/(1+\alpha^2)$ 来自 Gilbert→LL 转换；漏掉它会让有效温度偏高 $1+\alpha^2$ 倍
  （ $\alpha=0.3$ 时 9%， $\alpha=1$ 时 2 倍）。
* $1/|\mathbf S_i|$ 是**逐原子**归一化，不是全局常数：`gpu_fill_thermal_field` 在 kernel
  内用每个原子的实时 $|\mathbf S_i|$ 现算（`var = prefactor / |S_i|`），所以幅度随原子而异。
  代价是 $|\mathbf S_i|\to0$ 时幅度发散（R11 已知边界）；好处是不需要把 $|\mathbf S|$ 当成常量
  提出来，非等价磁矩体系才是对的。

$k_B=8.617333262\times10^{-5}$ eV/K（`SPIN_KB_EV_PER_K`）， $T$ 用 K， $\Delta t$ 用 fs。
Stage04 已冻结，r18/r19 未改。

**注意**： $\sigma\propto\Delta t^{-1/2}$，所以改时间步必须同时改噪声幅度。
`spin_llg` 在内部用同一 $\Delta t$ 计算，用户无需手动改。

## 6.3 三种积分格式

### 6.3.1 显式几何法（`scheme=geometric`）

先算 $\Delta\mathbf S$，再投影：

$$
\mathbf S^{n+1}=\frac{\mathbf S^{n}+\Delta\mathbf S}{\big|\mathbf S^{n}+\Delta\mathbf S\big|}\ |\mathbf S^{n}| .
\tag{6.6}
$$

便宜、显式，但步长受进动频率限制，且投影引入 $\mathcal O(\Delta t^2)$ 的
能量误差（不守恒）。

### 6.3.2 随机隐式中点（`scheme=midpoint`）

$$
\mathbf S^{n+1}=\mathbf S^{n}+\Delta t\ \mathbf f \Big(\frac{\mathbf S^{n}+\mathbf S^{n+1}}{2}\Big)
+\sqrt{\Delta t}\ \boldsymbol\sigma\circ\boldsymbol\xi^n .
\tag{6.7}
$$

同一步的 $\boldsymbol\xi^n$ 在迭代中**固定**（Stratonovich 的正确做法），
确定性场每轮在中点重算。迭代上限 50，容差 $10^{-6}$，不收敛计数写进轨迹头。

实测（S08A-R1，二聚体，1.6 M 步）：**未收敛计数 0**，最大残差
$9.998\times10^{-7}$。

### 6.3.3 保守自旋–晶格（`scheme=conservative`，见 07 篇）

不是独立的单自旋积分器，而是把 LLG 与晶格 velocity-Verlet 按 S–V–S
对称分裂耦合。`spin_llg ... T=0` 时随机项为零，格式对时间可逆，
能量误差二阶。

## 6.4 Itô 还是 Stratonovich

固定模长的旋转噪声在 **Stratonovich** 解释下才保持球面均匀测度。
按 Itô 离散需要 $\propto\alpha^2$ 的漂移修正。本实现用隐式中点，
天然对应 Stratonovich，不需要修正项——这也是 midpoint 的静态磁化分布
能对上解析值的原因（§6.6）。

## 6.5 模长锚点投影

每步把 $|\mathbf S|$ 拉回初值（`project_onto_norm_anchor`），并记录投影前的
原始误差 `pre_projection_norm_error`，用于区分"积分器漂移"与"float32 游走"。

实测（Fe 1024 原子，10 ps）：投影前模长漂移 $\le9.8\times10^{-5}\ \mu_B$
（相对 $2.2\ \mu_B$ 为 $4.5\times10^{-5}$），与 float32 的 $\sqrt{N}$ 游走一致。
投影把漂移压到 0，但**不改变**能量漂移量级——说明能量误差主要不来自模长漂移。

## 6.6 二聚体验证

两自旋、交换 $J$、易轴 $K$。S08A-R1 用 8 个种子、两种 scheme 各 1.6 M 步：

| 量 | geometric | midpoint | 精确求积 | 蒙特卡洛 |
|---|---|---|---|---|
| $\langle\mathbf S\_1\cdot\mathbf S\_2\rangle$ | 0.2583 | 0.2709 | 0.2625 | 0.2615 |

midpoint 偏差 +3.2%（在统计误差内），geometric 偏差 −1.6%（ $\mathcal O(\Delta t)$
投影误差）。两者都好于"Euler + 事后归一化"一个量级。

## 6.7 一个必须知道的坑：`seed` 在确定性 NVE 里是空操作

`spin_llg alpha=0 T=0` 时整条路径**没有任何随机数**：同一 `run.in` 连跑 5 次，
`thermo.out` 的 sha256 完全相同（PV1 实测 `334d9c31…`）。因此：

* 想检验"种子收敛"，必须换**初态**（重新抽样 Maxwell 速度），
  改 `seed` 关键字无效；
* 这同时是一份可复现性证据：确定性分支逐位可复现。

## 6.8 代码入口

* `src/spin/spin.cu::Spin::integrate`（分派三种 scheme）
* `src/spin/spin.cu::propagate_stochastic_midpoint`
* `src/spin/spin.cu::project_onto_norm_anchor`
* `src/spin/spin.cuh`：`sllg_scheme`、`spin_gamma`、`spin_alpha`、`spin_temperature`
* `src/measure/dump_restart.cu`：写 `spin_restart.bin`（含 curandState）

## 6.9 限制与未验证项

| 项 | 状态 |
|---|---|
| 固定模长 SLLG（ $\alpha>0,T>0$）温度标定 | PASS（±20%，smoke T3） |
| 二聚体静态关联 vs 精确求积 | PASS（midpoint 3%） |
| 耦合体系里"自旋温度 = 晶格温度" | **NOT_VERIFIED**：没有独立测温手段 |
| 自旋波线宽 | NOT_IMPLEMENTED（13 篇） |
| 外磁场 $-\sum\mathbf S_i\cdot\mathbf B$ | **已实现**：`add_bfield <Bx> <By> <Bz> [<omega_in_THz>]`（`src/main_gpumd/run.cu`），单位 $\mathrm{eV}/\mu_B$，必须先设 `spin_llg`； $\omega>0$ 为 THz 交变场（FMR / 自旋泵浦）。外场所做的功在功分解中单列一项 $W_{\rm ext}$ |
