# 01 — Hamiltonian、局域能量分解与基本量纲

本篇定义整个代码库唯一的能量函数。后续所有导数（力、磁梯度、virial、Hessian）
都是它的偏导，不存在第二套"物理"定义。

## 1.1 完整 Hamiltonian

体系由 $N$ 个原子组成，原子 $i$ 携带位置 $\mathbf R_i\in\mathbb R^3$ 与
局域磁矩 $\mathbf S_i\in\mathbb R^3$（$|\mathbf S_i|=\mu_i$，单位 $\mu_B$）：

$$
\boxed{\;
E\big(\{\mathbf R\},\{\mathbf S\}\big)
=\sum_i E_i^{(2)}\big(\{\mathbf R\},\{\mathbf S\}\big)
+\sum_i E_i^{(3)}\big(\{\mathbf R\},\{\mathbf S\}\big)
+\sum_i E_i^{\rm mag}\big(\{\mathbf S\}\big)
\;}
\tag{1.1}
$$

* $E^{(2)}$：两体径向项。主要依赖 $\mathbf R$，但每条键的权重被环境自旋投影
  $a_{ij}=\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j$ 调制（§1.3）。没有这一层调制，
  模型无法描述磁致伸缩与自旋–晶格耦合。
* $E^{(3)}$：三体角向项。依赖 $\cos\theta_{jik}$，同样被自旋环境调制。
* $E^{\rm mag}$：局域磁项。只依赖自旋方向，是固定晶格、共线自旋下唯一
  非零的"纯自旋"能量。

三者的截断：

| 参数 | 符号 | Fe 取值 | 作用对象 |
|---|---|---|---|
| 径向截断 | $r_c^{\rm rad}$ | 6.0 Å | $E^{(2)}$、$S^{\rm ex}$、$S^{\rm an}$、$S^{\rm dm}$、$S^{\rm ia}$ |
| 角向截断 | $r_c^{\rm ang}$ | 3.0 Å | $E^{(3)}$ 的 $j$–$i$–$k$ 三角形 |
| 磁截断 | $r_c^{\rm mag}$ | 6.0 Å（默认＝径向） | $S^{\rm mag}$ |

`nep.in` 里 `cutoff 6.0 3.0` 一次给出前两个。**注意**：截断半径决定
`num_cell`，而 `num_cell` 决定邻居表的镜像枚举范围；PV1 阶段曾因把
`num_cell` 按错截断算小，导致参考求值器与生产代码差 1.3%（见 02 篇 §2.7）。

## 1.2 描述符到能量的两级映射

每个原子先被映射为描述符向量，再由前馈网络映射为能量：

$$
\mathbf q_i=\Big[\,q^{S}_{i},\ q^{2}_{i},\ q^{3}_{i},\ q^{\rm mag}_{i}\Big]
\in\mathbb R^{n_{\rm desc}},
\qquad
E_i=\mathcal N\big(\mathbf q_i\big).
\tag{1.2}
$$

$\mathcal N$ 是单隐层网络（`neuron 30`），激活 $\tanh^2$（NEP4 约定），
输出层线性。

**关键设计选择**：所有自旋依赖都进入**描述符**，而不是作为一个显式的
$\mathbf S_i\cdot\mathbf S_j$ 项加进能量。交换作用、DMI、各向异性都是网络的
隐式拟合结果。因此不能把拟合参数直接解释为 $J_{ij}$、$D_{ij}$，只能通过
后处理（09、10 篇）反演；反演出来的"有效交换"是温度与磁场相关的量，
不是微观常数。

## 1.3 环境自旋如何进入径向/角向项

对每一对 $(i,j)$：

$$
a_{ij}=\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j\in[-1,1],
\qquad
\mathbf c_{ij}=\hat{\mathbf S}_i\times\hat{\mathbf S}_j\in\mathbb R^3 .
\tag{1.3}
$$

径向项展开为

$$
E_i^{(2)}=\sum_{j\in n(i)}\sum_{n=1}^{n_{\max}} c_n^{2}\big(a_{ij}\big)\;
f_n\big(r_{ij}\big),
\qquad
f_n(r)=T_n\!\Big(2\frac{r-r_c}{r_c}+1\Big)\Theta(r_c-r),
\tag{1.4}
$$

$T_n$ 是 Chebyshev 多项式，$c_n^{(2)}(a)$ 是 $a$ 的多项式（阶数由
`basis_size` 控制）。式 (1.4) 的物理含义：

* $a_{ij}=+1$（铁磁对齐）与 $a_{ij}=-1$（反铁磁对齐）可以有**不同的键刚度**，
  这是磁致伸缩的微观来源；
* 因为没有 $\mathbf c_{ij}$ 的线性项，DM 相互作用不能由 $E^{(2)}$ 直接给出，
  必须由 $S^{\rm dm}$（一个反对称标量，02 篇 §2.4）承担。

角向项同理：

$$
E_i^{(3)}=\sum_{(j,k)\in n(i)}\sum_{n} c_n^{3}\big(a_{ij},a_{ik},\cos\theta_{jik}\big)\,
f_n(r_{ij})f_n(r_{ik}).
\tag{1.5}
$$

## 1.4 磁项

$$
E_i^{\rm mag}=\sum_{j\in n_{\rm mag}(i)}\sum_{n} c_n^{\rm mag}
\big(\hat{\mathbf S}_i\cdot\hat{\mathbf S}_j\big)\,
g_n(r_{ij}),
\tag{1.6}
$$

$g_n$ 是另一组径向基。注意 $E^{\rm mag}$ **不含** $\mathbf S_i$ 与晶轴的点积，
因此单离子各向异性（例如 bcc Fe 的 $\langle100\rangle$ 易轴）只能通过邻域
自旋的联合取向间接出现。这是本模型的结构限制，见
`03_SYMMETRY_PARITY_CHIRALITY.md` §3.5 与 `KNOWN_LIMITATIONS.md` L-4。
实测后果：把固定晶格上的全部自旋整体旋转，能量严格不变，即**没有**
单离子各向异性能垒。

## 1.5 单位制度（唯一权威表）

内部单位与输入输出单位不同，混用是这类代码最常见的错误来源。

| 量 | 输入/输出 | 内部 | 换算 |
|---|---|---|---|
| 长度 | Å | Å | — |
| 时间（`time_step`） | fs | nat（$1$ nat $=10.1805$ fs） | $\Delta t_{\rm nat}=\Delta t_{\rm fs}/10.1805$ |
| 能量 | eV | eV | — |
| 力 | eV/Å | eV/Å | — |
| 速度（`dump_xyz velocity`） | Å/fs | Å/nat | $\times 10.1805$ |
| 速度（`thermo.out` 隐含） | — | Å/nat | 同上 |
| 磁矩 $|\mathbf S|$ | $\mu_B$ | $\mu_B$ | — |
| 磁梯度 $\mathbf g=-\partial E/\partial\mathbf S$ | eV/$\mu_B$ | eV/$\mu_B$ | — |
| 旋磁比 $\gamma$ | rad·$\mu_B$/(eV·ps) | 同一数值 | 内部 $\times10^{-3}$ 转 rad·$\mu_B$/(eV·fs) |
| 动能 | — | eV | $\frac12 m v^2\times1.0364269\times10^{-4}$（$m$ 用 u，$v$ 用 Å/fs） |
| 热流（`heat_flux.out`） | eV/(Å²·ps)？ | eV/(Å²·nat) | $\times 98.23$ **（归一化存疑，11 篇 §11.7）** |

**动能那一行必须记住**：`thermo.out` 的 KE 已经乘过 $1.0364269\times10^{-4}$。
自己从 `dump_xyz` 的 Å/fs 速度重算时若漏掉它，KE 会大 $9.6\times10^{3}$ 倍；
PV1 阶段就因为漏乘质量与漏乘该因子，两次算出荒谬的温度。

## 1.6 局域能量的可加性

$E_i$ 是局域能量，$E=\sum_i E_i$。两体项在 $i,j$ 之间对称分配，
因此 $E_i$ 只包含每条键的一半。两个后果：

1. `dump_xyz potential` 的每原子能量之和严格等于 `thermo.out` 的总 PE。
   在 `heat_audit` 的 128 原子算例上逐帧验证，偏差 < 1 ulp。
2. 由 $E_i$ 直接构造的"能量矩" $\sum_i\mathbf R_i\dot e_i$ 只在非周期体系里
   等于键矢形式；周期体系多出面项 $\sum \mathbf n\mathbf L\,\dot e_{ij}$。
   这一点在 PV1 被实测否定（11 篇 §11.6）。

## 1.7 与 NEP4 上游的关系

本代码是 NEP4 的自旋扩展：

* 描述符基函数、Chebyshev 展开、网络拓扑、训练器主循环与上游一致；
* 新增 `spin` 开关、五个自旋描述符块（掩码位）、`spin_forces` 标签、
  磁梯度反向传播、virial 的自旋通道、SLLG/SLD 积分器、重启格式；
* `spin 0` 时走的是上游同一条路径，因此纯晶格结果与上游数值可比，
  这也是 `tests/smoke.py::T1` 的判据之一。

## 1.8 本篇验证状态

| 命题 | 状态 | 证据 |
|---|---|---|
| 式 (1.1) 的分解与 `dump_xyz potential` 一致 | **PASS** | `heat_audit` 128 原子，逐帧 < 1 ulp |
| $\mathbf g=-\partial E/\partial\mathbf S$ | **PASS** | float64 参考求值器，最大相对差 2.6e-5（05 篇） |
| $\mathbf F=-\partial E/\partial\mathbf R$ | **PASS** | 同参考求值器，$3\times10^{-6}$ eV/Å |
| 单离子各向异性可被表示 | **FAIL（结构限制）** | 09 篇 §9.6：整体旋转自旋能量不变 |
| 混合导数 $\partial^2E/\partial R\,\partial S$ | **PASS** | Frobenius 相对差 1.3e-3（05 篇 §5.6） |
