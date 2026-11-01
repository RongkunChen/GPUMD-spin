# 11 — 局域能量连续性、热流表达式与规范

本篇记录的是**否定性结论**：最初写进文档的连续性判据在周期性边界下不成立，
四个算例（含无自旋对照）一致失败。这里保留全过程，因为"判据错了"与
"代码错了"必须分开，否则后面所有热导率计算都建在沙地上。

## 11.1 局域能量

$E=\sum_i e_i$，$e_i$ 是局域原子能（`dump_xyz potential`）。连续性方程要求

$$
\frac{\partial e_i}{\partial t}+\sum_{j\neq i}\mathbf j_{ij}=0,
\qquad \mathbf j_{ij}=-\mathbf j_{ji},
\tag{11.1}
$$

即"$i$ 的能量变化只能通过与 $j$ 的成对通量流走"。

## 11.2 从 Hamiltonian 推热流（教科书形式）

对 $e_i(\mathbf R,\mathbf S)$ 求全导数：

$$
\dot e_i=\sum_j\frac{\partial e_i}{\partial\mathbf R_j}\!\cdot\!\dot{\mathbf R}_j
+\sum_j\frac{\partial e_i}{\partial\mathbf S_j}\!\cdot\!\dot{\mathbf S}_j .
\tag{11.2}
$$

利用 $\mathbf F_j=-\partial E/\partial\mathbf R_j$、$\mathbf g_j=-\partial E/\partial\mathbf S_j$，
把成对项对称化（每对只算一次，各分一半），得到

$$
\mathbf J=\sum_i \mathbf R_i\,\dot e_i
=\underbrace{\sum_i \mathbf R_i\,\mathbf F_i\cdot\dot{\mathbf R}_i}_{\mathbf J_{\rm L}}
+\underbrace{\sum_i \mathbf R_i\,\mathbf g_i\cdot\dot{\mathbf S}_i}_{\mathbf J_{\rm S}} .
\tag{11.3}
$$

$\mathbf J_{\rm L}$ 是常规 virial 型热流（`compute_heat` 算的就是它），
$\mathbf J_{\rm S}$ 是自旋通道。交叉项来自 $\partial e_i/\partial\mathbf S_j$ 与
$\partial e_i/\partial\mathbf R_j$ 的混合：它不能靠"$-\mathbf g\cdot\dot{\mathbf S}=0$"
一句话消掉，因为 $\mathbf g_i\cdot\dot{\mathbf S}_i$ 只在稳态/单自旋时为零。

`dump_heat_flux` 输出的三组数（每帧 11 列）对应：

| 列 | 含义 |
|---|---|
| 0 | 时间（ps） |
| 1–3 | $\mathbf J_{\rm tot}$（总） |
| 4–6 | $\mathbf J_{\rm p}$（声子，含 virial 与 $\dot{\mathbf S}$ 无关部分） |
| 7 | 标量（当前未使用） |
| 8–10 | $\mathbf J_{\rm m}$（磁子通道） |

## 11.3 规范非唯一性

$\mathbf j_{ij}$ 只在"成对、反对称"约束下唯一；把能量在 $i,j$ 之间重新分配
（$\mathbf R_i\to\mathbf R_i+\delta$）会改变 $\mathbf J$ 而保持 $\nabla\cdot\mathbf J$。
因此：

* $\kappa$ 在热力学极限下规范不变；
* 分解 $\kappa=\kappa_{\rm LL}+\kappa_{\rm SS}+\kappa_{\rm LS}+\kappa_{\rm SL}$
  **依赖规范**，尤其 $\kappa_{\rm cross}$ 可能为负；
* 报分解时必须同时报规范（本实现用"原子中心 + 成对平分"）。

## 11.4 连续性判据为什么不成立（PV1 的核心发现）

教科书判据是

$$
\frac{d}{dt}\sum_i \mathbf R_i e_i \stackrel{?}{=}\mathbf J .
\tag{11.4}
$$

四个算例：

| 算例 | 体系 | 结果 |
|---|---|---|
| `cont_N128` | 128 原子，NVE，每步 dump，400 步 | relRMS = **7.96**，corr = −0.94 |
| `cont2_N128` | 同上 + 每原子 virial/力 | 用 dump 的 virial 重建 $\mathbf J_p$ 与 dump 值一致（差 0.5/18.9）；仍不闭合 |
| `cont_nospin_N128` | **去掉自旋**的同一实验 | relRMS = **18**，一样不闭合 |
| `cont3_N1024` | 1024 原子（19.4×19.5×30 Å > $2r_c$） | relRMS = **3.81**，corr = 0.62 |

**先排除"我的记账写错了"**：同一批 dump 上

* $\mathrm{rms}\big(\sum_i \dot e_i^{\rm pot}\big)=18.2$ eV/ps，
* $\mathrm{rms}\big(\sum_i \mathbf F_i\!\cdot\!\mathbf v_i\big)=18.2$ eV/ps，
* 两者之和的 rms 只有 **0.08**（0.4%）。

即逐原子势能、力、速度三者互相自洽到 0.4%。

**结论：判据本身不成立，不是热流算子的 bug。** 原因：周期性边界下
$\mathbf r_i$ 只在模 $\mathbf L$ 意义下有定义，而相互作用用键矢
$\mathbf r_{ij}=\mathbf r_i-\mathbf r_j+\mathbf n\mathbf L$。于是

$$
\sum_i \mathbf r_i\dot e_i
=\underbrace{\sum_{i<j}\mathbf r_{ij}\,\dot e_{ij}}_{J_{\rm 键矢}}
+\sum_{i<j}\mathbf n\mathbf L\,\dot e_{ij},
\tag{11.5}
$$

第二项是**面项**，在小胞里不消失也不等于零。128 原子胞的 $L_x=9.70$ Å、
$L_y=9.75$ Å 都小于 $2r_c=12$ Å，几乎每对都带镜像；把胞放大到 1024 原子
（三方向都超过 $2r_c$）后 relRMS 仍从 7.96 只降到 3.81，说明残余项不是
"胞太小"，而是**这个判据在 PBC 下没有定义**。无自旋对照同样失败（18×），
进一步排除自旋通道。

**状态：`NOT_VALIDATED_BY_DESIGN`。** 要补的判据是键矢形式
$\sum_{i<j}\mathbf r_{ij}\dot e_{ij}$，需要 $\dot e_{ij}$ 的成对分解
（NEP 不直接给），或用解析二体模型（LJ）在非周期团簇上做。两条都超出
本阶段范围，写进 `KNOWN_LIMITATIONS.md` L-11。

## 11.5 平移不变性：PASS

同一条 128 原子轨迹，把所有坐标整体平移 (0.317, 0.191, 0.233) Å（再折回盒子）
后重跑，400 步内两条 $\mathbf J(t)$ 序列的相对 RMS 偏差 **1.11e-05**，
第一步偏差 9.4e-08（$\mathbf J$ 本身量级 0.011）。这个量级正是 float32
坐标平移后舍入不同造成的。**PASS。**

## 11.6 绝对归一化：三个数字差 7 个数量级（`NOT_VALIDATED`）

对**同一条**轨迹（1024 原子 Fe，250 K）：

| 来源 | $\kappa_{zz}$ (W/(m·K)) |
|---|---|
| 自研积分器，用 `dump_heat_flux` 的 $\mathbf J$（eV/(Å²·ps)） | **5.94e+05** |
| 发布版 `compute_hac`（同一轨迹，z 列 rtc 平台） | **−6.4e−02** |
| 物理量级（Fe 在 250 K） | ~80 |

三者两两相差 $10^{7}$ 与 $10^{4}$。自查过两遍：

* 估计器本身对 OU 过程给出 0.978（12 篇 §12.4），所以**不是积分器或 SI 换算写错**；
* `compute_hac` 用同一个 `compute_heat` 核，只是乘的常数不同。

数量级线索：把 dump 的 $\mathbf J$ 再除以 $1000/10.1805=98.23$，得到
$\kappa_{zz}=5.94\times10^5/98.23^2=\mathbf{62}$ W/(m·K)——正好是 Fe 的量级。
这与"`dump_heat_flux` 多做了一次 natural→ps 换算"的假设吻合；
而 `compute_hac` 与修正值又差约 $10^3$，且**符号为负**（rtc 在 1 ps 内翻负），
说明它的 5 分量分解（`jx_in/jx_out` 各自相关、丢掉交叉项）在自旋体系上不可用。

**这是假设，不是结论。** 最小可判定实验：用一个 $\kappa$ 已知的解析模型
（LJ 晶体或理想气体极限）跑同样两条路径，看哪条给出正确绝对值。
**状态：`NOT_VALIDATED`——热流的绝对归一化没有得到验证。**

## 11.7 与代码的对应

* `src/measure/compute_heat.cu`：逐原子 $e_i$ 与 $\sum_j \mathbf r_{ij}\otimes\mathbf f_{ij}$。
  **只看函数名不能证明它包含自旋通道**，要读它是否把 `g` 项加进去。
* `src/measure/dump_heat_flux.cu`：写出 $\mathbf J$ 的 11 列。
* `src/measure/hac.cu`：`compute_hac`，用 5 分量分别做自相关；因丢交叉项，
  在自旋体系上不可用。

## 11.8 S09 状态

| 前置条件 | 状态 | 证据 |
|---|---|---|
| 耦合能量守恒（0.2 fs） | FAIL | R2：$1.0201\times10^{-5}>10^{-5}$ |
| 连续性方程 | **NOT_VALIDATED_BY_DESIGN** | §11.4 |
| 热流绝对值 | **NOT_VALIDATED** | §11.6 |
| 相关函数形状 | PASS | 12 篇：$\tau=0.176$ ps，单调衰减 |
| 统计收敛 | **FAIL** | 5 ps 只有 14 个独立样本 |
| 估计器/单位 | PASS | OU 自检 0.978 ± 0.12 |

**在 §11.6 的归一化问题被解析模型钉死之前，任何 Fe 的 $\kappa$、
$\kappa_{ph}$、$\kappa_{mag}$、$\kappa_{cross}$ 都不允许出现在报告里。**
S09 维持 `BLOCKED`。
