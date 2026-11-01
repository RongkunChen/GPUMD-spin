# 02 — 五种自旋描述符

记 $\mathbf r_{ij}=\mathbf R_j-\mathbf R_i$，$r_{ij}=|\mathbf r_{ij}|$，
$\hat{\mathbf r}_{ij}=\mathbf r_{ij}/r_{ij}$，$f_c$ 为光滑截断，
$g_n(r)$ 为第 $n$ 个径向基。

掩码位：`Sex=1, San=2, Sdm=4, slot4=8, Smag=16`。
slot4 在 `layout 3` 下是 `Sia`，在 `layout 1/2` 下是 `Senv`。
`spin_blocks` 一行给出启用的块，例如 `spin_blocks Sex Smag`（mask17）、
`spin_blocks Sex San Sdm Sia Smag`（mask31）。

## 2.1 Sex（自旋交换，rank-0，T-偶，P-偶）

$$
q^{\rm Sex}_{i,n}=\sum_{j\in\mathcal N(i)} g_n(r_{ij})\,f_c(r_{ij})\;\mathbf S_i\!\cdot\!\mathbf S_j .
\tag{2.1}
$$

共线时 $\mathbf S_i\cdot\mathbf S_j=\pm|\mathbf S_i||\mathbf S_j|$，退化为标量交换；
非共线时等价于 Heisenberg 交换。这是 bcc Fe 铁磁关联的主要载体，也是
Tc 与自旋波刚度的来源。

$\partial q/\partial\mathbf S_i=\sum_j g_nf_c\,\mathbf S_j$，形式极简，
是唯一一个梯度不需要投影算子的块。

## 2.2 San（自旋—键各向异性，rank-0，T-偶，P-偶）

$$
q^{\rm San}_{i,n}=\sum_j g_n f_c\;(\mathbf S_i\!\cdot\!\hat{\mathbf r}_{ij})(\mathbf S_j\!\cdot\!\hat{\mathbf r}_{ij}).
\tag{2.2}
$$

它把自旋钉在键方向上，**破坏自旋空间整体旋转不变性**（03 篇 §3.3），
物理上对应各向异性交换与磁弹性耦合。若只用 `Sex`，固定晶格下把全部自旋
整体旋转能量不变；加入 `San` 后该对称性被显式破坏，代价是"易轴"由键方向
（而不是晶体学轴）决定，因此不能用它拟合立方各向异性常数 $K_1$。

## 2.3 Sdm（键向 DMI 型，赝标量，T-偶，P-奇）

$$
q^{\rm Sdm}_{i,n}=\sum_j g_n f_c\;\hat{\mathbf r}_{ij}\!\cdot\!(\mathbf S_i\times\mathbf S_j).
\tag{2.3}
$$

$\mathbf S$ 是轴矢量（反演不变），$\hat{\mathbf r}$ 是极矢量（反演变号），
所以三重积在反演下变号：$q^{\rm Sdm}$ 是 P-奇。能量必须是 P-偶，因此：

* 在中心对称晶体（bcc Fe）里，对每个键 $\mathbf r_{ij}$ 都存在 $-\mathbf r_{ij}$，
  两项严格抵消，$q^{\rm Sdm}\equiv0$；
* 在界面/表面/掺杂打破中心对称时它非零，这才是 DMI 的合法来源。

**不要**把 `Sdm` 的非零解释为"数据里有 SOC"：它只说明构型缺中心对称。

## 2.4 Sia（单离子各向异性，rank-2 四极矩）

$$
q^{\rm Sia}_{i,n}=\sum_j g_n f_c\left(\tfrac32(\mathbf S_i\!\cdot\!\hat{\mathbf r}_{ij})^2-\tfrac12|\mathbf S_i|^2\right).
\tag{2.4}
$$

这是 $\mathbf S\mathbf S^{\mathsf T}-\tfrac13|\mathbf S|^2 I$ 的 $l=2$ 分量。
它能表示**单轴**各向异性，但在严格立方环境中所有 $l=2$ 求和为零，
所以 bcc Fe 的 $K_1$（$l=4$ 立方项）**不能**由 Sia 表示。这是能力限制，
不是数值问题——见 §2.10 的实测。

## 2.5 Smag（模长通道，T-偶，P-偶）

$$
q^{\rm Smag}_{i,n}=\big(\text{scale}\cdot|\mathbf S_i|\big)^{\,n+1},
\qquad \text{scale}=\texttt{spin\_mag\_scale}.
\tag{2.5}
$$

在位项（不求和邻居），负责让能量依赖磁矩大小。**这是本模型唯一能把
"基态磁矩"作为可观测量输出的通道**，因此 P2 检验（$E$ 对 $|S|$ 的极小）
本质上是检验 Smag 子网络。

$\partial q/\partial\mathbf S=(n+1)u^{n}\,\text{scale}\,\mathbf S/|\mathbf S|$
在原点不可微（方向依赖）。实现约定 $|\mathbf S|<10^{-8}$ 取 0，并把该约定
写进声明表（`spin_block_analytic_at_zero_moment`）。

**`spin_mag_scale` 的取值规则（r18 起）**：`nep.in` 不写这一行时，训练器
从训练集读出 $1/\max|\mathbf S|$ 作为 scale（Fe 数据为 0.234148785，
因为 $\max|\mathbf S|=4.27\,\mu_B$）。手写固定值会把 $u$ 压在 $[0,1]$ 的
一小段里，网络的有效分辨率下降——r17 用固定 0.25 就是一个隐性缺陷
（见 15 篇 F-24）。

## 2.6 Senv（废弃）

旧 layout 的 rank-1、T-奇通道。无外磁场时它没有物理依据，代码保留读取
但标记 deprecated；`spin_blocks` 里写 `Senv` 会打印警告。

## 2.7 径向/角向基与镜像枚举

$$
g_n(r)=\sum_k c_{n,k}\,T_k\!\Big(2\frac{r}{r_c}-1\Big)f_c(r),
\tag{2.6}
$$

角向部分用三阶/四阶球谐组合：

$$
s_{abc}=\sum_j \hat r^a\hat r^b\hat r^c\,g_n f_c,
\qquad
q^{\rm ang}=\sum_{abc}\sum_{a'b'c'}C_{abc,a'b'c'}\,s_{abc}s_{a'b'c'},
\tag{2.7}
$$

$C_{abc}$ 是预算好的球谐系数（`C3B`、`C4B`）。

**镜像枚举的陷阱**：邻居表按 `num_cell` 枚举周期镜像。`num_cell` 必须由
**径向截断**（$r_c^{\rm rad}$）决定，而不是角向截断；若按角向截断算，
`num_cell` 会偏小，$r_c^{\rm rad}$ 边缘的键会漏掉镜像。PV1 阶段参考求值器
最初就是这么错的，导致与生产代码差 1.3%；改成"按 $r_c^{\rm rad}$ 枚举
$-n\ldots n$ 全部镜像、不做最小镜像折叠"后，相对差降到 $1.2\times10^{-8}$。

## 2.8 掩码、layout 与原子数

| mask | 位组合 | 块 | rank | T | P | 典型用途 |
|---|---|---|---|---|---|---|
| 17 | 1+16 | Sex, Smag | 0,0 | +,+ | +,+ | Fe 铁磁（r18 A 模型） |
| 31 | 1+2+4+8+16 | Sex, San, Sdm, Sia, Smag | 0,0,0,2,0 | 全 + | 含一奇 | 含 SOC 的参数化（r18 B 模型） |
| 19 | 1+2+16 | Sex, San, Smag | — | — | — | 单轴各向异性 |

`layout` 决定 slot4 的解释：`layout 3` → `Sia`（rank-2），`layout 1/2` → `Senv`。
`spin_blocks` 的书写顺序不影响结果，但**影响 `nep.txt` 里的参数块布局**，
因此预测时必须用训练时同一个 `nep.in`（只用 `nep.txt` 而不给 `nep.in`，
`nep.exe` 会按默认架构重建描述符，静默算出错误的力——PV1 阶段实测
F_MAE 从 0.09 变成 0.81 eV/Å）。

## 2.9 代码入口

* 声明表：`src/utilities/spin_descriptor.cuh`（宇称/时间反演/rank/退化条件/是否需 SOC）
* 内核：`src/utilities/spin_descriptor_kernels.cuh`（small-box 与 large-box 两个变体）
* 累加精度：`nep_acc_t`（`double`，见 14 篇）；循环内 double 累加，出循环窄化一次
* 打印与回读：`src/utilities/spin_descriptor_report.cuh`
* 训练器侧：`src/main_nep/nep.cu`、`nep_specialized.cu`

## 2.10 验证

| 命题 | 实测 | 状态 |
|---|---|---|
| Sdm 为 P-奇：对称排布下为零 | $\lvert q^{\rm Sdm}\rvert\sim3\times10^{-8}$ | PASS |
| 非对称排布下非零 | $1.3\times10^{-2}$ | PASS |
| Sia 在立方环境恒零 | $3.3\times10^{-8}$ | PASS |
| Sia 在单轴环境非零 | $0.357$ | PASS |
| Smag 能量—导数一致 | 相对差 $4.7\times10^{-4}$ | PASS |
| 描述符与 float64 参考一致 | 相对差 $1.2\times10^{-8}$ | PASS |

## 2.11 限制

* `Sdm` 只覆盖键向 DMI；界面 DMI 需要更复杂的几何（未实现）。
* `Sia` 无 $l=4$ 立方项，bcc Fe 的 $K_1$ 不可表示。
* `Sdm` 的 P-奇不保证总能 P-even：只有构型本身中心对称时两项才抵消。
* mask31 是"类 SOC 的参数化"，不是"数据里证明了 SOC"。
* 所有描述符都是**局域**的（$r<6$ Å），长程偶极相互作用不在模型内；
  这意味着形状各向异性、退磁场无法描述。
