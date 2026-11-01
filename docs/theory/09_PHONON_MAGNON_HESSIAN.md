# 09 — 声子、磁子与混合 Hessian

## 9.1 三个矩阵

$$
\Phi_{ij}^{ab}=\frac{\partial^2E}{\partial R_i^a\partial R_j^b},\qquad
K_{ij}^{ab}=\frac{\partial^2E}{\partial S_i^a\partial S_j^b},\qquad
C_{ij}^{ab}=\frac{\partial^2E}{\partial R_i^a\partial S_j^b}.
$$

由 $F=-E_R$、$g=-E_S$，有限差分形式为

$$
\Phi_{ij}^{ab}=-\frac{\partial F_j^b}{\partial R_i^a},\quad
K_{ij}^{ab}=-\frac{\partial g_j^b}{\partial S_i^a},\quad
C_{ij}^{ab}=-\frac{\partial g_j^b}{\partial R_i^a}=-\frac{\partial F_i^a}{\partial S_j^b}.
$$

两种算 $C$ 的方式必须一致（Maxwell），这是抓符号错误的最便宜判据。

## 9.2 求和规则与零模

**声学求和规则**（平移不变）：$\sum_j\Phi_{ij}^{ab}=0$ 对每个 $i,a$。
实测（16 原子，48×48）：$5.6\times10^{-14}$ eV/Å²——机器精度。

**自旋 Goldstone**：若 $E$ 在整体自旋旋转下不变（无 SOC，即 mask17），
则旋转方向 $\delta\mathbf S_j=\mathbf n\times\mathbf S_j$ 必须是切空间 Hessian 的零模。
注意两点陷阱（本项目踩过）：

1. $\mathbf K\cdot(\mathbf n\times\mathbf S)\neq0$ 不是“破坏对称性”的证据：
   对称方向上梯度随态一起转，$\mathbf K\mathbf m=\mathbf n\times\mathbf g$ 本来就不为零；
2. 直线二阶导 $\mathbf m^{\mathsf T}\mathbf K\mathbf m$ 会混进 Smag 的径向刚度
   （直线运动在二阶上改变 $|\mathbf S|$）。

正确做法：取每个自旋的切平面基 $\mathbf e_{i,1},\mathbf e_{i,2}\perp\mathbf S_i$，
构造 $2N\times2N$ 矩阵 $\tilde K_{ia,jb}=\mathbf e_{i,a}^{\mathsf T}\mathbf K_{ij}\mathbf e_{j,b}$，
再求其特征值；整体旋转应对应 3 个零模。本项目**未完成**这一步，故 Goldstone 判据记 `NOT_VERIFIED`。

## 9.3 动力学矩阵

**声子**：$D_{ij}^{ab}(\mathbf q)=\frac{1}{\sqrt{m_im_j}}\sum_{\mathbf R}e^{i\mathbf q\cdot\mathbf R}\Phi^{ab}_{i0,j\mathbf R}$，
本征值 $\omega^2$，本征矢是极化。质量加权保证 $\mathbf q=0$ 的 3 个声学模为零（若 ASR 成立）。

**磁子**：无阻尼 LLG 线性化。以 $\mathbf S_j=s_j\hat{\mathbf n}_j$ 为基，
$\delta\mathbf S_j$ 只在切平面内，运动方程 $\dot{\delta\mathbf S}_j=-\gamma\,\mathbf S_j\times\delta\mathbf H_j$，
$\delta\mathbf H_j=\sum_k K_{jk}\delta\mathbf S_k$。写成 $\dot{\mathbf u}=\mathcal A\mathbf u$，
本征值 $\pm i\omega$（无阻尼）或复频率（有阻尼）。

**磁声混合**：联立

$$
\begin{pmatrix}\Phi & C\\ C^{\mathsf T} & K\end{pmatrix}
$$

的广义本征问题（质量加权 + 切空间投影），本征矢给出混合模；
反交叉处的分裂 $\Delta\omega\propto|C|$ 才是可观测的耦合强度，
$\lVert C\rVert$ 单独不是物理量。

## 9.4 代码入口

* 有限差分驱动：`tools/one_shot/phase7_hessian.py`（本轮新增，输出 `phi.npy`/`K.npy`/`C.npy`）
* 单点求值：`tools/spin_validation/spin_singlepoint_stage05b.cu`
* 上游声子：`phonon/hessian.cu`（只做晶格，不含自旋）

## 9.5 实测（本轮）

| 量 | mask17（16 原子） | mask31（4 原子） |
|---|---|---|
| ASR $\max|\sum_j\Phi_{ij}|$ | $5.6\times10^{-14}$ eV/Å² | $2.4\times10^{-4}$ |
| $\max|\Phi|$ | 10.6 eV/Å² | 98.0 |
| $\max|\Phi-\Phi^{\mathsf T}|$ | $1.4\times10^{-3}$ | $7.8\times10^{-3}$ |
| Maxwell 相对差 | $3.1\times10^{-3}$ | $1.0\times10^{-3}$ |
| $\max|K-K^{\mathsf T}|$ | $1.7\times10^{-4}$ | $1.8\times10^{-4}$ |

## 9.6 限制

* 只做了 $\mathbf q=0$ 的 Hessian，没有超胞、没有色散、没有 DOS、没有寿命。
* 有限差分步长 $h=10^{-3}$（Å 与 μB），受 float32 输出精度限制（见 14）。
* Goldstone 检验需要切空间形式，未完成 → `NOT_VERIFIED`。

## 9.6 能力状态（r18）

| 能力 | 状态 | 说明 |
|---|---|---|
| 静态 $\Phi$（声子 Hessian，中心差分） | **可用** | 16 原子 48×48 矩阵，ASR 残差 $5.6\times10^{-14}$ |
| 静态 $K$（切空间投影后的自旋 Hessian） | **PARTIAL** | 矩阵能算，但 Goldstone 零模判据未完成（§9.2） |
| 混合 $C$ | **可用（验证过）** | Frobenius 相对差 $1.3\times10^{-3}$（05 篇 §5.6） |
| 声子谱 $D(\mathbf q)$ + 本征值 | **NOT_IMPLEMENTED** | 需要超胞力常数与 Fourier 求和，代码未写 |
| 磁子谱 | **NOT_IMPLEMENTED** | 需要切空间线性化与本征求解，代码未写 |
| 磁声混合谱（反交叉） | **NOT_IMPLEMENTED** | 依赖上两项 |
| Goldstone 零模计数 | **NOT_VERIFIED** | 正确做法见 §9.2，未完成 |

**为什么不做**：磁子谱要求先把 $K$ 投影到切空间再对角化 $6N\times6N$
复矩阵；在 432 原子上是 $2592^2$，需要 LAPACK 级实现与内存规划。
这属于"新功能"而不是"修缺陷"，不在 r18 的范围。前两项矩阵已经在
`tools/numerical_accuracy/nep_double_ref.cpp --mixed-matrix` 里可导出，
后续做谱分析的人可以直接用。

## 9.7 一条必须写下来的限制

`Sia` 在立方环境中恒为零（02 篇 §2.4 实测 $3.3\times10^{-8}$），
而 `Smag` 只有在位项。因此 **bcc Fe 的立方各向异性 $K_1$ 在本模型中不可表示**：
把固定晶格上的全部自旋整体旋转任意角度，能量严格不变（数值上到 $10^{-8}$ eV）。
后果：

* 不能用本模型计算磁晶各向异性能、矫顽力、易轴取向；
* 磁子谱在 $\mathbf q=0$ 处不会有各向异性 gap（真实的 bcc Fe 有 $\sim0.4$ meV）；
* "基态磁矩方向"在模型里是简并的，任何方向都同样合理。

这一条不是数值问题，是模型的结构限制，写进 `KNOWN_LIMITATIONS.md` L-4。
