# 10 — Berry 相位、拓扑与手性

> 状态：**NOT_RUN_MODEL_GATE / NOT_REPRESENTABLE（按机制）**。本文件规定“什么条件下才能报”，
> 以及离散算法与负例；不提供未经检验的材料结论。

## 1. 适用范围（先判断能不能算）

| 机制 | 需要 | 本项目的 mask17 | 本项目 mask31 |
|---|---|---|---|
| 磁子 Berry 曲率 / 热霍尔 | SOC 或 DMI 打破 \(\omega(\mathbf q)=\omega(-\mathbf q)\) | 对称性强制为零 → **NOT_REPRESENTABLE** | 表示上“可能”，但数据未证明 SOC → **NOT_VALIDATED** |
| 手性声子（圆偏振角动量） | 破反演 + 破镜像的晶体 + 简并/近简并支 | 由结构决定，与自旋无关 | 同左 |
| 拓扑声子（Weyl/Dirac 点、边界态） | 非平庸带交叉 + 全局隙 | 需实算，未做 | 未做 |
| Chern 数 | 二维周期 + 规范不变离散 | 未做 | 未做 |

**判据顺序**：先看对称性是否强制为零；再看模型是否真能表达该机制；最后才看数值。

## 2. 离散 Berry 相位与 Wilson loop

对占据带子空间 \(\{|u_n(\mathbf k)\rangle\}\)，沿闭合路径分 \(N\) 段：

$$
\theta=-\operatorname{Im}\ln\prod_{m=0}^{N-1}\big\langle u(\mathbf k_m)\big|u(\mathbf k_{m+1})\big\rangle,
\qquad \mathbf k_N=\mathbf k_0 .
$$

- **规范不变性**：每个 \(|u\rangle\) 乘相位不改变 \(\theta\)；数值上要用重叠矩阵的**行列式**（多带时）
  \(\det M^{(m)}=\det\langle u_n(\mathbf k_m)|u_{m'}(\mathbf k_{m+1})\rangle\)，而不是逐带相位差之和。
- **简并子空间**：在子空间内做 SVD/正交化后再取 det，避免随机相位导致的 \(2\pi\) 跳变。
- 曲率：\(\Omega(\mathbf k)=\partial_{k_x}A_y-\partial_{k_y}A_x\)，\(A=i\langle u|\nabla_{\mathbf k}u\rangle\)；
  离散用 Wilson loop 的小回路版本（plaquette）更稳。

## 3. 度量的选择

| 系统 | 本征问题 | 度量 | 常见错误 |
|---|---|---|---|
| 声子 | \(D(\mathbf q)\mathbf e=\omega^2\mathbf e\) | 质量加权欧氏度量 | 忘记 \(1/\sqrt{m_im_j}\) → 简并错位 |
| 磁子（玻色） | \(i\dot\eta=\mathcal H\eta\) | paraunitary：\(\sigma_3\) 或 \(\mathcal J=\mathrm{diag}(1,-1)\) | 用厄米度量 → 本征值非物理 |
| 磁子（BdG） | \(\mathcal H_{\rm BdG}\Psi=E\sigma_3\Psi\) | \(\sigma_3\) 不定度量 | 直接对 \(\mathcal H\) 取本征值 |

## 4. 手性 vs 拓扑（概念区分）

- **圆偏振 / 角动量**：模式 \(\mathbf e\) 的 \(\langle \mathbf e|\mathbf L|\mathbf e\rangle\)，是**局域可观测量**；
- **Berry 曲率**：\(\mathbf k\) 空间的几何量；
- **Chern 数**：曲率在闭合流形上的积分（整数）；
- **拓扑边界态**：开边界谱中的隙内态。

四者互不等价：算出非零角动量**不**等于拓扑非平庸；算出非零 Berry 曲率也不等于有边界态。

## 5. 负例（必须先过）

| 负例 | 期望结果 |
|---|---|
| 一维二原子链（有反演对称） | Berry 相位 \(0\) 或 \(\pi\)，与规范无关 |
| 无 SOC 的立方磁体 | 磁子 Berry 曲率恒零（时间反演 + 反演） |
| 简并两支的随机相位注入 | Wilson loop 结果不变（检验子空间处理） |

## 6. 本版本状态

| 项 | 状态 |
|---|---|
| 离散 Wilson loop 实现 | 未实现（仅规范） |
| 声子/磁子度量核算 | 公式已给；代码未接 |
| Fe 的拓扑/手性结论 | **NOT_REPRESENTABLE / NOT_VALIDATED**（无 SOC 证据） |
| 任何“材料有拓扑态”的表述 | **禁止** |
