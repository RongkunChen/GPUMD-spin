# 10 — 磁声耦合矩阵元

## 10.1 定义

$$
C_{ij}^{ab}=\frac{\partial^2E}{\partial R_i^a\partial S_j^b}
=-\frac{\partial g_j^b}{\partial R_i^a}
=-\frac{\partial F_i^a}{\partial S_j^b}.
$$

单位：eV/(Å·μB)。它描述“挪动原子如何改变磁梯度”以及“转一个自旋如何改变力”。

## 10.2 矩阵元的物理含义

* 对角块 $C_{ii}$：单离子磁弹性（自旋—局域畸变）；
* 近邻块 $C_{ij}$（ $j$ 是近邻）：交换伸缩（spin–phonon 交换耦合），
  在金属里主导磁振子—声子散射；
* 长程部分：本模型截断在 $r_c$，所以超过 6 Å 的耦合恒为零。

## 10.3 与对称性的关系

* 若 $E$ 在整体自旋旋转下不变（mask17），则 $\sum\_j C\_{ij}\mathbf S\_j$ 必须为零：
  **均匀旋转不能产生力**。这是 $C$ 的一条独立求和规则，可用作额外检查。
* mask31 含 San/Sdm/Sia， $C$ 不必满足该规则——这也是两种参数化的可测区别。

## 10.4 模态耦合与反交叉

线性化后，磁声混合模由 $(\Phi,C;C^{\mathsf T},K)$ 的广义本征问题给出。
当声子支与磁子支在某 $\mathbf q$ 交叉时，若 $C\neq0$，本征值会避开交叉：

$$
\omega_\pm^2=\frac{\omega_{\rm ph}^2+\omega_{\rm mag}^2}{2}
\pm\sqrt{\Big(\frac{\omega_{\rm ph}^2-\omega_{\rm mag}^2}{2}\Big)^2+|\tilde C|^2},
$$

$\tilde C$ 是 $C$ 在两个模矢上的投影。**只有 $\tilde C$ 是可观测耦合强度**；
$\lVert C\rVert_F$ 依赖基的选择，不是物理量。

## 10.5 代码入口与状态

* $\Phi$、 $K$、 $C$ 的数值：`tools/one_shot/phase7_hessian.py`
* 目前**没有**实现混合模的广义本征求解器 → 反交叉与模态投影为 `NOT_IMPLEMENTED`。
* 已能给出 $C$ 的矩阵与 Maxwell 一致性检查（ $1.0\times10^{-3}$（mask31）/ $3.1\times10^{-3}$（mask17）相对）。

## 10.6 限制

* 没有 SOC 证据， $C$ 的各向异性部分不可信。
* 只有 $\mathbf q=0$ 的 $C$；真正的耦合强度需要 $\mathbf q$ 分辨的 $C(\mathbf q)$。
