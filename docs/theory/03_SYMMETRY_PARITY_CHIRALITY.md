# 03 — 对称性、宇称、旋转协变性与手性

## 3.1 三类变换

| 变换 | 对 $\mathbf R$ | 对 $\mathbf S$ | 对能量的要求 |
|---|---|---|---|
| 平移 | $\mathbf R_i+\mathbf a$ | 不变 | 不变 |
| 空间反演 $P$ | $-\mathbf R_i$ | $+\mathbf S_i$（轴矢量） | 不变（无外场） |
| 时间反演 $T$ | 不变 | $-\mathbf S_i$ | 无耗散时不变 |
| 整体旋转 $R$ | $Q\mathbf R_i$ | $Q\mathbf S_i$ | 不变（无 SOC） |
| 自旋空间旋转 | 不变 | $Q\mathbf S_i$ | **仅在描述符不含键向耦合时**不变 |

## 3.2 为什么“描述符的宇称”≠“能量的宇称”

$P$ 作用后 $q^{\rm Sdm}\to-q^{\rm Sdm}$， $q^{\rm Sex}\to+q^{\rm Sex}$。
ANN 是 $q$ 的非线性函数，所以

$$
E(-q^{\rm Sdm},q^{\rm other})\neq E(q^{\rm Sdm},q^{\rm other})
$$

一般成立。只有当构型满足 $q^{\rm Sdm}\equiv0$（例如中心对称晶体、或训练集在反演下成对出现且权重对称）时能量才可能是 P-even。
**判据是对完整势做反演测试，不是对单个描述符贴标签。**

代码里 `spin_block_parity_inversion()` 给出的标签只描述**该描述符**，不描述 $E$。

## 3.3 旋转协变性的逐块分类

| 块 | 整体旋转 $\mathbf S\_i\to Q\mathbf S\_i$ | 整体旋转 $\mathbf R$ |
|---|---|---|
| Sex | 不变（ $\mathbf S\_i\cdot\mathbf S\_j$） | 不变 |
| San | **变**（依赖 $\hat{\mathbf r}$） | 与自旋一起转则不变 |
| Sdm | **变**（三重积） | 同上 |
| Sia | **变** | 同上 |
| Smag | 不变（只有 $\vert \mathbf S\vert$） | 不变 |

所以只有 `mask17 = Sex|Smag` 的模型具有“自旋空间整体旋转不变性” → 它必须有一个 Goldstone 模
（整体旋转零能）。实测：旋转 0.3 rad， $\Delta E=0.0$ eV（mask17）， $\Delta E=-2.9\times10^{-3}$ eV（mask31）。
这是**区分两种参数化的一个可测判据**，也是判断能否谈论各向异性的前提。

## 3.4 手性与拓扑的门槛

非零 Berry 曲率/Chern 数、手性声子、以及“磁振子霍尔效应”需要
时间反演或空间反演的特定破缺 + 自旋—轨道耦合。本项目：

* mask17 无 SOC 机制 → 拓扑不变量必须为零（由对称性强制），不是“算出来为零”；
* mask31 含 Sdm（P-奇）但数据未证明 SOC，且模型只到 $r_c=6$ Å；
* 因此任何非零拓扑数都不可信。状态记为 `NOT_REPRESENTABLE` / `NOT_VALIDATED`，不虚构数字。

## 3.5 代码入口

* 标签表：`utilities/spin_descriptor.cuh`
* 反演/旋转测试的驱动：`tools/spin_validation/spin_symmetry_stage06.py`
* 本次新增：`tools/one_shot/phase3_model_check.py`（对选定模型做整体旋转测试）

## 3.6 验证与限制

* 已验证：mask17 的整体旋转不变性（0.0 eV）；mask31 的不变性破缺（ $-2.9\times10^{-3}$ eV）。
* 未验证：真正的反演测试需要对完整势做 $P$ 变换（需要成对构型），本项目只做了描述符级与
  单构型级的检查；这一点写进 `KNOWN_LIMITATIONS.md`。
