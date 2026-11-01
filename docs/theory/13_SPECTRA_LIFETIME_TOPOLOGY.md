# 13 — 谱学、寿命与拓扑

## 13.1 动态结构因子

自旋：$S(\mathbf q,\omega)=\big|\sum_j\int dt\,e^{i(\mathbf q\cdot\mathbf R_j-\omega t)}\mathbf S_j(t)\big|^2$；
速度/SED：对速度自相关做 FFT 得声子 DOS。

有限采样窗 $T_{\rm win}$ 的展宽 $\Delta\omega\approx2\pi/T_{\rm win}$。
要把“窗函数展宽”、“阻尼（物理线宽）”、“非谐/散射”分开，必须做 $T_{\rm win}$ 扫描：
窗展宽随 $1/T_{\rm win}$ 缩小，物理线宽不随窗变。

## 13.2 寿命

若有统计显著的线宽 $\Gamma$（半高半宽），则 $\tau=1/\Gamma$，误差由拟合的置信区间给出。
**谐波线性矩阵不能给有限温寿命**：它给的是 $\mathbf q=0$ 的简正频率，没有散射通道。

## 13.3 Berry 曲率与 Wilson loop

离散规范不变的写法：把 $\mathbf q$ 空间分成小方格，取格点上的本征矢 $|u_n(\mathbf k)\rangle$
（简并时取子空间投影），定义

$$
U_\mu(\mathbf k)=\det\langle u_m(\mathbf k)|u_n(\mathbf k+\delta\hat\mu)\rangle ,
\qquad
\Omega(\mathbf k)=-2\,\mathrm{Im}\ln\prod_{\square}U_\mu .
$$

Chern 数 $C=\frac{1}{2\pi}\sum_{\square}\Omega$。
要点：(a) 简并带必须用子空间行列式而不是逐带相位；(b) 声子要用质量加权内积，
磁子（bosonic BdG）要用 $\sigma_3$ 的 paraunitary 度量，否则本征矢的“归一化”是错的；
(c) 结果必须与规范选择无关（换相位后数值不变）——这是一条自检。

## 13.4 手性 vs 拓扑

* 圆极化声子的**角动量投影**是可算的量（$\langle u|\hat L_z|u\rangle$），不需要拓扑；
* **Berry 曲率**是 $\mathbf k$ 空间的几何量；
* **Chern 数**是它的积分（拓扑不变量）；
* **边界态**需要开边界计算。
四者不同，不能互相替代。手性声子不等于拓扑声子。

## 13.5 本项目的能力边界

| 对象 | 机制要求 | 本模型 | 结论 |
|---|---|---|---|
| 磁子 Chern 数 | 需 DMI/SOC 破缺 | mask17 无；mask31 只有未证明的 Sdm | `NOT_REPRESENTABLE` |
| 声子 Berry 曲率 | 需手性晶格/破缺 | 立方 Fe，无 | 平凡（由对称性强制） |
| 磁声混合模 | 需 $C\neq0$ | 有 $C$，但无混合本征求解器 | `NOT_IMPLEMENTED` |
| 有限温寿命 | 需长轨迹 + 拟合 | 未跑 | `NOT_RUN` |

**不虚构非零拓扑数**：若模型缺少必要机制，正确答案是“由对称性强制为零”，不是“算出一个小数”。

## 13.6 代码入口

* 上游：`measure/sed.cu`、`measure/dos.cu`、`phonon/hessian.cu`
* 本项目未新增 Berry/Wilson 实现（见上表）。

## 13.7 限制

* 没有实现 paraunitary 的磁子 BdG 求解器。
* $S(\mathbf q,\omega)$ 需要对大量帧做 FFT，本项目的 4–16 原子体系太小，色散不可分辨。

## 13.6 能力状态（r18）：分清"有代码"和"验证过"

r18 的核对结果比 r17 文档更细：`src/measure/` 里**确实存在**谱与磁化相关的模块，
但它们在自旋体系上的正确性**没有被本项目验证**。两者必须分开报。

| 能力 | 代码 | 本项目验证 | 说明 |
|---|---|---|---|
| 声子 SED（`compute_sed ... mode velocity`） | **存在** | **NOT_VALIDATED** | 上游 GPUMD 模块，按速度自相关 + 窗函数 |
| 磁子 SED（`compute_sed ... mode spin`） | **存在** | **NOT_VALIDATED** | 读 `atom.spin`（float），本仓库扩展；未做与解析磁子谱的对比 |
| 磁化强度 / 磁化率（`dump_magnetization`） | **存在** | **PASS（冒烟）** | 输出 `Mx My Mz |M| varM chi Msat`，`chi=(<M·M>-<M>·<M>)/(N k_B T)`；见 `USER_MANUAL` §6 |
| 声子 DOS（`compute_dos`） | 存在 | NOT_VALIDATED | 依赖速度自相关 |
| 谱线宽 / 寿命 | 部分（SED 可给线宽） | **NOT_VALIDATED** | 需要长时采样与线型拟合，未做 |
| Berry 曲率 / Chern 数 | **不存在** | NOT_IMPLEMENTED | 无布洛赫基与 $\mathbf q$ 网格 |
| Wilson loop / 拓扑不变量 | **不存在** | NOT_IMPLEMENTED | 同上 |
| 手性序参量 | 后处理可得 | **未做** | 从 `dump_xyz spin` 直接算 |

**结论**：交付物中不含任何谱、线宽或拓扑数值。`compute_sed` 与
`dump_magnetization` 能跑（冒烟级），但在没有与解析结果对比之前，
它们的输出只能当"工程可用"，不能当物理结论。`FINAL_TEST_MATRIX.csv`
里相应条目写 `NOT_VALIDATED` 或 `NOT_IMPLEMENTED`，不得读成 PASS。

## 13.7 如果想自己验证磁子 SED

1. 用 `examples/fe_bcc_magnetic/` 的 432 原子 bcc 胞，`spin_llg alpha=0.01`，
   `compute_sed 10 1000 1 x spin`；
2. 与解析磁子色散 $\omega(q)=\gamma J(0)-J(q)$ 对比（$J(q)$ 由 09 篇的 $K$
   在切空间投影后得到）；
3. 判据：第一布里渊区边界处的峰位相对误差 < 15%，且 $\mathbf q	o0$ 时
   $\omega	o0$（Goldstone）。

第 2 步是瓶颈：$K$ 的切空间投影还没实现（09 篇 §9.2），所以这条验证链
目前断在中间。
