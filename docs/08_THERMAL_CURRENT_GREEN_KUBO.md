# 08 — 热流与 Green–Kubo

> **当前版本状态：`BLOCKED_CONSERVATION`。** 耦合能量门禁（`07` 的 B 组）未通过
> （ $\Delta t=0.2$ fs， $\max|\Delta E|/\text{atom}=1.067\times10^{-5}>10^{-5}$），
> 因此本文件给出**推导、实现映射与验收方案**，但**不报告任何 Fe 的 $\kappa$ 数值**。

## 1. 局域能量与连续性

模型是局域的： $E=\sum_i e_i$， $e_i$ 只依赖截断内的 $(\mathbf r,\mathbf S)$。对时间求导：

$$
\frac{dE}{dt}=\sum_i\dot e_i=\sum_i\Big[\frac{\partial e_i}{\partial \mathbf r_i} \cdot \dot{\mathbf r}_i
+\frac{\partial e_i}{\partial \mathbf S_i} \cdot \dot{\mathbf S}_i\Big]+\text{(共享项)},
$$

GPUMD 的局域能量采用**半分配**（每一对 $i,j$ 的键能对半分给两端、三体项按参与原子均分），
因此可以写出离散连续性方程

$$
\frac{d e_i}{dt}+\sum_{j}j_{ij}=0,\qquad j_{ij}=-j_{ji},
$$

其中 $j_{ij}$ 是 $i\to j$ 的能流。**这是构造热流算符的前提**：热流不是“把三列加起来”，
而是必须让 $\sum_i \mathbf r_i \dot e_i$ 与 $\mathbf J$ 的散度项对应。

## 2. 热流算符

广延热流

$$
\mathbf J=\sum_i \mathbf r_i\ \dot e_i
=\underbrace{\sum_i \mathbf r_i\ (-\mathbf F_i \cdot \dot{\mathbf r}_i)}_{\mathbf J_L}
+\underbrace{\sum_i \mathbf r_i\ (-\mathbf g_i \cdot \dot{\mathbf S}_i)}_{\mathbf J_S},
$$

- $\mathbf J_L$ 的可用形式： $\sum_i \dot{\mathbf r}_ i\ \epsilon_i+\sum_{i<j}\mathbf r_{ij}(\mathbf F_{ij} \cdot \dot{\mathbf r}_ i)$（依能量分配约定）。
- $\mathbf J_S=-\sum_i \mathbf g_i\ (\dot{\mathbf S}_ i \cdot \mathbf r_i)$ 的**非局域**部分来自 $\mathbf g_i$ 依赖邻居 $\mathbf S_j$，
  必须按键拆分，不能只写 $-\mathbf g_i \cdot \dot{\mathbf S}_ i$（后者散度不为零）。
- **规范非唯一**：能量零点、 $\mathbf r_i$ 的原点、是否包含 $\mathbf J_S$ 的“对流项”都会改变分解；
  只有 $\mathbf J_{\rm total}$ 的散度是物理的， $\kappa_{LL}/\kappa_{SS}/\kappa_{LS}$ 的拆分依赖规范。

代码入口：`src/measure/compute_heat.cu`、`src/measure/dump_heat_flux.cu`（列名与上面两项对应），
自旋项用 `atom.spin_gamma/alpha` 由 $(\mathbf S,\mathbf g)$ 重建 $\dot{\mathbf S}$。

## 3. Green–Kubo

广延量：

$$
\kappa_{\alpha\beta}=\frac{1}{Vk_BT^2}\int_0^\infty\big\langle J_\alpha(t)J_\beta(0)\big\rangle\ dt .
$$

密度量 $\mathbf j=\mathbf J/V$：

$$
\kappa_{\alpha\beta}=\frac{V}{k_BT^2}\int_0^\infty\big\langle j_\alpha(t)j_\beta(0)\big\rangle\ dt .
$$

两者**不能混用**：差别是 $V^2$。实现时固定其一并在输出里写清楚。

单位换算（可重算）：

| 量 | 输入单位 | 换算 |
|---|---|---|
| $J$ | eV·Å/fs | $1\ \mathrm{eV\ \text{\AA}/fs}=1.602176634\times10^{-19}\times10^{-10}/10^{-15}\ \mathrm{J\ m/s}=1.602176634\times10^{-14}$ W |
| $\int\langle JJ\rangle dt$ | (eV·Å/fs)²·fs | 乘以 $(1.602176634\times10^{-14})^2\times10^{-15}$ |
| $V$ | Å³ | $\times10^{-30}$ m³ |
| $k_BT^2$ | eV·K | $\times1.602176634\times10^{-19}$ J |

于是 $\kappa$ 的数值 = 上式组合后 × 相应因子，得到 W/(m·K)。脚本里一律先换到 SI 再乘 $\frac{1}{Vk_BT^2}$。

## 4. 验收方案（本版本未执行完）

| 检查 | 判据 | 状态 |
|---|---|---|
| 能量矩守恒： $d(\sum_i \mathbf r_i e_i)/dt$ vs $\mathbf J$ | 相对偏差 < 1%（解析二体模型） | 设计完成，未运行 |
| 平移不变性、最小像 | 平移整体坐标 $\mathbf J$ 不变 | 未运行 |
| 力/场功率闭合 | $\sum\mathbf F \cdot \dot{\mathbf r}+\sum\mathbf g \cdot \dot{\mathbf S}+\dot E=0$ | 部分（`spin_work` 记账） |
| EMD 平台 | $\kappa(t)$ 出现平台，块误差收敛 | 未运行 |
| NEMD 互证 | 与 EMD 同量级（±30%） | 未运行 |

脚本骨架：`tools/` 下的分析脚本接受 `dump_heat_flux` 输出，按块平均 + 积分平台出 $\kappa(t)$；
**在 B 门禁通过前，这些脚本只用于诊断，不得作为物理结果发布。**

## 5. 分解与符号

$\mathbf J=\mathbf J_L+\mathbf J_S$ 时

$$
\kappa_{\rm total}=\kappa_{LL}+\kappa_{SS}+\kappa_{LS}+\kappa_{SL},
$$

$\kappa_{\rm cross}\equiv\kappa_{LS}+\kappa_{SL}$ **不保证为正**，且随规范变化。
报告中必须写明：分解方式、能量零点、是否含对流项、是否做了质心速度扣除。

## 6. 本版本状态

| 项 | 状态 |
|---|---|
| 连续性推导与代码映射 | 已完成（本文） |
| `compute_heat.cu` 审计 | 已完成（列名/单位对照，见阶段6 证据） |
| 数值连续性测试 | NOT_RUN |
| EMD/NEMD $\kappa$ | **BLOCKED_CONSERVATION** |
| Fe 的 $\kappa_{ph}/\kappa_{mag}/\kappa_{cross}$ | **禁止报告** |
