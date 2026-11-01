# 14 — 可发表的方法学（Methods / SI 素材）

> 只写**本项目真的做过并能复现**的部分；未验证项在 §6 单列，投稿时必须原样保留。

## 1. 模型

局域能量 \(E(\mathbf R,\mathbf S)=\sum_i e_i\)，五个自旋块（Sex/San/Sdm/Sia/Smag）与结构 NEP 描述符耦合，见 `04`。
本工作使用两条**预先声明**的假设：H1 = `Sex|Smag`（无 SOC），H2 = 五块全开（SOC-like 表示）。
H2 只表示能力更强，**不**作为“数据含 SOC”的证据。

## 2. 训练与划分

- 数据：`Fe/train.xyz`（33527 帧，SHA `e8f0114c…`），`Fe/test.xyz`（800 帧，SHA `dc1570d9…`），只读。
- 去重：按 `species+pos+spin` 哈希 → 32730 唯一状态；打乱后前 256 为 DEV，其余为训练池。
- **官方 800 帧不参与任何选择**；其 19 帧（14 状态）与训练重叠，另报 781 帧口径。
- 标签契约：\(\mathbf S=\texttt{spin\_length}\times\texttt{initial\_magmoms}\)，
  \(\mathbf g\approx\texttt{spin\_forces\_vert}\)（`ASSUMED_LABEL_CONTRACT`，未做 DFT 溯源核验）。

## 3. 误差统计

MAE/RMSE 定义见 `11` §4；\(R^2\) 用中心化 SST；置信区间按**帧**分块（不是按原子分量）。
磁梯度分解为 \(g_\parallel,g_\perp\)，并报告 \(\boldsymbol\tau=\mathbf S\times\mathbf g\)。

## 4. 动力学

- LL 形式，\(\gamma=\gamma_{\rm LL}\)，噪声方差 \(\sigma^2=2\alpha k_BT/[\gamma\Delta t(1+\alpha^2)|\mathbf S|]\)。
- 离散：几何（Rodrigues + 解析收缩）与随机隐式中点（噪声步内固定）。
- 约束：每原子 \(|\mathbf S|\) 锚点投影；投影前记录原始映射误差。
- 自旋–晶格：S-V-S Strang 分裂，每次求力对应真实中间态（`07`）。

## 5. 统计与可复现

- 固定种子（训练 20261031/2/3；SLLG 4242；REMD 12345）。
- 每条轨迹记录：模型 SHA、源码 SHA、GPU、`nep.in`、`run.in`、退出码。
- 误差条：replica 间标准误 + 自相关修正后的有效样本数标准误（双口径）。
- 时间步与采样间隔以 \(\tau_G=1/(\gamma\alpha k_BT)\) 为单位给出。

## 6. 必须声明的限制

| 限制 | 影响的结论 |
|---|---|
| G6 逐原子能量差 1.19e-6 > 1.05e-6 | 不能声称 trainer 与 MD 完全等价 |
| 耦合能量漂移 1.067e-5 > 1e-5（阶 1.30 < 1.5） | **不能**报告耦合热导/耦合动力学定量结果 |
| 单自旋平衡 +2.9%（4σ 未过） | 不能声称任意步长都精确热平衡 |
| 无 SOC 数据 | 不能报告 DMI/K1/MAE/Tc 的定量值 |
| 模型只到 2k/8k 帧 | 不能声称材料级精度 |
| 官方 800 已曝光 | 不能称为盲测 |

## 7. 引用说明

本文件不附文献 DOI：文中理论（LL/Gilbert、FDT、GK、Wilson loop）均为标准结果，
投稿时请引用原始文献（如 Gilbert 2004、Brown 1963、Kubo 1957、King-Smith–Vanderbilt 1993），
并**明确区分**“引用的是理论”与“本项目实测的是哪一步”。

## 8. 可复现清单（SI 建议附）

1. `SOURCE_MANIFEST.sha256`（源码逐文件）
2. 模型 `nep.txt` + `nep.in` + `loss.out`（每个候选）
3. `run.in` 与 `smoke_result.json`
4. `PHASE_MATRIX.csv`（全部门禁与状态）
5. 本文档 `13`（已知问题）作为 Limitations 小节
