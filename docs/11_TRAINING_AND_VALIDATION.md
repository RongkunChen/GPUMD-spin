# 11 — 训练与验证（Fe 数据约定）

> 数据来源状态：`ASSUMED_LABEL_CONTRACT`。按用户决定，**原始 DFT/SOC 来源不作为开发阻断**，
> 也不再追索；本文只声明“标签怎么读”，**不**声称 DFT 已核验。

## 1. 数据事实（只读、逐字节保留）

| 文件 | SHA256 | 帧数 | 原子数 |
|---|---|---|---|
| `Fe/train.xyz` | `e8f0114ca495497fb7e4b6486a40437f524fa847dafbd27994a14409a47bc247` | 33527 | 537984 |
| `Fe/test.xyz` | `dc1570d9cf603779268aa4799e23b1e2a6b31972842e1b066709cfa4b044f728` | 800 | 12832 |

列契约：`species, pos, forces, spin_length, initial_magmoms, spin_forces_vert`，头部带 `energy=`、`virial=`。

$$
\mathbf S_i=\texttt{spin\\_length}_i\times\texttt{initial\\_magmoms}_i\ [\mu_B],\qquad
\mathbf g_i\approx\texttt{spin\\_forces\\_vert}_i\ [\mathrm{eV/\mu_B}].
$$

`spin_forces_vert` 的字段名不可读作“已投影”：实测 $|\mathbf w \cdot \mathbf u|/|\mathbf w|$ 中位 0.915，
即标签以**纵向分量为主**。这一点必须在任何训练报告里写明。

## 2. 划分（输入状态隔离，不是轨迹级独立）

- 按 `species+pos+spin` 的文本哈希去重：33527 帧 → **32730 个唯一状态**（797 个重复帧）。
- 打乱（种子 20261031）后：前 256 状态 = **DEV**，其余 = 训练池（**32474**）。
- **train ∩ dev = 0**（按状态）；官方 800 帧中 19 帧（14 个状态）与训练池同状态 →
  同时报 **800** 与**去重 781** 两个口径，且**都不称“独立盲测”**。
- 母轨迹未知（帧序不是轨迹序，逐帧位移中位 0.63 Å），所以这是“输入状态隔离”。

## 3. 两条预先声明的假设

| 假设 | 掩码 | 含义 | 不能声称 |
|---|---|---|---|
| H1 无 SOC | `Sex Smag`（17） | 各向同性交换 + 模长项 | 不能声称有各向异性 |
| H2 SOC-like | 五块全开（31） | 表示能力更强 | **不能**声称数据含 SOC/DMI/MAE |

选择只看 **DEV** 的预注册多目标（E/F/g 标准化误差 + 横向力矩误差 + 稳定性），
**官方 800 分数不参与**任何选择（历史已曝光，如实声明）。

## 4. 误差定义

$$
\mathrm{MAE}=\frac1N\sum|y^{\rm pred}-y^{\rm ref}|,\qquad
\mathrm{RMSE}=\sqrt{\frac1N\sum(y^{\rm pred}-y^{\rm ref})^2},\qquad
R^2=1-\frac{\sum(y^{\rm pred}-y^{\rm ref})^2}{\sum(y^{\rm ref}-\bar y^{\rm ref})^2}.
$$

- 能量：meV/atom；力：eV/Å（逐分量）；磁梯度：eV/ $\mu_B$（逐分量）。
- **置信区间按帧分块**（不是按原子分量，那会把相关性当独立样本）。
- 磁矩方向分解： $g_\parallel=\mathbf g \cdot \hat{\mathbf S}$、 $g_\perp=|\mathbf g-g_\parallel\hat{\mathbf S}|$；
  角力矩 $\boldsymbol\tau=\mathbf S\times\mathbf g$（eV）。

## 5. 本版本实测（2k / 8k 帧，20000 代；工程候选）

| run | 训练帧 | DEV E | official800 E | 781 E | 800 F | 800 g |
|---|---|---|---|---|---|---|
| m17 s1 | 2048 | 12.95 | 11.90 | 11.87 | 0.082 | 0.0227 |
| m17 s2 | 2048 | 14.67 | 13.99 | 13.86 | 0.084 | 0.0223 |
| m31 s1 | 2048 | 11.73 | 11.51 | 11.42 | 0.091 | 0.0217 |
| m31 s2 | 2048 | 13.31 | 13.20 | 13.09 | 0.094 | 0.0240 |
| m17 s1 | 8192 | 见 `phase_4` | — | — | — | — |

（单位：E meV/atom，F eV/Å，g eV/ $\mu_B$。）

## 6. 禁止事项

1. 不得用官方 800 选 mask/seed/超参/checkpoint。
2. 不得把 mask31 的 E 稍优解读为“物理更合适”。
3. 不得声称 8k/全量学习曲线“已完成”，除非 `phase_4/runs` 里真有对应 `run.json`。
4. 未达预注册精度 → 标 `ENGINEERING_ONLY`，可用于演示，不可报告材料定量性质。

## 7. 复现

```powershell
python tools/one_shot/phase4_train.py <evidence_dir> prepare
python tools/one_shot/phase4_train.py <evidence_dir> train 17 20261031 8k 20000
python tools/one_shot/phase4_train.py <evidence_dir> eval  <run_dir>
```

## 8. 学习曲线（2026-10-31 实测，固定 20000 代）

训练预算固定为 20000 代、batch 32，因此"帧数越多"等价于"每帧被看的次数越少"。
这是一条**固定算力**曲线，不是收敛曲线。

| 掩码 | 帧数 | 种子数 | DEV E (meV/atom) | 种子间极差 | 官方 800 E | 去重 781 E |
|---|---|---|---|---|---|---|
| 17 | 8192 | 3 | 16.94 | 14.1% | 13.82–15.74 | 13.76–15.68 |
| 17 | 32474 | 3 | **15.23** | 11.2% | 12.77–14.83 | 12.62–14.70 |
| 31 | 8192 | 3 | 14.85 | 12.6% | 12.07–13.72 | 12.06–13.71 |
| 31 | 32474 | 3 | **12.38** | 13.8% | 10.64–12.62 | 10.60–12.64 |

同预算下加数据是有效的（17：16.94→15.23；31：14.85→12.38）。
mask31 的 E 更好但 F 更差（0.091–0.095 vs 0.082–0.087 eV/Å），
所以**不能**只凭 E 宣布 mask31 在物理上更合适。

选型规则在看分数之前写死：`score = E/20 + F/0.10 + g/0.030`，只用 DEV，
种子间 E 极差超过 15% 的组不可选。四组都通过该稳定性判据。
官方 800 与 781 两列仅作事后披露，不参与任何选择。

状态：**ENGINEERING_ONLY**。没有任何一组达到"可直接做材料定量预测"的精度（E ≲ 5 meV/atom）。
