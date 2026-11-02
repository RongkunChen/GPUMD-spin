# 5 描述符 50 万步训练的磁性能检查点系列

## 这是什么

`work_20261101_retrain/C_5desc_500k/` 里那一次训练（5 描述符全开、λ_e=5、λ_m=1、
batch 1000、generation 500000）每 10 万步取一次权重，对每个权重跑**同一套**磁性能
测试，然后把结果画在一起。

## 检查点是怎么拿到的（一个必须说清楚的事实）

`nep.exe` 每 `output_interval`（5000）代重写一次 `nep.txt`，**不留历史**。
所以：

* **100k 的权重已经丢了**——本轮 11:32 才开始抓，当时已经到 110k。第一个检查点
  实际是 **110k**（差 10k 代，1%），后面 200k/300k/400k/500k 是精确的。
* 抓取由 `series_orchestrator.py` 完成：它轮询 `loss.out`，一旦读到代数 ≥ 边界就
  sleep 8 s 再复制 `nep.txt`（此时它正是该边界的权重），然后立刻跑电池。
  第一版用 `nohup ... &` 起的 shell 守护在一分钟内就被回收了（`ps` 查无进程），
  所以改成这个常驻 Python 进程。

## 每个检查点跑什么（`tools/physics_validation/model_series.py`）

| 项 | 方法 | 代价 |
|---|---|---|
| DEV 误差 | `nep.exe` 预测模式跑 256 帧留出集 | 10 s |
| $a_0$、 $B_0$ | EOS 扫描 2.76–2.92 Å（54 原子）+ 局部 5 点二次细化 | 10 s |
| $m_{eq}$ | 0–3 μB 步长 0.25 扫能量，二次拟合顶点 | 6 s |
| 磁基态 | FM / NC-90 / AFM-A / DLM(3 组) / NM / AFM-G 在**该模型的** $a_0$ 上 | 15 s |
| 磁子 | $8^3$ 常规胞冻结螺旋，共格 $q$， $q\to0$ 外推 $D$ | 60 s |
| 声子 | FM 原胞 $7^3$，有限位移 IFC， $\omega_{\max}$ 与 $\omega(N)$ | 25 s |
| $T_c$ | SLLG-Langevin， $L=6$（432 原子），6 个温度 × 1 ps，半磁化交叉 | 6–10 min |

**协议对每个检查点完全相同**，所以曲线之间的差别只来自权重。

## 两个已知的协议限制

1. $T_c$ 用的是**单一尺寸 + 单一采样器**。SLLG-Langevin 的稳态依赖 dt（见主报告的
   D3：dt 从 0.05 到 2 fs 会让 $|M|$ 从 1.54 掉到 0.16），所以这里的 $T_c$ 是
   **可比较的相对值**，不是收敛值；没有 Binder 交叉（那需要 3 个尺寸 × 每点 ≥50 ps）。
2. $D$ 的误差主要来自 $q\to0$ 外推：naive 斜率比外推值低 12%。表里给的是外推值。

## 文件

```
checkpoints/            每个边界的 nep.txt + 当时的 loss.out + orchestrator.log
series/<tag>.json       每个检查点的完整结果（缓存，重跑会跳过已完成项）
series/eos|moment|states|spiral|tc|eval/  原始 run.in / model.xyz / 输出
figures/series_*.png    对比图（DEV、静态量、M(T)、磁子、磁基态、声子、汇总表）
```

重跑（只补没做完的）：

```powershell
cd tools\physics_validation
python model_series.py --scan ..\..\work_20261101_retrain\C_5desc_500k\checkpoints
python make_series_figures.py
```

## 已经跑出来的结果

### gen_110k（当前唯一可用的检查点；100k 已被覆盖）

| 项 | 值 | 参考 | 判读 |
|---|---|---|---|
| DEV 能量 MAE | **6.53 meV/atom** | — | 优于 A 模型 50k 的 10.38 |
| DEV 力 MAE | 0.0872 eV/Å | — | 优于 A 的 0.0901 |
| DEV 力矩 MAE | 0.0212 eV/μB | — | 与 A 的 0.0206 持平 |
| $a_0$ | **2.8328 Å** | 2.866 | −1.16% → 过 2% 判据 |
| $B_0$ | 286.6 GPa | 170 | +69%（模型整体偏硬，与 A 的 305 同向） |
| $m_{eq}$ | **2.455 μB** | 2.2 | **+11.6% → 超出 10% 判据** |
| 磁基态 | FM 最低 | FM | ✓（NC-90 +169、AFM-A +190、DLM +208、NM +437、AFM-G +518 meV/atom） |
| 磁子刚度 $D$ | ~~439.2~~ → **303 / 310 meV·Å²** | 280–330 | ✓ PASS（原 FAIL 系换算因子错误，见 §15） |
| $\hbar\omega(H)$ | 1243 meV | — | — |
| 声子 $\omega_{\max}$ | 11.09 THz | ~9.5 | +17% |
| $T_c$ | **885 K** | 1043 | **−15.2%**（A 模型是 1101 K） |

**和 A 模型（2 描述符、50k 代）比**：能量和力都更好，但**磁矩偏大 11.6%（越界）**
且 **$T_c$ 从 1101 掉到 885 K**。注意 $T_c$ 的下降**不能**只拿 $D$ 解释：按 §15 更正后
D 已落在实验区间内（303 / 310）， $T_c$ 仍偏低 15%，说明决定 $T_c$ 的是 $m_{eq}$ 偏大
与交换在短波偏强这一组偏差（ $J_2$ 几乎为零、 $J_4$ 符号相反），
不能只看 $D$ 判读磁性的好坏。

### 还没到的检查点

| 边界 | 预计时刻 | 状态 |
|---|---|---|
| 200k | ~13:10 | 自动抓取 + 自动跑电池 + 自动刷新图 |
| 300k | ~14:20 | 同上 |
| 400k | ~15:30 | 同上 |
| 500k | ~16:45 | 同上（跑完自动出最终图并退出） |

编排器 `series_orchestrator.py` 正在后台跑（`orchestrator.log` 有每次抓取与电池的开始/结束时间）。

## per-checkpoint battery (updated)

`model_series.py` now also computes, for every 100k checkpoint, the full
spin-lattice physics set (see `tools/physics_validation/series_physics.py`):

* exchange constants J1..J6 from frozen spirals (3 directions, 8^3 conventional cell)
* magnon dispersion two ways: LSWT of the fitted J_n along Gamma-H-N-Gamma-P-H,
  and direct spiral energies omega = 2S[E(q)-E(FM)]/N
* phonons of every magnetic state (FM, NC-90, AFM-A, AFM-G, DLM) at each state's
  own relaxed lattice constant
* the magnon-phonon vertex (transverse vs longitudinal) and the magnon-phonon
  crossings
* D(|S|) at fixed |S|, i.e. how much of the "D too stiff" failure is just the
  moment being too large
* the magnetostrictive shift d ln(omega)/d ln|S|
