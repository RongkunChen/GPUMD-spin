# 03 — 快速上手与示例

三条命令跑通：**解压 → 编译 → smoke**。

```powershell
Expand-Archive GPUMD-spin-V3-SOURCE-v3.0.0-r17.zip -DestinationPath .
cd GPUMD-spin-V3-SOURCE-v3.0.0-r17
.\INSTALL.ps1            # 编译 + 6 项 smoke
```

`examples/synthetic/` 里的模型是**合成模型**（16 Fe 原子，`mask 17`，仅用于功能验证，**不能**用于材料物理）。

## 1. 静态单点

```text
# run.in
potential nep.txt
time_step 0.0001
ensemble nve
dump_thermo 1
run 1
```

```powershell
cd examples\synthetic
..\..\src\gpumd.exe
```

- 输入：`nep.txt`（模型）、`model.xyz`（16 原子 + `spin:R:3`）。
- 输出：`thermo.out`。第 1 行是表头，**第 3 列是 PE（eV，总量）**，第 2 列 KE。
- 单位：时间 `time_step` 用 **fs**；能量 eV；磁矩 $\mu_B$；力 eV/Å；磁梯度 eV/ $\mu_B$。

## 2. 训练一个极小模型

```text
# nep.in
type 1 Fe
cutoff 6.0 3.0
n_max 4 4
basis_size 8 8
l_max 4 0 0
neuron 30
spin 1
spin_blocks Sex Smag
spin_mag_scale 0.25
lambda_e 1.0
lambda_f 1.0
lambda_m 1.0
batch 32
population 20
seed 20261031
generation 200
output_interval 50
nep_compile off
```

```powershell
..\..\src\nep.exe            # 需要同目录有 train.xyz（含 energy=/forces/spin 列）
```

输出：`nep.txt`（模型）、`loss.out`（每 `output_interval` 行）、`nep.restart`（可续训）。
**`loss.out` 首行的 `total` 只等于 `L1+L2`，不含数据项**，不要用它判断收敛；看分项 RMSE。

## 3. 最小自旋演化

```text
potential nep.txt
time_step 0.5
ensemble nve
dump_thermo 20
dump_xyz 20 traj.xyz spin
spin_llg gamma=3.04 alpha=0.5 T=300 seed=4242 scheme=midpoint
run 2000
```

- `dump_xyz` 默认只写 species+pos；要别的量必须**显式请求**（`spin`、`force`、`velocity`、`potential`、`torque`、`virial`），
  否则 xyz 头部没有 `Properties=`，后处理会读错列。
- `gamma=3.04` 是物理值；小于 0.5×会在日志里打印“噪声主导”的警告。

## 4. 重启

```text
spin_llg gamma=3.04 alpha=0.5 T=300 seed=4242 scheme=midpoint restart=spin_restart.bin
dump_restart 1000
```

分两段跑（详见 `07`）：

```powershell
# 第一段：run 1000，得到 restart.xyz + spin_restart.bin
copy restart.xyz model.xyz
# 第二段：run.in 里加 restart=spin_restart.bin，run 1000
```

与一次性 `run 2000` 的结果应当**逐位相同**（`tests/smoke.py` T5 就是这条判据）。

## 5. Replica（REMD）

```text
potential nep.txt
time_step 0.0001
multi_replica remd replicas 2 exchange 50 temp 300 400 seed 12345
ensemble nvt_bdp 300 400 100
dump_xyz -1 0 50 rep.xyz
run 200
```

注意三个易错点：`time_step` 在 replica 里**不带步数**；`ensemble` 只能用 `nvt_bdp`，
且首尾温度必须等于 `temp` 的上下限；`dump_xyz` 语法是 `-1 0 <interval> <prefix>`。

## 6. 结果怎么看

| 文件 | 内容 | 陷阱 |
|---|---|---|
| `thermo.out` | T, KE, PE, 应力张量分量 | PE/KE 是**总量**不是每原子 |
| `traj.xyz` | 每 `dump_xyz` 间隔一帧 | 头部 `Time=` 不是 `Step=` |
| `restart.xyz` | 可直接当 `model.xyz` 用 | 只有 R,v,S；磁自由度另见 `spin_restart.bin` |
| `spin_work.out` | $W_{\rm rot},W_{\rm noise},W_{\rm ext}$ | 累积量，单位 eV |
| `loss.out` | 训练损失分项 | 首列是 generation |

## 7. 示例目录

```
examples/
  synthetic/          model.xyz + nep.txt（16 原子，mask17）
  spin_llg/           run.in.{conservative,geometric,midpoint}
  restart/            run.in.{prefix,restart}
  replica/            run.in
```

> 所有示例都用合成模型，绝对能量（ $\sim-0.73$ eV/16 原子）没有物理意义。
