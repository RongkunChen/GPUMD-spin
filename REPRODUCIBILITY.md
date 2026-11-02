# 复现指南（r18）

目标：任何人拿到这个包，能在**全新目录**里从源码编译出三个可执行文件，
并用包内模型复现本文列出的数字。

## 1. 最短路径（Windows）

```bat
Expand-Archive GPUMD-spin-V3-SOURCE-v3.1.0-r18.zip D:\fresh
cd D:\fresh\GPUMD-spin-V3-SOURCE-v3.1.0-r18
BUILD.cmd -Arch sm_89 -Jobs 8
python tests\smoke.py --bin src
python tests\acceptance.py --bin src
python tests\physics_checks.py --bin src --model models\fe_mask17_2desc_r18.nep.txt --quick
```

环境：Windows 10/11 + CUDA 12.x + VS 2022（含 VC++ 工具集）+ Python 3.9+（numpy）。
本包实测环境：RTX 4060 Ti（sm_89）、CUDA 12.8、驱动 610.60。

## 2. 逐项复现表

| 目标 | 命令 | 期望（r18 实测） |
|---|---|---|
| 编译 | `BUILD.cmd -Arch sm_89 -Jobs 8` | rc=0，`src/` 下三个 exe（约 3–5 min） |
| 代码路径 | `python tests/smoke.py --bin src` | 7/7 PASS |
| 全链路 | `python tests/acceptance.py --bin src` | 5/5 PASS（训练→预测→MD→SLLG→重启） |
| 晶格常数 | `python tests/physics_checks.py --bin src --quick` | $a_0=2.8258$ Å（−1.4%） |
| 基态磁矩 | 同上（P2） | $\lvert S\rvert_0=2.413\ \mu_B$ |
| 能量漂移 | 同上（P4） | $3.5$– $4.4\times10^{-7}$ eV/atom（0.1 ps） |
| guard 回归 | 同上（P5） | 600 K 起步 2000 步：rc=0、无 RUNAWAY |
| $T_c$ | 同上（P3，3 点快速模式） | $\approx1009$ K（粗） |
| $a_0$ 精细扫描 | `cd examples/fe_bcc_ground_state && python make_cells.py && python scan.py` | 7 点表 + `a0 = 2.8258` |
| $T_c$ 曲线 | `cd examples/fe_curie_temperature && python tc_scan.py` | `tc_points.csv`， $T_c\approx950$ K |
| NVE 步长扫描 | `cd examples/fe_nve_stability && python nve_drift.py --dt 0.05 --ps 1` | $\sim6\times10^{-8}$ eV/atom |

所有示例都**不需要**额外数据：bcc 胞由 `tools/physics_validation/make_fe_system.py`
在线构造，模型在 `models/`。

## 3. 模型

| 文件 | 说明 | DEV $E_{\rm MAE}$ | $a_0$ | $\lvert S\rvert_0$ | $T_c$ |
|---|---|---|---|---|---|
| `models/fe_mask17_2desc_r18.nep.txt` | Sex Smag，50k 代 | 10.38 meV/atom | 2.8258 Å | 2.413 $\mu_B$ | ≈950 K |
| `models/fe_mask31_5desc_r18.nep.txt` | 全 5 块，50k 代 | 8.75 meV/atom | 2.8273 Å | 2.432 $\mu_B$ | ≈850 K |
| `models/fe_mask17_r17_baseline.nep.txt` | r17 基线（对照） | 15.23 meV/atom | 2.823 Å | 2.60 $\mu_B$ | — |

训练输入在 `models/nep_A_2desc.in`、`models/nep_B_5desc.in`；
指标在 `models/DEV_METRICS.json`。

**重新训练**：把训练集复制成 `train.xyz`，用上面的 `.in` 跑 `nep.exe` 5 万代。
同 seed + 同数据 → 逐位相同的模型（这是本项目的可复现性判据之一）。

## 4. 关键数字的来历

| 数字 | 出处 |
|---|---|
| $4.6\times10^{-9}$ eV/atom/ps | Fe 1024 原子 1/5/10 ps 的斜率（5 ps 与 10 ps 一致到 3%） |
| $2.2$ ns | $10^{-5}/(4.6\times10^{-9})$，到 1e-5 门禁的时间 |
| $\tau=0.176$ ps | 热流自相关降到 $0.1C_0$ 的积分时间（1024 原子，250 K） |
| 0.978 ± 0.123 | OU 过程 60 次实现的估计器自检（解析 $\kappa$ 已知） |
| $5.94\times10^5$ / $-6.4\times10^{-2}$ / ~80 | 同一条轨迹三种 $\kappa_{zz}$（差 $10^7$，故不报数） |
| 1.023e-5 | 0.2 fs、4 原子强交换场的耦合能量漂移（FAIL，保留） |
| 7.153e-7 | G6 逐原子门禁最差值（混合判据 PASS） |

## 5. 环境记录

```
GPU      : NVIDIA GeForce RTX 4060 Ti (sm_89), driver 610.60
CUDA     : 12.8 (V12.8.61)
MSVC     : Visual Studio 2022, vcvars64
Python   : 3.13 (numpy 2.x)
OS       : Windows 11
```

Linux 构建脚本（`build.sh`）**未在本轮验证**（`UNTESTED`）。
`BUILD.ps1` 会自动探测 nvcc / vcvars / GPU 架构，也可以显式给 `-Arch`。

## 6. 复现失败时先查什么

| 现象 | 先查 |
|---|---|
| 编译报 `nep_acc_t undefined` | 是否用了旧版 `spin_descriptor_kernels.cuh`（typedef 必须在那里且带守卫） |
| `gpumd.exe` 一跑就 `RUNAWAY GUARD ... limit 300` | 用了 r17 的二进制（F26），换 r18 |
| 预测的力差一个量级 | `nep.in` 是否是训练时那一份（架构从 `nep.in` 重建） |
| `thermo.out` 有两段表头 | 目录被重跑过（追加写），删文件或换目录 |
| 温度离谱 | 速度单位（Å/fs vs Å/nat）与 KE 因子 `1.0364269e-4` |
| 第二段之后文件不再增长 | r18 及更早的 F-25；r19 已修复（单文件双段 = 两次调用，逐位相同） |
