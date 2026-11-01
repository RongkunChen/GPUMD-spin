# 15 — 复现手册（命令级）

所有命令都在源码包根目录执行（`<PKG>` = 解压目录）。每条都注明**预期产物**与**失败如何判读**。
标 `[未执行]` 的条目在本版本没有真实运行记录，不得当作已验证。

## 0. 一键

```powershell
.\INSTALL.ps1                     # 依赖检查 + 编译 + 6 项 smoke
```
预期：末尾 `== INSTALL OK ==`，生成 `src/gpumd.exe`、`src/nep.exe`、`src/gpumd_replica.exe`、`smoke_result.json`。

## 1. 编译（分步）

```powershell
.\BUILD.ps1 -Arch sm_89 -Jobs 8   # 只编译
python tests\smoke.py --bin src   # 只测
```
失败判读：`BUILD.ps1` 返回非零并在 `build.log` 末尾给出缺失项；`smoke.py` 打印每项 PASS/FAIL/SKIP。

## 2. 单点

```powershell
mkdir scratch\sp; copy examples\synthetic\* scratch\sp
cd scratch\sp
..\..\src\gpumd.exe    # run.in 见 docs/03
```
预期 `thermo.out` 第 3 列 PE ≈ −0.7335 eV（16 原子，合成模型，无物理意义）。

## 3. 单自旋 FDT

```powershell
# 单原子 model.xyz + 常场；SPIN_NOISE_DEBUG=1 可导出每步热场
$env:SPIN_NOISE_DEBUG="1"
..\..\src\gpumd.exe
```
预期 `spin_noise.out` 存在；方差与 `2αk_BT/(γΔt(1+α²)|S|)` 相对偏差 <4%。

## 4. 二聚体（MC vs SLLG）`[未执行：需 Python 工具]`

```powershell
python tools\spin_validation\check_sllg_stage08a.py --part dimer --dir <ev> --exe src\sllg_stage08a.exe
```
注意该脚本的 `--exe` 是**测试驱动**，不是 gpumd；见 `evidence/stage08a/20261031_131143/build/`。

## 5. 重启（最关键的一条）

```powershell
python tests\smoke.py --bin src            # T5 就是它
```
或手工三段（`docs/07` §4）。判据：`max|ΔS| == 0` 且末态 RNG 块逐字节相同。

## 6. Fe 训练 `[需用户数据]`

```powershell
python tools\one_shot\phase4_train.py <ev> prepare       # 去重/划分
python tools\one_shot\phase4_train.py <ev> train 17 20261031 8k 20000
python tools\one_shot\phase4_train.py <ev> eval  <run_dir>
```
预期：`<run_dir>/nep.txt`、`loss.out`、`metrics.json`。**数据不在包内**，需要自备并核对 SHA（`docs/11`）。

## 7. 保守耦合 B 门禁 `[需 S05B harness]`

```powershell
python tools\spin_validation\check_coupling_stage05b.py --part B ...
```
本版本**未重跑**；历史值见 `docs/07` §2。

## 8. 热流（受门禁控制）

`compute_heat`/`dump_heat_flux` 可运行，但**在 B 门禁通过前不得作为物理结果**：
`docs/08` 给公式与单位换算，脚本只用于诊断。

## 9. 谱学/拓扑

`[未执行]`。工具骨架与判据见 `docs/09`、`docs/10`；没有模型门禁就标 `NOT_RUN_MODEL_GATE`。

## 10. 常见失败与判读

| 现象 | 判读 |
|---|---|
| smoke T2/T3 报 `spin_norm_drift` 大 | 先确认 model.xyz 的 spin 列顺序（SoA/AoS 陷阱） |
| smoke T5 `max_spin_diff != 0` | checkpoint 没被加载（`restart=` 拼写/路径/哈希不符） |
| `nep.exe` 报 `'energy' is missing in the second line` | 预测用的 xyz 缺 `energy=` 属性 |
| `gpumd` 报 `Unrecognized argument in dump_xyz` | 语法是 `dump_xyz <interval> <file> [quantities]`，不是 `dump_xyz <file> <interval>` |
| 训练 loss 不动 | `total` 首行只含 L1+L2；看分项 RMSE |
