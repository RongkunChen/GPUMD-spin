# 用户手册（GPUMD-spin V3, r18）

适用版本：`VERSION = r18`（2026-11-01）。
所有命令都在"工作目录"执行（该目录含 `nep.txt`、`model.xyz`、`run.in`）。

---

## 0. 五分钟上手

```bash
# 1) 编译（Windows）
BUILD.cmd -Arch sm_89 -Jobs 8
#    Linux: ./build.sh   （需要 nvcc + g++ + make）

# 2) 冒烟测试：7 项，全部跑真二进制
python tests/smoke.py --bin src

# 3) 验收测试：训练 → 预测 → MD → SLLG → 重启
python tests/acceptance.py --bin src

# 4) 物理检查：晶格常数 / 基态磁矩 / Tc / 能量漂移 / guard 回归
python tests/physics_checks.py --bin src --model models/fe_mask17_2desc_r18.nep.txt --quick
```

三个可执行文件：

| 文件 | 作用 |
|---|---|
| `nep.exe` | 训练（`nep.in` + `train.xyz`）与预测（加 `prediction 1`） |
| `gpumd.exe` | MD / SLLG / 保守自旋–晶格动力学 |
| `gpumd_replica.exe` | 副本交换（replica exchange）MD |

---

## 1. 输入文件一览

| 文件 | 必需 | 说明 |
|---|---|---|
| `nep.txt` | 是 | 模型参数（文本，人类可读，见 `MODEL_FORMAT.md`） |
| `model.xyz` | 是 | 初始构型（extended XYZ，含 `spin` 列） |
| `run.in` | 是 | 运行指令 |
| `nep.in` | 训练/预测时 | 架构与训练超参 |
| `train.xyz` | 训练/预测时 | 训练集或待预测帧 |
| `spin_restart.bin` | 重启时 | `dump_restart` 写出的检查点 |

`model.xyz` 的最小形式：

```
2
Lattice="6.0 0 0 0 6.0 0 0 0 6.0" Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3
Fe  0.0 0.0 0.0   0.0 0.0 0.0   0.0 0.0 2.2
Fe  1.5 1.5 1.5   0.0 0.0 0.0   0.0 0.0 2.2
```

`spin` 列写成三个数（ $\mu_B$）。**注意排列**：多帧文件的 `spin` 是逐原子三元组，
不是"先所有 $x$ 再所有 $y$"；`tests/physics_checks.py` 与
`tools/physics_validation/make_fe_system.py` 都按逐原子三元组写。

---

## 2. 训练（`nep.exe`）

### 2.1 `nep.in`

```
type 1 Fe
cutoff 6.0 3.0        # 径向 / 角向截断 (Å)
n_max 4 4
basis_size 8 8
l_max 4 0 0
neuron 30
spin 1                # 启用自旋
spin_blocks Sex Smag  # mask17；全 5 块写 "Sex San Sdm Sia Smag"
lambda_e 5.0
lambda_f 1.0
lambda_m 5.0
batch 32
population 20
seed 20261101
generation 50000
output_interval 500
nep_compile off
```

| 参数 | 含义 | 建议 |
|---|---|---|
| `type` | 元素种类与名称 | 必须与 xyz 的 species 一致 |
| `cutoff` | 径向、角向截断 | Fe 用 6.0 / 3.0 |
| `n_max`, `basis_size`, `l_max` | 描述符基组 | 与上游 NEP4 相同，改大需重训 |
| `neuron` | 隐藏层宽度 | 30 够用；>50 收益小、慢 |
| `spin` | 1 启用自旋通道 | 关掉时 `spin` 列被忽略 |
| `spin_blocks` | 启用哪些自旋描述符块 | 见 §2.2 |
| `lambda_e/f/m` | 能量/力/磁矩损失的权重 | 能量与磁矩 5、力 1 是 r18 的推荐 |
| `batch`, `population` | 进化策略的种群与批大小 | 默认即可 |
| `seed` | 随机种子 | 同 seed 同数据 → **逐位相同**的模型 |
| `generation` | 进化代数 | 50k 在单卡上约 40 分钟（Fe 256 帧） |
| `nep_compile off` | 不做 CUDA 源码特化编译 | 保持 off，否则每个模型要多花几分钟 |

### 2.2 `spin_blocks` 怎么选

| 值 | 块 | 适用 |
|---|---|---|
| `Sex Smag` | 交换 + 模长（mask17） | 铁磁金属、无 SOC；**推荐默认** |
| `Sex San Smag` | 加键向各向异性 | 需要单轴各向异性 |
| `Sex San Sdm Sia Smag` | 全 5 块（mask31） | 含 SOC 型参数化；能量略好但更慢，且不保证物理上更对 |

**`spin_mag_scale` 不要手写**：不写时训练器按 $1/\max|\mathbf S|$ 自动定标
（Fe 数据为 0.234148785）。写死一个值会让磁矩通道的有效分辨率下降（F-24）。

### 2.3 训练集格式

`train.xyz` 每帧头必须带 `energy=`、`force=`、`spin_force=`（可选 `virial=`）：

```
256
Lattice="..." Properties=species:S:1:pos:R:3:force:R:3:spin:R:3:spin_force:R:3 energy=-1.234 force=1 spin_force=1
```

`energy=` 是**总能量**（eV），`force=1` 表示下面有 3 列力，
`spin_force=1` 表示有 3 列磁梯度（ $-\partial E/\partial\mathbf S$）。
若要做预测，这几列仍要占位（训练器会校验列数），详见 `MODEL_FORMAT.md`。

### 2.4 训练输出

| 文件 | 内容 |
|---|---|
| `nep.txt` | 模型（可直接给 `gpumd.exe`） |
| `loss.out` | 每代的 4 个损失分量 + 正则项 |
| `energy_train.out` / `force_train.out` / `spin_train.out` | 逐帧/逐原子预测 vs 参考 |
| `nep.restart` | 中断后可续训 |

---

## 3. 预测

在 `nep.in` 末尾加一行 `prediction 1`，把待预测帧写成 `train.xyz`（可只带占位列），
运行 `nep.exe`。输出 `energy_train.out`（2 列：预测、参考）、
`force_train.out`（6 列：预测 3 + 参考 3）、`spin_train.out`（同）。

**必须用训练时的 `nep.in`**：`nep.exe` 从 `nep.in` 重建描述符架构，
只从 `nep.txt` 读参数。给一个默认 `nep.in` 会静默算错
（实测 F_MAE 从 0.09 变 0.81 eV/Å）。

---

## 4. `run.in` 关键字参考

### 4.1 必需

| 关键字 | 语法 | 说明 |
|---|---|---|
| `potential` | `potential nep.txt` | 指定模型 |
| `time_step` | `time_step <fs> [<interval>]` | 积分步长（fs）与采样间隔 |
| `ensemble` | `ensemble nve` / `nvt_lan T T tau` / `nvt_bdp` / `npt_scr` … | 系综 |
| `run` | `run <steps>` | 运行步数（可多次出现） |

### 4.2 自旋

| 关键字 | 语法 | 说明 |
|---|---|---|
| `spin_llg` | `spin_llg gamma=3.0385 alpha=0.1 T=300 seed=1 scheme=geometric` | 启用 SLLG。`alpha=0 T=0` → 确定性保守分支 |
| | 可选 `restart=spin_restart.bin` | 从检查点恢复（含 RNG） |
| `add_bfield` | `add_bfield Bx By Bz [omega_THz]` | 外磁场（eV/ $\mu_B$）；`omega>0` 时为正弦驱动（FMR） |
| `magfield` | `magfield <H> <nx> <ny> <nz> [unit=tesla\|ev] [g=<f>] [wave=static\|sin\|cos] [freq=THz] [phase=rad]` | **V4**。LAMMPS `fix precession/spin zeeman` 风格；能量 $-\sum_i \mathbf S_i\cdot\mathbf B$ 计入 `potential_per_atom`。默认 `g=1`（自旋列已是 $\mu_B$，`gamma` 已含 g=2）， $B[\mathrm{eV}/\mu_B]=5.7883818\times10^{-5}H[\mathrm T]$。必须写在 `spin_llg` 之后 |
| `magelec` | `magelec <cut_A> <kappa> <Ex> <Ey> <Ez>` | **V4**。LAMMPS `pair_style spin/magelec`： $H=\sum_{i<j}\kappa(\mathbf E\times\mathbf r_{ij})\cdot(\mathbf S_i\times\mathbf S_j)$， $\kappa$ 单位 eV/(Å· $\mu_B^2$) |
| `magelec_pair` | `magelec_pair <ti> <tj> <kappa>` | **V4**。按元素对覆盖 $\kappa$（0-based 类型索引） |
| `minimize` | `minimize <sd\|fire> <ftol> <steps> [box_change] [hydrostatic] [spin=on\|off\|auto] [spin_tol=] [alpha_damp=] [discrete_factor=]` | **V4**。`spin=auto`（默认）在读过 `spin_llg` 时把磁矩一并最小化；收敛需 **力** 与 **横向力矩** 同时达标 |
| `spin_substeps` | `spin_substeps <n>` | 每个 MD 步内做 n 个自旋子步 |
| `spin_torque_refresh` | `spin_torque_refresh on/off` | 子步之间是否只刷新力矩（省一次全力计算） |
| `spin_split` | `spin_split strang` 等 | 分裂方式（默认对称 Strang） |

### 4.3 输出与测量

| 关键字 | 语法 | 输出 |
|---|---|---|
| `dump_thermo` | `dump_thermo <interval>` | `thermo.out`：T, KE, PE, virial… |
| `dump_xyz` | `dump_xyz <interval> <file> [props]` | 轨迹；props 可取 `potential velocity force spin torque virial unwrapped_position` |
| `dump_restart` | `dump_restart <interval> [file]` | 检查点；自旋部分写进 `spin_restart.bin` |
| `dump_magnetization` | `dump_magnetization <interval> [file] [warmup N]` | `magnetization.out`：`Mx My Mz |M| varM chi Msat` |
| `dump_heat_flux` | `dump_heat_flux <interval>` | `heat_flux.out`，11 列（见提示） |
| `compute_hac` | `compute_hac <interval> <max_lag> <num_corr>` | `hac.out`；**自旋体系上不可用**（见 §8） |
| `compute_sed` | `compute_sed <interval> <num_q> <x\|y\|z> [velocity\|spin] [0\|1]` | 谱能量密度 |
| `compute_dos` | `compute_dos ...` | 声子 DOS |
| `minimize` | `minimize fire` / `minimize sd` | 能量最小化（见 §5.1） |

`dump_xyz` 的属性名必须与内核支持的一致；写错会报
`Unknown dump_xyz property`，不会静默忽略。

---

## 5. 常用工作流

### 5.1 晶格常数（静态扫描）

```bash
for a in 2.75 2.775 2.80 2.825 2.85 2.875 2.90; do
  # 生成 16 原子 bcc 胞（用 tools/physics_validation/make_fe_system.py）
  python -c "import sys;sys.path.insert(0,'tools/physics_validation');import make_fe_system as G;\
fr=G.bcc(2,2,2,$a);s,_=G.set_spins(fr,'fm',2.2,1);G.write_xyz('model.xyz',fr,s)"
  printf 'potential nep.txt\ntime_step 0.001 1\nensemble nve\nspin_llg gamma=3.0385 alpha=0 T=0\ndump_thermo 1\nrun 1\n' > run.in
  gpumd.exe > /dev/null
  echo "$a $(head -1 thermo.out | awk '{print $3}')"
done
```

把 7 个点做二次拟合取极小，得到 $a_0$。R18 的 Fe 模型给 **2.8258 Å**（实验 2.866，−1.4%）。
一条命令版：`python tests/physics_checks.py --bin src --quick`（含 P1）。

### 5.2 基态磁矩

固定 $a=a_0$，扫 $|\mathbf S|\in\lbrace 2.0,2.2,2.4,2.6,2.8\rbrace$，取 $E$ 极小。
R18 给 **2.413 $\mu_B$**（实验 2.22，+8.7%）。

### 5.3 居里温度（推荐用 `dump_magnetization`）

```bash
# 432 原子 bcc，每个温度 1 ps
printf 'potential nep.txt\ntime_step 0.05 100\nensemble nvt_lan 900 900 100\n\
spin_llg gamma=3.0385 alpha=0.3 T=900 seed=7\ndump_magnetization 200 magnetization_900.out warmup 5\n\
seed 7\nrun 20000\n' > run.in
gpumd.exe > stdout.txt
```

`magnetization.out` 的 `|M|` 列直接给磁化强度，`chi` 列给磁化率
（ $=(\langle\mathbf M\cdot\mathbf M\rangle-\langle\mathbf M\rangle\cdot\langle\mathbf M\rangle)/(Nk_BT)$，单位 $\mu_B^2$/eV）。

**判据**： $|M|(T)$ 在 $T_c$ 附近陡降， $\chi$ 在 $T_c$ 出现峰。
R18 的两个模型：A（mask17） $\approx950$ K，B（mask31） $\approx850$ K，实验 1043 K。
**误差棒约 ±100 K**（每点 1 ps、432 原子）。

要更准：每点 ≥50 ps、多尺寸（ $L=4,6,8$）做 Binder cumulant 交点。
本项目**没做**这一步（见 `docs/06_SPIN_LLG_FDT_AND_MC.md`）。

### 5.4 保守动力学（无热浴）

```
ensemble nve
spin_llg gamma=3.0385 alpha=0 T=0
```

时间步按 `docs/07_SPIN_LATTICE_INTEGRATION_RESTART.md`：**0.05 fs 参考、0.1 fs 平衡、
0.2 fs 不建议**。Fe 1024 原子 250 K 下漂移速率 4.6e-9 eV/atom/ps。

### 5.5 分段续算

```bash
# 第一段：跑到一半，写检查点
printf '...\ndump_restart 10000\nrun 10000\n' > run.in && gpumd.exe

# 第二段：从检查点继续（模型、gamma、alpha、T、dt、scheme 必须一致）
printf '...\nspin_llg gamma=3.0385 alpha=0 T=0 restart=spin_restart.bin\nrun 10000\n' > run.in && gpumd.exe
```

不符（模型换过、scheme 改过、载荷损坏）会以 **rc=4** 拒绝启动，不会静默重播种。
`tests/smoke.py::T5` 与 `tests/acceptance.py` 的第 5 项都检查"连续 vs 分段逐位相同"。

### 5.6 副本交换

用 `gpumd_replica.exe`，`run.in` 里给多个温度与交换间隔；
`examples/replica/` 有最小可跑例子。

---

## 6. 单位（最容易出错的地方）

| 量 | 单位 | 备注 |
|---|---|---|
| 位置 | Å | |
| 时间步 | **fs** | 内部是 nat（1 nat = 10.1805 fs） |
| 能量 | eV | |
| 力 | eV/Å | |
| 速度（dump） | Å/fs | 内部是 Å/nat，要 ×10.1805 |
| 自旋 $\lVert \mathbf S\rVert$ | $\mu_B$ | |
| 磁梯度/力矩列 | eV/ $\mu_B$ | `torque` 列存的是 $-\partial E/\partial\mathbf S$，不是 $\mathbf S\times\mathbf g$ |
| 外磁场 | eV/ $\mu_B$ | `add_bfield` |
| 动能 | — | $\frac12 m v^2\times1.0364269\times10^{-4}$（ $m$ 用 u， $v$ 用 Å/fs） |
| 热流 | eV/(Å²·ps)？ | **归一化存疑**，见 §8 |

自己从 `dump_xyz` 重算温度时，务必用上表最后两行；漏乘因子会让 KE 差 4 个数量级。

---

## 7. 性能参考（RTX 4060 Ti, sm_89）

| 体系 | 配置 | 速度 |
|---|---|---|
| 1024 原子 Fe | dt=0.05 fs，SLLG + NVT | ≈2.1 ms/步（纯 MD），≈5 ms/步（含两次力） |
| 3456 原子 Fe | 同上 | ≈14 ms/步 |
| 训练 50k 代 | 256 帧，mask17，neuron 30 | ≈40 min |

**I/O 是隐形瓶颈**：每 50 步 dump 一次轨迹会让 20 ps 运行从 2.1 ms/步变成 6.3 ms/步。
建议 `dump_xyz` 间隔 ≥1000 步，`dump_thermo` ≥200 步。

`energy_diagnostic`（P1 双精度能量诊断）会让墙钟 +64%，**只在诊断时打开**。

---

## 8. 已知缺陷与禁止事项（务必先读）

1. **0.2 fs 的耦合能量门禁仍 FAIL**（ $1.02\times10^{-5}>10^{-5}$）。用 0.05 或 0.1 fs。
2. **热流的绝对归一化未验证**：同一条轨迹三个 $\kappa_{zz}$ 相差 $10^{7}$。
   **不要**用 `compute_hac` 或 `dump_heat_flux` 报热导率；`compute_hac` 在自旋体系上
   给出负值。详见 `docs/08_THERMAL_CURRENT_GREEN_KUBO.md`。
3. ~~NVT→NVE 写在同一个 `run.in` 里， $N\ge432$ 会挂死~~ —— **r19 已修复**（F-25）。现在可以写 `ensemble nvt_lan ...` + `run` + `ensemble nve` + `run`，两个块的 `dump_thermo` / `dump_xyz` 会接在同一个文件后面。
   拆成两次 `gpumd.exe` 调用（第一次 dump 末帧，第二次从该帧起跑）。
4. **没有单离子各向异性**：固定晶格整体旋转自旋能量不变， $K_1$ 不可表示。
5. **`seed` 在 `alpha=0 T=0` 的 NVE 里是空操作**：换初态才算换种子。
6. **`thermo.out` 是追加写**：同一目录重跑会得到两段拼接的文件，分析时取最后一块。
7. `dump_heat_flux` 的列顺序：`time, Jtot(3), Jp(3), 标量, Jm(3)`。

---

## 9. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| `RUNAWAY GUARD STOP: lattice temperature excursion (limit 300)` | r17 的 F26 缺陷 | 用 r18（已修）；若仍出现，检查是否用错 `gpumd.exe` |
| rc=4 且提示 restart 不匹配 | 模型/参数/文件损坏 | 确认 `nep.txt` 与检查点同源 |
| 温度爆炸（ $10^5$ K） | 单位错（速度当 Å/nat 用）或 dt 太大 | 用 `velocity 300` 让程序自己采样；dt ≤ 0.1 fs |
| 力全是 0 | 用了非训练 `nep.in` 做预测，架构不匹配 | 用训练时的 `nep.in` |
| `thermo.out` 有两段表头 | 追加写 | 删旧文件或用新目录 |
| 第二段之后文件不再增长 | 使用的是 r18 及更早的二进制（F-25） | 升级到 r19，或拆成两次调用 |
| 训练 loss 不降 | 标签列数不对 / `spin_force` 缺失 | 按 §2.3 检查表头 |
| 能量漂移快 | dt 太大 / 不是保守分支（`alpha>0` 会耗散） | 见 §5.4 |

---

## 10. 相关文档

| 想了解 | 看 |
|---|---|
| 理论推导 | `docs/04_HAMILTONIAN_AND_DESCRIPTOR_THEORY.md` |
| 安装与构建 | `INSTALL_WINDOWS.md`、`BUILD_FROM_SOURCE.md` |
| 开发与扩展 | `DEVELOPER_GUIDE.md` |
| 模型文件格式 | `MODEL_FORMAT.md` |
| 验证状态与限制 | `VALIDATION_REPORT.md`、`KNOWN_LIMITATIONS.md` |
| 复现步骤 | `REPRODUCIBILITY.md`、`examples/` |
