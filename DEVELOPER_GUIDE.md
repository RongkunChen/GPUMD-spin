# 开发者指南（GPUMD-spin V3, r18）

面向要改代码、加功能或做审计的人。读之前建议先读
`docs/theory/05`（导数结构）、`docs/theory/14`（精度）、`docs/theory/15`（缺陷日志）。

---

## 1. 代码结构

```
src/
  main_gpumd/     gpumd 主程序；run.cu 是控制流（含 S–V–S 分裂、R19 guard）
  main_nep/       nep 训练/预测；nep.cu（通用）、nep_specialized.cu（JIT 特化）、dataset.cu
  main_replica/   副本交换 MD（源码副本 + 共享部分的 -I 路径）
  main_gnep/      图形化前端（本交付未构建）
  force/          力/能量/磁梯度的生产者；nep.cu 是主体，force.cu 是分发
  integrate/      系综与积分器；ensemble*.cu、thermo.cu、integrate.cu
  spin/           自旋动力学：SLLG、保守中点、重启 I/O、R19 guard 的 spin 侧
  measure/        所有 dump_* / compute_* 测量模块（43 个）
  model/          Atom/Box/Group 数据容器、xyz 读写
  mc/             蒙特卡洛（自旋与晶格）
  minimize/       能量最小化（SD / FIRE）
  phonon/         声子相关（静态部分）
  utilities/      描述符内核、邻居表、GPU 容器、错误处理、单位常数
  lib/            第三方（cusolver 等封装）
```

规模：384 个 `.cu/.cuh`，约 15.1 万行。

### 1.1 一次 MD 步的调用链

```
Run::compute()                      main_gpumd/run.cu
 ├─ Spin::propagate(dt/2)           spin/spin.cu        （S 半步）
 ├─ Force::compute(atom, box, ...)  force/force.cu      （重算 F, g, virial）
 ├─ Integrate::integrate(...)       integrate/ensemble_*.cu（V 整步）
 ├─ Spin::propagate(dt/2)           （S 半步）
 ├─ Force::compute(...)             （等时点重算，保证输出与状态一致）
 ├─ Measure::post_force(...)        measure/measure.cu
 └─ Spin::check_runaway(...)        spin/spin.cu → R19 guard
```

**两条不变的约定**：

1. 每个宏观步内**力被算两次**（中间态一次、末态一次）。删掉第二次会让
   `dump_xyz` 的力与输出的 $(\mathbf R,\mathbf S)$ 不对应，重启不再逐位一致。
2. `Measure::post_force` 之前必须完成第二次力计算，否则所有测量看到中间态。

---

## 2. 构建

### 2.1 Windows（推荐）

```bat
BUILD.cmd -Arch sm_89 -Jobs 8 -Log build.log
```

`BUILD.ps1` 会：找 `nvcc`（PATH → `CUDA_PATH` → 常见安装根）→ 用 `vswhere`
定位 `vcvars64.bat` → 探测 GPU 的 compute capability（未给 `-Arch` 时）→
`make -j` → 检查三个 exe 是否都存在。返回码非 0 表示失败。

**执行策略**：`BUILD.cmd` 只对当前进程 `Bypass`，不改系统设置。

### 2.2 Linux

```bash
./build.sh            # 或 cd src && make -j8
```

需要 `nvcc`、`g++`、`make`；GPU 架构由 `makefile` 的 `GENCODE` 控制。

### 2.3 从零验证构建（发布流程必做）

```bash
unzip GPUMD-spin-V3-SOURCE-v3.1.0-r18.zip -d /tmp/fresh
cd /tmp/fresh/GPUMD-spin-V3-SOURCE-v3.1.0-r18
BUILD.cmd -Arch sm_89 -Jobs 8
python tests/smoke.py --bin src
python tests/acceptance.py --bin src
```

发布包里的 `FINAL_TEST_MATRIX.csv` 与 `fresh_build/` 就是这条流程的产物。

---

## 3. 加一个 `run.in` 关键字

以 `dump_magnetization` 为例，完整改动是 3 处：

1. **新建模块** `src/measure/my_dump.cu` + `.cuh`，模仿
   `dump_magnetization.cu`：构造函数里 `parse(tokens)`，实现
   `pre_run/post_force/finalize`，`action_name` 设为关键字名。
2. **注册**：`src/measure/measure.cu` 的关键字分支里加
   `else if (tokens[0] == "my_dump") { my_dump = std::make_unique<MyDump>(tokens); }`，
   并在 `pre_run/post_force/finalize` 的分发处调用它。
3. **编译**：`src/makefile` 里 `OBJ` 列表加 `measure/my_dump.obj`；
   `main_replica/makefile` 若共用该模块也要加。

**注意**：`measure/` 下的模块共享 `Atom`、`Box`、`Group`、`Force` 的只读视图。
需要新的物理量（例如自旋通量）时，先在 `Atom` 里加字段并在
`run.cu` 里填好（参考 `atom.spin_dynamics`、`atom.spin_gamma`、`atom.spin_flux_A`），
不要让测量模块自己去 `dynamic_cast`。

---

## 4. 加一个自旋描述符块

1. `src/utilities/spin_descriptor.cuh` 的声明表加一行：名称、掩码位、
   rank、时间反演、空间反演、是否需 SOC、退化条件。
2. `spin_descriptor_kernels.cuh`：在 `find_descriptor` 的
   small-box 与 large-box 两个变体里都实现前向与对 $\mathbf S$、 $\mathbf R$ 的导数。
   **两个变体必须给出一致的结果**（`tests/smoke.py` 不含这一项，需要自己写
   16 原子 vs 1024 原子的对比）。
3. `main_nep/nep.cu`、`nep_specialized.cu`：同步训练器侧，保证标签与梯度一致。
4. 掩码位与 `spin_blocks` 字符串的映射加进 `nep.in` 解析。

**必须做的验证**（缺一不可）：

* 与 float64 参考求值器 `tools/numerical_accuracy/nep_double_ref.cpp` 对比
  （相对差 < $10^{-6}$）；
* 对称性：该块的 rank/parity 与声明表一致（构造对称/反对称作测试）；
* 小胞/大胞一致；
* `spin_blocks` 里加/不加该块，其它块的数值不变。

---

## 5. 精度策略（不要随手改）

| 位置 | 类型 | 能不能改 |
|---|---|---|
| 描述符累加 `nep_acc_t` | double | **已定**，循环内 double、出循环窄化一次 |
| 描述符存储 `q[]` | float32 | 不能改：改了旧模型全废 |
| ANN 权重 | float32 | 不能改：`nep.txt` 格式 |
| 坐标/速度/力 | double | 已是 double |
| 自旋/力矩 | float32 | 改它需要重写 ~800 行求导链（P3，已 BLOCKED） |
| 能量聚合 `sum_pe` | double | 已定 |

要动精度，先读 `docs/theory/14` §14.8 的决策表：P2/P3/P4 的重启条件
（当前一条都不满足）。**不要**为了"看起来更准"把某处单独改 double——
混合精度会在相邻实现之间引入 $10^{-6}$ 量级的差，正好落在门禁阈值上，
会让所有历史对比失效。

---

## 6. 测试

| 脚本 | 内容 | 用时 |
|---|---|---|
| `tests/smoke.py` | 7 项代码路径（NEP 单点、固定晶格、随机自旋、二聚体、重启、副本…） | ~1 min |
| `tests/acceptance.py` | 训练 → 预测 → MD → SLLG → 重启，全链路 | ~3 min |
| `tests/physics_checks.py` | 晶格常数 / 基态磁矩 / Tc / 能量漂移 / guard 回归 | 3 min（`--quick`）/ 15 min |
| `tools/physics_validation/run_lt.py` | 长时间稳定性矩阵（dt/尺寸/时长/种子） | 数十分钟 |
| `tools/physics_validation/run_heat.py` | 热流与 GK 前置审计 | 数十分钟 |

约定：

* 一个测试只有"跑通"不算 PASS，必须检查**数值**；
* 缺前置条件的测试记 `SKIP`，**不允许**记 PASS；
* 历史 FAIL 保留在矩阵里，换更好判据时新增一行，不要删旧行。

---

## 7. 改代码的工作流

```bash
git -c core.autocrlf=false apply my.patch     # 必须带这个开关
```

* 源码 `run.cu` 等是 **CRLF**，`integrate.cuh` 是 **LF**。改动时保持原行尾，
  否则补丁会把整文件标成改动。
* 生成补丁时保留目标文件的行尾：

```python
a = open(old, 'rb').read().decode().splitlines(keepends=True)
b = open(new, 'rb').read().decode().splitlines(keepends=True)
open(out, 'w', newline='').write(''.join(difflib.unified_diff(a, b, ...)))
```

* 改完立刻跑 `smoke.py`；如果碰了力/描述符，再跑
  `tools/numerical_accuracy/run_A_magnetic_gradient.py`。
* 提交前更新：`CHANGELOG.md`、`docs/theory/15_DEFECTS_AND_FIXES.md`、
  `VALIDATION_REPORT.md`。

---

## 8. 调试经验（都是踩过的坑）

| 症状 | 第一个该查的东西 |
|---|---|
| 结果随运行次数变 | 输出文件是追加写；先截断再跑 |
| 能量不守恒 | 是否 `alpha>0`（耗散系综本来不守恒）；dt 是否 ≤0.1 fs |
| 温度离谱 | 速度单位（Å/fs vs Å/nat）；KE 因子 `1.0364269e-4` |
| 力与参考差很多 | 预测用的 `nep.in` 架构是否与训练一致 |
| 重启不逐位一致 | 是否漏了第二次力计算；RNG 状态是否入盘 |
| guard 乱报 | 看 `docs/theory/15` 的 F19/F20/F21/F26；判据语义比数值更容易错 |
| 第二段不产数据 | F-25（r19 已修复）：`Measure::post_run` 曾清空 actions_ |
| 编译报 `nep_acc_t undefined` | typedef 必须在 `spin_descriptor_kernels.cuh` 且带守卫 |
| `main_replica` 链接重复 `main` | 排除 `main_gpumd/main.obj` |

### 8.1 一个方法论上的教训

本项目出现过三次"看起来是物理问题、其实是判据问题"：

1. guard 的限温（F26）——判据把动能当势能、回退条件用了"是否为正"；
2. 热流连续性（11 篇）——判据在 PBC 下不成立，含无自旋对照也失败；
3. 梯度差分（05 篇）——判据的分辨率低于噪声。

三次的共同特征：**判据失败时先怀疑判据**，用"同一物理量的第二条独立路径"
（参考求值器、无自旋对照、解析模型）去交叉验证，再决定是不是代码错。

---

## 9. 性能调优

* 力计算占 90% 以上；`spin_torque_refresh on` 可以省掉子步之间的全力计算。
* I/O：`dump_xyz` 间隔 <200 步时变成 I/O bound（实测 2.1 → 6.3 ms/步）。
* `energy_diagnostic`（P1）让墙钟 +64%，只用于诊断。
* 邻居表重建频率由 `time_step` 的第二个参数控制；高温体系要更频繁。

---

## 10. 未完成的功能（想接手的看这里）

| 功能 | 难点 | 起点 |
|---|---|---|
| 磁子谱 | 切空间投影 + $6N\times6N$ 复本征 | `docs/theory/09` §9.2、§9.6 |
| Goldstone 零模计数 | 同上 | 同上 |
| 热流归一化钉死 | 需要解析模型（LJ 晶体）对照 | `docs/theory/11` §11.6 |
| NVT→NVE 第二段无输出（F-25） | r19 已修复（dump action 跨段保留） | `docs/theory/15` |
| P3 全 double 求导链 | ~800 行、与训练器共用 | `docs/theory/14` §14.8 |
| Binder cumulant / 有限尺寸标度 | 需要多尺寸 + 长采样 | `docs/theory/08` |
