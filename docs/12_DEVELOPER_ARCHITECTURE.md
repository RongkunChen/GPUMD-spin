# 12 — 开发者架构

## 1. 目录与职责

```
src/
  main_gpumd/      gpumd 主程序：run.cu 负责 run.in 分发与 S-V-S 分裂时序
  main_nep/        nep 训练器：structure.cu（数据集/邻居）、nep.cu（损失/优化）、parameters.cu
  main_replica/    独立编译的 REMD/PRD 驱动（有自己的 force/model 副本）
  main_gnep/       多目标训练（本版本未用于 Fe）
  spin/            spin.cu(h)：LLG、FDT 噪声、中点求解、锚点投影、checkpoint
  force/           nep.cu(h)：生产 NEP 求值（GPU），含自旋五块与 virial
  measure/         dump_restart.cu（xyz + spin_restart.bin）、compute_heat.cu、dump_xyz.cu …
  integrate/       ensemble_*.cu、thermo.cu
  model/           atom/box/group/read_xyz
  utilities/       spin_descriptor.cuh（声明式对称表）、spin_descriptor_kernels.cuh（三处共享核）、
                   gpu_vector、run_input、error
```

## 2. 依赖方向（简化）

```
run.cu ──► Integrate ──► Ensemble(…) ──► Force::compute ──► NEP kernels
   │                                          ▲
   ├──► Spin::propagate ──────────────────────┘ (SpinFieldProvider 回填 g)
   └──► Measure::end_of_step ──► Dump_Restart ──► Spin::save_restart_state
```

- `Force::compute(..., spin, torque)` 是唯一的力/矩来源；`Spin` 通过 `SpinFieldProvider` 在**中点**回填场。
- `Atom` 持有 `spin_gamma/spin_alpha/spin_dynamics`：measure 类只能看到 `Atom`，这是刻意的最小耦合。

## 3. 数据布局（最易错处）

| 缓冲 | 布局 | 说明 |
|---|---|---|
| `atom.spin` | **SoA**：`[Sx(N)][Sy(N)][Sz(N)]` | 写 `model.xyz` 时按原子交错（AoS）输出 |
| `atom.position_per_atom` | SoA | 同上 |
| `q_spin` | 按 layout 顺序拼接 | `dim_spin=25`（5 块 × 5 分量） |
| virial | MD 9 分量 `xx yy zz xy xz yz yx zx zy` | trainer 6 分量，按名映射 |

> 初态 SoA 写错一格，就会得到“|S1|=1.166”这类幽灵（S08A 的真实事故）。

## 4. 编译链

| 目标 | 入口 | 产物 |
|---|---|---|
| gpumd | `src/makefile` → `SOURCES_GPUMD` | `gpumd(.exe)` |
| nep | 同上 `SOURCES_NEP` | `nep(.exe)` |
| gpumd_replica | `make -C main_replica`（子 makefile） | `gpumd_replica(.exe)` |

- Windows 需要 `OS=Windows_NT`（对象后缀 `.obj`、cl 主机编译器）；`-arch=sm_XY` 必须匹配 GPU。
- `main_replica` 是**独立编译的代码副本**：它有自己的 `force/nep.cu`。v3.0.0 修了它的
  `-I..` 包含路径与 `ParaMB::spin_block_mask`。**长期风险：副本会与主树漂移**（见 `13`）。

## 5. 新增一个测量量的步骤

1. 在 `measure/` 建 `xxx.cu(h)`，继承对应基类，实现 `parse/pre_run/end_of_step/post_run`；
2. 在 `measure/measure.cu` 的关键字表注册；
3. 若需要自旋状态，从 `atom.spin/torque` 读，**不要**在 measure 里重新求力；
4. 加进 `makefile` 的 `SOURCES_GPUMD`；
5. `tests/smoke.py` 加一条真实断言。

## 6. 新增一个描述符块的步骤（破坏性）

1. `utilities/spin_descriptor.cuh` 加对称/能力声明；
2. `spin_descriptor_kernels.cuh` 加值/导数（三处调用点共用）；
3. `force/nep.cu` 与 `main_nep/nep.cu` 的解析与 `dim_spin` 计算；
4. **升 layout 版本 + 重训**：旧权重的含义变了，不能静默复用。

## 7. API（C++）

```cpp
Spin spin;
spin.parse({"spin_llg", "gamma=3.04", "alpha=0.5", "T=300", "seed=1",
            "scheme=midpoint", "restart=spin_restart.bin"});
spin.initialize(N, /*has_spin=*/true);
spin.check_discretisation(dt_fs);           // 这里校验并加载 checkpoint
spin.propagate(dt, S, torque, N, stream, &provider);
// 诊断
spin.sllg_pre_projection_norm_error();
spin.save_restart_state("spin_restart.bin", step, time);
```

## 8. Python 接口

`tools/` 下的脚本只依赖 `numpy`：`check_sllg_stage08a.py`（SLLG/MC/精确求积）、
`phase4_train.py`（训练/评估）、`tests/smoke.py`（安装验收）。
它们不链接 CUDA，只用可执行文件与文本输出，便于在别的机器复现。

## 9. 版本约定

- `VERSION` = 上游 r17；本项目 tag = `v3.0.0-r17`。
- 补丁：`ALL_CHANGES.patch` 是相对上游 r17 的**全部**改动（含本轮 S08B 与 replica 修复）。
- 二进制摘要只标识单次构建（嵌入构建时间），代码身份看补丁 + 源码 SHA。
