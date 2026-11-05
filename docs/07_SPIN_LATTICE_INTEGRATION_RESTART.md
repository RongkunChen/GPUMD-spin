# 07 — 自旋–晶格耦合积分与完整重启

## 1. Strang 分裂的时序（生产实现）

`spin_split strang`（默认）在每个 MD 步内执行：

```
      S(dt/2)          V(dt)                    S(dt/2)
  ─────────────►  ───────────────►  ─────────────────────►
  A   S: S_n → S_{n+1/2}      场取 (r_n, S_n)
  B   F(r_n, S_{n+1/2})       第一次 VV kick
  C   VV: 半 kick + drift + 半 kick   r_n → r_{n+1}
  D   F(r_{n+1}, S_{n+1/2})   第二次 VV kick（与 C 的半步构成完整 V）
  E   （VV 的第二半 kick）
  F   S: S_{n+1/2} → S_{n+1}  场取 (r_{n+1}, S_{n+1/2})
  G   F(r_{n+1}, S_{n+1})     等时输出状态（thermo/xyz/restart 都读它）
```

要点：
- 每次求力对应一个**真实存在**的 $(R,S)$ 组合，没有“用上一步的力凑”；
- 步末再求一次力，保证 `thermo`、`dump_xyz`、`restart.xyz`、磁化输出读的是**同一状态**；
- 代码：`src/main_gpumd/run.cu` L384–L470；`compute_force()` 在 L312/L426 被调用。

历史遗留的 `spin_split staggered` 是**一阶**耦合（VV kick 用上一步的自旋），只为复现 pre-S05B 结果保留，选择时会打印警告。

## 2. 能量、动量与误差阶（必须实测，不靠推断）

保守情形（ $\alpha=T=0$）的判据：

| 量 | 门槛 | 历史实测（S05B-R1，4 Fe 原子） |
|---|---|---|
| $\max\lVert \Delta E\rVert /\text{atom}$， $\Delta t=0.2$ fs | $1.0\times10^{-5}$ eV | **$1.0673\times10^{-5}$（FAIL）** |
| 同， $\Delta t=0.1$ fs | — | $4.3321\times10^{-6}$ |
| 同， $\Delta t=0.05$ fs | — | $1.6758\times10^{-6}$ |
| 误差阶 $p=\log_2(\Delta E_{0.2}/\Delta E_{0.1})$ | $\ge1.5$ | $\approx1.30$ |
| 固定晶格（A 组） | PASS | PASS |
| 等时单点（C 组） | PASS | PASS |

> **当前版本状态：`COUPLED_NOT_VALIDATED`。** 本轮未重跑该 harness（见 `VALIDATION_REPORT.md`），因此
> 耦合热流（`08`）与耦合输运一律 BLOCKED；不得用“能量差很小”替代门禁。

解析玩具模型（两原子磁弹性）用于机制反例： $E=-J\ \mathbf S_1 \cdot \mathbf S_2+\frac{k}{2}(r-r_0)^2+\lambda(r-r_0)(\mathbf S_1 \cdot \mathbf S_2)$。
在该模型上可以直接算出精确能量守恒，用来判断生产分裂是否「结构正确」而非仅「数值小」。

## 3. Checkpoint 格式（`spin_restart.bin`）

`restart.xyz` 是 xyz 文件，只能装 $(\mathbf R,\mathbf v,\mathbf S)$。磁自由度的另外两块状态**不在其中**：

1. **每原子 curand 状态**：热噪声是**有状态**随机流，第 $k$ 步之后的位置无法由 `seed` 反推；
   重新 seed 会从第 0 步重放——分布对、轨迹错。
2. **每原子 $|\mathbf S|$ 锚点**：约束投影的参考值。

二进制布局（小端，逐字段 `fwrite`）：

| 偏移 | 类型 | 含义 |
|---|---|---|
| 0 | `char[16]` | magic `GPUMDSPINRST01` |
| 16 | `u32` | schema 版本（=1） |
| 20 | `u32` | natoms |
| 24 | `u64` | `nep.txt` 的 FNV-1a 哈希（0 = 文件不存在） |
| 32 | `f64 ×4` | gamma, alpha, temperature, dt_fs |
| 64 | `i32 ×3` | scheme, has_rng, has_anchor |
| 76 | `i64 ×2` | sllg_steps, physics_step |
| 92 | `f64` | global_time |
| 100 | `f32[n]` | 每原子 $\lVert \mathbf S\rVert$ 锚点 |
| … | `u8[n×48]` | 每原子 `curandState`（`sizeof=48`） |
| 末 8B | `u64` | 前面全部字节的 FNV-1a 载荷校验和 |

写入是**原子式**（`*.tmp` + `rename`），崩溃不会留下半个文件。读取时校验：magic、版本、natoms、
载荷校验和、`nep.txt` 哈希、 $\gamma/\alpha/T/\Delta t$/scheme；任一不符**直接 `exit(4)`**，不静默接受。

## 4. 重启语义与验证

run.in：

```
spin_llg gamma=3.04 alpha=0.5 T=300 seed=4242 scheme=midpoint restart=spin_restart.bin
dump_restart 1000
```

三段式验证（`tests/smoke.py` T5 + 阶段3 完整版）：

```
cont    : run 2000, dump_restart 2000      → 轨迹 A + 末态 checkpoint
prefix  : run 1000, dump_restart 1000      → 轨迹 B + 中途 checkpoint
restart : cp prefix/restart.xyz model.xyz  ; cp prefix/spin_restart.bin .
          spin_llg ... restart=spin_restart.bin ; run 1000
```

判据（实测结果）：

| 量 | 结果 |
|---|---|
| $\max\lVert \Delta \mathbf S\rVert$（float32 逐位） | **0**（三者全同） |
| $\max\lVert \Delta \mathbf R\rVert$、 $\max\lVert \Delta \mathbf v\rVert$ | **0** |
| $\max\lVert \Delta \mathrm{PE}\rVert$、 $\max\lVert \Delta \mathrm{KE}\rVert$ | **0** |
| 末态 RNG 块（768 B = 16×48） | **逐字节相同** |
| 损坏载荷 / 改 scheme | 均 `rc=4` 拒绝 |
| 去掉 `restart=` 只重播种 | $\max\lVert \Delta\mathbf S\rVert =5.68\ \mu_B$（说明测试有分辨力） |

> midpoint 求解器是**步内**迭代，没有跨步中间状态，因此 checkpoint 不需要保存「半步阶段」。

## 5. 兼容矩阵

| 场景 | 支持 | 说明 |
|---|---|---|
| $\alpha=T=0$（保守） | ✅ 逐位 | 无 RNG，只有 R/v/S |
| geometric $T>0$ | ✅ 逐位 | RNG 必须续接 |
| midpoint $T>0$ | ✅ 逐位 | 同上 |
| NPT / 热浴链 | ❌ | 恒压器/热浴内部状态不在本文件内；本轮未验证 |
| 跨构建（不同编译器/架构） | ⚠️ | 只能要求容差一致，不能要求逐位 |
| 改变 `nep.txt` | ❌ | 哈希不符直接拒绝 |
| 改变 dt/scheme/ $\gamma/\alpha/T$ | ❌ | 直接拒绝 |

## 6. 代码入口

| 功能 | 文件 |
|---|---|
| 分裂时序 | `src/main_gpumd/run.cu`（L384 起） |
| checkpoint 写 | `src/measure/dump_restart.cu` + `Spin::save_restart_state` |
| checkpoint 读 | `src/spin/spin.cu` → `Spin::load_restart_state`（在 `check_discretisation` 中触发） |
| 解析 | `Spin::parse` 的 `restart=` |

## 7. 本版本状态

| 项 | 状态 |
|---|---|
| 重启逐位一致（三组） | **PASS** |
| 负例拒绝（损坏/错模型/错 scheme） | **PASS** |
| 保守耦合能量门禁 B | **FAIL**（1.067e-5 > 1e-5） |
| NPT/热浴链重启 | **NOT_VALIDATED** |
