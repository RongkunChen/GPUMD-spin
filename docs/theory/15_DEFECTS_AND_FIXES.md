# 15 — 缺陷日志：症状、根因、修复与验证

本篇只记录**已经定位到根因**的缺陷。每条都给出"如何复现"和"如何验证修好了"，
因为"看起来好了"在数值代码里是最不可靠的判据。

编号 F 系列从 r17 开发期开始，跨版本连续编号；r18 新增 F26、F27。

| 编号 | 组件 | 症状 | 根因 | 状态 |
|---|---|---|---|---|
| F19 | R19 runaway guard | DFT 快照起步的 MD 被误判为"势能崩塌" | 参考值在未弛豫构型上 priming | 已修 |
| F20 | R19 guard | 固定晶格 + 自旋动力学被误判为晶格温升 | 没有"冻结晶格"识别 | 已修 |
| F21 | R19 guard | 能量注入判据随机触发 | 判据定义错误（$w_{\rm net}$ 被阻尼主导） | 已禁用 |
| F22 | 测试工具 | 二聚体初态 $\vert \mathbf S_1\vert =1.16619$ | AoS 数组被按 SoA 读 | 已修（测试侧） |
| F23 | 构建 | `main_replica` 10 处 `nep_acc_t undefined` | typedef 不在副本可见的头里 | 已修 |
| F24 | 训练 | 磁矩通道分辨率低 | `spin_mag_scale` 被写死 0.25 | 已修（改为数据驱动） |
| F25 | 积分器/主循环 | NVT→NVE 同文件切换在 $N\ge432$ 挂死 | 未定位（进程存活但无输出） | **未修**，有绕法 |
| F26 | R19 guard | NVE 下 $T\ge300$ K 一律被判"晶格温升"并停机 | `Integrate::temperature` 未初始化，读到 denormal | 已修 |
| F27 | `Integrate` | 同上（隐患） | 三个温度成员无初值 | 已修（初始化为 0） |

---

## F19 — guard 在未弛豫构型上 priming

**症状**：用 DFT 快照作初态跑 Fe NVE，前 ~1000 步被报
`potential-energy collapse` 并停机。

**根因**：guard 在 step 0 采样 $\mathbf R$ 的势能作为参考。DFT 快照相对模型
极小值有 $0.1$–$0.5$ eV/atom 的落差，前 1000 步的弛豫是**合法**的能量下降，
被当成失稳。

**修法**：参考值改为从"已弛豫"的构型 priming（`spin_guard_primed` 标志），
并在 priming 时重设 `spin_guard_reference_energy`。

**验证**：同一 Fe 快照重跑，前 1000 步不再触发；人为注入 $+5$ eV 仍能触发。

---

## F20 — 冻结晶格被误判

**症状**：`ensemble nve` + 固定原子（或 $T=0$ 无初速）时，自旋动力学正常，
但 guard 报"晶格温度 excursion"。

**根因**：晶格温度为 0，任何数值抖动都让 $T_{\rm lat}$ 相对变化无穷大。

**修法**：检测 `frozen_lattice`，切换为 spin-only 模式：关闭晶格温度与
能量守恒判据，重新 priming 参考值。

**验证**：`examples/spin_llg/run.in.conservative` 在 4 原子上跑 10 万步不触发；
`tests/smoke.py::T2` 覆盖。

---

## F21 — 能量注入判据被禁用（不是修好，是删掉）

**症状**：某些算例随机报"energy injection"。

**根因**：判据用 $w_{\rm net}/|w_{\rm guard}[0]|>1.05$ 且 $w_{\rm net}/{\rm atom}>0.02$ eV。
但 $w_{\rm net}$ 被"阻尼项向噪声场松弛"主导（$\sigma_{\rm SLLG}\sim19$ rms$|B_\perp|$），
它与 $W_{\rm rot}$ 的 2048 次抵消后剩下的残差是 $\mathcal O(dS^2)$ 离散伪迹。
Fe 3456 原子实测：$w_{\rm net}=-2.34\times10^{3}$ eV，而总能量只动了 $+5.9$ eV。

**决策**：**删除判据**，而不是调阈值。删掉的是一个定义错误的量，
不是放宽一个正确的门禁。判据 (3)(4)（直接读 `thermo.out`）保留。

**验证**：删除后所有历史算例不再误触发；人为破坏（把力乘 2）仍由判据 (4) 抓到。

---

## F22 — 二聚体初态被按 AoS 读成 SoA

**症状**：S08A 的二聚体 $|\mathbf S_1|=\sqrt{1.36}=1.16619$、$|\mathbf S_2|=0.8$。

**根因**：`spin` 在 xyz 里按 `[S1x S2x][S1y S2y][S1z S2z]`（SoA）排列，
测试生成器按 `[S1x S1y S1z][S2x S2y S2z]`（AoS）填充。

**修法**：按 SoA 填充，并在积分前加 $|\mathbf S|=1$ 断言（失败 rc=4）。

**验证**：断言在错误填充下立即失败；正确填充下 8 种子全部通过。

**教训**：这条是**测试侧**的 bug，不是生产代码的 bug。它一度被误读成
"生产积分器不保模长"。

---

## F23 — `main_replica` 编译失败

**症状**：`nep_acc_t undefined` 出现 10 次。

**根因**：`main_replica/` 是源码副本，用父树的
`spin_descriptor_kernels.cuh`，但有自己的 `nep_utilities.cuh`。
typedef 放在后者里，前者看不到。

**修法**：把 typedef 移进 `spin_descriptor_kernels.cuh`，加
`#ifndef NEP_ACC_T_DEFINED` 守卫，两处都能看到且不重复定义。
链接时排除 `main_gpumd/main.obj`（否则重复 `main()`）。

**验证**：三个可执行文件全部构建成功；`BUILD.ps1` 返回 0。

---

## F24 — `spin_mag_scale` 写死

**症状**：磁矩通道（Smag）的有效分辨率低，训练出的基态磁矩偏大
（r17 的 2.60 $\mu_B$ vs 实验 2.22）。

**根因**：`nep.in` 里手写 `spin_mag_scale 0.25`，而 Fe 训练集
$\max|\mathbf S|=4.27\,\mu_B$。$u={\rm scale}\cdot|\mathbf S|$ 因此只落在
$[0,1.07]$，网络在 $u\approx1$ 附近的分辨率被压缩。

**修法**：`nep.in` **不写**该行，训练器按 $1/\max|\mathbf S|=0.234148785$
自动定标（数据驱动）。

**验证**：r18 两个 50k 模型基态磁矩 2.413 / 2.432 $\mu_B$（旧 2.60），
DEV 能量 10.38 / 8.75 meV/atom（旧 15.23）。
注意：这部分改善来自**重训**，不是来自 F26/F27 补丁——补丁已证明逐位不改动力学。

---

## F25 — NVT→NVE 同文件切换在 $N\ge432$ 挂死（未修）

**症状**：`run.in` 里先 `ensemble nvt_lan` 跑一段，再 `ensemble nve` 跑一段，
在 $N=432$ 与 $N=1024$ 上各复现一次：进程存活、CPU 占用、26 分钟无任何输出。

**定位**：两次独立复现都在第二个 `run` 块的 6000 步附近。
$N=128$ 不复现；拆成两次 `gpumd.exe` 调用（第一次 dump 末帧、第二次从该帧起跑）
完全正常。

**状态**：**r19 已修复**。根因不在积分器，而在 measure：

```cpp
void Measure::post_run(...) {
  for (auto& action : actions_) action->post_run(...);
  actions_.clear();          // <-- 第一个 run 块结束就把所有 measure 删了
}
```

第二个 `run` 块于是只做积分：`thermo.out` / `traj.xyz` / `restart.xyz` 都不再更新，
而 stdout 仍在打印进度（重定向到文件时被块缓冲，肉眼更看不到）。"进程存活、CPU 占用、
文件不动"被读成了挂死；$N$ 越大单块跑得越久、越容易被提前判死，所以阈值看上去在
$N\approx432$——实际上是"等多长时间"的问题，不是 $N$ 的物理阈值。

修复（r19，`src/measure/measure.cu` + `dump_thermo.cu[h]`）：
`post_run()` 不再清空，dump 类 action（`dump_thermo`/`dump_xyz`/`dump_exyz`/
`dump_netcdf`/`dump_restart`）跨段保留，`pre_run()` 以 `"a"` 重新打开文件；
`dump_thermo` 用 `header_written_` 保证同一次进程只写一次表头。`compute_*` 仍在段末
总结并丢弃（累加器按段定义，静默续用会把两个 ensemble 平均到一起）。

验收（`validation/BCC_Fe_full_physics/raw_data/nvt_nve_switch/`）：

| 检查 | 结果 |
|---|---|
| $N$=250/432/1024，NVT 3000 + NVE 12000，`dump_thermo 100` | 150/150 行（旧版 31 行） |
| 同上 + `dump_xyz 500` | 18/18 帧 |
| 单块 NVT / NVE 各 2000 步，新旧二进制 | `thermo.out` **逐字节相同** |
| 单文件双段 vs 两次调用（重启文件接力） | 末态 T/PE/KE **完全相同** |
| 第二段 60000 步 | 105 s 正常结束 |

**这条缺陷的教训**：一个"没有输出"的进程既可能是死锁，也可能是"没人再写文件"。
先看代码里谁负责写，再怀疑积分器。

---

## F26 — guard 的限温恒为 300 K（r18 的主要修复）

**症状**：任何 `ensemble nve` 且 $T\ge300$ K 的运行，在一定步数后被停机：

```
RUNAWAY GUARD STOP: lattice temperature excursion (limit 300.000 K)
```

即使初温是 200 K 或 600 K，limit 都打印 300.000。

**根因（两层）**：

1. `Integrate::temperature` 在 NVE 下**从未赋值**，是未初始化的 `double`。
   发布版二进制里它恰好读到 **6.9514519871473713e-310**（denormal）；
2. 旧代码 `if (!(target_temperature > 0.0))` 用"是否为正"做回退判据，
   而 denormal 是**正数**，回退分支永不执行。于是
   `max(1.5*t, t+300)` 在 $t\approx7\times10^{-310}$ 时塌成常数 **300 K**。

调试打印（临时插桩，已移除）：

```
[DBG-F26] step=1000 T=309.339 raw=6.9514519871473713e-310 T_target=0.000 T_limit=300.000
```

**修法（两处，互补）**：

* `main_gpumd/run.cu`：回退链改成 `>1.0` 判据，逐级回退到
  primed 参考温度 → 本次实测温度 → 300 K；
* `integrate/integrate.cuh`：`temperature1/2/temperature` 初始化为 `0.0`（F27），
  从根上消除未定义读。

顺带修掉两个同源的槽位错误（同一段代码里）：

* PE 原来取 `thermo[1]`（那是**动能**），改为 `thermo[2]`；
* 总能量原来是 `thermo[1] + 1.5 N k_B T` = $2\times$KE，改为 `thermo[2] + thermo[1]`。

**验证（三重）**：

1. 同一输入、同一模型，修复前后 `thermo.out` 与 `traj.xyz` 的 sha256 **完全相同**
   → 修复不动物理；
2. 历史误杀算例（`evidence/stage_PV1/fe_lt/cases/dt_0.2`）：
   旧版 rc=2、停在第 1000/5000 步；新版 rc=0、跑满 5000 步，limit 609.3 K；
3. `tests/physics_checks.py::P5`：600 K 起步的 NVE 2000 步，rc=0、无 RUNAWAY 标记。

---

## F27 — `Integrate` 的三个温度成员无初值

见 F26 第 2 条。单独列出是因为它是一类**通用的**缺陷模式：
"成员没初值 → 读到 denormal / 垃圾 → 所有 `if (x > 0)` 型保护失效"。

**修法**：

```cpp
double temperature1 = 0.0;
double temperature2 = 0.0;
double temperature  = 0.0;
```

**检查清单**（给后续开发者）：任何参与判据的浮点成员都必须有初值；
任何用"是否非零"做回退的代码，都要考虑"非零但是垃圾"的可能。

---

## 附：构建期问题（不是物理缺陷）

| 问题 | 现象 | 修法 |
|---|---|---|
| PowerShell 执行策略 | `.\BUILD.ps1` 被拒（Restricted） | 提供 `BUILD.cmd`，仅对本进程 `Bypass` |
| `git apply` 换行 | 补丁把 LF 写成 CRLF | `git -c core.autocrlf=false apply` |
| `read_bytes` 少取地址 | 重启文件读错 | 全部改 `&name` |
| `GPU_Vector::copy_to_host` 非 const | 成员函数编译失败 | 成员加 `mutable` |
| `thermo.out` 是追加写 | 重跑目录得到两段拼接的文件 | 运行前截断输出文件；分析取最后一块 |
| `dump_xyz velocity` 单位 | 误当 Å/nat 用，KE 大 1e4 倍 | 见 01 篇 §1.5 单位表 |
| Python 的 `/tmp` 与 Git Bash 不同 | 补丁往返验证失败 | 用相对路径 / Windows 可见路径 |
