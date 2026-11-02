# 14 — 数值精度、误差来源与适用边界

## 14.1 精度分层

| 层 | 类型 | 说明 |
|---|---|---|
| 文件读入 | double | `model.xyz`、`train.xyz` 坐标按 double 解析 |
| MD 坐标 / 速度 / 力 | double | `g_x`、`g_v`、`g_f` |
| 自旋 / 力矩状态 | **float32** | 唯一仍是单精度的物理状态 |
| 描述符累加 | **double**（`nep_acc_t`） | r17 起从 float32 改成 double，见 §14.2 |
| 描述符存储 | float32 | `q[]` 与 `nep.txt` 格式不变（旧模型无需重训） |
| ANN 前向 | float32 | `w0`、`b0`、`w1` |
| 能量聚合 | double | `sum_pe` |
| 训练器坐标 | **float32** | `main_nep/structure.cu` 的 `float* x` |
| 训练器邻居向量 | float32 | `x[n2] + delta - x1` 全 float 运算 |

`nep_acc_t` 的定义放在 `utilities/spin_descriptor_kernels.cuh`（而不是
`nep_utilities.cuh`），因为有 `#ifndef NEP_ACC_T_DEFINED` 守卫：
`main_replica/` 是源码副本，用父树的描述符内核但自己的 `nep_utilities.cuh`，
typedef 必须能被两边看到。

## 14.2 累加顺序误差（已修）

逐原子描述符是邻居求和。训练器（镜像三重循环）与 MD（cell list）走**不同顺序**，
float32 每次加法都舍入，$N_{\rm nb}\sim10^2$ 项后可差 3–4 ulp。
$|q|\sim10$ 时 $\Delta q\sim5\times10^{-6}$，经 $J=\partial e/\partial q$ 放大成
$10^{-6}$ eV 的 $e_i$ 差。

修法：`nep_acc_t = double`，循环内 double 累加，出循环窄化一次。

效果：Stage03A 逐原子门禁从 3/5 FAIL（$1.192\times10^{-6}$）变成
5/5 PASS（最差 $7.153\times10^{-7}$）。混合判据 PASS；纯绝对 $10^{-6}$ 判据
仍 4/5，历史 FAIL 保留在矩阵里。

## 14.3 残余误差（未修）

修完累加后仍有 $7\times10^{-7}$ eV 量级残差。已定位：

* 训练器邻居向量用 **float32 算术**：`x[n2] + delta - x1`；MD 用 double 相减再窄化。
  对最差 case 的原子，328 个周期像对的向量差到 $9.5\times10^{-7}$ Å；
* 这个 $\Delta\mathbf r$ 让 $q$ 变 $10^{-6}$，进而 $e_i$ 变 $10^{-7}$。

**一次被证伪的假设**：曾认为"训练器坐标存 float32"是主因，设计了实验
（把 fixture 坐标 cast 成 float32 再跑 MD 探针），结果 `md32 - md = 0`——
实验**无效**，因为 MD 探针的位置来自 bundle 而不是 xyz。该结论已撤回。

要把残差降到 $10^{-8}$，必须把训练器坐标与邻居向量改成 double。这是流水线级
改动（内存布局、邻居表、kernel 签名），没有物理内容，**r18 未做**（P3 `BLOCKED`）。

## 14.4 有限差分的分辨率地板

中心差分验证梯度时

$$
\delta(\text{grad})\approx\frac{\text{ulp}(E)}{2h}+\frac{h^2}{6}|E'''| .
\tag{14.1}
$$

实测（$E_0=105.286$ eV，float32 ulp $7.6\times10^{-6}$ eV，$g_x=0.040$）：

| $h$ | 相对差 |
|---|---|
| $10^{-3}$ | $1.40\times10^{-2}$ |
| $3\times10^{-4}$ | $2.96\times10^{-2}$ |
| $10^{-4}$ | $1.88\times10^{-1}$ |

误差随 $h$ 减小而**增大** → 噪声主导。这条门禁在 float32 能量输出下不可达。

**结论（r17 完成）**：不要试图用差分验证梯度，改用 float64 参考求值器
`tools/numerical_accuracy/nep_double_ref.cpp`。它把 $\mathbf g$ 的对比精度
从 $10^{-2}$ 提到 $2.6\times10^{-5}$，并且是"同一模型的第二实现"，
检验的是实现而不是模型。

## 14.5 能量求值地板与长时间行为

Fe 1024 原子、250 K、$\Delta t=0.05$ fs，用 P1 双精度能量诊断直接测同一轨迹：

| 量 | P0（float32 能量） | P1（double 能量） |
|---|---|---|
| 0.2 fs，$\delta E_{\max}$ | $1.0201\times10^{-5}$ | $1.0105\times10^{-5}$ |

P1 只把漂移降了 1%（1.05%），**说明 0.2 fs 超限的 85% 是积分截断，不是 float32**。
这是 R2 阶段最重要的归因结论：换精度救不了时间步。

纯晶格对照（同一 $\Delta t$ 序列）：$3.38\to0.81\to0.24\to0.073\times10^{-6}$，
比值 $\approx4$，标准二阶。耦合体系在 $\Delta t\le0.05$ 出现
$\propto\sqrt{N}$ 的自旋扇区地板：$1.22\times10^{-6}$@2000 步 →
$1.58\times10^{-6}$@4000 步（$\sqrt2=1.41$ 的量级）。

代价：P1 让墙钟增加 **64%**（4.98 → 8.16 s）。因此 P1 只作为**诊断开关**
（`energy_diagnostic` 关键字 → `energy_diag.out`），默认生产路径逐字节不变。

## 14.6 长时间漂移速率（PV1，最有用的一个数）

Fe `mask17`、250 K、$\Delta t=0.05$ fs、$\alpha=0$、NVE、1024 原子：

| 时长 | $\max\vert \Delta E\vert /N$ | 末态 $\vert \Delta E\vert /N$ | 后半段斜率 (eV/atom/ps) |
|---|---|---|---|
| 1 ps | $5.94\times10^{-8}$ | $+2.86\times10^{-8}$ | 1 ps 太短，不可靠 |
| 5 ps | $9.02\times10^{-8}$ | $+8.95\times10^{-8}$ | $+4.74\times10^{-9}$ |
| 10 ps | $1.87\times10^{-7}$ | $+1.44\times10^{-7}$ | $+4.58\times10^{-9}$ |

5 ps 与 10 ps 的**斜率一致到 3%** → 漂移是匀速累积，不是加速失稳。外推：

$$
\dot E_{\max}\approx4.6\times10^{-9}\ \mathrm{eV/atom/ps}
\quad\Longrightarrow\quad
\delta E=10^{-5}\ \mathrm{eV/atom}\ \text{需}\ \approx2.2\ \mathrm{ns}.
\tag{14.2}
$$

适用范围写死为 Fe / `mask17` / $\alpha=0$ / NVE / 250 K；不得外推到其它材料。

## 14.7 误差预算（Fe 单点）

| 来源 | 量级（eV/atom） |
|---|---|
| ANN float32 前向 | $10^{-8}$ |
| 描述符累加（double 后） | $10^{-12}$ |
| 训练器邻居向量 float32 | $10^{-7}$ |
| 能量输出格式 | $10^{-9}$ |
| **总** | $\sim10^{-7}$ |

这解释了为什么"混合判据"（$10^{-6}+10^{-6}|e|$）能过而"纯绝对判据"（$10^{-6}$）
还差一点。

## 14.8 双精度重构的决策（R2-F）

| 模式 | 内容 | 决策 |
|---|---|---|
| P0 | 当前生产路径 | **接受为基线** |
| P1 | 双精度能量诊断（默认关闭） | **保留**，仅诊断 |
| P2 | 自旋/力矩改 double | **暂停**：原子位置/速度/力本就是 double，只剩 spin/torque 是 float32，收益未证 |
| P3 | 整条求导链改 double | **BLOCKED**：约 800 行 float32 专用代码，`get_f12_*` 与训练器共用 |
| P4 | 训练器全 double | **BLOCKED**：同上，且无物理必要 |

解除条件（当前一条都不满足）：(a) 有物理结论被证明受 float32 限制；
(b) P3 有可维护的实现路径；(c) 有第三方复现需求。

## 14.9 适用边界

* 单点能量在 $10^{-7}$ eV/atom 内可信；
* 力在 $10^{-5}$ eV/Å 内可信；
* 长时间守恒受**求值地板**限制：逐步骤抖动 $\sim2\times10^{-6}$ eV，与积分器无关；
* 任何依赖 $<10^{-7}$ eV 差异的物理结论（例如微小各向异性能、$K_1$）在本精度下**不可信**；
* 时间步政策见 07 篇 §7.5：0.05 fs 参考、0.1 fs 平衡、0.2 fs 不建议。
