# 07 — 保守自旋—晶格积分与时间步政策

## 7.1 分裂格式

对称（Strang）分裂，一个宏观步 $\Delta t$：

$$
\underbrace{S(\Delta t/2)}_{\text{自旋半步}}\;\underbrace{V(\Delta t)}_{\text{晶格 Verlet}}\;\underbrace{S(\Delta t/2)}_{\text{自旋半步}} .
\tag{7.1}
$$

* $S(\Delta t/2)$：在**固定** $\mathbf R$ 下推进自旋。$\alpha=T=0$ 时是保守中点，
  等价于把 $\mathbf S$ 沿有效场旋转一个角度，能量守恒到求值精度。
* $V(\Delta t)$：速度 Verlet，力取中间态 $(\mathbf R_n,\mathbf S_{n+1/2})$。

为什么这样分：$S$ 与 $V$ 各自的向量场在子流形上可逆且保辛，对称复合给出
二阶、时间可逆、保体积的格式。代价是每步要算两次力。

## 7.2 求力时刻与等时输出

| 时刻 | 状态 | 动作 |
|---|---|---|
| $t_n$ | $(\mathbf R_n,\mathbf S_n)$ | 输出（等时点） |
| $t_n+\Delta t/2$ | $\mathbf S_{n+1/2}$ | 第一个自旋半步 |
| — | $(\mathbf R_n,\mathbf S_{n+1/2})$ | 重算力（$V$ 的 kick 用中间态） |
| $t_{n+1}$ | $(\mathbf R_{n+1},\mathbf S_{n+1/2})$ | 晶格推进 |
| $t_{n+1}$ | $\mathbf S_{n+1}$ | 第二个自旋半步 |
| $t_{n+1}$ | $(\mathbf R_{n+1},\mathbf S_{n+1})$ | **再算一次力/能量**，输出 |

最后一次重算是关键：否则输出帧的力/能量属于中间态，与输出的
$(\mathbf R,\mathbf S)$ 不对应——这是"续算不逐位一致"的常见原因。

S05B 的 C 组门禁用单点工具从 dump 的 $(\mathbf R,\mathbf S)$ 重算
$E,F,g,\text{virial}$，与运行写出的值比：实测 $|\Delta E|\le2.3\times10^{-12}$ eV、
$\Delta F=\Delta g=0$。

## 7.3 能量误差与实测阶

$$
\Delta E_{\rm tot}(t)=E(\mathbf R_t,\mathbf S_t)-E(\mathbf R_0,\mathbf S_0).
\tag{7.2}
$$

分裂误差 $\propto\Delta t^2$。判据：$\max_t|\Delta E|/N\le10^{-5}$ eV/atom，
且把 $\Delta t$ 减半后误差比值的 $\log_2$ 平均 $\ge1.5$。

本版本实测（4 原子强交换场，100 fs）：

| $\Delta t$/fs | $\max|\Delta E|/N$ | 纯晶格对照 | 末态 $|\Delta E|$ |
|---|---|---|---|
| 0.2 | $1.023\times10^{-5}$ | $3.77\times10^{-6}$ | $1.04\times10^{-6}$ |
| 0.1 | $3.18\times10^{-6}$ | $1.44\times10^{-6}$ | $1.03\times10^{-6}$ |
| 0.05 | $1.64\times10^{-6}$ | $1.39\times10^{-6}$ | $9.0\times10^{-8}$ |

误差阶 **1.68**（阈值 1.5 → PASS）；漂移仍超阈值 2.3% → **FAIL**（保留）。

诊断：逐步骤能量抖动 $\approx1.99\times10^{-6}$ eV 与自旋是否演化无关，
它等于 float32 求值地板（Stage04A：对称等价求值间最大差 $1.91\times10^{-6}$ eV/atom）。
所以该项不是分裂误差，但**不能**据此宣布门禁通过——阈值未改，仍记 FAIL。

## 7.4 可逆性与时间对称

对称分裂要求两个半步严格互逆。工程上靠：同一有效场迭代器、同一收敛容差、
同一噪声序列（$T=0$ 时无噪声）。

`restart=` 从 `spin_restart.bin` 恢复 $(\mathbf R,\mathbf S,\mathbf v)$、
每原子 curandState、$\gamma,\alpha,T,\Delta t$、scheme、物理步数与模型哈希。
三组连续 vs 分段重启实测**逐位相同**（$\max|\Delta S|=0$，RNG 768 B 相同），
负向控制（损坏载荷、改 scheme、改模型）全部 rc=4。

## 7.5 时间步政策（r18 正式采用）

历史 0.2 fs 的 S05B FAIL 保留，但**不再作为所有后续研究的统一阻断条件**。
依据 PV1 的 28 算例矩阵：

| 档位 | $\Delta t$ | 依据 |
|---|---|---|
| **参考（推荐）** | **0.05 fs** | Fe 1024 原子、250 K、1–10 ps：$\delta E=5.9\times10^{-8}\to1.9\times10^{-7}$ eV/atom，漂移速率 $4.6\times10^{-9}$ eV/atom/ps |
| 平衡 | 0.1 fs | 同条件 1 ps：$8.4\times10^{-8}$，余量 119× |
| 不建议 | 0.2 fs | 同条件 1 ps：$1.22\times10^{-7}$（比 0.05 高 54%）；强场小胞直接失守 |
| 只做诊断 | 0.025 fs | 与 0.05 同量级（$9.0\times10^{-8}$），无收益，慢一倍 |

适用边界写死为：**Fe / `mask17` / $\alpha=0$ / NVE / $T\approx250$ K / $\le10$ ps**。
外推依据是"漂移速率恒定"（5 ps 与 10 ps 的斜率一致到 3%），
由此 $10^{-5}$ 门禁约在 **2.2 ns** 才到线。**不得声称对其它材料或更长时长已验证。**

## 7.6 代码入口

* `src/main_gpumd/run.cu`：S–V–S 调用顺序、两次 `compute_force()`
* `src/spin/spin.cu::Spin::propagate`（半步长 $=\Delta t/2$ 由调用方传入）
* `src/measure/dump_restart.cu`：重启文件写出
* 验证脚本：`tools/physics_validation/run_lt.py`、`analyze_lt.py`

## 7.7 限制

* 只验证 NVE 晶格 + 保守自旋。NPT/热浴组合的重启未序列化。
* 0.2 fs 的耦合能量门禁仍 FAIL，因此**热输运结论仍被阻断**（11、12 篇）。
* NVT→NVE 写进同一个 `run.in` 在 $N\ge432$ 会挂死（F-25），必须拆成两次调用。
