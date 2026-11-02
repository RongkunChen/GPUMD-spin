# 05 — 原子力、磁梯度与 virial

## 5.1 定义与符号

$$
\mathbf F_i=-\frac{\partial E}{\partial \mathbf R_i},\qquad
\mathbf g_i=-\frac{\partial E}{\partial \mathbf S_i},\qquad
\boldsymbol\tau_i=\mathbf S_i\times \mathbf g_i .
\tag{5.1}
$$

* $\mathbf F_i$：eV/Å
* $\mathbf g_i$：eV/$\mu_B$
* $\boldsymbol\tau_i$：eV（力矩，进 LLG 的 $\mathbf S\times\mathbf H$）

符号约定与 GPUMD 一致：`dump_xyz` 的 `torque` 列存的是 $-\partial E/\partial\mathbf S$，
**不是** $\mathbf S\times\mathbf g$。名字容易误解，真正的力矩要自己叉乘。
SLLG 内核里也是用 $\mathbf g$ 而不是力矩，因为 LLG 的右端是
$-\gamma\,\mathbf S\times\mathbf g$。

## 5.2 链式法则与反向累加

$$
\frac{\partial e_i}{\partial R_i^a}=\sum_{d}\frac{\partial e_i}{\partial q_d}\frac{\partial q_d}{\partial R_i^a},
\qquad
\frac{\partial e_i}{\partial S_i^a}=\sum_d\frac{\partial e_i}{\partial q_d}\frac{\partial q_d}{\partial S_i^a}.
\tag{5.2}
$$

$\partial q/\partial R$ 的累加是"反向"的：对每一对 $(i,j)$ 先算
$\partial q/\partial \mathbf r_{ij}$，再按 $-\hat{\mathbf r}_{ij}$ 给 $i$、
$+\hat{\mathbf r}_{ij}$ 给 $j$。这保证：

* 牛顿第三定律 $\sum_i\mathbf F_i=0$ 到机器精度；
* 平移不变（只依赖 $\mathbf r_{ij}$），因此在 PBC 下与原点选择无关；
* 旋转协变（$\partial q/\partial\mathbf R$ 是矢量）。

自旋通道的导数不需要额外的对称化：每个 $\mathbf S_i$ 只属于原子 $i$，
$\partial q/\partial\mathbf S_i$ 是单边贡献。

**计算复杂度**：一次描述符求值给出 $q$，反向传播的代价约为前向的 2–3 倍。
GPU 实现把 $\partial q/\partial\mathbf S$ 与 $\partial q/\partial\mathbf R$ 合并到同一个内核，
共享邻居表（`find_force_spin` / `find_force`）。

## 5.3 virial

`dump_xyz virial` 输出 9 分量，逐原子按 $\mathbf r\otimes\mathbf f$ 累加，
顺序为

```
xx xy xz  yx yy yz  zx zy zz
```

自旋块的贡献来自 $\partial q/\partial\mathbf R$ 里与 $\hat{\mathbf r}$ 有关的部分
（`San`、`Sdm`、`Sia` 都含 $\hat{\mathbf r}$）。`Sex` 与 `Smag` 不含
$\hat{\mathbf r}$，因此对 virial 无直接贡献。

训练侧只存 6 分量（`xx yy zz xy yz zx`）。比较时必须**按名称映射**，
不能假设 $xy=yx$：非平衡构型下 $xy\neq yx$（差一个 $\mathcal O(\Delta t)$ 的
反对称部分），把 9 分量硬塞进 6 分量会在训练中引入系统性偏差。

## 5.4 Maxwell 互易与混合 Hessian

能量是 $C^2$ 标量函数，因此

$$
\frac{\partial F_i^a}{\partial S_j^b}=\frac{\partial g_j^b}{\partial R_i^a}
\quad\Longleftrightarrow\quad
\frac{\partial^2 E}{\partial R_i^a\,\partial S_j^b}
=\frac{\partial^2 E}{\partial S_j^b\,\partial R_i^a}.
\tag{5.3}
$$

要小心两件事：

1. 混合 Hessian $C_{i\alpha,j\beta}=\partial^2E/\partial R_{i\alpha}\partial S_{j\beta}$
   是一个 $3N\times3N$ 的**矩形块**（不是对称矩阵）：$C\neq C^{\mathsf T}$ 在
   一般情况下成立，因为 $R$ 与 $S$ 是两套不同自由度。互易性说的是
   "$-\partial\mathbf F/\partial\mathbf S$ 与 $-\partial\mathbf g/\partial\mathbf R$
   **是同一个矩阵**"，而不是 $C=C^{\mathsf T}$。
2. 逐元素相对误差在**小元素**上必然发散（$10^{-6}$ 量级的元素配 $10^{-9}$ 的
   绝对误差就是 1e-3 相对误差）。因此判据只能用 Frobenius 范数，
   不能用 max-relative。

## 5.5 有限差分的分辨率陷阱

用能量中心差分验证 $\mathbf g$ 时误差预算为

$$
\delta g\approx\frac{\text{ulp}(E)}{2h}+\frac{h^2}{6}|E'''| .
\tag{5.4}
$$

$E\approx105$ eV 时 float32 ulp 为 $7.6\times10^{-6}$ eV；取 $h=10^{-3}\,\mu_B$
得第一项 $3.8\times10^{-3}$ eV/$\mu_B$，而典型 $g_x$ 只有 $0.04$ → 相对误差 10%。
把 $h$ 减半会让第一项翻倍，所以"减小步长"在这里是**反效果**。

**结论**：$\mathbf g$ 的差分验证必须换更高精度的能量参考（float64 求值器）
或换 $|g|$ 大的分量。PV1 阶段因此专门写了主机端 float64 参考求值器
`tools/numerical_accuracy/nep_double_ref.cpp`，把 $\mathbf g$ 的对比精度
从 $10^{-2}$ 提到 $2.6\times10^{-5}$。

另一条更实用的判据：比较**解析梯度与对同一模型的独立实现**，
而不是与差分比较。参考求值器是同一模型的第二个实现（同一常数表、
同一描述符定义，用 double 计算），因此它检验的是"实现有没有写错"，
而不是"模型对不对"。

## 5.6 实测

| 量 | 参考 | 相对差 | 阈值 | 状态 |
|---|---|---|---|---|
| $\mathbf F$ vs $-\partial E/\partial R$（float64 参考） | 32 原子 bcc + 数据集帧 | $3.0\times10^{-6}$ eV/Å | $10^{-5}$ | **PASS** |
| $\mathbf g$ vs $-\partial E/\partial S$（float64 参考） | 同上 | $2.6\times10^{-5}$ eV/$\mu_B$ | $10^{-4}$ | **PASS** |
| 混合 Hessian $C$（Frobenius） | 32 原子，中心差分 $h=10^{-3}$ | $1.3\times10^{-3}$ | $10^{-2}$ | **PASS** |
| Maxwell 互易（逐元素 1%） | 4/16 原子 | 小元素不成立 | — | **不可用判据**（见 §5.4） |
| $\mathbf g$ vs 能量中心差分（float32） | 4 原子 | $10^{-2}$ | $10^{-3}$ | **FAIL**（判据无分辨力，见 §5.5） |

最后一行是历史 FAIL，保留在 `FINAL_TEST_MATRIX.csv` 里，不因为换了更好的
判据就删掉。

## 5.7 固定模长下只用切向分量

SLLG 保持 $|\mathbf S_i|$ 不变，因此纵向分量 $\mathbf g_\parallel=\hat{\mathbf S}_i(\hat{\mathbf S}_i\cdot\mathbf g_i)$ 不产生运动：
$\mathbf S\times\mathbf g$ 对 $\mathbf g$ 的纵向部分恒为零。实现上仍然
计算完整的 $\mathbf g$（训练需要它），但积分时只用到切向部分。
后果：**纵向磁刚度不被动力学检验**，只被静态能量检验（P2 的 $E(|S|)$ 扫描）。

## 5.8 代码入口

* `src/force/nep.cu::NEP::compute`：$\mathbf F$、$\mathbf g$、virial 落盘
* `src/utilities/spin_descriptor_kernels.cuh`：自旋块对 $\mathbf S$、$\mathbf R$ 的导数
* `src/force/force.cu::Force::set_energy_diagnostic`：P1 双精度能量诊断的转发
* `tools/numerical_accuracy/nep_double_ref.cpp`：float64 参考求值器
  （`--mixed-matrix` 输出 $C$，`--dump-descriptor` 输出描述符）
