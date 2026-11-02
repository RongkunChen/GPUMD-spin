# 04 — Hamiltonian 与自旋描述符理论

本篇给出 $E(\mathbf R,\mathbf S)$ 的完整定义、五个自旋块的公式、对称性分类，以及每一条公式对应的生产代码位置。
记号： $\mathbf r_{ij}=\mathbf r_j-\mathbf r_i$， $\hat{\mathbf r}_ {ij}=\mathbf r_{ij}/|\mathbf r_{ij}|$，
$\mathbf S_i$ 单位 $\mu_B$，长度量 $\text{\AA}$，能量 eV。

## 1. 局域能量模型

$$
E(\mathbf R,\mathbf S)=\sum_{i=1}^{N} e_i\big(\lbrace \mathbf r_{ij}\rbrace _{j\in n(i)},\lbrace \mathbf S_i,\mathbf S_j\rbrace _{j\in n(i)}\big),
\qquad n(i)=\lbrace j: |\mathbf r_{ij}|<R_c\rbrace .
$$

- 代码入口：`src/force/nep.cu`（MD/GPU 路径）、`src/main_nep/structure.cu`（trainer 路径）。
- 半径基 $R_c=6.0$ Å、角度基 $R_c'=3.0$ Å（`nep.txt` 的 `cutoff` 行）。
- 局域性保证了 $e_i$ 只依赖截断内的邻居 → 力与磁梯度都是**解析局域导数**，不需要自动微分。

## 2. 结构（非磁）描述符

径向（两体）：

$$
q^{2}_{i,n}=\sum_{j\in n(i)} g_n(r_{ij}),\qquad
g_n(r)=\sum_{k} c^{n}_{k}\ B_k \left(\tfrac{r-R_c}{R_c}\right),
$$

角度（三体，Chebyshev 型）：

$$
q^{3}_{i,n,l}=\sum_{j<k\in n(i)} g_n(r_{ij})\ g_n(r_{ik})\ P_l(\cos\theta_{ijk}).
$$

## 3. 五个自旋块

设 $\mathbf u_i=\mathbf S_i/|\mathbf S_i|$（ $|\mathbf S_i|>0$）， $\tilde{\mathbf S}_i=s\ \mathbf S_i$， $s=$`spin_mag_scale`。

| 块 | 定义 | 变量 | 对称性 |
|---|---|---|---|
| **Sex** | $q^{\rm ex}_ {i,n}=\sum_{j\in n(i)} g_n(r_{ij})\ (\mathbf u_i \cdot \mathbf u_j)$ | 1 | 旋转不变（联合/自旋单独均可）、P-even、T-even |
| **San** | $q^{\rm an}_ {i,n,l}=\sum_{j<k} g_n g_n\ P_l(\hat{\mathbf r}_ {ij} \cdot \hat{\mathbf r}_ {ik})\ (\mathbf u_i \cdot \hat{\mathbf r}_ {ij})(\mathbf u_i \cdot \hat{\mathbf r}_ {ik})$ | $l$ 通道 | 旋转不变、P-even |
| **Sdm** | $q^{\rm dm}_ {i,n}=\sum_{j\in n(i)} g_n(r_{ij})\ \hat{\mathbf r}_ {ij} \cdot (\mathbf u_i\times\mathbf u_j)$ | 1 | **P-odd**、T-even、仅联合旋转不变 |
| **Sia** | $q^{\rm sia}_ {i,n,(ab)}=\sum_{j\in n(i)} g_n(r_{ij})\left(u_{i,a}u_{i,b}-\tfrac13\delta_{ab}\right)$ | 5（对称无迹） | 旋转不变、P-even、 $l=2$ |
| **Smag** | $q^{\rm mag}_{i,n}=\big(s\vert \mathbf S_i\vert \big)^{n+1}$ | $n=0..$ | 旋转不变、P-even、原点不可微 |

旧布局的 **Senv** 占据 slot 4，是 T-odd 通道；无外场/无破 T 物理时不应无条件启用，v3 已被 `Sia` 取代（`spin_layout 3`）。

### 3.1 为什么 Sdm 是 P-odd 而 $E$ 不必是 P-odd

反演： $\mathbf r_{ij}\to-\mathbf r_{ij}$（极矢量）， $\mathbf S\to+\mathbf S$（轴矢量）。于是
$\hat{\mathbf r} \cdot (\mathbf u_i\times\mathbf u_j)\to-\ \hat{\mathbf r} \cdot (\mathbf u_i\times\mathbf u_j)$。
所以**单个 Sdm 通道**在反演下变号。但总能量是这些通道经 ANN 的非线性组合：
$e_i=\mathcal F(\lbrace q\rbrace )$， $\mathcal F$ 一般不是奇函数，因此 $E$ 在反演下**既不自动偶也不自动奇**。
只有 DMI 型能量 $\propto \mathbf D_{ij} \cdot (\mathbf S_i\times\mathbf S_j)$（ $\mathbf D_{ij}\propto\hat{\mathbf r}_{ij}$）才是整体 P-odd 的物理项。
**因此：不能用“Sdm 是 P-odd 描述符”推断“模型表达了 DMI”，也不能推断“能量满足宇称”。**

### 3.2 Sia 的能力边界

$\left(u_a u_b-\tfrac13\delta_{ab}\right)$ 是 $l=2$ 的四极矩。对严格立方局域环境，
$\sum_{j}g_n(r_{ij})$ 权重下的一阶矩各向同性 → 该块**恒为零**。
而 bcc Fe 的单离子各向异性主导项是 $K_1(S_x^4+S_y^4+S_z^4)$，属 $l=4$。
**结论：Sia 无法表示 bcc 的四阶立方各向异性；要表示必须新增 $l=4$ 描述符并新 layout + 重训。**

### 3.3 Smag 的可微性

$q^{\rm mag}_n=(s|\mathbf S|)^{n+1}$。对 $n=0$： $q=s|\mathbf S|$，在 $\mathbf S=0$ 处方向导数依赖方向 → 不可微。
生产核在 $|\mathbf S|\ge 10^{-8}\ \mu_B$ 用精确导数

$$
\frac{\partial q_n}{\partial \mathbf S}=(n+1)\ (s|\mathbf S|)^{n}\ s\ \frac{\mathbf S}{|\mathbf S|},
$$

在 $|\mathbf S|<10^{-8}$ 处写 0 —— 这是**约定**，不是导数。代码：`src/utilities/spin_descriptor_kernels.cuh`（三处实现一致），常量 `SPIN_MAG_MIN_MOMENT`。

## 4. 掩码与布局

```
spin_layout 3 Sex San Sdm Sia Smag      # 顺序固定，核按此顺序拼接
spin_blocks Sex Smag                    # mask 17：只有两块的权重被训练/读取
spin_mag_scale 0.25
```

- 位：`Sex=1, San=2, Sdm=4, slot4=8, Smag=16`；mask17 = 1+16，mask31 = 全开。
- **掩码必须由 `nep.txt` 显式给出**：用全开掩码去评估只训练了两块的模型，会读到四组未训练权重。
- 代码：`src/force/nep.cu`（解析）、`src/main_replica/force/nep.cu`（v3.0.0 同步）、
  `src/utilities/spin_descriptor.cuh`（声明式对称/能力表，MD 与 trainer 打印同一份）。

## 5. 对称性核对表（模型级，不是描述符级）

| 变换 | mask17 | mask31 | 检验方式 | 本版本结果 |
|---|---|---|---|---|
| 整体自旋旋转（无 SOC） | 必须不变 | 不应不变（Sia/Sdm 破各向同性） | 旋转所有 $\mathbf S$ 比较 $E$ | mask17： $\Delta E=0.0$ eV（`VALIDATION_REPORT.md`） |
| 空间反演 | $E$ 不变（无 DMI） | 视权重而定 | 反演 $\mathbf r$、保持 $\mathbf S$ | NOT_VALIDATED |
| 时间反演 | $E$ 偶 | $E$ 偶（Senv 除外） | $\mathbf S\to-\mathbf S$ 后 $E$ 比较 | NOT_VALIDATED |

> 注意：只有**完整势能**的对称性才是物理结论；描述符分类只是必要条件。

## 6. 数值算法（求值顺序）

1. 邻居表：`src/force/neighbor.cu`（cell list，半邻域 + 对称化）。
2. 描述符：径向 → 角度 → 自旋五块，按 layout 顺序拼成 `q[dim]`（`dim=54`，见 `nep.txt` 头）。
3. 标度：`q_scaled = q * scaler + shift`（每分量）。
4. ANN：`54-30-1`，隐层 `tanh`，输出 $e_i$（`src/force/nep.cu` 的 `find_force_spin`）。
5. 力/磁梯度：在**同一** kernel 内由 $\partial q/\partial \mathbf r$、 $\partial q/\partial \mathbf S$ 链式求和。

## 7. 验收反例（必须能复现的负例）

- **Sia 恒零**：立方环境 + 单轴应变 → Sia 块从 $\sim10^{-8}$ 跳到 $0.357$（阶段06 记录）。
- **Sdm 宇称**：对称排布 $|q|\sim3.3\times10^{-8}$，非对称排布 $1.3\times10^{-2}$。
- **Smag 原点**： $|\mathbf S|=10^{-10}$ 时导数按约定为 0，有限差分不收敛 → 这是预期行为。

## 8. 本版本状态

| 项 | 状态 |
|---|---|
| 五块公式与生产核一致（MD/特殊化/JIT 三处） | PASS（阶段06） |
| `spin_blocks` 解析（主树 + replica） | PASS（v3.0.0） |
| 整体旋转不变性（mask17） | PASS（ $\Delta E=0$） |
| 反演/时间反演（模型级） | NOT_VALIDATED |
| mask31 的 SOC/DMI/K1 物理含义 | **不可宣称**（仅表示能力） |

## 9. 描述符求和的数值精度（2026-10-31 修订）

**数学定义。** 逐原子描述符是邻居上的求和：

$$
q^{i}_{n}=\sum_{j\in\mathcal N(i)} g_n(r_{ij})\ \Phi_c(r_{ij}),\qquad
s^{i}_{abc}=\sum_{j\in\mathcal N(i)} \hat r_{ij}^{\ a}\hat r_{ij}^{\ b}\hat r_{ij}^{\ c}\ g_n(r_{ij})\Phi_c(r_{ij}),
$$

其中 $g_n$ 是 Chebyshev 径向基乘系数， $s$ 再组合成角向 $q$。数学上求和顺序无关。

**物理假设。** 无。这一节不涉及模型形式，只涉及同一实数的求值方式。

**数值算法。** $\mathcal N(i)$ 的枚举顺序由实现决定：训练器按镜像三重循环
（`main_nep/dataset.cu` 的 `ia,ib,ic`）生成邻居表并在 `find_descriptors_radial/angular`
里顺序累加；MD 按 cell list 顺序在 `force/nep.cu` 的 `find_descriptor` 里累加。
float32 每次加法都重新舍入， $N_{\rm nb}\sim 10^2$ 项时可差 3–4 ulp。
$\lvert q\rvert\sim10$ 时这是 $5\times10^{-6}$，经 $J=\partial e/\partial q$ 放大成
$10^{-6}$ eV 的能量差——正好是冻结的逐原子门禁的量级。

**现在的做法。** 累加器类型 `nep_acc_t = double`，循环结束后一次性窄化：
$q^{\rm stored}_ n={\rm fl}_ {32}\big(\sum_j \cdots\big)$。一次确定性舍入与顺序无关，
两条路径在 $10^{-15}$ 相对精度上一致。

**代码入口。** `utilities/nep_utilities.cuh`（`nep_acc_t`、`accumulate_s`、`find_q_one` 模板）、
`force/nep.cu`、`main_nep/nep.cu`、`main_nep/nep_specialized.cu`、
`utilities/spin_descriptor_kernels.cuh`。

**单位。** 与描述符同（无量纲的基函数乘系数）；累加精度不影响单位。

**验收反例（必须能失败）。** Stage03A 冻结的 5 个 case，逐元素判据
$\lvert e^{\rm cand}_i-e^{\rm ref}_i\rvert\le10^{-6}+10^{-6}\lvert e^{\rm ref}_i\rvert$。
修复前 3/5 FAIL（最差 $1.192\times10^{-6}$ 对阈值 $1.055\times10^{-6}$）；
修复后 5/5 PASS（最差 $7.15\times10^{-7}$）。详细表在 `VALIDATION_REPORT.md` §2。

**本版本测试状态。** PASS。残余 $7\times10^{-7}$ 来自训练器坐标存 float32 而 MD 存 double，
不是累加顺序问题；要消掉它必须把训练器坐标改成 double（改变整条流水线的内存与格式），
未做，已在 `VALIDATION_REPORT.md` §3 记录。

**影响范围。** 存下来的 $q$ 仍是 float32，`q_scaler`、ANN 宽度、`nep.txt` 布局不变，
所以**旧模型无需重训**：同一个 `nep.txt` 的能量在 $10^{-6}$ eV 量级上更准确，含义不变。
附带效应：耦合能量守恒的误差阶从 1.30 回到 1.68（门禁 1.5），漂移
$1.067\times10^{-5}\to1.023\times10^{-5}$ eV/atom（门禁 $10^{-5}$，仍差 2.3%）。
