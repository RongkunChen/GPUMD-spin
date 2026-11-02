# 09 — 声子、磁子、磁声耦合与寿命

> 状态：**框架与解析检验就绪；Fe 的大规模色散/寿命 `NOT_RUN_MODEL_GATE`**（缺发布级模型）。
> 本文件给公式、算法、代码映射与验收反例，不声称已有 Fe 数值结果。

## 1. 线性响应量

$$
\Phi_{i\alpha,j\beta}=\frac{\partial^2 E}{\partial r_{i\alpha}\partial r_{j\beta}},\qquad
K_{i\alpha,j\beta}=\frac{\partial^2 E}{\partial S_{i\alpha}\partial S_{j\beta}},\qquad
C_{i\alpha,j\beta}=\frac{\partial^2 E}{\partial r_{i\alpha}\partial S_{j\beta}} .
$$

符号约定与 `05` 一致： $\mathbf F=-\partial E/\partial\mathbf R$、 $\mathbf g=-\partial E/\partial\mathbf S$，
因此 $\partial\mathbf F/\partial\mathbf S=\partial\mathbf g/\partial\mathbf R$（混合导数互易）。

**固定模长**： $\mathbf S_i$ 被约束在球面 $|\mathbf S\_i|=S\_i$ 上，磁 Hessian 必须投影到切平面

$$
K^{\perp}_{i\alpha,j\beta}=\Big(\delta_{\alpha\gamma}-\hat S_{i\alpha}\hat S_{i\gamma}\Big)
K_{i\gamma,j\delta}\Big(\delta_{\delta\beta}-\hat S_{j\delta}\hat S_{j\beta}\Big),
$$

并且要在**平衡磁构型**上取值。忽略投影会得到 3 个伪零模（整体旋转只有 2 个独立方向）。

## 2. 声子

质量加权动力学矩阵

$$
D_{\alpha\beta}(\mathbf q)=\frac{1}{\sqrt{m_i m_j}}\sum_{\mathbf R}\Phi_{i0,j\mathbf R}\ e^{i\mathbf q\cdot\mathbf R},
\qquad \omega^2=\text{eig}(D).
$$

验收：
- 声学求和规则（ASR）： $\sum\_j \Phi\_{i\alpha,j\beta}=0$ → 3 个 $\mathbf q=0$ 声学模频率 $\to0$；
- 有限差分步长收敛： $h$ 扫描必须出现误差平台；
- 超胞与 $\mathbf q$ 网格收敛。

代码：`src/phonon/hessian.cu`（IFC 由生产 `Force::compute` 的有限差分得到），分析脚本在 `tools/`。

## 3. 磁子

线性化无阻尼 LLG（ $\alpha=0$）：

$$
\frac{d\ \delta\mathbf S_i}{dt}=-\gamma\ \mathbf S_i^{(0)}\times\Big(\sum_j K_{ij}\delta\mathbf S_j\Big),
$$

在切平面基下写成 $\dot{\boldsymbol\eta}=i\ \mathcal H\ {\boldsymbol\eta}$，本征值给 $\omega(\mathbf q)$。
单位： $\gamma$ 用 $\mu_B/(\mathrm{eV\ fs})$； $K$ 用 $\mathrm{eV}/\mu_B^2$； $\omega$ 得到 rad/fs，乘 $658.2$ 得 THz。

无 SOC 的 mask17 有**整体自旋旋转 Goldstone 模**（ $\omega=0$）；有 SOC-like 项时才有 gap。
实测（本版本）：mask17 在整体旋转下 $E$ 差 $0.0$ eV → Goldstone 约束成立。

## 4. 磁声混合

联立 $\begin{pmatrix}\Phi & C\\\\ C^{T} & K^\perp\end{pmatrix}$ 求本征模；判断“耦合强度”必须看
**反交叉分裂**或模投影权重，**不能**把 $\lVert C\rVert _F$ 直接叫耦合常数（它与单位、超胞大小、规范都有关）。

## 5. 谱学与寿命

- $S(\mathbf q,\omega)$：对 $\mathbf S_i(t)$ 做时空傅里叶变换，窗函数与采样长度决定峰宽下限；
- SED：对速度自相关做 FFT；
- 寿命：Lorentzian 拟合 $\Gamma$， $\tau=1/\Gamma$，**必须报告拟合区间与窗函数**
  （否则“线宽”只是窗展宽）。谐波 Hessian 不能给有限温寿命。

## 6. 验收反例

| 反例 | 期望 |
|---|---|
| 一维单原子链解析色散 $\omega=2\sqrt{k/m}\vert \sin(qa/2)\vert$ | 数值误差 < 1% |
| 二聚体磁子（两自旋反铁磁） | 两个模，一个 Goldstone |
| 无 SOC 却报出各向异性 gap | 一定是数值假象 |

## 7. 本版本状态

| 项 | 状态 |
|---|---|
| $K$ 切平面投影与 Goldstone | 公式与检验方案就绪；mask17 旋转不变已实测 |
| 声子色散/DOS/群速度 | NOT_RUN_LARGE_SCALE |
| 磁子色散 | NOT_RUN_LARGE_SCALE |
| 磁声反交叉 | NOT_RUN_MODEL_GATE |
| 有限温寿命 | NOT_RUN_MODEL_GATE |
