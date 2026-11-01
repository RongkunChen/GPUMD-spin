# 04 — ANN 架构、训练损失与解析梯度

## 4.1 结构

单隐层，$\tanh$ 激活：

$$
e_i=\sum_{n=1}^{N_{\rm neu}} w^{(1)}_n\tanh\!\Big(\sum_{d=1}^{D} w^{(0)}_{nd}q_{i,d}-b^{(0)}_n\Big)-b^{(1)},
$$

$D=$ `dim`（本模型 54），$N_{\rm neu}=30$。参数按 `nep.txt` 的行序排布，
类型之间共享或按 `num_types` 复制（本模型 1 类 Fe）。

## 4.2 损失

$$
L=\lambda_e\sum_i\Big(\frac{e_i^{\rm pred}-e_i^{\rm ref}}{N}\Big)^2
+\lambda_f\sum_{i,a}\big(F^{\rm pred}_{ia}-F^{\rm ref}_{ia}\big)^2
+\lambda_m\sum_{i,a}\big(g^{\rm pred}_{ia}-g^{\rm ref}_{ia}\big)^2
+\lambda_v\sum(\text{virial})^2 .
$$

本项目的权重：$\lambda_e=\lambda_f=\lambda_m=1$（`nep.in`）。
$g$ 的标签按 `spin_forces` 列读入，状态记为 ASSUMED_LABEL_CONTRACT（来源未核实）。

## 4.3 解析梯度

对 $q$ 的导数由 $\tanh'=1-\tanh^2$ 直接给出：

$$
\frac{\partial e_i}{\partial q_{i,d}}=\sum_n w^{(1)}_n\big(1-x_n^2\big)w^{(0)}_{nd},
\qquad x_n=\tanh(\cdot).
$$

再乘 $\partial q/\partial\mathbf S_i$、$\partial q/\partial\mathbf R_i$ 得到 $\mathbf g_i$、$\mathbf F_i$。
代码里这两条链是手写的（`apply_ann_one_layer` 返回 `Fp[]`，再由描述符核的反向累加得到力）。

## 4.4 训练机制

`nep.exe` 用演化策略（population 20，batch 32）更新参数：每个 generation 只对一个 batch 求损失，
按适应度做变异/选择，不做反向传播。这解释了为什么“固定 generation 数、加数据”会让每个样本被看的次数变少
（本项目的学习曲线就是这个效应，见 `../11_TRAINING_AND_VALIDATION.md`）。

## 4.5 精度相关

* 前向是 float32；描述符的邻居求和改成 double 累加后，两条实现的 $e_i$ 一致到 $10^{-7}$ eV（见 14）。
* `q_scaler` 把描述符缩放到 O(1)，是训练稳定性的关键；它随模型一起存进 `nep.txt`。

## 4.6 代码入口

* 前向：`utilities/nep_utilities.cuh::apply_ann_one_layer`
* 训练：`main_nep/nep.cu`、`main_nep/fitness.cu`、`main_nep/dataset.cu`
* JIT：`main_nep/nep_compile.cu` 生成专用源码，`nep_specialized.cu` 编译执行

## 4.7 验证

* `tests/smoke.py::T1`：单点能量一致性；
* Stage03A：逐原子 $e_i$、$F$、$g$、virial、描述符逐元素比对（本版本 G6 混合判据 5/5 PASS）；
* 训练收敛：loss.out 的 E/F/g 三条曲线（`../11_TRAINING_AND_VALIDATION.md` 有实测表）。

## 4.8 限制

* 单隐层 30 神经元的表达能力有限；本项目最好的 Fe 模型 DEV E_MAE 12.4 meV/atom，
  不足以做定量材料预测（ENGINEERING_ONLY）。
* 训练集只有一种化学环境（Fe），外推到合金/表面无效。
