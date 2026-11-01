# 模型与数据格式

## 1. `nep.txt`

```
nep4_spin 1 Fe            # 版本、类型数、元素
spin 1                    # 启用自旋
spin_layout 3 Sex San Sdm Sia Smag
spin_mag_scale 0.25
spin_blocks Sex Smag      # 实际启用的块（掩码 17）
cutoff 6 3 84 14          # r_rad, r_ang, 角向展开参数
n_max 4 4
basis_size 8 8
l_max 4 0 0
neuron 30
<w0 矩阵 / b0 / w1 / b1>
<c 系数>
<q_scaler>
```

读取顺序由 `force/nep.cu::NEP::NEP` 决定；参数在文件里的顺序 = 内存里的顺序，
`actual_file_params_fingerprint` 就是这串 float 的 FNV 哈希，用来证明两条路径读的是同一份权重。

**兼容性**：`spin_blocks` 缺省时按 `spin 1` 给默认掩码；`spin_layout` 决定 slot4 是 `Sia` 还是
`Senv`。旧 layout 的模型照样能读，但 `Senv` 标记 deprecated。

## 2. `model.xyz`（MD 输入）

```
4
Lattice="20 0 0 0 20 0 0 0 20" Properties=species:S:1:pos:R:3:spin:R:3 pbc="T T T"
Fe 0.0 0.0 0.0 0.0 0.0 2.2
...
```

* `spin` 列单位 μB，直接是 $\mathbf S$（不是方向 × 模长）；
* `pbc` 与 `Lattice` 必须一致；
* 20 Å 的盒子 + 4 原子是 `examples/` 里的合成用例，不是物理体系。

## 3. 训练数据 `train.xyz`

```
1
Lattice="..." Properties=species:S:1:pos:R:3:forces:R:3:spin:R:3 spin_forces:R:3 energy=... pbc="T T T"
Fe x y z Fx Fy Fz Sx Sy Sz gx gy gz
```

* `energy` 是**总能量**（eV），训练器除以原子数得到 per-atom 目标；
* `spin_forces` 按项目约定当 $-\partial E/\partial\mathbf S$（eV/μB），
  来源未核实，状态 ASSUMED_LABEL_CONTRACT；
* 本项目的项目约定：$\mathbf S=\text{spin\_length}\times\text{initial\_magmoms}$。

## 4. 输出

| 文件 | 内容 |
|---|---|
| `loss.out` | 每 `output_interval` 代一行：代、总损失、E/F/g 分量 |
| `energy_train.out` | 两列：预测 / 参考（per atom） |
| `force_train.out` | 六列：预测 xyz / 参考 xyz |
| `spin_train.out` | 同上，磁梯度 |
| `thermo.out` | 温度、动能、势能、virial 9 分量、box |
| `traj.xyz` | 与输入同格式（含 `energy=` 与 `virial=` 头） |
| `spin_restart.bin` | 二进制检查点（见 `USER_MANUAL.md` §5） |

## 5. 版本与摘要

`VERSION` 给 tag；`MANIFEST.sha256` 给每个文件的 SHA256；
`checksums-SHA256.txt` 是上游原始校验；`FINAL_SOURCE_MANIFEST.sha256` 是本次交付的清单。
