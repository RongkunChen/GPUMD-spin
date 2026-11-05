# 示例：磁矩最小化 + 外场 + 磁电耦合

```bat
copy ..\..\..\tests\data\fege.nep.txt nep.txt
copy ..\..\..\tests\data\fege_b20.xyz     model.xyz
..\..\src\gpumd.exe
```

`fege_b20.xyz` 是 216 原子的 3x3x3 胞；示例 `run.in` 的参数对它同样适用
（只是 `Lattice` 边长不同）。

## 预期输出

```
Zeeman field (LAMMPS `fix precession/spin zeeman`): B = (0, 0, 0.000289419) eV/mu_B [5 T, g = 1: ...]
    step 0: total_potential = ..., f_max = ..., w_perp_max = ...
  Converged at step NN: f_max = ... AND w_perp_max = ..., E = ...
end of run: Zeeman energy = ...
```

两条收敛判据缺一不可：力（晶格）与**横向**力矩（磁矩）。固定 |S| 时力矩的
纵向分量不做功、永不归零，所以判据必须用横向分量。

## 已知坑

- `magfield` 写在 `spin_llg` 之前会报 `magfield requires spin_llg to be set first`。
- 盒子任一周期方向必须 >= 2x 最大径向截断（本例 6 A，即 >= 12 A）。
  `12x12x2` 这种 `Lz = 9.6 A` 的盒子会通过 GPUMD 的几何检查但结果无意义。
