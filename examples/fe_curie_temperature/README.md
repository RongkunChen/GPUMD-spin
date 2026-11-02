# 例：Fe 的居里温度（M(T) + 磁化率峰）

432 原子 bcc（ $6\times6\times6$），每个温度 1 ps，`dump_magnetization` 直接给
$|\mathbf M|$ 与 $\chi$。

```bash
python tc_scan.py --temps 300,600,800,900,1000,1100,1300
```

输出 `tc_points.csv`（T, |M|, χ）与估计的 $T_c$。

预期（r18 A 模型）： $T_c\approx950$ K（实验 1043），**误差棒约 ±100 K**——
每点只有 1 ps、单一尺寸。要更准需要每点 ≥50 ps 且做多尺寸 Binder 分析
（本包未实现，见 `docs/theory/08`）。

判据写法： $|\mathbf M|(T)$ 陡降区 + $\chi$ 峰位。**不要**用"最后一个非零 M"
当 $T_c$，那个数与采样长度直接相关。
