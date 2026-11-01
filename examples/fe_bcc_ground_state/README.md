# 例：bcc Fe 的晶格常数与基态磁矩

两条静态扫描，不需要长时动力学。

```bash
# 需要一个 Fe 模型
python make_cells.py            # 生成 7 个 bcc 胞
python scan.py                  # 逐个跑、打印 E/atom、二次拟合 → a0
```

预期（r18 A 模型）：`a0 = 2.8258 Å`（实验 2.866，−1.4%）。
再做 |S| 扫描：`python make_cells.py --mode moment && python scan.py --mode moment`
得 `|S|0 = 2.413 μB`（实验 2.22）。

一条命令版：`python ../../tests/physics_checks.py --bin ../../src --quick`。
