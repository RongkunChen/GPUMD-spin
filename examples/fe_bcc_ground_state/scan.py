"""Run every generated cell statically, print E/atom, fit the minimum.

usage:  python scan.py [--gpumd ../../src/gpumd.exe] [--mode lattice|moment]
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--gpumd", default=os.path.join(HERE, "..", "..", "src", "gpumd.exe"))
ap.add_argument("--model", default=os.path.join(HERE, "..", "..", "models",
                                                    "fe_mask17_2desc_r18.nep.txt"))
ap.add_argument("--mode", default="lattice")
args = ap.parse_args()

RUN = """potential nep.txt
time_step 0.001 1
ensemble nve
spin_llg gamma=3.0385 alpha=0 T=0
dump_thermo 1
run 1
"""

cells = sorted(f for f in os.listdir(os.path.join(HERE, "cells")) if f.endswith(".xyz"))
rows = []
for fn in cells:
    tag = fn[:-4]
    d = os.path.join(HERE, "runs", tag)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "model.xyz"), "w") as fh:
        fh.write(open(os.path.join(HERE, "cells", fn)).read())
    with open(os.path.join(d, "run.in"), "w") as fh:
        fh.write(RUN)
    if os.path.exists(os.path.join(d, "nep.txt")):
        os.remove(os.path.join(d, "nep.txt"))
    with open(os.path.join(d, "nep.txt"), "wb") as fh, open(args.model, "rb") as src:
        fh.write(src.read())
    th = os.path.join(d, "thermo.out")
    if os.path.exists(th):
        open(th, "w").close()
    subprocess.run([args.gpumd], cwd=d, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL, check=True)
    first = [l.split() for l in open(th) if not l.startswith("#")][0]
    n = int(open(os.path.join(d, "model.xyz")).readline())
    lat, s = open(os.path.join(HERE, "cells", tag + ".meta")).read().split()
    rows.append(dict(a=float(lat), S=float(s), E_per_atom=float(first[2]) / n))

x = np.array([r["a"] if args.mode == "lattice" else r["S"] for r in rows])
y = np.array([r["E_per_atom"] for r in rows])
i = int(np.argmin(y))
c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
x0 = -c[1] / (2 * c[0])
print(f"{'a (A)':>8s} {'|S| (muB)':>10s} {'E/atom (eV)':>14s}")
for r in rows:
    print(f"{r['a']:8.3f} {r['S']:10.2f} {r['E_per_atom']:14.6f}")
label = "a0" if args.mode == "lattice" else "|S|0"
print(f"\n{label} = {x0:.4f}   E0 = {np.polyval(c, x0):.6f} eV/atom")
ref = 2.866 if args.mode == "lattice" else 2.22
print(f"experiment = {ref}   deviation = {(x0 / ref - 1) * 100:+.1f}%")
json.dump(dict(mode=args.mode, rows=rows, x0=float(x0),
               E0=float(np.polyval(c, x0)), experiment=ref),
          open(os.path.join(HERE, f"scan_{args.mode}.json"), "w"), indent=1)
