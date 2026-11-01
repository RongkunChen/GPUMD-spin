"""|M|(T) and chi(T) for bcc Fe using dump_magnetization.

usage: python tc_scan.py --temps 300,600,900,1200 [--ps 1.0] [--rep 6,6,6]
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "physics_validation"))
import make_fe_system as G

ap = argparse.ArgumentParser()
ap.add_argument("--temps", default="300,600,800,900,1000,1100,1300")
ap.add_argument("--ps", type=float, default=1.0)
ap.add_argument("--rep", default="6,6,6")
ap.add_argument("--a", type=float, default=2.8258)
ap.add_argument("--model", default=os.path.join(HERE, "..", "..", "models",
                                                    "fe_mask17_2desc_r18.nep.txt"))
ap.add_argument("--gpumd", default=os.path.join(HERE, "..", "..", "src", "gpumd.exe"))
ap.add_argument("--seed", type=int, default=7)
args = ap.parse_args()

rep = tuple(int(x) for x in args.rep.split(","))
steps = int(args.ps * 1000 / 0.05)
rows = []
for T in [int(x) for x in args.temps.split(",")]:
    d = os.path.join(HERE, f"T{T}")
    os.makedirs(d, exist_ok=True)
    fr = G.bcc(*rep, args.a)
    spins, _ = G.set_spins(fr, "fm", 2.2, args.seed)
    fr["vel"] = G.maxwell_velocities(fr["n"], float(T), args.seed)
    G.write_xyz(os.path.join(d, "model.xyz"), fr, spins)
    with open(os.path.join(d, "nep.txt"), "wb") as fh, open(args.model, "rb") as src:
        fh.write(src.read())
    with open(os.path.join(d, "run.in"), "w") as fh:
        fh.write(
            "potential nep.txt\n"
            "time_step 0.05 100\n"
            f"ensemble nvt_lan {T} {T} 100\n"
            f"spin_llg gamma=3.0385 alpha=0.3 T={T} seed={args.seed}\n"
            "dump_magnetization 200 magnetization.out warmup 10\n"
            f"seed {args.seed}\nrun {steps}\n")
    for junk in ("magnetization.out", "stdout.txt"):
        p = os.path.join(d, junk)
        if os.path.exists(p):
            open(p, "w").close()
    subprocess.run([args.gpumd], cwd=d, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = np.loadtxt(os.path.join(d, "magnetization.out"), comments="#")
    # columns: step T Mx My Mz |M| varM chi ...
    half = data[len(data) // 2:]
    M = half[:, 5]
    chi = half[:, 8]
    rows.append(dict(T=T, M=float(M.mean()), M_std=float(M.std()),
                     chi=float(chi.mean()), n_atoms=fr["n"]))
    print(f"  T={T:5d} K   |M|={M.mean():.3f} +- {M.std():.3f} mu_B/atom"
          f"   chi={chi.mean():.3f}", flush=True)

with open(os.path.join(HERE, "tc_points.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

T = np.array([r["T"] for r in rows], float)
c = np.array([r["chi"] for r in rows])
i = int(np.argmax(c))
if 0 < i < len(T) - 1:
    coef = np.polyfit(T[i - 1:i + 2], c[i - 1:i + 2], 2)
    tc = -coef[1] / (2 * coef[0])
    print(f"\nTc(chi peak, parabolic) = {tc:.0f} K   (experiment 1043, "
          f"uncertainty ~100 K)")
else:
    print(f"\nchi peaks at the edge of the scan (T={T[i]:.0f} K); widen --temps")
