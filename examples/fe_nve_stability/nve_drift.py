"""NVE energy drift for a bcc Fe cell.

usage: python nve_drift.py --dt 0.05 --ps 1.0 [--rep 4,4,4] [--T 250]
"""
import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools", "physics_validation"))
import make_fe_system as G

ap = argparse.ArgumentParser()
ap.add_argument("--dt", type=float, default=0.05)
ap.add_argument("--ps", type=float, default=1.0)
ap.add_argument("--rep", default="4,4,4")
ap.add_argument("--T", type=float, default=250.0)
ap.add_argument("--a", type=float, default=2.8258)
ap.add_argument("--model", default=os.path.join(HERE, "..", "..", "models",
                                                    "fe_mask17_2desc_r18.nep.txt"))
ap.add_argument("--gpumd", default=os.path.join(HERE, "..", "..", "src", "gpumd.exe"))
args = ap.parse_args()

rep = tuple(int(x) for x in args.rep.split(","))
steps = int(args.ps * 1000 / args.dt)
d = os.path.join(HERE, f"dt{args.dt:g}_{args.ps:g}ps")
os.makedirs(d, exist_ok=True)
fr = G.bcc(*rep, args.a)
spins, _ = G.set_spins(fr, "fm", 2.2, 3)
fr["vel"] = G.maxwell_velocities(fr["n"], args.T, 3)
G.write_xyz(os.path.join(d, "model.xyz"), fr, spins)
with open(os.path.join(d, "nep.txt"), "wb") as fh, open(args.model, "rb") as src:
    fh.write(src.read())
with open(os.path.join(d, "run.in"), "w") as fh:
    fh.write("potential nep.txt\n"
             f"time_step {args.dt} 100\n"
             "ensemble nve\n"
             "spin_llg gamma=3.0385 alpha=0 T=0\n"
             "dump_thermo 20\n"
             f"run {steps}\n")
th = os.path.join(d, "thermo.out")
if os.path.exists(th):
    open(th, "w").close()
subprocess.run([args.gpumd], cwd=d, check=True,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
data = np.loadtxt(th, comments="#")
n = fr["n"]
e = (data[:, 1] + data[:, 2]) / n
dmax = float(np.abs(e - e[0]).max())
print(f"dt={args.dt} fs, {args.ps} ps, {n} atoms, T~{data[:,0].mean():.0f} K")
print(f"  max|dE|/N = {dmax:.3e} eV/atom   end = {e[-1]-e[0]:+.3e}   "
      f"criterion 1e-5 -> {'PASS' if dmax < 1e-5 else 'FAIL'}")
