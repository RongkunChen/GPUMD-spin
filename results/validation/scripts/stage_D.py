"""Stage D: Curie temperature by equilibrium spin sampling.

There is no spin Monte Carlo in this build (`mc/` only has canonical and
semi-grand-canonical *atom* MC; `run.in` has no MC ensemble), so the only
equilibrium sampler for the spin subsystem is the Langevin-thermostatted SLLG,
whose noise amplitude is fixed by the fluctuation-dissipation relation
(validated separately against the exact two-spin partition function and by the
temperature calibration in the r17/r18 release notes).

Protocol per (T, L):
  * bcc supercell L x L x L conventional cells, spins at the equilibrium moment
  * `ensemble nvt_lan T T 100`  (lattice and spins at the same T)
  * `spin_llg gamma=3.0385 alpha=0.3 T=T`  -> FDT noise
  * discard the first 40% of the samples as equilibration
  * block averages -> <|M|>, chi, U4 with an autocorrelation-aware error

The method is labelled `SLLG-Langevin (equilibrium sampling)`, never as
"dynamical Tc".
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess

import numpy as np

import feutils as U

OUT = os.path.join(U.VAL, "raw_data", "D")
os.makedirs(OUT, exist_ok=True)
A0 = json.load(open(os.path.join(U.VAL, "raw_data", "B", "eos_fit.json")))["a0_A"]
M0 = json.load(open(os.path.join(U.VAL, "raw_data", "B", "moment_equilibrium.json")))["m_eq_muB"]


def run_point(L, T, seed, ps=6.0, dt=2.0, interval=20):
    d = os.path.join(OUT, f"L{L}_T{int(T)}_s{seed}")
    os.makedirs(d, exist_ok=True)
    for junk in ("magnetization.out", "thermo.out", "stdout.txt"):
        p = os.path.join(d, junk)
        if os.path.exists(p):
            open(p, "w").close()
    cell = U.bcc_cell(L, L, L, A0)
    U.write_xyz(os.path.join(d, "model.xyz"), cell, U.fm_spins(cell["n"], M0))
    shutil.copyfile(U.MODELS["A"], os.path.join(d, "nep.txt"))
    steps = int(ps * 1000 / dt)
    open(os.path.join(d, "run.in"), "w").write(
        "potential nep.txt\n"
        f"time_step {dt} 100\n"
        f"ensemble nvt_lan {T} {T} 100\n"
        f"spin_llg gamma=3.0385 alpha=0.3 T={T} seed={seed}\n"
        f"dump_magnetization {interval} magnetization.out\n"
        f"dump_thermo {interval * 5}\n"
        f"seed {seed}\nrun {steps}\n")
    with open(os.path.join(d, "stdout.txt"), "w") as f:
        p = subprocess.run([U.EXE], cwd=d, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    if p.returncode != 0:
        return dict(L=L, T=T, seed=seed, status="RUN_FAILED")
    data = np.loadtxt(os.path.join(d, "magnetization.out"), comments="#")
    # columns: step T Mx My Mz |M| varM chi ...
    t = data[:, 1]
    Mabs = data[:, 5]
    chi_col = data[:, 8]
    n = cell["n"]
    m = Mabs / n
    cut = int(0.4 * len(m))
    ms = m[cut:]
    chis = chi_col[cut:]
    m2 = np.mean(ms ** 2)
    m4 = np.mean(ms ** 4)
    U4 = 1.0 - m4 / (3.0 * m2 ** 2) if m2 > 0 else float("nan")
    # autocorrelation time of |m|
    x = ms - ms.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0] if ac[0] > 0 else 1
    tau = 1.0
    for k in range(1, len(ac)):
        if ac[k] < 0:
            break
        tau += 2 * ac[k]
    nb = max(4, min(20, int(len(ms) / max(2 * tau, 1))))
    blocks = np.array_split(ms, nb)
    err = float(np.std([b.mean() for b in blocks], ddof=1) / np.sqrt(nb))
    return dict(L=L, T=T, seed=seed, n_atoms=n, steps=steps, dt_fs=dt, total_ps=ps,
                n_samples=len(ms), tau_samples=float(tau),
                independent_samples=float(len(ms) / max(tau, 1)),
                m_mean=float(ms.mean()), m_err=err, m_std=float(ms.std()),
                chi_mean=float(np.mean(chis)), chi_err=float(np.std(chis) / np.sqrt(len(chis))),
                U4=float(U4), T_mean=float(t.mean()), T_std=float(t.std()),
                status="PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="6,8")
    ap.add_argument("--temps", default="300,700,1000,1200,1400")
    ap.add_argument("--ps", type=float, default=6.0)
    ap.add_argument("--dt", type=float, default=2.0)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    rows = []
    for L in [int(x) for x in a.sizes.split(",")]:
        for T in [float(x) for x in a.temps.split(",")]:
            for s in [int(x) for x in a.seeds.split(",")]:
                r = run_point(L, T, s, ps=a.ps, dt=a.dt)
                rows.append(r)
                print(f"  L={L} T={T:.0f} s={s} m={r.get('m_mean', float('nan')):.4f} "
                      f"chi={r.get('chi_mean', float('nan')):.4f} U4={r.get('U4', float('nan')):.3f} "
                      f"Tmean={r.get('T_mean', float('nan')):.0f}", flush=True)
    with open(os.path.join(OUT, f"scan_{a.tag}.csv"), "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", f"scan_{a.tag}.csv", flush=True)
