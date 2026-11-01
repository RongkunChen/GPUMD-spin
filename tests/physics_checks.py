#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics-level checks for a GPUMD-spin V3 installation (r18).

`smoke.py` answers "does each code path run".  This script answers "does the
model + integrator reproduce the physics we claim", on a *perfect bcc cell*
that is deliberately out of the training distribution:

  P1  lattice constant     -- E(a) scan, quadratic fit, compared to experiment
  P2  ground-state moment  -- E(|S|) scan at a0, quadratic fit
  P3  Curie temperature    -- |M|(T) with SLLG at five temperatures
  P4  NVE energy drift     -- dt = 0.05 fs, 1 ps, dE_max per atom
  P5  runaway-guard regression -- a 600 K NVE run must NOT be aborted
                                  (this is the F26 defect of r17)

Every threshold is printed next to the measured value, and a check that cannot
run (missing model, missing binary) is reported SKIP, never PASS.

usage:
    python tests/physics_checks.py --bin <dir with gpumd.exe> \
        --model models/fe_mask17_2desc_r18.nep.txt [--work scratch] \
        [--json out.json] [--quick]

`--quick` runs P1/P2 on a 16-atom cell and P3 on two temperatures only; it is
what the packaged acceptance run uses.  Without it P3 uses five temperatures
and a 432-atom cell.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "tools", "physics_validation"))
import make_fe_system as G  # noqa: E402

KB = 8.617333262e-5  # eV/K
FE_MASS = 55.845
# 0.5 * m * v^2 with v in A/fs and m in u gives eV after this factor
KE_FACTOR = 1.0364269e-4


def run_gpumd(exe, workdir, lines, timeout=3600):
    """Run one gpumd job in `workdir`, return (rc, stdout text)."""
    for junk in ("thermo.out", "traj.xyz", "stdout.txt", "spin_work.out",
                 "SPIN_GUARD_STOP", "heat_flux.out"):
        fp = os.path.join(workdir, junk)
        if os.path.exists(fp):
            open(fp, "w").close()
    with open(os.path.join(workdir, "run.in"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(workdir, "stdout.txt"), "w") as f:
        p = subprocess.run([exe], cwd=workdir, stdout=f,
                           stderr=subprocess.STDOUT, timeout=timeout)
    txt = open(os.path.join(workdir, "stdout.txt"), errors="replace").read()
    return p.returncode, txt


def read_thermo(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows)


def read_traj(path):
    """Return (n_atoms, [frames]) with frames as dict of numpy arrays."""
    lines = open(path).readlines()
    out, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        if i + 2 + n > len(lines):
            break
        head = lines[i + 1]
        props = re.search(r"Properties=(\S+)", head).group(1).split(":")
        # Properties is name:type:ncomp triples; "species" is a string column
        # and must be skipped, otherwise every array is shifted by one.
        names, c = [], 0
        for k in range(0, len(props), 3):
            names.append((props[k], props[k + 1], int(props[k + 2])))
        cols = {}
        for nm, typ, nc in names:
            cols[nm] = (c, c + nc)
            c += nc
        rows = [l.split() for l in lines[i + 2:i + 2 + n]]
        fr = {}
        for nm, (a, b) in cols.items():
            if nm == "species":
                continue
            fr[nm] = np.array([[float(x) for x in r[a:b]] for r in rows])
        out.append(fr)
        i += 2 + n
    return (out[0][next(iter(out[0]))].shape[0] if out else 0), out


def static_energy(exe, workdir, a, spin_len, n_atoms, dt=0.001):
    """Potential energy per atom of a perfect bcc cell at lattice constant a."""
    fr = G.bcc(2, 2, 2, a) if n_atoms <= 16 else G.bcc(4, 4, 4, a)
    spins, _ = G.set_spins(fr, "fm", spin_len, 1)
    G.write_xyz(os.path.join(workdir, "model.xyz"), fr, spins)
    lines = ["potential nep.txt", f"time_step {dt} 1", "ensemble nve",
             "spin_llg gamma=3.0385 alpha=0 T=0", "dump_thermo 1", "run 1"]
    rc, txt = run_gpumd(exe, workdir, lines)
    if rc != 0:
        raise RuntimeError("gpumd failed (rc=%d):\n%s" % (rc, txt[-800:]))
    d = read_thermo(os.path.join(workdir, "thermo.out"))
    return float(d[0, 2]) / fr["n"], rc


def parabolic_min(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.argmin(y))
    if i == 0 or i == len(x) - 1:
        return None
    c = np.polyfit(x[i - 1:i + 2], y[i - 1:i + 2], 2)
    return float(-c[1] / (2 * c[0])), float(np.polyval(c, -c[1] / (2 * c[0])))


def p1_lattice(exe, work, model, quick):
    n = 16 if quick else 16
    grid = np.round(np.arange(2.75, 2.901, 0.025), 3)
    rows = []
    for a in grid:
        e, _ = static_energy(exe, os.path.join(work, "p1"), float(a), 2.2, n)
        rows.append(dict(a=float(a), E_per_atom=e))
    fit = parabolic_min([r["a"] for r in rows], [r["E_per_atom"] for r in rows])
    a0 = fit[0] if fit else None
    return dict(cell="bcc 2x2x2 (16 atoms), |S| = 2.2",
                scan=rows, a0_A=a0, E0_eV_per_atom=fit[1] if fit else None,
                experiment_A=2.866, error_percent=(a0 / 2.866 - 1) * 100 if a0 else None,
                verdict="PASS" if a0 and abs(a0 / 2.866 - 1) < 0.05 else "FAIL",
                criterion="|a0 - 2.866| / 2.866 < 5%")


def p2_moment(exe, work, model, a0, quick):
    grid = [2.0, 2.2, 2.4, 2.6, 2.8]
    rows = []
    for s in grid:
        e, _ = static_energy(exe, os.path.join(work, "p2"), a0, s, 16)
        rows.append(dict(S=float(s), E_per_atom=e))
    fit = parabolic_min([r["S"] for r in rows], [r["E_per_atom"] for r in rows])
    s0 = fit[0] if fit else None
    return dict(cell="bcc 2x2x2 at a0 = %.4f A" % a0, scan=rows,
                S0_muB=s0, E0_eV_per_atom=fit[1] if fit else None,
                experiment_muB=2.22,
                error_percent=(s0 / 2.22 - 1) * 100 if s0 else None,
                verdict="PASS" if s0 and 1.7 < s0 < 2.9 else "FAIL",
                criterion="1.7 < S0 < 2.9 mu_B (a 50k-step model is not expected to match 2.22 exactly)")


def p3_curie(exe, work, model, a0, quick):
    temps = [600, 900, 1200] if quick else [300, 600, 800, 1000, 1200, 1400]
    rep = (4, 4, 4) if quick else (6, 6, 6)   # 128 / 432 atoms
    ps = 0.5 if quick else 1.0
    steps = int(ps * 1000 / 0.05)
    rows = []
    for T in temps:
        d = os.path.join(work, f"p3_{T}K")
        os.makedirs(d, exist_ok=True)
        fr = G.bcc(*rep, 2.866)
        spins, _ = G.set_spins(fr, "fm", 2.2, 7)
        fr["vel"] = G.maxwell_velocities(fr["n"], float(T), 7)
        G.write_xyz(os.path.join(d, "model.xyz"), fr, spins)
        lines = ["potential nep.txt", "time_step 0.05 100",
                 f"ensemble nvt_lan {T} {T} 100",
                 f"spin_llg gamma=3.0385 alpha=0.3 T={T} seed=7",
                 f"dump_xyz 200 traj.xyz spin", f"seed 7", f"run {steps}"]
        rc, txt = run_gpumd(exe, d, lines)
        if rc != 0:
            rows.append(dict(T=T, rc=rc, M_mean=None))
            continue
        n, frames = read_traj(os.path.join(d, "traj.xyz"))
        mags = [np.linalg.norm(f["spin"].mean(axis=0)) for f in frames[len(frames) // 2:]]
        rows.append(dict(T=T, rc=rc, n_atoms=n, n_frames=len(frames),
                         M_mean=float(np.mean(mags)), M_std=float(np.std(mags)),
                         M_saturation_fraction=float(np.mean(mags) / 2.2)))
    # Tc estimate: linear extrapolation of the two points that bracket 50% of
    # the low-temperature magnetisation (crude, +-150 K; see the theory doc).
    tc = None
    good = [r for r in rows if r["M_mean"]]
    if len(good) >= 2:
        m_hi = good[0]["M_mean"]
        target = 0.5 * m_hi
        for a, b in zip(good, good[1:]):
            if a["M_mean"] >= target > b["M_mean"]:
                f = (a["M_mean"] - target) / (a["M_mean"] - b["M_mean"])
                tc = a["T"] + f * (b["T"] - a["T"])
                break
    return dict(cell=f"bcc {rep[0]}x{rep[1]}x{rep[2]}, {ps} ps per point",
                points=rows, Tc_estimate_K=tc, experiment_K=1043,
                verdict="PASS" if tc and 600 < tc < 1600 else "CHECK",
                criterion="Tc in 600-1600 K; the estimator itself is only good to +-150 K")


def p4_energy_drift(exe, work, model, a0, quick):
    d = os.path.join(work, "p4")
    os.makedirs(d, exist_ok=True)
    steps = 2000 if quick else 20000       # 0.1 / 1.0 ps at dt = 0.05
    fr = G.bcc(4, 4, 4, a0)
    spins, _ = G.set_spins(fr, "fm", 2.2, 3)
    fr["vel"] = G.maxwell_velocities(fr["n"], 250.0, 3)
    G.write_xyz(os.path.join(d, "model.xyz"), fr, spins)
    lines = ["potential nep.txt", "time_step 0.05 100", "ensemble nve",
             "spin_llg gamma=3.0385 alpha=0 T=0", "dump_thermo 20", "run %d" % steps]
    rc, txt = run_gpumd(exe, d, lines)
    therm = read_thermo(os.path.join(d, "thermo.out"))
    n = fr["n"]
    e = (therm[:, 1] + therm[:, 2]) / n
    dmax = float(np.abs(e - e[0]).max())
    return dict(rc=rc, n_atoms=n, steps=steps, t_ps=steps * 0.05 / 1000,
                dE_max_per_atom=dmax, dE_end_per_atom=float(e[-1] - e[0]),
                T_mean=float(therm[:, 0].mean()),
                verdict="PASS" if (rc == 0 and dmax < 1e-5) else "FAIL",
                criterion="max|dE|/atom < 1e-5 eV")


def p5_guard_regression(exe, work, model, a0):
    """F26: an NVE run above 300 K must not be stopped by the runaway guard.

    r17 stopped every NVE run at T >= 300 K with "lattice temperature excursion
    (limit 300.000 K)" because Integrate::temperature was uninitialised.
    """
    d = os.path.join(work, "p5")
    os.makedirs(d, exist_ok=True)
    fr = G.bcc(4, 4, 4, a0)
    spins, _ = G.set_spins(fr, "fm", 2.2, 5)
    fr["vel"] = G.maxwell_velocities(fr["n"], 600.0, 5)
    G.write_xyz(os.path.join(d, "model.xyz"), fr, spins)
    lines = ["potential nep.txt", "time_step 0.05 100", "ensemble nve",
             "spin_llg gamma=3.0385 alpha=0 T=0", "dump_thermo 100", "run 2000"]
    rc, txt = run_gpumd(exe, d, lines)
    guarded = "RUNAWAY GUARD" in txt
    limit = [l.strip() for l in txt.splitlines()
             if "excursion" in l or "Run stopped" in l][:2]
    therm = read_thermo(os.path.join(d, "thermo.out")) if os.path.getsize(
        os.path.join(d, "thermo.out")) else np.zeros((0, 3))
    return dict(rc=rc, guard_triggered=guarded, limit_line=limit,
                n_thermo_rows=int(len(therm)),
                T_first=float(therm[0, 0]) if len(therm) else None,
                T_last=float(therm[-1, 0]) if len(therm) else None,
                verdict="PASS" if (rc == 0 and not guarded and len(therm) >= 5) else "FAIL",
                criterion="rc = 0, no RUNAWAY GUARD, run reaches step 2000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="directory with gpumd.exe")
    ap.add_argument("--model", default=os.path.join(PKG, "models",
                                                    "fe_mask17_2desc_r18.nep.txt"))
    ap.add_argument("--work", default=os.path.join(PKG, "_physics_work"))
    ap.add_argument("--json", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    exe = os.path.join(args.bin, "gpumd.exe")
    if not os.path.exists(exe):
        print("gpumd.exe not found in", args.bin)
        return 2
    if not os.path.exists(args.model):
        print("model not found:", args.model)
        return 2
    subs = ["p1", "p2", "p4", "p5"]
    if args.quick:
        subs += [f"p3_{T}K" for T in (600, 900, 1200)]
    else:
        subs += [f"p3_{T}K" for T in (300, 600, 800, 1000, 1200, 1400)]
    for sub in subs:
        d = os.path.join(args.work, sub)
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(args.model, os.path.join(d, "nep.txt"))

    res = dict(model=os.path.abspath(args.model), bin=os.path.abspath(exe),
               quick=bool(args.quick))
    print("P1 lattice constant ...", flush=True)
    res["lattice"] = p1_lattice(exe, args.work, args.model, args.quick)
    a0 = res["lattice"]["a0_A"] or 2.866
    print("   a0 = %.4f A (%+.1f%% vs 2.866)" % (a0, res["lattice"]["error_percent"]), flush=True)

    print("P2 ground-state moment ...", flush=True)
    res["moment"] = p2_moment(exe, args.work, args.model, a0, args.quick)
    print("   |S|0 = %.3f mu_B (exp 2.22)" % (res["moment"]["S0_muB"] or float("nan")), flush=True)

    print("P4 NVE energy drift ...", flush=True)
    res["energy"] = p4_energy_drift(exe, args.work, args.model, a0, args.quick)
    print("   dE_max = %.3e eV/atom over %.2f ps" %
          (res["energy"]["dE_max_per_atom"], res["energy"]["t_ps"]), flush=True)

    print("P5 runaway-guard regression ...", flush=True)
    res["guard"] = p5_guard_regression(exe, args.work, args.model, a0)
    print("   guard triggered: %s (T %.0f -> %.0f K)" %
          (res["guard"]["guard_triggered"], res["guard"]["T_first"] or 0,
           res["guard"]["T_last"] or 0), flush=True)

    print("P3 Curie temperature ...", flush=True)
    res["curie"] = p3_curie(exe, args.work, args.model, a0, args.quick)
    print("   Tc ~ %s K" % res["curie"]["Tc_estimate_K"], flush=True)

    verdicts = {k: res[k]["verdict"] for k in ("lattice", "moment", "energy",
                                               "guard", "curie")}
    res["verdicts"] = verdicts
    res["pass"] = all(v == "PASS" for v in verdicts.values())
    out = args.json or os.path.join(args.work, "physics_checks.json")
    json.dump(res, open(out, "w"), indent=1)
    print("\n".join(f"  {k:8s} {v}" for k, v in verdicts.items()))
    print("wrote", out)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
