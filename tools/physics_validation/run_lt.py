"""Stage PV1 -- Fe spin-lattice long-time stability matrix.

Every case is one directory under ``evidence/stage_PV1/fe_lt/cases`` holding
run.in, the model.xyz it started from, stdout.txt, thermo.out and (when asked
for) traj.xyz.  The script is deliberately dumb: it writes run.in, calls the
released gpumd.exe, records the wall clock and the exit code, and parses what
the run produced.  All analysis lives in ``analyze_lt.py``.

Conventions
-----------
* Equilibration is NVT (Langevin on the lattice) + SLLG at the same
  temperature, alpha = 0.3.
* Production is NVE + SLLG with alpha = 0, T = 0, i.e. *conservative*
  spin-lattice dynamics.  Only those runs are used for energy conservation;
  a thermostatted run has no conserved total energy by construction and is
  never counted as an energy-drift measurement.
* dt is in fs, n_steps is in MD steps, so the physical length is
  dt * n_steps / 1000 ps.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAGE = os.path.join(ROOT, "evidence", "stage_PV1")
FE_LT = os.path.join(STAGE, "fe_lt")
SYS = os.path.join(FE_LT, "systems")
CASES = os.path.join(FE_LT, "cases_250K")
GPUMD = os.path.join(ROOT, "final_delivery_20261031_174555", "win_staging", "gpumd.exe")
NEP = os.path.join(
    ROOT, "final_delivery_20261031_174555", "selected_models", "mask17_baseline", "nep.txt"
)
GEN = os.path.join(ROOT, "tools", "physics_validation", "make_fe_system.py")


def sh(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}\n{p.stderr[-2000:]}")
    return p.stdout


def ensure_system(name, args):
    path = os.path.join(SYS, name + ".xyz")
    if not os.path.exists(path):
        cmd = f'python "{GEN}" --out "{path}" ' + " ".join(args)
        sh(cmd)
        print(f"  built {name}")
    return path


def build_systems():
    os.makedirs(SYS, exist_ok=True)
    base = ["--source", "dataset", "--xyz", os.path.join(ROOT, "Fe", "test.xyz"), "--frame", "0"]
    systems = {
        "fe128_ds_fm": base + ["--rep", "2,2,2", "--spin-mode", "fm", "--spin-length", "2.2"],
        "fe432_ds_fm": base + ["--rep", "3,3,3", "--spin-mode", "fm", "--spin-length", "2.2"],
        "fe1024_ds_fm": base + ["--rep", "4,4,4", "--spin-mode", "fm", "--spin-length", "2.2"],
        "fe3456_ds_fm": base + ["--rep", "6,6,6", "--spin-mode", "fm", "--spin-length", "2.2"],
        "fe1024_ds_raw": base + ["--rep", "4,4,4", "--spin-mode", "dataset"],
        "fe1024_bcc": ["--source", "bcc", "--bcc-rep", "8,8,8", "--a", "2.87",
                       "--spin-mode", "fm", "--spin-length", "2.2"],
    }
    out = {}
    for name, args in systems.items():
        out[name] = ensure_system(name, args)
    return out


def write_run_in(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def run_case(name, xyz, run_lines, keep_traj=False, timeout=7200):
    d = os.path.join(CASES, name)
    os.makedirs(d, exist_ok=True)
    # gpumd APPENDS to thermo.out / hac.out / spin_work.out.  Re-running a case
    # into an existing directory therefore concatenates two runs into one file
    # (found the hard way: a 400-row run came back with 805 rows and two
    # headers).  Remove every output this harness knows about before starting.
    for junk in ("thermo.out", "traj.xyz", "heat_flux.out", "hac.out",
                 "spin_work.out", "SPIN_GUARD_STOP", "neighbor.out", "stdout.txt"):
        fp = os.path.join(d, junk)
        if os.path.exists(fp):
            os.remove(fp)
    shutil.copyfile(NEP, os.path.join(d, "nep.txt"))
    shutil.copyfile(xyz, os.path.join(d, "model.xyz"))
    write_run_in(os.path.join(d, "run.in"), run_lines)
    t0 = time.time()
    with open(os.path.join(d, "stdout.txt"), "w") as f:
        p = subprocess.run([GPUMD], cwd=d, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    wall = time.time() - t0
    meta = dict(case=name, xyz=os.path.basename(xyz), wall_s=round(wall, 2),
                exit_code=p.returncode, run_in=run_lines)
    # pull the guard message and the speed out of stdout
    txt = open(os.path.join(d, "stdout.txt"), errors="replace").read()
    meta["guarded"] = "RUNAWAY GUARD" in txt
    for line in txt.splitlines():
        if line.startswith("Speed of this run"):
            meta["speed_atom_step_per_s"] = float(line.split("=")[1].split()[0])
        if "Run stopped safely after" in line:
            meta["guard_stop"] = line.strip()
    with open(os.path.join(d, "case.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"  {name}: {wall:.1f}s rc={p.returncode}"
          + (" GUARD" if meta["guarded"] else ""), flush=True)
    if not keep_traj:
        for junk in ("traj.xyz",):
            p2 = os.path.join(d, junk)
            if os.path.exists(p2) and os.path.getsize(p2) > 200e6:
                os.remove(p2)
    return meta


def eq_lines(dt, n_steps, temperature, seed, dump_file=None):
    lines = [
        "potential nep.txt",
        f"time_step {dt} 100",
        "ensemble nvt_lan %g %g 100" % (temperature, temperature),
        f"spin_llg gamma=3.0385 alpha=0.3 T={temperature} seed={seed}",
        "dump_thermo 50",
    ]
    if dump_file:
        lines.append(f"dump_xyz {n_steps} {dump_file} potential velocity spin")
    lines += [f"seed {seed}", f"run {n_steps}"]
    return lines


def nve_lines(dt, n_steps, seed, traj=None, traj_every=None, heat_flux=None,
              thermo_every=50):
    """NVE + deterministic (alpha = 0, T = 0) SLLG: the conservative dynamics.

    ``traj`` writes the spin (and torque) so that |M|(t) and |S|(t) can be
    followed; ``traj_every`` defaults to one frame at the end of the run.
    """
    lines = [
        "potential nep.txt",
        f"time_step {dt} 100",
        "ensemble nve",
        "spin_llg gamma=3.0385 alpha=0 T=0",
        f"dump_thermo {thermo_every}",
    ]
    if traj:
        every = traj_every or n_steps
        lines.append(f"dump_xyz {every} {traj} spin torque force")
    if heat_flux:
        lines.append(f"dump_heat_flux {heat_flux}")
    lines += [f"seed {seed}", f"run {n_steps}"]
    return lines


def two_stage_lines(dT, eq_steps, dt, n_steps, seed, traj=None, traj_every=None,
                    thermo_every=50):
    """0.5 ps NVT equilibration at dT, then NVE production from the relaxed state.

    Used by the size and control series, whose cells are built from scratch and
    are therefore not at a stationary point of the model.
    """
    lines = [
        "potential nep.txt",
        f"time_step {dt} 100",
        f"ensemble nvt_lan {dT} {dT} 100",
        f"spin_llg gamma=3.0385 alpha=0.3 T={dT} seed={seed + 1000}",
        f"seed {seed + 1000}",
        f"run {eq_steps}",
        "ensemble nve",
        "spin_llg gamma=3.0385 alpha=0 T=0",
        f"dump_thermo {thermo_every}",
    ]
    if traj:
        every = traj_every or n_steps
        lines.append(f"dump_xyz {every} {traj} spin torque")
    lines += [f"seed {seed}", f"run {n_steps}"]
    return lines


def resample_velocities(src_xyz, dst_xyz, temperature, seed):
    """Copy a frame and replace its velocities by a fresh Maxwell sample.

    Needed for a real seed test: with alpha = 0 and T = 0 the dynamics has no
    random number in it at all, so changing `seed` in run.in changes nothing
    (verified: five runs, one sha256).  The randomness that matters is the
    initial microstate, so the seed is applied here, to the velocities.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_fe_system as G
    import numpy as np

    fr = G.read_xyz_frames(src_xyz)[0]
    fr["vel"] = G.maxwell_velocities(fr["n"], temperature, seed)
    G.write_xyz(dst_xyz, fr, fr["spin"])
    return dst_xyz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["eq", "dt", "duration", "size", "seed", "seed2", "magnetic",
                             "control", "all"])
    ap.add_argument("--eq-dir", default="eq_1024_T250")
    args = ap.parse_args()

    os.makedirs(CASES, exist_ok=True)
    systems = build_systems()

    if args.stage in ("eq", "all"):
        # One 2 ps equilibration at 300 K; the last frame seeds every other case.
        run_case(args.eq_dir, systems["fe1024_ds_fm"],
                 eq_lines(0.05, 40000, 250.0, 11, dump_file="eq_last.xyz"), keep_traj=True)

    eq_frame = os.path.join(CASES, args.eq_dir, "eq_last.xyz")

    if args.stage in ("dt", "all"):
        for dt, steps in ((0.2, 5000), (0.1, 10000), (0.05, 20000), (0.025, 40000)):
            run_case(f"dt_{dt:g}", eq_frame,
                     nve_lines(dt, steps, 21, traj="traj.xyz", traj_every=50),
                     keep_traj=True)

    if args.stage in ("duration", "all"):
        # The trajectory is only needed for |M|(t) and |S|(t); dumping it every
        # 0.1 ps keeps a 20 ps run at ~40 MB and costs no measurable time, while
        # dumping every 50 steps made the run I/O bound (6.3 vs 2.1 ms/step).
        for ps, steps, every in ((1, 20000, 50), (5, 100000, 2000), (10, 200000, 2000)):
            run_case(f"dur_{ps}ps", eq_frame,
                     nve_lines(0.05, steps, 31, traj="traj.xyz", traj_every=every,
                               thermo_every=200),
                     keep_traj=True)

    if args.stage in ("size", "all"):
        # Two gpumd invocations, not two `run` blocks in one file: with spin
        # dynamics, an NVT -> NVE switch inside a single run.in hangs at
        # N >= 432 (observed twice, see FE_LT_REPORT.md).  Equilibrate to a
        # frame first, then start a fresh NVE run from that frame.
        for tag in ("fe128_ds_fm", "fe432_ds_fm", "fe1024_ds_fm", "fe3456_ds_fm"):
            eqname = f"eqsz_{tag}"
            run_case(eqname, systems[tag],
                     eq_lines(0.05, 20000, 250.0, 41, dump_file="eq_last.xyz"))
            frame = os.path.join(CASES, eqname, "eq_last.xyz")
            run_case(f"size_{tag}", frame,
                     nve_lines(0.05, 20000, 41, traj="traj.xyz", traj_every=500))

    if args.stage in ("seed", "all"):
        # Seed 21 is the dt_0.05 case; add four more independent streams.
        for s in (22, 23, 24, 25):
            run_case(f"seed_{s}", eq_frame, nve_lines(0.05, 20000, s, traj="traj.xyz",
                                                      traj_every=100))

    if args.stage in ("seed2", "all"):
        # Independent microstates: same equilibrated structure, fresh Maxwell
        # velocities from four seeds, then 1 ps of deterministic NVE each.
        for k in (1, 2, 3, 4):
            xyz = os.path.join(CASES, f"seed2_{k}.xyz")
            resample_velocities(eq_frame, xyz, 250.0, 1000 + k)
            run_case(f"seed2_{k}", xyz, nve_lines(0.05, 20000, 200 + k),
                     keep_traj=True)

    if args.stage in ("magnetic", "all"):
        # Spin-lattice equilibration at several temperatures: the only claim is
        # whether |M| relaxes and whether the two subsystems share one T.
        for T in (300, 600, 900, 1200):
            run_case(f"mag_T{T}", systems["fe1024_ds_fm"],
                     eq_lines(0.05, 20000, float(T), 51, dump_file="final.xyz"),
                     keep_traj=True)

    if args.stage in ("control", "all"):
        for tag, seed in (("fe1024_bcc", 61), ("fe1024_ds_raw", 62)):
            eqname = f"eqctl_{tag}"
            run_case(eqname, systems[tag], eq_lines(0.05, 20000, 250.0, seed,
                                                   dump_file="eq_last.xyz"))
            frame = os.path.join(CASES, eqname, "eq_last.xyz")
            run_case(f"ctrl_{tag}_nve250", frame,
                     nve_lines(0.05, 20000, seed, traj="traj.xyz", traj_every=500),
                     keep_traj=True)


if __name__ == "__main__":
    main()
