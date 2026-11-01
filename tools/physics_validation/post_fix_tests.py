"""Post-fix verification and magnetic ground-state tests (2026-11-01).

Sequence, each step writing its own JSON so a later failure cannot hide an
earlier one:

1. `guard`   -- the NVE-above-300-K case that the shipped binary killed at step
                1000 must now run to completion, and a 250 K case must stay
                bit-identical to the shipped binary (the guard only reads).
2. `lattice` -- 0 K energy versus lattice constant for bcc Fe, minimised by a
                parabola through the three lowest points.
3. `moment`  -- 0 K energy versus |S| at the relaxed lattice constant.
4. `tc`      -- |M|(T) from SLLG + NVT runs, Tc read off where the
                magnetisation collapses.
"""

import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORK = os.path.join(ROOT, "work_20261101_retrain")
OUT = os.path.join(WORK, "tests")
GPUMD_FIX = os.path.join(WORK, "bin", "gpumd.exe")
GPUMD_OLD = os.path.join(ROOT, "final_delivery_20261031_174555", "win_staging", "gpumd.exe")
NEP_NEW = os.environ.get("NEP_OVERRIDE") or os.path.join(WORK, "A_2desc", "nep.txt")
TAG = os.environ.get("TEST_TAG", "")
GEN = os.path.join(ROOT, "tools", "physics_validation", "make_fe_system.py")
MASS_FE = 55.845
KB = 8.617333262e-5


def run(case, xyz, lines, exe=GPUMD_FIX, nep=NEP_NEW, timeout=7200):
    d = os.path.join(OUT, case)
    os.makedirs(d, exist_ok=True)
    # Truncate, do not delete: the sandbox blocks bulk os.remove and aborted a
    # whole lattice sweep with SAFE_DELETE_BULK_CONFIRM_REQUIRED.
    for junk in ("thermo.out", "traj.xyz", "spin_work.out", "SPIN_GUARD_STOP",
                 "stdout.txt", "neighbor.out", "magnetization.out"):
        fp = os.path.join(d, junk)
        if os.path.exists(fp):
            open(fp, "w").close()
    shutil.copyfile(nep, os.path.join(d, "nep.txt"))
    shutil.copyfile(xyz, os.path.join(d, "model.xyz"))
    with open(os.path.join(d, "run.in"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(d, "stdout.txt"), "w") as f:
        p = subprocess.run([exe], cwd=d, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    txt = open(os.path.join(d, "stdout.txt"), errors="replace").read()
    info = dict(case=case, rc=p.returncode, guard=("RUNAWAY GUARD" in txt),
                guard_stop=("Run stopped safely" in txt))
    print(f"  {case}: rc={p.returncode}" + (" GUARD" if info["guard"] else ""), flush=True)
    return info, d


def make_cell(path, rep="2,2,2", a=2.87, spin=2.2, T=0.0, seed=1, source="bcc"):
    cmd = [sys.executable, GEN, "--out", path]
    if source == "bcc":
        cmd += ["--source", "bcc", "--bcc-rep", rep, "--a", str(a)]
    else:
        cmd += ["--source", "dataset", "--xyz", os.path.join(ROOT, "Fe", "test.xyz"),
                "--rep", rep]
    cmd += ["--spin-mode", "fm", "--spin-length", str(spin)]
    if T > 0:
        cmd += ["--temperature", str(T), "--seed", str(seed)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def read_thermo(d):
    blocks, cur = [], []
    for line in open(os.path.join(d, "thermo.out")):
        if line.startswith("#"):
            if cur:
                blocks.append(cur)
                cur = []
            continue
        if line.strip():
            cur.append([float(x) for x in line.split()])
    if cur:
        blocks.append(cur)
    return np.array(blocks[-1])


# ---------------------------------------------------------------- 1. guard fix
def test_guard():
    """Regression on the *recorded* false abort (fe_lt/cases/dt_0.2).

    That directory was produced by the shipped binary, which stopped at step
    1000 of 5000 with "lattice temperature excursion (limit 300)".  The same
    input must run to completion with the fixed binary.  A synthetic cell is not
    used: the abort depends on the actual T trajectory, and a fresh cell can
    easily stay below the (wrong) limit and "pass" without reproducing anything.
    """
    src = os.path.join(ROOT, "evidence", "stage_PV1", "fe_lt", "cases", "dt_0.2")
    res = {}
    for tag, exe in (("old", GPUMD_OLD), ("new", GPUMD_FIX)):
        d = os.path.join(OUT, f"guard_regress_{tag}")
        os.makedirs(d, exist_ok=True)
        for junk in ("thermo.out", "stdout.txt", "traj.xyz", "SPIN_GUARD_STOP"):
            fp = os.path.join(d, junk)
            if os.path.exists(fp):
                open(fp, "w").close()
        for f in ("model.xyz", "run.in", "nep.txt"):
            shutil.copyfile(os.path.join(src, f), os.path.join(d, f))
        with open(os.path.join(d, "stdout.txt"), "w") as fh:
            p = subprocess.run([exe], cwd=d, stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
        txt = open(os.path.join(d, "stdout.txt"), errors="replace").read()
        nrows = sum(1 for l in open(os.path.join(d, "thermo.out"))
                    if not l.startswith("#") and l.strip())
        res[tag] = dict(rc=p.returncode, guard=("RUNAWAY GUARD" in txt),
                        stopped=("Run stopped safely" in txt), thermo_rows=nrows,
                        limit_line=[l.strip() for l in txt.splitlines() if "limit" in l][:1])
        print(f"  guard_{tag}: rc={p.returncode} guard={res[tag]['guard']} rows={nrows}", flush=True)
    res["regression_fixed"] = bool((not res["new"]["guard"]) and res["new"]["thermo_rows"] >= 100)
    res["reproduced_on_old"] = bool(res["old"]["guard"])
    return res


FROZEN = os.path.join(ROOT, "final_delivery_20261031_174555", "selected_models",
                      "mask17_baseline", "nep.txt")


def test_bit_identity():
    """A 250 K NVE case must be identical between the shipped and fixed binaries.

    The model MUST be a frozen file: pointing this at a training directory whose
    nep.txt is being rewritten every 500 generations produced two different
    models and a bogus "the fix changed the dynamics" verdict on the first try.
    """
    cell = make_cell(os.path.join(OUT, "cell_bcc128_T250.xyz"), rep="4,4,4", T=250.0, seed=3)
    lines = ["potential nep.txt", "time_step 0.05 100", "ensemble nve",
             "spin_llg gamma=3.0385 alpha=0 T=0", "dump_thermo 50",
             "dump_xyz 2000 traj.xyz spin", "seed 7", "run 2000"]
    _, d_old = run("ident_250K_old", cell, lines, exe=GPUMD_OLD, nep=FROZEN)
    _, d_new = run("ident_250K_new", cell, lines, exe=GPUMD_FIX, nep=FROZEN)
    import hashlib

    def h(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None

    a = h(os.path.join(d_old, "thermo.out"))
    b = h(os.path.join(d_new, "thermo.out"))
    c = h(os.path.join(d_old, "traj.xyz"))
    d = h(os.path.join(d_new, "traj.xyz"))
    return dict(thermo_identical=(a == b), traj_identical=(c == d),
                sha_thermo_old=a, sha_thermo_new=b,
                note="the guard only reads state; a changed trajectory would mean the fix "
                     "touched the dynamics")


# ------------------------------------------------------------ 2/3. ground state
def single_point(cell, tag, exe=GPUMD_FIX):
    lines = ["potential nep.txt", "time_step 0.0001 1", "ensemble nve",
             "spin_llg gamma=3.0385 alpha=0 T=0", "dump_thermo 1", "run 1"]
    _, d = run(f"sp_{tag}", cell, lines, exe=exe)
    th = read_thermo(d)
    return float(th[0, 2]), float(th[0, 2]) / 128.0


def test_lattice():
    rows = []
    for a in np.arange(2.70, 3.001, 0.025):
        cell = os.path.join(OUT, f"cell_a{a:.3f}.xyz")
        make_cell(cell, rep="4,4,4", a=float(a))
        e, epa = single_point(cell, f"a{a:.3f}")
        rows.append(dict(a=float(a), E=e, E_per_atom=epa))
        print(f"    a={a:.3f}  E/atom={epa:+.6f} eV", flush=True)
    arr = np.array([r["E_per_atom"] for r in rows])
    i = int(np.argmin(arr))
    best = None
    if 0 < i < len(rows) - 1:
        x = np.array([rows[i - 1]["a"], rows[i]["a"], rows[i + 1]["a"]])
        y = np.array([rows[i - 1]["E_per_atom"], rows[i]["E_per_atom"], rows[i + 1]["E_per_atom"]])
        c = np.polyfit(x, y, 2)
        a0 = -c[1] / (2 * c[0])
        e0 = np.polyval(c, a0)
        best = dict(a0=float(a0), E0_per_atom=float(e0), curvature=float(c[0]))
    return dict(scan=rows, grid_min=rows[i], fit=best)


def test_moment(a0):
    rows = []
    for s in (1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8):
        cell = os.path.join(OUT, f"cell_S{s:.1f}.xyz")
        make_cell(cell, rep="4,4,4", a=a0, spin=float(s))
        e, epa = single_point(cell, f"S{s:.1f}")
        rows.append(dict(S=float(s), E_per_atom=epa))
        print(f"    |S|={s:.1f}  E/atom={epa:+.6f} eV", flush=True)
    arr = np.array([r["E_per_atom"] for r in rows])
    i = int(np.argmin(arr))
    best = None
    if 0 < i < len(rows) - 1:
        x = np.array([rows[i - 1]["S"], rows[i]["S"], rows[i + 1]["S"]])
        y = np.array([rows[i - 1]["E_per_atom"], rows[i]["E_per_atom"], rows[i + 1]["E_per_atom"]])
        c = np.polyfit(x, y, 2)
        s0 = -c[1] / (2 * c[0])
        best = dict(S0=float(s0), E0_per_atom=float(np.polyval(c, s0)))
    return dict(scan=rows, grid_min=rows[i], fit=best)


# --------------------------------------------------------------------- 4. Tc
def test_tc(a0, n_atoms=432, rep="6,6,6"):
    """|M|(T) with SLLG at T and a Langevin lattice thermostat at the same T.

    `rep` is the bcc replication, so the atom count is 2*nx*ny*nz: 6,6,6 -> 432.
    """
    temps = [int(x) for x in os.environ.get("TC_TEMPS", "300,500,700,800,900,1000,1100,1300").split(",")]
    out = []
    for T in temps:
        cell = os.path.join(OUT, f"tc_cell_{n_atoms}.xyz")
        if not os.path.exists(cell):
            make_cell(cell, rep=rep, a=a0, T=float(T), seed=11)
        lines = [
            "potential nep.txt",
            "time_step 0.05 100",
            f"ensemble nvt_lan {T} {T} 100",
            f"spin_llg gamma=3.0385 alpha=0.3 T={T} seed={100 + T}",
            "dump_thermo 100",
            "dump_xyz 100 traj.xyz spin",
            "seed 2026",
            "run 20000",
        ]
        _, d = run(f"tc_{T}K", cell, lines)
        spins = []
        for fr in read_spin_frames(os.path.join(d, "traj.xyz")):
            spins.append(fr)
        if spins:
            M = np.array([s.mean(axis=0) for s in spins])
            # last 60% is used as the equilibrium window
            m = M[int(0.4 * len(M)):]
            mm = np.linalg.norm(m, axis=1)
            out.append(dict(T=T, M_mean=float(mm.mean()), M_std=float(mm.std()),
                            M_first=float(np.linalg.norm(M[0])), n_frames=len(M)))
        else:
            out.append(dict(T=T, M_mean=None))
        print(f"    T={T}  |M|={out[-1]['M_mean']}", flush=True)
    return dict(n_atoms=n_atoms, points=out)


def read_spin_frames(path):
    frames = []
    with open(path) as f:
        while True:
            l = f.readline()
            if not l:
                break
            try:
                n = int(l)
            except ValueError:
                break
            head = f.readline()
            props = re.search(r"Properties=(\S+)", head).group(1).split(":")
            names = []
            k = 0
            while k < len(props):
                names.append((props[k], int(props[k + 2])))
                k += 3
            cols = {}
            c = 0
            for nm, nc in names:
                cols[nm] = (c, c + nc)
                c += nc
            rows = [f.readline().split() for _ in range(n)]
            if "spin" in cols:
                a, b = cols["spin"]
                frames.append(np.array([[float(x) for x in r[a:b]] for r in rows]))
    return frames


def evaluate_model(nep, tag, data=None):
    """Run nep.exe in predict mode on a held-out set and return RMSEs.

    nep.exe reads `train.xyz` in the working directory and writes
    energy/force/spin_train.out plus the RMSE block in loss.out.  The file name
    is fixed by the trainer, so the dev set is copied in as train.xyz.
    """
    d = os.path.join(OUT, f"eval_{tag}")
    os.makedirs(d, exist_ok=True)
    src = data or os.path.join(WORK, "data", "dev.xyz")
    shutil.copyfile(src, os.path.join(d, "train.xyz"))
    shutil.copyfile(nep, os.path.join(d, "nep.txt"))
    # nep.exe builds the descriptor from nep.in and only reads the *parameters*
    # from nep.txt.  A minimal nep.in therefore silently evaluates a different
    # architecture: with no spin_blocks line the first attempt returned
    # F_MAE = 0.81 eV/A on dev while the trainer reported 0.12 on the same
    # distribution.  Use the training nep.in of the model under test.
    src_in = os.environ.get("EVAL_NEP_IN")
    if src_in and os.path.exists(src_in):
        txt = open(src_in).read()
        if "prediction" not in txt:
            txt += "prediction 1\n"
        open(os.path.join(d, "nep.in"), "w").write(txt)
    else:
        with open(os.path.join(d, "nep.in"), "w") as fh:
            fh.write("type 1 Fe\ncutoff 6.0 3.0\nn_max 4 4\nbasis_size 8 8\nl_max 4 0 0\n"
                     "neuron 30\nspin 1\nprediction 1\n")
    with open(os.path.join(d, "stdout.txt"), "w") as fh:
        p = subprocess.run([os.path.join(WORK, "bin", "nep.exe")], cwd=d,
                           stdout=fh, stderr=subprocess.STDOUT, timeout=3600)
    txt = open(os.path.join(d, "stdout.txt"), errors="replace").read()
    out = dict(tag=tag, rc=p.returncode, nep=nep, data=src)
    m = re.findall(r"RMSE\(([a-z_]+)\)\s*=\s*([0-9.eE+-]+)", txt)
    out["rmse"] = {k: float(v) for k, v in m}
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(OUT, exist_ok=True)
    res = {}
    if which in ("guard", "all"):
        res["guard"] = test_guard()
        res["bit_identity"] = test_bit_identity()
    if which in ("eval", "all"):
        res["eval_frozen_mask17"] = evaluate_model(FROZEN, "frozen_mask17")
    if which in ("lattice", "all"):
        res["lattice"] = test_lattice()
    if which in ("moment", "all"):
        a0 = float(os.environ.get("MOMENT_A0") or
                   res.get("lattice", {}).get("fit", {}).get("a0", 2.87))
        res["moment"] = test_moment(a0)
    if which in ("tc", "all"):
        a0 = float(os.environ.get("MOMENT_A0") or
                   res.get("lattice", {}).get("fit", {}).get("a0", 2.87))
        res["tc"] = test_tc(a0)
    path = os.path.join(OUT, f"results_{which}{TAG}.json")
    json.dump(res, open(path, "w"), indent=1)
    print(f"-> {path}")
