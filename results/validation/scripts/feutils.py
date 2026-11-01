"""Shared helpers for the BCC Fe validation.

Only calls the production binaries; nothing here re-implements the potential.
Every runner writes its inputs, command line and raw stdout next to the data so
that a failed/odd result can be traced back.
"""
import json
import os
import re
import shutil
import subprocess

import numpy as np

ROOT = r"D:/researche_test/rongkunchen/GPUMD-spin/V3"
VAL = os.path.join(ROOT, "validation", "BCC_Fe_full_physics")
SRC = os.path.join(ROOT, "final_delivery_20261101_r18", "staging",
                   "GPUMD-spin-V3-SOURCE-v3.1.0-r18")
EXE = os.path.join(ROOT, "final_delivery_20261101_r18", "win_staging", "bin", "gpumd.exe")
MODELS = {
    "A": os.path.join(SRC, "models", "fe_mask17_2desc_r18.nep.txt"),
    "B": os.path.join(SRC, "models", "fe_mask31_5desc_r18.nep.txt"),
    "r17": os.path.join(SRC, "models", "fe_mask17_r17_baseline.nep.txt"),
}
A_FE = 2.8258          # from the r18 EOS scan (model value, not the experimental one)
MASS_FE = 55.845       # u
KB_EV = 8.617333262e-5
TIME_UNIT = 10.1805    # fs per natural time unit


# ----------------------------------------------------------------- xyz helpers
def bcc_cell(nx, ny, nz, a):
    """Conventional bcc cell replicated nx x ny x nz (2 atoms per cell)."""
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    pts = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for b in basis:
                    pts.append((np.array([i, j, k]) + b) * a)
    P = np.array(pts)
    L = np.array([[nx * a, 0, 0], [0, ny * a, 0], [0, 0, nz * a]])
    return dict(n=len(P), lattice=L, pos=P)


def write_xyz(path, cell, spins, comment=""):
    L, n = cell["lattice"], cell["n"]
    with open(path, "w") as f:
        f.write(f"{n}\n")
        lat = " ".join(f"{x:.10f}" for x in L.reshape(-1))
        f.write(f'Lattice="{lat}" Properties=species:S:1:pos:R:3:spin:R:3')
        f.write(f" {comment}\n" if comment else "\n")
        for i in range(n):
            row = ["Fe"] + [f"{x:.10f}" for x in cell["pos"][i]]
            row += [f"{s:.10f}" for s in spins[i]]
            f.write(" ".join(row) + "\n")


def read_frame(path, index=0):
    lines = open(path).read().splitlines()
    i, k = 0, 0
    while i < len(lines):
        n = int(lines[i])
        if k == index:
            head = lines[i + 1]
            L = np.array([float(x) for x in
                          re.search(r'Lattice="([^"]+)"', head).group(1).split()]).reshape(3, 3)
            rows = [l.split() for l in lines[i + 2:i + 2 + n]]
            # columns follow Properties= ; assume species:1 pos:3 vel:3 spin:3
            props = re.search(r"Properties=(\S+)", head).group(1).split(":")
            names, c = [], 0
            for j in range(0, len(props), 3):
                names.append((props[j], props[j + 1], int(props[j + 2])))
            cols = {}
            for nm, typ, nc in names:
                cols[nm] = (c, c + nc)
                c += nc
            out = dict(n=n, lattice=L, header=head, cols=cols)
            alias = {"forces": "force", "vel": "velocity", "positions": "pos"}
            for nm, (a, b) in cols.items():
                if nm == "species":
                    continue
                out[alias.get(nm, nm)] = np.array([[float(x) for x in r[a:b]]
                                                  for r in rows])
            return out
        i += 2 + n
        k += 1
    raise IndexError(index)


# ----------------------------------------------------------------- run helpers
def run_case(workdir, cell, spins, run_lines, nep=None, timeout=3600, keep=()):
    """Create a case directory, run gpumd once, return (rc, stdout, thermo)."""
    os.makedirs(workdir, exist_ok=True)
    for junk in ("thermo.out", "traj.xyz", "stdout.txt", "magnetization.out",
                 "omega2.out", "D.out", "eigenvector.out", "spin_work.out",
                 "SPIN_GUARD_STOP", "force_constant.out", "H.out"):
        p = os.path.join(workdir, junk)
        if os.path.exists(p):
            open(p, "w").close()
    write_xyz(os.path.join(workdir, "model.xyz"), cell, spins)
    shutil.copyfile(nep or MODELS["A"], os.path.join(workdir, "nep.txt"))
    with open(os.path.join(workdir, "run.in"), "w") as f:
        f.write("\n".join(run_lines) + "\n")
    with open(os.path.join(workdir, "stdout.txt"), "w") as f:
        p = subprocess.run([EXE], cwd=workdir, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout)
    txt = open(os.path.join(workdir, "stdout.txt"), errors="replace").read()
    th = os.path.join(workdir, "thermo.out")
    thermo = read_thermo(th) if os.path.exists(th) and os.path.getsize(th) else np.zeros((0, 3))
    return p.returncode, txt, thermo


def read_thermo(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows)


def static_energy(workdir, cell, spins, nep=None, dt=1e-3):
    """Potential energy (eV) of a fixed configuration, no dynamics."""
    lines = ["potential nep.txt", f"time_step {dt} 1", "ensemble nve",
             "spin_llg gamma=3.0385 alpha=0 T=0", "dump_thermo 1", "run 1"]
    rc, txt, th = run_case(workdir, cell, spins, lines, nep=nep)
    if rc != 0 or len(th) == 0:
        raise RuntimeError(f"gpumd failed rc={rc} in {workdir}\n{txt[-800:]}")
    return float(th[0, 2])


# ----------------------------------------------------------------- spin states
def fm_spins(n, m):
    s = np.zeros((n, 3))
    s[:, 2] = m
    return s


def afm1_spins(cell, m):
    """AFM-I: (001) layers alternate; bcc basis (0,0,0) up / (.5,.5,.5) down."""
    s = fm_spins(cell["n"], m)
    frac = cell["pos"] / np.diag(cell["lattice"])
    for i in range(cell["n"]):
        # the two sublattices of the conventional cell alternate along z
        zc = frac[i, 2]
        s[i, 2] = m if (np.floor(zc + 1e-6) % 2 == 0) == (abs(frac[i, 0] % 1 - 0.5) > 1e-6) else -m
    return s


def checkerboard_spins(cell, m):
    """AFM-II: two sublattices of the bcc lattice are antiparallel (rocksalt-like)."""
    s = fm_spins(cell["n"], m)
    frac = cell["pos"] / np.diag(cell["lattice"])
    # sublattice A = integer sites, B = half-integer sites
    is_b = np.all(np.abs(frac % 1 - 0.5) < 0.25, axis=1)
    s[~is_b, 2] = +m
    s[is_b, 2] = -m
    return s


def random_spins(n, m, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    return v * m


def spiral_spins(cell, m, q, axis=2):
    """Flat spiral: spin rotates in the plane perpendicular to `axis` as q.r."""
    s = np.zeros((cell["n"], 3))
    for i in range(cell["n"]):
        ang = q @ cell["pos"][i]
        if axis == 2:
            s[i] = [m * np.sin(ang), 0.0, m * np.cos(ang)]
        elif axis == 0:
            s[i] = [0.0, m * np.sin(ang), m * np.cos(ang)]
        else:
            s[i] = [m * np.sin(ang), m * np.cos(ang), 0.0]
    return s


def rotate_all(spins, R):
    return spins @ np.array(R).T


def save_json(path, obj):
    json.dump(obj, open(path, "w"), indent=1, default=float)
    return path
