"""Parse the Stage PV1 Fe long-time matrix into JSON + CSV.

Only quantities that a reader can recompute from the raw files are produced:
thermo.out gives T, KE and PE; traj.xyz (when present) gives the spins.  The
energy of a *conservative* run is KE + PE with both columns taken from the same
thermo row, and the reference is the first row of that run -- not a separately
recomputed t = 0 (the R2 lesson: mixing two definitions manufactures drift).
"""

import glob
import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FE_LT = os.path.join(ROOT, "evidence", "stage_PV1", "fe_lt")
CASES = os.path.join(FE_LT, "cases_250K")


def read_thermo(path):
    """Read the LAST run in thermo.out.

    gpumd appends, so a directory that was run twice holds two header blocks
    back to back.  Only the final block belongs to the case as it is described
    by run.in.
    """
    blocks = []
    cur = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                if cur:
                    blocks.append(cur)
                    cur = []
                continue
            if not line.strip():
                continue
            cur.append([float(x) for x in line.split()])
    if cur:
        blocks.append(cur)
    return np.array(blocks[-1]) if blocks else np.array([])


def read_xyz(path, max_frames=None):
    """Yield (lattice, positions, spins, potential) per frame."""
    frames = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        n = int(lines[i])
        head = lines[i + 1]
        lat = np.array([float(x) for x in re.search(r'Lattice="([^"]+)"', head).group(1).split()])
        lat = lat.reshape(3, 3)
        props = re.search(r"Properties=(\S+)", head)
        names = []
        if props:
            toks = props.group(1).split(":")
            k = 0
            while k < len(toks):
                names.append((toks[k], int(toks[k + 2])))
                k += 3
        cols = {}
        c = 0
        for name, ncol in names:
            cols[name] = (c, c + ncol)
            c += ncol
        rows = [ln.split() for ln in lines[i + 2 : i + 2 + n]]
        pos = np.array([[float(x) for x in r[cols["pos"][0] : cols["pos"][1]]] for r in rows])
        spin = None
        if "spin" in cols:
            spin = np.array(
                [[float(x) for x in r[cols["spin"][0] : cols["spin"][1]]] for r in rows]
            )
        pot = None
        if "energy_atom" in cols:
            pot = np.array([float(r[cols["energy_atom"][0]]) for r in rows])
        frames.append(dict(lat=lat, pos=pos, spin=spin, pot=pot, head=head))
        i += 2 + n
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def summarize(case_dir):
    meta = json.load(open(os.path.join(case_dir, "case.json")))
    th = os.path.join(case_dir, "thermo.out")
    if not os.path.exists(th):
        return None
    d = read_thermo(th)
    if len(d) < 3:
        return None
    T, KE, PE = d[:, 0], d[:, 1], d[:, 2]
    # number of atoms from the header
    n = None
    for line in open(th):
        if line.startswith("# num_atoms"):
            n = int(line.split()[-1])
    if n is None:
        return None
    E = KE + PE
    dt_out = None
    for line in open(th):
        if line.startswith("# dt_output"):
            dt_out = float(line.split()[-2])
    # the timestep itself is only in run.in
    dt_fs = None
    ensemble = None
    for line in meta.get("run_in", []):
        tok = line.split()
        if tok and tok[0] == "time_step":
            dt_fs = float(tok[1])
        if tok and tok[0] == "ensemble":
            ensemble = tok[1]
    t_ps = np.arange(len(d)) * dt_out / 1000.0
    dE = E - E[0]
    # linear slope over the second half (skip the initial transient)
    half = len(t_ps) // 2
    if len(t_ps) - half > 2:
        slope = np.polyfit(t_ps[half:], dE[half:] / n, 1)[0]
    else:
        slope = float("nan")
    out = dict(
        case=os.path.basename(case_dir),
        n_atoms=n,
        dt_fs=dt_fs,
        dt_output_fs=dt_out,
        ensemble=ensemble,
        conservative=(ensemble == "nve"),
        n_rows=len(d),
        t_ps=float(t_ps[-1]),
        exit_code=meta["exit_code"],
        guarded=meta.get("guarded", False),
        wall_s=meta.get("wall_s"),
        T_mean=float(T.mean()),
        T_first=float(T[0]),
        T_last=float(T[-1]),
        T_std=float(T.std()),
        dE_max_per_atom=float(np.abs(dE).max() / n),
        dE_end_per_atom=float(dE[-1] / n),
        dE_slope_eV_per_atom_per_ps=float(slope),
        PE_first_per_atom=float(PE[0] / n),
        PE_last_per_atom=float(PE[-1] / n),
    )

    traj = os.path.join(case_dir, "traj.xyz")
    if os.path.exists(traj):
        try:
            frames = read_xyz(traj)
        except Exception as exc:  # a truncated trajectory is itself a result
            out["traj_error"] = str(exc)
            frames = []
        if frames and frames[0]["spin"] is not None:
            spins = np.stack([f["spin"] for f in frames])  # (nf, n, 3)
            mags = np.linalg.norm(spins, axis=2)
            M = spins.mean(axis=1)
            out.update(
                spin_frames=len(frames),
                S_mag_mean=float(mags.mean()),
                S_mag_drift=float(np.abs(mags - mags[0]).max()),
                M_abs_first=float(np.linalg.norm(M[0])),
                M_abs_last=float(np.linalg.norm(M[-1])),
                M_abs_mean=float(np.linalg.norm(M, axis=1).mean()),
                Mz_first=float(M[0, 2]),
                Mz_last=float(M[-1, 2]),
            )
    return out


def main():
    rows = []
    for case_dir in sorted(glob.glob(os.path.join(CASES, "*"))):
        if not os.path.isdir(case_dir):
            continue
        if not os.path.exists(os.path.join(case_dir, "case.json")):
            continue
        # directories renamed with a leading underscore are kept as evidence of
        # a run that did not finish (the R19 guard stop and the NVT->NVE hang)
        if os.path.basename(case_dir).startswith("_"):
            continue
        r = summarize(case_dir)
        if r:
            rows.append(r)
            print(f"{r['case']:26s} dt={r['dt_fs']:<6} t={r['t_ps']:6.2f} ps  "
                  f"dEmax={r['dE_max_per_atom']:.3e}  T={r['T_mean']:.1f}"
                  + ("  GUARD" if r["guarded"] else ""))
    json.dump(rows, open(os.path.join(FE_LT, "fe_lt_results.json"), "w"), indent=1)
    keys = ["case", "n_atoms", "dt_fs", "t_ps", "T_mean", "T_std", "dE_max_per_atom",
            "dE_end_per_atom", "dE_slope_eV_per_atom_per_ps", "S_mag_drift",
            "M_abs_first", "M_abs_last", "exit_code", "guarded", "wall_s"]
    with open(os.path.join(FE_LT, "fe_lt_matrix.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + "\n")
    print(f"\n{len(rows)} cases -> fe_lt_results.json / fe_lt_matrix.csv")


if __name__ == "__main__":
    main()
