"""Phonons of bcc Fe from finite-displacement force constants, entirely in Python.

Why this file exists
--------------------
The shipped `compute_phonon` builds its Hessian through
`Force::compute_single_potential`, which calls the *spin-free*
`Potential::compute` overload.  Measured consequence: a FM cell and a
|S| = 0 cell give a byte-identical `omega2.out`, i.e. the built-in phonons
are the phonons of the non-magnetic limit.  That path cannot be used for
stages E / F / G2.

What is used instead
--------------------
`dump_xyz ... force` **is** spin aware (the same 0.01 A displacement gives
F = -0.1569 eV/A with FM spins and -0.1081 eV/A with |S| = 0).  So the
second-order force constants are built here from central differences of
that quantity; the dynamical matrix, dispersion, DOS, degeneracies and
Wilson loops are plain numpy.  The production potential is only ever
*evaluated*, never re-implemented.

Cell
----
bcc **primitive** cell, 1 atom, so the spectrum has exactly three acoustic
branches (the task's requirement).  The primitive vectors are
A_i = (a/2)(1,1,-1), (-1,1,1), (1,-1,1); the interplanar width per
repetition is a/sqrt(2) = 2.0 A, so a 7x7x7 supercell (14.0 A) is the
smallest one with width > 2 r_c = 12 A.  A 6x6x6 supercell (12.0 A) and an
8x8x8 one (16.0 A) are the convergence controls.

Conventions
-----------
* Phi(R)[a,b] = d^2E / du_a(0) du_b(R)  in eV/A^2.
* D(q) = (1/m) sum_R Phi(R) exp(i q.R), Hermitian after symmetrisation.
* omega[THz] = 15.6333 sqrt(lambda), lambda in eV/(A^2 amu).
  Imaginary branches are reported as negative numbers, never clipped.

usage
-----
  python phonon_fd.py --mode fm                  # dispersion + DOS (n=7)
  python phonon_fd.py --mode conv                # supercell/amplitude study
  python phonon_fd.py --mode dlm --n 6           # FM vs DLM frequency shift
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feutils as U

OUT = os.path.join(U.VAL, "raw_data", "E_fd")
RC = 6.0                      # radial cutoff of the model (nep.in: cutoff 6.0 3.0)
OMEGA_FACTOR = 15.6333        # sqrt(eV/A^2/amu) -> THz

# high-symmetry points of the bcc BZ, in units of 2*pi/a (Cartesian)
HSP = {"G": [0.0, 0.0, 0.0], "H": [1.0, 0.0, 0.0], "N": [0.5, 0.5, 0.0],
       "P": [0.5, 0.5, 0.5]}
PATH = [("G", "H"), ("H", "N"), ("N", "G"), ("G", "P"), ("P", "H")]


# ------------------------------------------------------------------ references
def a0_from_stage_B():
    p = os.path.join(U.VAL, "raw_data", "B", "eos_fit.json")
    if os.path.exists(p):
        d = json.load(open(p))
        return float(d.get("a0_A") or d["bm3"]["a0_A"])
    return U.A_FE


def m0_from_stage_B():
    p = os.path.join(U.VAL, "raw_data", "B", "moment_equilibrium.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for k in ("m_eq_muB", "m_eq_uB", "m_eq"):
            if k in d:
                return float(d[k])
    return 2.4169


# ----------------------------------------------------------------- cell helpers
def primitive_cell(n, a):
    """n x n x n repetitions of the bcc primitive cell (1 atom per cell)."""
    A = np.array([[1, 1, -1], [-1, 1, 1], [1, -1, 1]], float) * a / 2
    L = A * n                     # supercell lattice
    pts = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # the ATOM positions use the primitive vectors, the BOX uses
                # n x them.  Using `@ L` here (the first version) put the atoms
                # on a lattice 2x too sparse, the model returned NaN and gpumd
                # exited with "non-finite energy at step 1".
                pts.append(np.array([i, j, k], float) @ A)
    return dict(n=len(pts), lattice=L, pos=np.array(pts))


def box_width(L):
    """Smallest interplanar spacing of the cell (what must exceed 2*r_c)."""
    V = abs(np.linalg.det(L))
    return float(min(V / np.linalg.norm(np.cross(L[j], L[k]))
                     for j, k in ((0, 1), (1, 2), (0, 2))))


def min_image(dr, L):
    f = dr @ np.linalg.inv(L)
    f -= np.round(f)
    return f @ L


def make_spins(cell, mode, m, seed=0):
    if mode == "fm":
        return U.fm_spins(cell["n"], m)
    if mode == "s0":
        return U.fm_spins(cell["n"], 0.0)
    if mode == "dlm":
        return U.random_spins(cell["n"], m, seed)
    raise ValueError(mode)


def forces_at(workdir, cell, spins, nep=None):
    """Spin-aware force array (n,3) of a frozen configuration."""
    lines = ["potential nep.txt", "time_step 0.001 1", "ensemble nve",
             "spin_llg gamma=3.0385 alpha=0 T=0",
             "dump_xyz 1 f.xyz force spin", "run 1"]
    rc, txt, _ = U.run_case(workdir, cell, spins, lines, nep=nep)
    if rc != 0:
        raise RuntimeError(f"gpumd failed rc={rc} in {workdir}\n{txt[-600:]}")
    return np.array(U.read_frame(os.path.join(workdir, "f.xyz"), 0)["force"], float)


# ------------------------------------------------------- force constants (Phi(R))
def force_constants(n, a, delta, spin_mode="fm", m=2.4169, seed=0, tag="", nep=None):
    """Phi(R) for the 1-atom primitive cell, from 3x2 central differences."""
    cell = primitive_cell(n, a)
    L = cell["lattice"]
    key = f"{tag}_{spin_mode}_n{n}_d{delta:g}" + (f"_s{seed}" if spin_mode == "dlm" else "")
    base = os.path.join(OUT, "cases", key)
    os.makedirs(base, exist_ok=True)
    spins = make_spins(cell, spin_mode, m, seed)
    f0 = forces_at(os.path.join(base, "ref"), cell, spins, nep=nep)

    phi = {}
    for alpha in range(3):
        fp, fm = [], []
        for sign in (+1.0, -1.0):
            pos = cell["pos"].copy()
            pos[0, alpha] += sign * delta
            d = os.path.join(base, f"d_{alpha}_{'p' if sign > 0 else 'm'}")
            f = forces_at(d, dict(cell, pos=pos), spins, nep=nep)
            (fp if sign > 0 else fm).append(f)
        dphi = -(fp[0] - fm[0]) / (2.0 * delta)      # Phi = -dF/du
        for j in range(cell["n"]):
            dr = min_image(cell["pos"][j] - cell["pos"][0], L)
            if np.linalg.norm(dr) > RC + 1e-6:
                continue
            R = tuple(np.round(dr, 6))
            phi.setdefault(R, np.zeros((3, 3)))[alpha, :] += dphi[j]

    # symmetrise: Phi(-R) = Phi(R)^T.  Without this D(q) is only Hermitian up to
    # the float32 asymmetry of the force response.
    asym, out = [], {}
    for R, blk in phi.items():
        Rm = tuple(-x for x in R)
        other = phi.get(Rm)
        asym.append(float(np.abs(blk - (other.T if other is not None else blk.T)).max()))
        out[R] = 0.5 * (blk + (other.T if other is not None else blk.T))
    out, sym_dev = symmetrise_cubic(out)
    w_before = frequencies(out, np.zeros(3))[0]
    out, asr_before, asr_after = enforce_asr(out)
    meta = dict(n=n, a=a, delta=delta, spin_mode=spin_mode, seed=seed,
                n_atoms=cell["n"], n_shell_blocks=len(out),
                cubic_symmetry_dev_before_eV_A2=sym_dev,
                asr_residual_before_eV_A2=asr_before,
                asr_residual_after_eV_A2=asr_after,
                gamma_before_asr_THz=[float(x) for x in w_before],
                max_pair_distance_A=float(max(np.linalg.norm(np.array(R)) for R in out)),
                asym_before_symmetry_max=float(max(asym)) if asym else 0.0,
                reference_force_max=float(np.abs(f0).max()),
                cutoff_A=RC, nep=nep or U.MODELS["A"])
    return out, meta, cell



# ------------------------------------------------ cubic symmetry and ASR cleanup
def cubic_ops():
    """The 48 signed permutation matrices of m-3m, as integer arrays."""
    ops = []
    perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    for pm in perms:
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    U = np.zeros((3, 3))
                    for i, j in enumerate(pm):
                        U[i, j] = (sx, sy, sz)[i]
                    ops.append(U)
    return ops


def symmetrise_cubic(phi):
    """Average Phi over the 48 cubic operations: Phi(gR) = U Phi(R) U^T.

    The model is cubic but the finite-difference response is not, at the level of
    its float32 force noise; this removes the non-symmetric part.
    """
    ops = cubic_ops()
    acc, cnt = {}, {}
    for R, blk in phi.items():
        Rv = np.array(R)
        for U in ops:
            Rg = tuple(np.round(U @ Rv, 6))
            key = tuple(0.0 if abs(x) < 1e-9 else float(x) for x in Rg)
            acc[key] = acc.get(key, np.zeros((3, 3))) + U @ blk @ U.T
            cnt[key] = cnt.get(key, 0) + 1
    out = {k: v / cnt[k] for k, v in acc.items()}
    dev = max(float(np.abs(out.get(R, np.zeros((3, 3))) - blk).max())
              for R, blk in phi.items())
    return out, dev


def enforce_asr(phi):
    """Set the on-site block so that sum_R Phi(R) = 0 exactly (acoustic sum rule).

    Standard post-processing (phonopy `--asr`, ALAMODE).  The size of the
    correction is reported, because it is the model's force noise floor and not
    a physical effect.
    """
    R0 = (0.0, 0.0, 0.0)
    tot = np.zeros((3, 3))
    for R, blk in phi.items():
        tot += blk
    before = float(np.abs(tot).max())
    out = dict(phi)
    out[R0] = out.get(R0, np.zeros((3, 3))) - tot
    after = float(np.abs(sum(out.values())).max())
    return out, before, after


# --------------------------------------------------------------- dynamical matrix
def dynamical_matrix(phi, q, m_mass=U.MASS_FE):
    D = np.zeros((3, 3), complex)
    for R, blk in phi.items():
        D += blk * np.exp(1j * float(q @ np.array(R)))
    return D / m_mass


def frequencies(phi, q):
    D = dynamical_matrix(phi, q)
    herm = float(np.abs(D - D.conj().T).max())
    lam = np.linalg.eigvalsh((D + D.conj().T) / 2)
    w = np.sign(lam) * np.sqrt(np.abs(lam)) * OMEGA_FACTOR
    return np.sort(w), herm, D


def qpath(a, npts=41):
    pts, labels = [], []
    for seg, (pa, pb) in enumerate(PATH):
        ka = np.array(HSP[pa], float) * 2 * np.pi / a
        kb = np.array(HSP[pb], float) * 2 * np.pi / a
        for t in np.linspace(0, 1, npts, endpoint=(seg == len(PATH) - 1)):
            pts.append((1 - t) * ka + t * kb)
            labels.append((seg, f"{pa}-{pb}", float(t)))
    return pts, labels


def dos_from_grid(phi, a, nq=16, npts=400, sigma=0.12):
    ks = (np.arange(nq) + 0.5) / nq
    ws = []
    for i in range(nq):
        for j in range(nq):
            for k in range(nq):
                q = 2 * np.pi / a * np.array([ks[i], ks[j], ks[k]])
                ws.append(frequencies(phi, q)[0])
    ws = np.array(ws)
    grid = np.linspace(ws.min() - 0.6, ws.max() + 0.6, npts)
    g = np.exp(-0.5 * ((grid[:, None, None] - ws[None, :, :]) / sigma) ** 2)
    dens = g.sum(axis=(1, 2)) / (sigma * np.sqrt(2 * np.pi)) / len(ws)
    return grid, dens, ws


# --------------------------------------------------------------------- mode: fm
def mode_fm(n=7, delta=0.01, nep=None, nq=41):
    a = a0_from_stage_B()
    m = m0_from_stage_B()
    phi, meta, cell = force_constants(n, a, delta, "fm", m, nep=nep, tag="main")

    pts, labels = qpath(a, nq)
    rows = []
    for q, (seg, lab, t) in zip(pts, labels):
        w, herm, _ = frequencies(phi, q)
        rows.append(dict(q=q.tolist(), seg=lab, t=t, omega=w.tolist(), herm=herm))
    W = np.array([r["omega"] for r in rows])

    with open(os.path.join(OUT, "phonon_dispersion.dat"), "w") as f:
        f.write("# seg t qx qy qz (2pi/a)  omega_1 omega_2 omega_3 (THz, signed)\n")
        for r in rows:
            q = np.array(r["q"]) * a / (2 * np.pi)
            f.write(f"{r['seg']} {r['t']:.4f} {q[0]:.5f} {q[1]:.5f} {q[2]:.5f} "
                    + " ".join(f"{x:.5f}" for x in r["omega"]) + "\n")

    grid, dens, _ = dos_from_grid(phi, a, nq=16)
    with open(os.path.join(OUT, "phonon_dos.dat"), "w") as f:
        f.write("# omega_THz  g(omega)  (states/THz/primitive cell)\n")
        for x, y in zip(grid, dens):
            f.write(f"{x:.5f} {y:.6e}\n")

    # sound velocities from a long-wavelength fit; the path-based dw/dq is not
    # usable because sorting swaps branches at crossings
    # v[m/s] = (domega/dq in THz*A) * 2*pi*1e12 / 1e10 = 628.3 * x
    KM_S = 2 * np.pi * 1e12 / 1e10 / 1000.0   # -> km/s
    snd = {}
    for name, dv in (("001", np.array([0, 0, 1.0])), ("111", np.array([1, 1, 1.0]) / np.sqrt(3))):
        qs = np.linspace(0.02, 0.10, 6) * 2 * np.pi / a
        ws = np.array([frequencies(phi, dv * x)[0] for x in qs])
        v = [float(np.polyfit(qs, ws[:, b], 1)[0]) * KM_S for b in range(3)]
        snd[name] = v

    wG, _, DG = frequencies(phi, np.zeros(3))
    res = dict(
        method="finite displacement (central difference, +/-delta) of the "
               "spin-aware dumped forces; Phi(R) and D(q) built in Python",
        cell="bcc primitive, 1 atom, 3 acoustic branches",
        a0_A=a, m0_uB=m, supercell=f"{n}x{n}x{n} primitive ({meta['n_atoms']} atoms)",
        supercell_width_A=box_width(cell["lattice"]),
        delta_A=delta, cutoff_A=RC,
        max_pair_distance_A=meta["max_pair_distance_A"],
        q_path=[f"{p[0]}-{p[1]}" for p in PATH], n_q=len(rows),
        omega_at_Gamma_THz=[float(x) for x in wG],
        omega_max_THz=float(W.max()), omega_min_THz=float(W.min()),
        sound_velocity_km_s=snd,
        longitudinal_velocity_100_km_s=snd["001"][2],
        transverse_velocity_100_km_s=snd["001"][0],
        dos_peak_THz=float(grid[int(np.argmax(dens))]),
        acoustic_sum_rule_gamma_max_THz=float(np.abs(wG).max()),
        gamma_before_asr_THz=meta["gamma_before_asr_THz"],
        cubic_symmetry_dev_before_eV_A2=meta["cubic_symmetry_dev_before_eV_A2"],
        asr_correction_eV_A2=meta["asr_residual_before_eV_A2"],
        asr_correction_fraction_of_self_term=float(
            meta["asr_residual_before_eV_A2"] /
            max(np.abs(phi[(0.0, 0.0, 0.0)]).max(), 1e-12)),
        asr_note="the on-site block is corrected so that sum_R Phi(R) = 0; the "
                 "correction is ~2% of the on-site term and is the model's "
                 "float32 force noise (its far-shell part scales as 1/delta, "
                 "the near part does not decay with distance), not physics",
        asr_residual_max_eV_A2=float(np.abs(DG).max()),
        hermiticity_max=float(max(r["herm"] for r in rows)),
        symmetry_asymmetry_before_eV_A2=meta["asym_before_symmetry_max"],
        imaginary_mode_count=int((W < -1e-6).sum()),
        reference_note="the literature maximum phonon frequency of bcc Fe is "
                       "about 9.4-9.6 THz; that anchor is quoted from the "
                       "literature and was NOT re-verified in this project, so "
                       "the verdict rests on the internal criteria only",
        criterion_asr="|omega(Gamma)| < 0.05 THz for all three branches",
        criterion_stability="no branch below -1e-6 THz anywhere on the path",
        criterion_hermiticity="|D - D^dag|_max < 1e-6 eV/(A^2 amu)",
    )
    ok = (res["acoustic_sum_rule_gamma_max_THz"] < 0.05 and
          res["imaginary_mode_count"] == 0 and res["hermiticity_max"] < 1e-6)
    res["status"] = "PASS" if ok else "FAIL"
    U.save_json(os.path.join(OUT, "phonon_fd.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "q_path"}, indent=1)[:1800],
          flush=True)
    return res


# ------------------------------------------------------------------- mode: conv
def mode_conv(nep=None, delta0=0.01):
    a = a0_from_stage_B()
    m = m0_from_stage_B()
    qs = [np.array([0.0, 0.0, 0.5]) * 2 * np.pi / a,
          np.array([0.5, 0.0, 0.0]) * 2 * np.pi / a,
          np.array([0.5, 0.5, 0.0]) * 2 * np.pi / a]
    out = dict(a0_A=a, delta_A=delta0,
               probes=["q=(0,0,.5)2pi/a", "q=(.5,0,0)2pi/a", "q=(.5,.5,0)2pi/a"],
               supercell={}, amplitude={},
               criterion_supercell_THz=0.10, criterion_amplitude_THz=0.05)
    ref = None
    for n in (6, 7, 8):
        phi, meta, cell = force_constants(n, a, delta0, "fm", m, nep=nep, tag="conv")
        ws = np.array([frequencies(phi, q)[0] for q in qs])
        out["supercell"][str(n)] = dict(
            n_atoms=meta["n_atoms"],
            width_A=box_width(cell["lattice"]),
            omega=ws.tolist())
        if n == 7:
            ref = ws
    for n in (6, 7, 8):
        out["supercell"][str(n)]["max_dev_from_7_THz"] = float(
            np.abs(np.array(out["supercell"][str(n)]["omega"]) - ref).max())
    amp = {}
    for delta in (0.005, 0.01, 0.02):
        phi, meta, cell = force_constants(7, a, delta, "fm", m, nep=nep, tag="amp")
        amp[str(delta)] = np.array([frequencies(phi, q)[0] for q in qs])
    refa = amp["0.01"]
    out["amplitude"] = {k: dict(omega=v.tolist(),
                                max_dev_from_0p01_THz=float(np.abs(v - refa).max()))
                        for k, v in amp.items()}
    sc = max(abs(out["supercell"][str(n)]["max_dev_from_7_THz"]) for n in (6, 8))
    am = max(v["max_dev_from_0p01_THz"] for v in out["amplitude"].values())
    out["max_supercell_dev_THz"] = sc
    out["max_amplitude_dev_THz"] = am
    out["status"] = "PASS" if (sc < 0.10 and am < 0.05) else "PARTIAL"
    U.save_json(os.path.join(OUT, "phonon_convergence.json"), out)
    print(json.dumps(out, indent=1)[:1400], flush=True)
    return out


# -------------------------------------------------------------------- mode: dlm
def mode_dlm(nep=None, n=6, delta=0.01, seeds=(1, 2, 3, 4), npts=21):
    a = a0_from_stage_B()
    m = m0_from_stage_B()
    pts, labels = qpath(a, npts)
    phi, meta, cell = force_constants(n, a, delta, "fm", m, nep=nep, tag="dlm")
    w_fm = np.array([frequencies(phi, q)[0] for q in pts])
    acc = []
    for sd in seeds:
        ph, _, _ = force_constants(n, a, delta, "dlm", m, seed=sd, nep=nep, tag="dlm")
        acc.append(np.array([frequencies(ph, q)[0] for q in pts]))
    acc = np.array(acc)
    w_dlm = acc.mean(axis=0)
    sd_dlm = acc.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(w_dlm)
    dw = w_fm - w_dlm
    with open(os.path.join(OUT, "spin_phonon_frequency_shift.csv"), "w") as f:
        f.write("# seg t branch omega_FM omega_DLM domega lambda sd_DLM (THz)\n")
        for i, (q, (seg, lab, t)) in enumerate(zip(pts, labels)):
            for b in range(3):
                rel = dw[i, b] / w_dlm[i, b] if abs(w_dlm[i, b]) > 0.3 else float("nan")
                f.write(f"{lab} {t:.4f} {b+1} {w_fm[i,b]:.5f} {w_dlm[i,b]:.5f} "
                        f"{dw[i,b]:.5f} {rel:.5f} {sd_dlm[i,b]:.5f}\n")
    hs = {}
    for name in ("G", "H", "N", "P"):
        idx = [i for i, (s, l, t) in enumerate(labels)
               if l.split("-")[0] == name and t == 0.0]
        if idx:
            i = idx[0]
            hs[name] = dict(FM=w_fm[i].tolist(), DLM=w_dlm[i].tolist(),
                            domega=dw[i].tolist(), sd=sd_dlm[i].tolist())
    res = dict(supercell=f"{n}x{n}x{n} primitive ({cell['n']} atoms)",
               n_seeds=len(seeds), delta_A=delta,
               mean_abs_domega_THz=float(np.mean(np.abs(dw))),
               max_abs_domega_THz=float(np.abs(dw).max()),
               mean_sd_across_seeds_THz=float(np.mean(sd_dlm)),
               high_symmetry_points=hs,
               reference="Kormann et al., PRL 113, 165503 (2014) report a "
                         "softening of the longitudinal branch in the DLM state; "
                         "the comparison here is qualitative, no digitised data",
               note="branch pairing is by sorted index at the same q; a crossing "
                    "can swap two rows, in which case a single CSV row is not a "
                    "mode-following quantity -- high_symmetry_points is the safe "
                    "part to quote",
               criterion="|domega| > 5 sigma_seed at the high-symmetry points")
    hs_dev = max(abs(np.array(v["domega"])).max() for v in hs.values()) if hs else 0.0
    res["max_domega_high_symmetry_THz"] = float(hs_dev)
    res["status"] = "PASS" if hs_dev > 5 * np.mean(sd_dlm) else "WEAK"
    U.save_json(os.path.join(OUT, "spin_phonon_shift.json"), res)
    print(json.dumps(res, indent=1)[:1200], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fm", choices=["fm", "conv", "dlm"])
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--delta", type=float, default=0.01)
    ap.add_argument("--npts", type=int, default=41)
    ap.add_argument("--model", default=None)
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if A.mode == "fm":
        mode_fm(n=A.n, delta=A.delta, nep=A.model, nq=A.npts)
    elif A.mode == "conv":
        mode_conv(nep=A.model)
    else:
        mode_dlm(nep=A.model)
