"""Relax the lattice of each magnetic state, then compute that state's phonons.

Two things this fixes relative to the first pass:

1. **Relaxation first.** `minimize` cannot be used (it evaluates the spin-free
   energy, see stage E), so the relaxation is done here with a spin-aware energy
   scan + quadratic fit, one lattice constant per magnetic state, and the
   residual force at the optimum is recorded as the equilibrium check.

2. **Phonons for the magnetic cell, not for the primitive cell.**  For a
   collinear AFM the force constants are periodic with the *magnetic* cell, so
   displacing one atom and folding with the primitive periodicity is wrong.  The
   generalised code below displaces every atom of the magnetic basis and builds
   the 3N x 3N dynamical matrix:

      Phi[k, k', R] = d2E / du(k,0) du(k',R)
      D_{k k'}(q) = (1/m) sum_R Phi[k,k',R] exp(i q.(R + tau_k' - tau_k))

   Magnetic states used (all spins along z unless stated):
      FM       1 atom  (primitive bcc)          -> 3 branches
      AFM-G    2 atoms (conventional cube), the two sublattices antiparallel
      AFM-A    4 atoms (1x1x2 conventional), ferromagnetic (001) sheets
               stacked antiferromagnetically with period 2a
      NC-90    2 atoms, second sublattice canted 90 deg in the x-z plane
      DLM      primitive bcc, independent random directions, averaged over
               several configurations (there is no periodic order to respect)

usage: python relax_and_phonons.py [--n 5] [--delta 0.02] [--stage relax|phonon|all]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feutils as U
import phonon_fd as P

OUT = os.path.join(U.VAL, "raw_data", "H_states")
RC = 6.0
OMEGA = P.OMEGA_FACTOR


# --------------------------------------------------------------- magnetic cells
def prim_lattice(a):
    return np.array([[1, 1, -1], [-1, 1, 1], [1, -1, 1]], float) * a / 2


def state_cell(kind, a):
    """Lattice, basis positions (Cartesian) and spin directions of one magnetic cell."""
    if kind == "FM":
        return dict(L=prim_lattice(a), tau=np.zeros((1, 3)),
                    sp=np.array([[0.0, 0.0, 1.0]]), nb=1)
    if kind == "DLM":
        return dict(L=prim_lattice(a), tau=np.zeros((1, 3)),
                    sp=np.array([[0.0, 0.0, 1.0]]), nb=1)   # replaced per config
    conv = np.eye(3) * a
    tau2 = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]) * a
    if kind == "AFM-G":
        return dict(L=conv, tau=tau2,
                    sp=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), nb=2)
    if kind == "NC-90":
        return dict(L=conv, tau=tau2,
                    sp=np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]), nb=2)
    if kind == "AFM-A":
        L = np.diag([a, a, 2 * a])
        tau = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0, 0, 1.0], [0.5, 0.5, 1.5]]) * a
        sp = np.array([[0, 0, 1.0], [0, 0, -1.0], [0, 0, -1.0], [0, 0, 1.0]])
        return dict(L=L, tau=tau, sp=sp, nb=4)
    raise ValueError(kind)


def replicate(st, n):
    """n x n x n supercell of the magnetic cell, with atom -> (basis, R) map."""
    L = st["L"] * n
    tau, sp = st["tau"], st["sp"]
    nb = st["nb"]
    pos, spins, kidx, ridx = [], [], [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                R = np.array([i, j, k], float) @ st["L"]
                for b in range(nb):
                    pos.append(tau[b] + R)
                    spins.append(sp[b])
                    kidx.append(b)
                    ridx.append(R)
    return dict(n=len(pos), lattice=L, pos=np.array(pos),
                spins=np.array(spins), kidx=np.array(kidx),
                ridx=np.array(ridx), nb=nb, magcell=st["L"].copy(),
                tau=tau)


def min_image(dr, L):
    f = dr @ np.linalg.inv(L)
    f -= np.round(f)
    return f @ L


def forces_at(workdir, cell, spins, nep=None):
    lines = ["potential nep.txt", "time_step 0.001 1", "ensemble nve",
             "spin_llg gamma=3.0385 alpha=0 T=0",
             "dump_xyz 1 f.xyz force spin", "run 1"]
    rc, txt, _ = U.run_case(workdir, cell, spins, lines, nep=nep)
    if rc != 0:
        raise RuntimeError(f"gpumd failed rc={rc} in {workdir}\n{txt[-500:]}")
    return np.array(U.read_frame(os.path.join(workdir, "f.xyz"), 0)["force"], float)


# ------------------------------------------------------------------- relaxation
def relax_a(kind, m, nep=None, lo=2.76, hi=2.94, step=0.02, n=None, seeds=(1, 2, 3)):
    """Equilibrium lattice constant of one magnetic state (spin-aware energies).

    The supercell must be wide enough for the 6 A cutoff (width > 2 r_c), which
    is why the first version -- 8 atoms for FM -- gave 2.8368 A instead of the
    2.8254 A from the proper EOS: an 8-atom cell aliases the cutoff.  Sizes used
    here: FM/DLM primitive 7x7x7 (343 atoms), AFM-G / NC-90 conventional 5x5x5
    (250 atoms), AFM-A 5x5x5 of the 4-atom cell (500 atoms).
    """
    if n is None:
        n = dict(FM=7, DLM=7, **{"AFM-G": 5, "NC-90": 5, "AFM-A": 5})[kind]

    def energy(a):
        vals = []
        cfgs = seeds if kind == "DLM" else (None,)
        for sd in cfgs:
            st = replicate(state_cell(kind, a), n)
            if kind == "DLM":
                rng = np.random.default_rng(sd)
                v = rng.normal(size=(st["n"], 3))
                v /= np.linalg.norm(v, axis=1)[:, None]
                spins = v * m
            else:
                spins = st["spins"] * m
            d = os.path.join(OUT, "relax", f"{kind}_n{n}_a{a:.3f}"
                             + (f"_s{sd}" if sd is not None else ""))
            vals.append(U.static_energy(d, st, spins, nep=nep) / st["n"])
        return float(np.mean(vals))

    grid = np.arange(lo, hi + 1e-9, step)
    E = np.array([energy(a) for a in grid])
    i = int(np.argmin(E))
    i0, i1 = max(0, i - 1), min(len(grid) - 1, i + 1)
    c = np.polyfit(grid[i0:i1 + 1], E[i0:i1 + 1], 2)
    a0 = float(-c[1] / (2 * c[0])) if c[0] > 0 else float(grid[i])
    # one refinement pass on a 4x finer grid around the parabola vertex
    # local refinement: 5 points on a 0.003 A grid (the global quartic EOS of
    # stage B is biased by +-0.01 A on this stiff curve; see the report)
    fine = np.linspace(a0 - 0.006, a0 + 0.006, 5)
    Ef = np.array([energy(a) for a in fine])
    c2 = np.polyfit(fine, Ef, 2)
    a0 = float(-c2[1] / (2 * c2[0])) if c2[0] > 0 else a0
    st = replicate(state_cell(kind, a0), n)
    spins = (st["spins"] * m if kind != "DLM" else
             (lambda r: r / np.linalg.norm(r, axis=1)[:, None])(
                 np.random.default_rng(1).normal(size=(st["n"], 3))) * m)
    f = forces_at(os.path.join(OUT, "relax", f"{kind}_check"), st, spins, nep=nep)
    return dict(state=kind, a0_A=a0, E_min_eV_per_atom=float(np.polyval(c2, a0)),
                supercell=f"{n}x{n}x{n}", n_atoms=st["n"],
                width_A=float(P.box_width(st["lattice"])),
                grid=[float(x) for x in grid], E=[float(x) for x in E],
                fine_grid=[float(x) for x in fine], E_fine=[float(x) for x in Ef],
                curvature_eV_per_A2=float(2 * c2[0]),
                max_force_at_a0_eV_per_A=float(np.abs(f).max()),
                note="explicit spin-aware energy scan; `minimize` was NOT used "
                     "because it evaluates the S=0 energy (stage E).  For collinear "
                     "states the force vanishes by symmetry anyway, so the "
                     "equilibrium check is dE/da = 0 from the quadratic fit, not "
                     "max|F|.")


# ------------------------------------------------------- force constants, general
def force_constants(st, delta, nep=None, tag="", symmetrise=False):
    """Phi[k][k'][R] for the magnetic cell; displaces every basis atom."""
    L = st["lattice"]
    nb = st["nb"]
    base = os.path.join(OUT, "cases", tag)
    os.makedirs(base, exist_ok=True)
    # one representative atom per basis index
    reps = [int(np.where(st["kidx"] == b)[0][0]) for b in range(nb)]
    phi = {}
    for b, ia in enumerate(reps):
        for alpha in range(3):
            fp = fm_ = None
            for sign in (+1.0, -1.0):
                pos = st["pos"].copy()
                pos[ia, alpha] += sign * delta
                c2 = dict(st)
                c2["pos"] = pos
                f = forces_at(os.path.join(base, f"k{b}_{alpha}{'p' if sign > 0 else 'm'}"),
                              c2, st["spins"], nep=nep)
                if sign > 0:
                    fp = f
                else:
                    fm_ = f
            dphi = -(fp - fm_) / (2 * delta)
            for j in range(st["n"]):
                dr = min_image(st["pos"][j] - st["pos"][ia], L)
                if np.linalg.norm(dr) > RC + 1e-6:
                    continue
                kj = int(st["kidx"][j])
                R = dr - (st["tau"][kj] - st["tau"][b])
                key = (b, kj, tuple(np.round(R, 6)))
                phi.setdefault(key, np.zeros((3, 3)))[alpha, :] += dphi[j]

    # Phi[k,k',R] = Phi[k',k,-R]^T
    out, asym = {}, []
    for (b, k2, R), blk in phi.items():
        Rm = tuple(-x for x in R)
        other = phi.get((k2, b, Rm))
        asym.append(float(np.abs(blk - (other.T if other is not None else blk.T)).max()))
        out[(b, k2, R)] = 0.5 * (blk + (other.T if other is not None else blk.T))

    if symmetrise:
        ops = P.cubic_ops()
        acc, cnt = {}, {}
        for (b, k2, R), blk in out.items():
            for U in ops:
                Rg = tuple(np.round(U @ np.array(R), 6))
                key = (b, k2, tuple(float(x) if abs(x) > 1e-9 else 0.0 for x in Rg))
                acc[key] = acc.get(key, np.zeros((3, 3))) + U @ blk @ U.T
                cnt[key] = cnt.get(key, 0) + 1
        out = {k: v / cnt[k] for k, v in acc.items()}

    tot = np.zeros((nb, nb, 3, 3))
    for (b, k2, R), blk in out.items():
        tot[b, k2] += blk
    asr = float(np.abs(tot).max())
    for b in range(nb):
        R0 = (0.0, 0.0, 0.0)
        key = (b, b, R0)
        corr = sum(out.get((b, k2, R), 0) for (bb, k2, R) in list(out) if bb == b)
        out[key] = out.get(key, np.zeros((3, 3))) - corr
    meta = dict(nb=nb, n_atoms=st["n"], delta_A=delta, n_blocks=len(out),
                asr_before_eV_A2=asr,
                asymmetry_before_eV_A2=float(max(asym)) if asym else 0.0,
                cubic_symmetrised=bool(symmetrise))
    return out, meta


def frequencies(phi, st, q):
    nb = st["nb"]
    tau = st["tau"]
    D = np.zeros((3 * nb, 3 * nb), complex)
    for (b, k2, R), blk in phi.items():
        ph = np.exp(1j * float(q @ (np.array(R) + tau[k2] - tau[b])))
        D[3 * b:3 * b + 3, 3 * k2:3 * k2 + 3] += blk * ph
    D /= U.MASS_FE
    herm = float(np.abs(D - D.conj().T).max())
    lam = np.linalg.eigvalsh((D + D.conj().T) / 2)
    w = np.sign(lam) * np.sqrt(np.abs(lam)) * OMEGA
    return np.sort(w), herm, D


HSP = {"G": [0.0, 0.0, 0.0], "H": [1.0, 0.0, 0.0], "N": [0.5, 0.5, 0.0],
       "P": [0.5, 0.5, 0.5]}
PATH = [("G", "H"), ("H", "N"), ("N", "G"), ("G", "P"), ("P", "H")]


def qpath(a, npts=41):
    pts, labs = [], []
    for seg, (pa, pb) in enumerate(PATH):
        ka = np.array(HSP[pa]) * 2 * np.pi / a
        kb = np.array(HSP[pb]) * 2 * np.pi / a
        for t in np.linspace(0, 1, npts, endpoint=(seg == len(PATH) - 1)):
            pts.append((1 - t) * ka + t * kb)
            labs.append((f"{pa}-{pb}", float(t)))
    return pts, labs


def phonons_for_state(kind, a, m, n, delta, seeds=(1, 2, 3, 4), nep=None):
    """Dispersion of one magnetic state at its own relaxed lattice constant."""
    if kind == "DLM":
        st0 = replicate(state_cell("DLM", a), n)
        pts, _ = qpath(a)
        acc = []
        metas = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            v = rng.normal(size=(st0["n"], 3))
            v /= np.linalg.norm(v, axis=1)[:, None]
            st = dict(st0)
            st["spins"] = v * m
            phi, meta = force_constants(st, delta, nep=nep, tag=f"DLM_n{n}_s{sd}",
                                        symmetrise=True)
            metas.append(meta)
            acc.append(np.array([frequencies(phi, st, q)[0] for q in pts]))
        acc = np.array(acc)
        W = acc.mean(axis=0)
        sd_W = acc.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(W)
        meta = dict(nb=1, n_atoms=st0["n"], delta_A=delta,
                    asr_before_eV_A2=float(np.mean([x["asr_before_eV_A2"] for x in metas])),
                    asymmetry_before_eV_A2=float(np.mean([x["asymmetry_before_eV_A2"] for x in metas])),
                    cubic_symmetrised=True, n_configs=len(seeds))
    else:
        st = replicate(state_cell(kind, a), n)
        st["spins"] = st["spins"] * m
        # cubic averaging only makes sense when the magnetic order keeps the
        # cubic group: FM (no SOC -> spin decoupled) and DLM (disordered average)
        phi, meta = force_constants(st, delta, nep=nep, tag=f"{kind}_n{n}",
                                    symmetrise=(kind == "FM"))
        pts, _ = qpath(a)
        W = np.array([frequencies(phi, st, q)[0] for q in pts])
        sd_W = np.zeros_like(W)
    return W, sd_W, meta, pts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["relax", "phonon", "all"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--model", default=None)
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    m = P.m0_from_stage_B()
    kinds = ["FM", "AFM-G", "AFM-A", "NC-90", "DLM"]

    rel_path = os.path.join(OUT, "relaxation.json")
    if A.stage in ("relax", "all"):
        rel = {}
        for k in kinds:
            r = relax_a(k, m, nep=A.model)
            rel[k] = r
            print(f"{k:7s} a0 = {r['a0_A']:.4f} A   E = {r['E_min_eV_per_atom']:+.5f} "
                  f"eV/atom   max|F| = {r['max_force_at_a0_eV_per_A']:.2e} eV/A", flush=True)
        U.save_json(rel_path, rel)
    else:
        rel = json.load(open(rel_path))

    if A.stage in ("phonon", "all"):
        res = {}
        for k in kinds:
            a0 = rel[k]["a0_A"]
            n = 7 if k in ("FM", "DLM") else A.n
            W, sdW, meta, pts = phonons_for_state(k, a0, m, n, A.delta, nep=A.model)
            res[k] = dict(a0_A=a0, supercell=f"{n}x{n}x{n}", omega=W.tolist(),
                          omega_sd=sdW.tolist(), meta=meta,
                          omega_max_THz=float(W.max()), omega_min_THz=float(W.min()),
                          gamma_THz=[float(x) for x in W[0]],
                          imaginary=int((W < -1e-6).sum()))
            print(f"{k:7s} a0={a0:.4f}  n={n}  w_max={W.max():6.3f}  "
                  f"w(Gamma)={np.round(W[0],3)}  imaginary={res[k]['imaginary']}  "
                  f"asr={meta['asr_before_eV_A2']:.3f}", flush=True)
        U.save_json(os.path.join(OUT, "phonons_by_state.json"), res)
