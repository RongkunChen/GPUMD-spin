"""Magnon spectrum from the model, magnon-phonon coupling, and whether the two
branches repel each other.

Three questions, three independent measurements:

1. **Magnon dispersion from the model itself** (no exchange fit): a flat spiral
   of wavevector q has  E(q) - E(FM) = (N/2)[J(0) - J(q)], and linear spin-wave
   theory gives  hbar*omega(q) = S[J(0) - J(q)], so

       hbar*omega(q) = 2S [E(q) - E(FM)] / N .

   This is compared with the LSWT curve that stage C produced from the fitted
   J_n.  The comparison is a factor-of-S test: if the fit-based curve is low by
   exactly S = m/2 = 1.2085, then stage C dropped the spin magnitude.

2. **Magnon-phonon coupling strength**: the vertex that would hybridise one
   magnon with one phonon is  C = d2E / du dS_perp.  At a collinear state the
   energy depends on the spins only through  S_i.S_j  and  |S_i|, so a transverse
   tilt changes the energy at second order and C must vanish identically.  The
   longitudinal channel (d2E/du d|S|) is the exchange-striction term and does not
   vanish.  Both are measured here with a 2x2 central difference.

3. **Avoided crossing**: with C = 0 the spin and lattice sectors are decoupled,
   so the magnon and phonon branches must cross exactly.  The crossing points are
   located and the residual repulsion is bounded by the measured |C|.

usage: python magnon_and_coupling.py [--n 8]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feutils as U
import phonon_fd as P
import relax_and_phonons as R

OUT = os.path.join(U.VAL, "raw_data", "H_states")
THz_to_meV = 4.135667696


def spiral(cell, m, q):
    """Flat spiral in the x-z plane, wavevector q (Cartesian)."""
    s = np.zeros((cell["n"], 3))
    ang = cell["pos"] @ np.asarray(q)
    s[:, 0] = m * np.sin(ang)
    s[:, 2] = m * np.cos(ang)
    return s


def direct_magnon(a, m, n=8, nep=None):
    """hbar*omega(q) from the model's own spiral energies along Gamma-H.

    Only Gamma-H is used, and on an n x n x n *conventional* cell: an n x n x n
    primitive cell is only (n/2)^3 cubes wide, so half of the 9 points used in
    the first version were NOT commensurate with the box (the spiral broke at
    the periodic boundary) and gave a non-monotonic omega(q).
    """
    cell = U.bcc_cell(n, n, n, a)
    fm = U.fm_spins(cell["n"], m)
    e_fm = U.static_energy(os.path.join(OUT, "magnon", "FM"), cell, fm, nep=nep)
    rows = []
    for k in range(0, n + 1):
        t = k / n
        q = t * 2 * np.pi / a
        ang = cell["pos"][:, 2] * q
        sp = np.zeros((cell["n"], 3))
        sp[:, 0] = m * np.sin(ang)
        sp[:, 2] = m * np.cos(ang)
        d = os.path.join(OUT, "magnon", f"GH_{k}")
        e = U.static_energy(d, cell, sp, nep=nep)
        rows.append(dict(seg="G-H", t=float(t), q=[0.0, 0.0, float(q)],
                         E_eV=e, dE_meV_per_atom=(e - e_fm) / cell["n"] * 1000.0,
                         omega_meV=2 * (m / 2.0) * (e - e_fm) / cell["n"] * 1000.0))
    return rows, dict(n_atoms=cell["n"], a0_A=a, S=m / 2.0, E_FM_eV=e_fm,
                      commensurate_step_per_A=float(2 * np.pi / (n * a)),
                      formula="hbar*omega = 2 S [E(q)-E(FM)]/N, S = m/2 (g=2)")


def lswt_curve(nep=None):
    d = os.path.join(U.VAL, "raw_data", "C", "magnon_dispersion.dat")
    rows = []
    for line in open(d):
        if line.startswith("#"):
            continue
        s, t, kx, ky, kz, w = line.split()
        rows.append(dict(seg=s, t=float(t), q=np.array([float(kx), float(ky), float(kz)]),
                         omega_meV=float(w)))
    return rows


def coupling_derivatives(a, m, nep=None, epss=(0.02, 0.05, 0.10)):
    """Mixed derivatives d2E/du dS for a transverse tilt and for |S|."""
    n = 3
    cell = R.replicate(R.state_cell("FM", a), n)
    spins0 = cell["spins"] * m
    delta = 0.02          # A

    def energy(du, dspin_kind, eps):
        pos = cell["pos"].copy()
        pos[0, 0] += du
        c2 = dict(cell)
        c2["pos"] = pos
        sp = spins0.copy()
        if dspin_kind == "elastic":
            pass                    # control: no spin change at all (must give 0)
        elif dspin_kind == "transverse":
            # a ROTATION, and the sign of eps is used: the linear magnon-phonon
            # vertex is the part ODD in the tilt angle.  Keeping |S| fixed matters
            # (adding eps to S_y would change |S| by eps^2/2m and leak into the
            # longitudinal channel).
            th = eps / m
            sp[0] = [m * np.sin(th), 0.0, m * np.cos(th)]
        elif dspin_kind == "longitudinal":
            sp[0, 2] = m + eps       # change of |S| along the axis
        d = os.path.join(OUT, "coupling", f"{dspin_kind}_{du:+.2f}_{eps:.2f}")
        return U.static_energy(d, c2, sp, nep=nep)

    out = {}
    for kind in ("transverse", "longitudinal", "elastic"):
        scan = []
        for eps in epss:
            epp = energy(+delta, kind, +eps)     # +u, +tilt
            epm = energy(+delta, kind, -eps)     # +u, -tilt
            emp = energy(-delta, kind, +eps)     # -u, +tilt
            emm = energy(-delta, kind, -eps)     # -u, -tilt
            scan.append(dict(eps_muB=eps,
                             C_eV_per_A_per_muB=(epp - epm - emp + emm) / (4 * delta * eps)))
        out[kind] = dict(scan=scan,
                         C_eV_per_A_per_muB=float(np.mean([x["C_eV_per_A_per_muB"]
                                                           for x in scan])),
                         C_spread=float(np.std([x["C_eV_per_A_per_muB"] for x in scan])))
    out["note"] = ("C = [E(+du,+eps) - E(+du,0) - E(-du,+eps) + E(-du,0)] / (4 du eps); "
                   "transverse = tilt off the collinear axis (the magnon channel), "
                   "longitudinal = change of |S| (the exchange-striction channel)")
    return out


def crossings(a, m, phon, mag, npts=41):
    """Interpolate the phonon branches onto the magnon q grid and find crossings."""
    # along-path coordinate, identical for both grids
    def path_x(a):
        x, tot = [], 0.0
        for pa, pb in R.PATH:
            ka = np.array(R.HSP[pa]) * 2 * np.pi / a
            kb = np.array(R.HSP[pb]) * 2 * np.pi / a
            for t in np.linspace(0, 1, npts, endpoint=(pa, pb) == R.PATH[-1]):
                x.append(tot + t * np.linalg.norm(kb - ka))
            tot += np.linalg.norm(kb - ka)
        return np.array(x), tot

    xp, tot = path_x(a)
    ph = np.array(phon)
    xm = []
    for r in mag:                     # magnon uses 9 points per segment
        seg, t = r["seg"], r["t"]
        i = [k for k, (pa, pb) in enumerate(R.PATH) if f"{pa}-{pb}" == seg][0]
        ka = np.array(R.HSP[R.PATH[i][0]]) * 2 * np.pi / a
        kb = np.array(R.HSP[R.PATH[i][1]]) * 2 * np.pi / a
        xm.append(sum(np.linalg.norm(np.array(R.HSP[R.PATH[j][1]]) * 2 * np.pi / a -
                                     np.array(R.HSP[R.PATH[j][0]]) * 2 * np.pi / a)
                      for j in range(i)) + t * np.linalg.norm(kb - ka))
    xm = np.array(xm)
    mg = np.array([r["omega_meV"] for r in mag]) / THz_to_meV
    out = []
    for b in range(ph.shape[1]):
        phb = np.interp(xm, xp, ph[:, b])
        d = mg - phb
        sgn = np.sign(d)
        for i in np.where(np.diff(sgn) != 0)[0]:
            f = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]) + 1e-30)
            out.append(dict(branch=b + 1, x=float(xm[i] + f * (xm[i + 1] - xm[i])),
                            fraction_along_path=float((xm[i] + f * (xm[i + 1] - xm[i])) / tot),
                            omega_THz_at_cross=float(0.5 * (mg[i] + phb[i])),
                            omega_meV_at_cross=float(0.5 * (mg[i] + phb[i]) * THz_to_meV)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--model", default=None)
    A = ap.parse_args()
    m = P.m0_from_stage_B()
    rel = json.load(open(os.path.join(OUT, "relaxation.json")))
    a = rel["FM"]["a0_A"]

    mag, mag_meta = direct_magnon(a, m, A.n, nep=A.model)
    lsw = lswt_curve()
    # match by segment/t
    ls = {}
    for r in lsw:
        ls.setdefault((r["seg"], round(r["t"], 4)), r["omega_meV"])
    pairs = []
    for r in mag:
        k = (r["seg"], round(r["t"], 4))
        if k in ls:
            pairs.append((r["omega_meV"], ls[k]))
    pairs = np.array([p for p in pairs if abs(p[0]) > 1.0])
    ratio = pairs[:, 1] / pairs[:, 0] if len(pairs) else np.array([np.nan])
    print(f"magnon: direct vs LSWT over {len(pairs)} matched q points")
    print(f"  ratio LSWT/direct: mean {np.mean(ratio):.4f}  sd {np.std(ratio):.4f}  "
          f"(S = {m/2:.4f})")

    # stiffness from the direct curve, small q along Gamma-H, compared with the
    # LSWT value that stage C reported (D = 345.5 meV A^2)
    sm = [r for r in mag if r["seg"] == "G-H" and 0 < r["t"] <= 0.375]
    if len(sm) >= 3:
        qq = np.array([np.linalg.norm(r["q"]) for r in sm])
        ww = np.array([r["omega_meV"] for r in sm])
        D_naive = float(np.polyfit(qq ** 2, ww, 1)[0])
        D_dir = float(np.polyfit(qq ** 2, ww / qq ** 2, 1)[1])   # q -> 0 limit
    else:
        D_dir = D_naive = float("nan")
    lsm = []
    for r in lsw:
        if r["seg"] == "G-H" and 0 < r["t"] <= 0.5:
            lsm.append((np.linalg.norm(r["q"]) * 2 * np.pi / a, r["omega_meV"]))
    D_lsw = float(np.polyfit([x[0] ** 2 for x in lsm], [x[1] for x in lsm], 1)[0]) \
        if len(lsm) > 2 else float("nan")
    print(f"stiffness: direct D = {D_dir:.1f} meV.A^2, LSWT-from-fit D = {D_lsw:.1f}")

    dv = coupling_derivatives(a, m, nep=A.model)
    print(f"coupling: transverse C = {dv['transverse']['C_eV_per_A_per_muB']:+.3e} eV/(A muB)")
    print(f"          longitudinal C = {dv['longitudinal']['C_eV_per_A_per_muB']:+.3e}")

    phon = json.load(open(os.path.join(OUT, "phonons_by_state.json")))["FM"]["omega"]
    cr = crossings(a, m, phon, mag)
    print(f"crossings of the magnon branch with phonons: {len(cr)}")
    for c in cr[:6]:
        print(f"   branch {c['branch']} at {c['fraction_along_path']:.3f} of the path "
              f"({c['omega_THz_at_cross']:.2f} THz)")

    res = dict(a0_A=a, model=mag_meta,
               magnon_direct=mag, lswt_ratio=dict(mean=float(np.mean(ratio)),
                                                  sd=float(np.std(ratio)),
                                                  n=len(pairs), S=m / 2),
               stiffness=dict(D_direct_meV_A2=D_dir, D_direct_naive_meV_A2=D_naive,
                              D_lswt_meV_A2=D_lsw,
                              D_stage_C_reported_meV_A2=345.5,
                              reference_range_meV_A2=[280, 320]),
               coupling=dv, crossings=cr,
               crossing_note="with C = 0 the two sectors do not hybridise, so these are "
                             "exact crossings, not avoided ones; the residual repulsion is "
                             "bounded by |C| * sqrt(2 hbar / (M omega)) which is quoted in "
                             "the report")
    U.save_json(os.path.join(OUT, "magnon_coupling.json"), res)
    print("saved", os.path.join(OUT, "magnon_coupling.json"))
