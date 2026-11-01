"""Exchange constants, magnon spectrum and per-magnetic-state phonons, per checkpoint.

This is the part of the per-checkpoint battery that the first version was missing.
`model_series.py` only produced scalar magnetic numbers (a0, m_eq, D, Tc) plus a
single FM phonon curve; the physics that actually has to be checked for a
spin-lattice potential is the full set:

  * **exchange constants J1..J6** -- least squares on the model's own frozen-spiral
    energies in the convention  H = -1/2 sum_ij J_ij e_i.e_j  (e_i = unit spin),
    i.e.  E(q)/N - E_FM/N = (1/2) sum_n J_n [z_n - S_n(q)].
  * **magnon dispersion** -- two independent numbers for the same quantity:
      - LSWT from the fitted J_n:   hbar*omega(q) = S * sum_n J_n [z_n - S_n(q)]
        with S = m/2 (g = 2).  The earlier stage-C curve omitted this factor S
        and was therefore low by exactly 1/S = 0.827; the stiffness that comes
        out of it (345.5 meV.A^2) is 345.5/S = 433.7 once corrected.
      - direct frozen spirals:      hbar*omega(q) = 2S [E(q) - E(FM)] / N
    The two must agree; the ratio is stored as a self-check.
  * **phonons of every magnetic state** (FM, NC-90, AFM-A, AFM-G, DLM), each at
    its own relaxed lattice constant.  A collinear AFM has force constants that
    are periodic with the *magnetic* cell, so a primitive-cell folding would be
    wrong; the generalised dynamical matrix of relax_and_phonons.py is reused.
  * **magnon-phonon coupling** -- the mixed derivative d2E/du dS (transverse and
    longitudinal), and the crossings of the magnon branch with the phonon
    branches.  At a collinear state the transverse vertex vanishes identically,
    so the crossings must be exact, not avoided; the measurement is there to show
    that the model does this and to bound any residual repulsion.
  * **dispersion correction due to magnetism** -- omega(state) - omega(FM) on the
    same q grid, i.e. how much the spin order shifts the phonons.

Everything goes through the production binaries.  Nothing is fitted to make a
number look right; the only fits are the J_n (which is a definition) and the
quadratic/parabolic interpolations that locate minima.

usage (normally driven by model_series.run_one)
  python series_physics.py --add tag=path/to/nep.txt
  python series_physics.py --scan DIR
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                "validation", "BCC_Fe_full_physics", "scripts"))
sys.path.insert(0, _HERE)
import feutils as U
import phonon_fd as P
import relax_and_phonons as R

SER = os.path.join(U.ROOT, "work_20261101_retrain", "C_5desc_500k", "series")
THz_to_meV = 4.135667696
RC = 6.0                      # model cutoff, nep.in: cutoff 6.0 3.0

# (name, supercell repetitions of the magnetic cell) -- sizes chosen so the box
# is wider than 2*rc = 12 A; see relax_and_phonons.relax_a for why 8 atoms is
# not enough.
STATES = [("FM", 7), ("NC-90", 5), ("AFM-A", 5), ("AFM-G", 5), ("DLM", 7)]
DIRS = {"001": (0.0, 0.0, 1.0), "110": (1.0, 1.0, 0.0), "111": (1.0, 1.0, 1.0)}
SCHEMA = 5      # bump when a cached section changes definition


def _log(msg):
    print(f"    [{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- neighbour shells
def shells(a, rmax=RC):
    """Neighbour shells of one bcc site (union of both sublattice offsets)."""
    vecs = []
    for off in (np.zeros(3), np.array([0.5, 0.5, 0.5])):
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    d = (np.array([i, j, k], float) + off) * a
                    if np.linalg.norm(d) < 1e-6:
                        continue
                    vecs.append(d)
    vecs = np.array(vecs)
    r = np.linalg.norm(vecs, axis=1)
    out, used = [], np.zeros(len(vecs), bool)
    for i in np.argsort(r):
        if used[i] or r[i] > rmax:
            continue
        m = np.abs(r - r[i]) < 1e-4
        used |= m
        out.append(np.unique(np.round(vecs[m], 6), axis=0))
    return out


def _S_of_q(q, sh):
    return np.array([np.sum(np.cos(s @ q)) for s in sh])


# --------------------------------------------------------------------- exchange J
def spiral_energy(nep, a, m, q, ncell, tag, workdir):
    """Model energy per atom of a flat spiral in the x-z plane, wavevector q."""
    cell = U.bcc_cell(ncell, ncell, ncell, a)
    ang = cell["pos"] @ q
    sp = np.zeros((cell["n"], 3))
    sp[:, 0] = m * np.sin(ang)
    sp[:, 2] = m * np.cos(ang)
    e = U.static_energy(os.path.join(workdir, tag), cell, sp, nep=nep)
    return e / cell["n"], cell["n"]


def fit_exchange(nep, a0, m, tag, ncell=8):
    """J1..J6 from the model's spiral energies; also the LSWT stiffness.

    q points: three directions ([001], [110], [111] in units of 2*pi/a, *not*
    normalised -- the same convention as stage C, so the numbers are directly
    comparable) with mq = 0..ncell/2 on an ncell^3 *conventional* cell, which is
    what makes every 2*pi/(ncell*a) commensurate with the periodic box.
    """
    wd = os.path.join(SER, "runs", tag, "spiral")
    sh = shells(a0)
    z = np.array([len(s) for s in sh], float)
    qs, es = [], []
    for name, dv in DIRS.items():
        for mq in range(0, ncell // 2 + 1):
            q = np.array(dv, float) * 2 * np.pi * mq / (ncell * a0)
            e, nat = spiral_energy(nep, a0, m, q, ncell, f"{name}_m{mq}", wd)
            qs.append(q)
            es.append(e)
    e_fm = es[0]
    A = np.array([0.5 * (z - _S_of_q(q, sh)) for q in qs])
    b = np.array([e - e_fm for e in es])
    J, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    rms = float(np.sqrt(np.mean((b - A @ J) ** 2)))
    S = m / 2.0
    # closed form for the LSWT stiffness of a cubic crystal:
    # hbar*omega = D q^2 with D = (1/(6 S)) sum_n J_fit,n z_n r_n^2
    # (the J_fit are unit-vector coefficients, hence the 1/S, not S)
    D_closed = float(np.sum(J * z * np.array([np.linalg.norm(s[0]) ** 2
                                              for s in sh])) * 1000 / (6.0 * S))
    rows = [dict(shell=n + 1, r_A=float(np.linalg.norm(s[0])), z=int(len(s)),
                 J_meV=float(J[n] * 1000), J_per_m2_meV=float(J[n] * 1000 / m ** 2))
            for n, s in enumerate(sh)]
    J_S = J * 1000 / S ** 2                 # coefficients of S_i.S_j with |S| = S
    return dict(J_meV=[float(x * 1000) for x in J],
                J_S_meV=[float(x) for x in J_S],
                J0_meV=float(np.sum(J_S * z)),
                shells=rows, n_shells=len(sh),
                fit_rms_meV_per_atom=rms, rank=int(rank), n_q=len(qs), ncell=ncell,
                a0_A=a0, moment_muB=m, S=S,
                D_lswt_closed_form_meV_A2=D_closed,
                q_points=[dict(dir=k, m=int(round(np.linalg.norm(q) * ncell * a0 /
                                                  (2 * np.pi))),
                               q=[float(x) for x in q],
                               E_eV_per_atom=float(e))
                          for k, q, e in zip([k for k in DIRS for _ in
                                              range(ncell // 2 + 1)], qs, es)],
                convention="H = -1/2 sum_ij J_ij e_i.e_j; J_n are the Fourier "
                           "components of the model's angular energy functional "
                           "within the %.1f A cutoff" % RC,
                note="J in meV for the model's own |S| = %.4f muB; J_per_m2 is the "
                     "same number divided by m^2, for comparison with literature "
                     "Heisenberg parameters" % m)


def lswt_omega(q, J_meV, sh, S):
    """hbar*omega(q) in meV.

        hbar*omega = S [J^S(0) - J^S(q)]        (Holstein-Primakoff, |S| = S)

    The fitted J_n are the coefficients of a sum over *unit* vectors, so
    J^S = J_fit / S^2 and

        hbar*omega = [J_fit(0) - J_fit(q)] / S = 2 [E(q)-E(FM)] / (N S).

    The first version of this function -- and of the direct-spiral route next to
    it -- multiplied by S instead of dividing, which over-reported every magnon
    energy and every stiffness by S^2 = 1.49.  The two routes agreed with each
    other because they carried the same error, so nothing caught it until the
    precession frequency was measured directly (frozen_magnon_test.py).
    """
    z = np.array([len(s) for s in sh], float)
    return float(np.sum(np.array(J_meV) * (z - _S_of_q(np.asarray(q, float), sh))) / S)


def path_geometry(a, npts=41):
    """Cumulative distance along Gamma-H-N-Gamma-P-H, identical to the phonon grid."""
    x, lab, tot = [], [], 0.0
    for seg, (pa, pb) in enumerate(R.PATH):
        ka = np.array(R.HSP[pa]) * 2 * np.pi / a
        kb = np.array(R.HSP[pb]) * 2 * np.pi / a
        for t in np.linspace(0, 1, npts, endpoint=(seg == len(R.PATH) - 1)):
            x.append(tot + t * np.linalg.norm(kb - ka))
            lab.append((f"{pa}-{pb}", float(t)))
        tot += np.linalg.norm(kb - ka)
    return np.array(x), lab, float(tot)


def magnon_path(ex, a0, npts=41):
    """LSWT dispersion of the fitted J_n along the phonon q path."""
    sh = shells(a0)
    x, lab, tot = path_geometry(a0, npts)
    w, qs = [], []
    for segment, (pa, pb) in enumerate(R.PATH):
        ka = np.array(R.HSP[pa]) * 2 * np.pi / a0
        kb = np.array(R.HSP[pb]) * 2 * np.pi / a0
        for t in np.linspace(0, 1, npts, endpoint=(segment == len(R.PATH) - 1)):
            q = (1 - t) * ka + t * kb
            w.append(lswt_omega(q, ex["J_meV"], sh, ex["S"]))
            qs.append([float(v) for v in q])
    marks = [0.0]
    tt = 0.0
    for pa, pb in R.PATH:
        tt += np.linalg.norm(np.array(R.HSP[pb]) * 2 * np.pi / a0 -
                             np.array(R.HSP[pa]) * 2 * np.pi / a0)
        marks.append(tt)
    return dict(x=[float(v) for v in x], omega_meV=w, q=qs, path_length=float(tot),
                labels=[f"{pa}-{pb}" for pa, pb in R.PATH], npts=npts,
                high_symmetry=[dict(label=l, x=float(xx)) for l, xx in
                               zip(["$\\Gamma$", "H", "N", "$\\Gamma$", "P", "H"], marks)],
                formula="hbar*omega(q) = S sum_n J_n [z_n - S_n(q)], S = m/2")


def direct_spiral_magnon(nep, a0, m, tag, ncell=8):
    """hbar*omega(q) = 2S [E(q) - E(FM)]/N directly from spiral energies."""
    wd = os.path.join(SER, "runs", tag, "spiral_direct")
    S = m / 2.0
    e_fm, nat = spiral_energy(nep, a0, m, np.zeros(3), ncell, "FM", wd)
    out = {}
    for name, dv in DIRS.items():
        rows = []
        for mq in range(0, ncell + 1):
            q = np.array(dv, float) * 2 * np.pi * mq / (ncell * a0)
            e, _ = spiral_energy(nep, a0, m, q, ncell, f"{name}_m{mq}", wd)
            rows.append(dict(m=mq, q=[float(x) for x in q],
                             q_norm=float(np.linalg.norm(q)),
                             dE_meV_per_atom=float((e - e_fm) * 1000),
                             # 2*S -> 2/S: see lswt_omega(); the direct spiral and
                             # the LSWT curve must use the same relation
                             omega_meV=float(2 * (e - e_fm) * 1000 / S)))
        qq = np.array([r["q_norm"] for r in rows])
        ww = np.array([r["omega_meV"] for r in rows])
        sel = (np.arange(len(rows)) >= 1) & (np.arange(len(rows)) <= 3)
        D = float(np.polyfit(qq[sel] ** 2, ww[sel] / qq[sel] ** 2, 1)[1])
        D_naive = float(np.polyfit(qq[sel] ** 2, ww[sel], 1)[0])
        out[name] = dict(points=rows, D_meV_A2=D, D_naive_meV_A2=D_naive,
                         omega_at_zone_boundary_meV=float(ww[1:].max()))
    out["n_atoms"] = nat
    out["S"] = S
    # [001] is the headline value: its q grid is 2*pi/(8a) wide, i.e. three times
    # finer than [111] (whose points sit at sqrt(3)*2*pi/(8a)), so its q^2 -> 0
    # extrapolation carries the least quartic contamination.
    out["D_meV_A2"] = float(out["001"]["D_meV_A2"])
    out["D_mean_over_directions_meV_A2"] = float(np.mean([out[k]["D_meV_A2"]
                                                          for k in DIRS]))
    out["formula"] = "hbar*omega(q) = 2S [E(q)-E(FM)]/N, S = m/2 (g=2)"
    return out


# ------------------------------------------------- phonons of each magnetic state
def state_phonons(nep, m, tag, kinds=None, delta=0.02, npts=41):
    """Relax every magnetic state, then its full dispersion along the q path."""
    kinds = kinds or STATES
    R.OUT = os.path.join(SER, "runs", tag)      # keep this checkpoint's files apart
    out = {}
    for kind, n in kinds:
        seeds = (1, 2) if kind == "DLM" else (1, 2, 3)
        rel = R.relax_a(kind, m, nep=nep, n=n, seeds=seeds)
        W, sdW, meta, pts = R.phonons_for_state(kind, rel["a0_A"], m, n, delta,
                                                seeds=(1, 2), nep=nep)
        x, lab, tot = path_geometry(rel["a0_A"], npts)
        out[kind] = dict(
            a0_A=rel["a0_A"], supercell=f"{n}x{n}x{n}", n_atoms=rel["n_atoms"],
            omega=[[float(v) for v in row] for row in W],
            omega_sd=[[float(v) for v in row] for row in sdW],
            omega_max_THz=float(W.max()), omega_min_THz=float(W.min()),
            gamma_THz=[float(v) for v in W[0]],
            omega_N_THz=[float(v) for v in W[np.argmin(np.abs(x - _x_of("N", rel["a0_A"])))]],
            imaginary=int((W < -1e-6).sum()),
            # a genuine soft mode vs the ~1e-3 THz negative acoustic values that
            # the float32 force noise leaves behind after the ASR correction
            imaginary_below_0p1_THz=int((W < -0.1).sum()),
            softest_THz=float(W.min()),
            asr_before_eV_A2=float(meta["asr_before_eV_A2"]),
            gamma_abs_max_THz=float(np.abs(W[0]).max()),
            cubic_symmetrised=bool(meta.get("cubic_symmetrised", False)),
            n_configs=int(meta.get("n_configs", 1)),
            x=[float(v) for v in x], path_length=tot,
            E_min_eV_per_atom=rel["E_min_eV_per_atom"],
            max_force_at_a0=rel["max_force_at_a0_eV_per_A"],
            delta_A=delta, npts=npts)
        _log(f"{tag} phonon {kind:6s} a0={rel['a0_A']:.4f} w_max={W.max():6.3f} "
             f"w_min={W.min():7.3f} imaginary={out[kind]['imaginary']}")
    return out


def _x_of(label, a):
    """Along-path coordinate of a high-symmetry point (G-H-N-G-P-H)."""
    marks, tot = [0.0], 0.0
    for pa, pb in R.PATH:
        tot += np.linalg.norm(np.array(R.HSP[pb]) * 2 * np.pi / a -
                              np.array(R.HSP[pa]) * 2 * np.pi / a)
        marks.append(tot)
    return marks[["G", "H", "N", "G", "P", "H"].index(label)]


# ------------------------------------------------------- magnon-phonon coupling
def coupling_vertex(nep, a0, m, tag, epss=(0.02, 0.05, 0.10), delta=0.02, n=3):
    """Mixed derivative d2E/du dS: transverse (magnon) and longitudinal channels."""
    cell = R.replicate(R.state_cell("FM", a0), n)
    spins0 = cell["spins"] * m
    wd = os.path.join(SER, "runs", tag, "coupling")

    def energy(du, kind, eps):
        pos = cell["pos"].copy()
        pos[0, 0] += du
        c2 = dict(cell)
        c2["pos"] = pos
        sp = spins0.copy()
        if kind == "transverse":
            th = eps / m                     # a rotation: |S| is unchanged
            sp[0] = [m * np.sin(th), 0.0, m * np.cos(th)]
        elif kind == "longitudinal":
            sp[0, 2] = m + eps
        elif kind == "elastic":
            pass                             # control: no spin change, must be 0
        d = os.path.join(wd, f"{kind}_{du:+.2f}_{eps:.2f}")
        return U.static_energy(d, c2, sp, nep=nep)

    out = {}
    for kind in ("transverse", "longitudinal", "elastic"):
        scan = []
        for eps in epss:
            epp = energy(+delta, kind, +eps)
            epm = energy(+delta, kind, -eps)
            emp = energy(-delta, kind, +eps)
            emm = energy(-delta, kind, -eps)
            scan.append(dict(eps_muB=eps,
                             C_eV_per_A_per_muB=(epp - epm - emp + emm) /
                                                (4 * delta * eps)))
        vals = [s["C_eV_per_A_per_muB"] for s in scan]
        out[kind] = dict(scan=scan, C_eV_per_A_per_muB=float(np.mean(vals)),
                         C_spread=float(np.std(vals)),
                         C_abs_max=float(np.max(np.abs(vals))))
    # Noise floor.  Two independent estimators, both of which must be beaten:
    #   * the elastic control, where the spin block is not touched at all, so its
    #     "C" is pure finite-difference noise of the float32 forces;
    #   * the spread of a channel over the three tilt amplitudes, which measures
    #     the same noise without assuming the control is representative.
    # Anything within 2x of that floor is not a measured vertex.
    out["noise_floor_eV_per_A_per_muB"] = max(
        out["elastic"]["C_abs_max"], out["transverse"]["C_spread"],
        out["longitudinal"]["C_spread"])
    out["transverse_is_below_noise"] = bool(
        abs(out["transverse"]["C_eV_per_A_per_muB"]) <=
        2 * out["noise_floor_eV_per_A_per_muB"] + 1e-12)
    out["note"] = ("C = [E(+du,+eps) - E(+du,-eps) - E(-du,+eps) + E(-du,-eps)]/"
                   "(4 du eps) at 54 atoms, du = %.2f A; transverse = tilt off the "
                   "collinear axis (magnon channel, must vanish for a spin-rotational "
                   "invariant potential); longitudinal = change of |S| "
                   "(exchange-striction channel)" % delta)
    return out


def coupling_verdict(cp):
    """Re-derive the "is this vertex resolved?" test from a stored cache.

    Kept separate from coupling_vertex so that caches written before this test
    was defined can be re-judged without re-running the force evaluations.
    """
    if not cp:
        return dict(floor=None, transverse_resolved=None, longitudinal_resolved=None)
    floor = max(cp["elastic"]["C_abs_max"], cp["transverse"]["C_spread"],
                cp["longitudinal"]["C_spread"])
    return dict(floor_eV_per_A_per_muB=float(floor),
                transverse_resolved=bool(abs(cp["transverse"]["C_eV_per_A_per_muB"])
                                         > 2 * floor),
                longitudinal_resolved=bool(abs(cp["longitudinal"]["C_eV_per_A_per_muB"])
                                           > 2 * floor))


def magnon_phonon_crossings(ex, phon, a0, npts=41):
    """Where the LSWT magnon branch meets each phonon branch along the path."""
    sh = shells(a0)
    x, lab, tot = path_geometry(a0, npts)
    wmag = []
    for segment, (pa, pb) in enumerate(R.PATH):
        ka = np.array(R.HSP[pa]) * 2 * np.pi / a0
        kb = np.array(R.HSP[pb]) * 2 * np.pi / a0
        for t in np.linspace(0, 1, npts, endpoint=(segment == len(R.PATH) - 1)):
            wmag.append(lswt_omega((1 - t) * ka + t * kb, ex["J_meV"], sh, ex["S"]))
    wmag = np.array(wmag)
    ph = np.array(phon["omega"]) * THz_to_meV          # THz -> meV, same grid
    n = min(len(wmag), len(ph))
    wmag, ph, x = wmag[:n], ph[:n], x[:n]
    out = []
    for b in range(ph.shape[1]):
        d = wmag - ph[:, b]
        s = np.sign(d)
        for i in np.where(np.diff(s) != 0)[0]:
            f = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]) + 1e-30)
            out.append(dict(branch=b + 1, x=float(x[i] + f * (x[i + 1] - x[i])),
                            fraction_along_path=float((x[i] + f * (x[i + 1] - x[i])) / tot),
                            omega_meV=float(0.5 * (wmag[i] + ph[i, b])),
                            omega_THz=float(0.5 * (wmag[i] + ph[i, b]) / THz_to_meV)))
    out.sort(key=lambda r: r["omega_meV"])
    # Both branches go to zero at Gamma, so the raw sign change there is not a
    # crossing of two dispersing modes; keep only interior crossings with a
    # finite frequency.
    interior = [r for r in out if r["omega_meV"] > 1.0 and 0.01 < r["fraction_along_path"] < 0.99]
    return dict(crossings=interior, n=len(interior), n_raw=len(out),
                x=[float(v) for v in x],
                magnon_meV=[float(v) for v in wmag],
                note="exact crossings (not avoided): the transverse magnon-phonon "
                     "vertex vanishes at a collinear state, so the two sectors do "
                     "not hybridise; the measured |C| bounds any residual gap. "
                     "The degenerate Gamma points (both branches at zero, and the "
                     "3e-3 THz acoustic residue) are excluded from the count")


def exchange_at_moment(nep, a0, tag, m=2.2):
    """Refit J at the experimental moment.

    J_fit scales like m^1.06 (not m^2), so J^S = J_fit/S^2 is *not* invariant
    under a change of the moment.  Literature exchange parameters refer to the
    DFT ground state, whose moment is 2.2 uB, while this model's own moment is
    2.44 uB; quoting only the latter would overstate the comparison.
    """
    ex = fit_exchange(nep, a0, m, tag + "_m2p2")
    return dict(m_uB=m, J_meV=ex["J_meV"], J_S_meV=ex["J_S_meV"],
                J0_meV=ex["J0_meV"], fit_rms_meV_per_atom=ex["fit_rms_meV_per_atom"],
                D_lswt_closed_form_meV_A2=ex["D_lswt_closed_form_meV_A2"],
                note="refit with the spiral spins rescaled to the experimental "
                     "moment; J^S is the standard Heisenberg coefficient "
                     "(H = -1/2 sum_ij J S_i.S_j, |S| in hbar)")


# ------------------------------------------- why is D too stiff?  D(m) at fixed a
def stiffness_vs_moment(nep, a0, tag, m_eq, moments=(1.8, 2.0, 2.2, 2.4, 2.6),
                        ncell=8):
    """D along [001] at several fixed |S| -- the model's D scales like m^3.

    hbar*omega = 2S [E(q)-E(FM)]/N with S = m/2 and E quadratic in m gives
    D ~ m^3.  A moment that is 11% too large therefore inflates D by ~37% on its
    own, which matters for judging the 'D is 40% too stiff' failure.  Evaluating
    D at the experimental moment separates the two effects.
    """
    wd = os.path.join(SER, "runs", tag, "spiral_m")
    ms = sorted(set([float(x) for x in moments] + [float(m_eq)]))
    rows = []
    for m in ms:
        e_fm, _ = spiral_energy(nep, a0, m, np.zeros(3), ncell, f"FM_m{m:.2f}", wd)
        qs, es = [], []
        for mq in (1, 2, 3):
            q = np.array([0.0, 0.0, 1.0]) * 2 * np.pi * mq / (ncell * a0)
            e, _ = spiral_energy(nep, a0, m, q, ncell, f"m{m:.2f}_q{mq}", wd)
            qs.append(np.linalg.norm(q))
            es.append(e)
        qq, ww = np.array(qs), 2 * (np.array(es) - e_fm) * 1000 / (m / 2.0)
        rows.append(dict(m_uB=m, D_meV_A2=float(np.polyfit(qq ** 2, ww / qq ** 2, 1)[1]),
                         omega_at_H_meV=float(2 * (
                             spiral_energy(nep, a0, m, np.array([0, 0, 2 * np.pi / a0]),
                                           ncell, f"H_m{m:.2f}", wd)[0] - e_fm) * 1000
                             / (m / 2.0))))
    lnm = np.log([r["m_uB"] for r in rows])
    lnD = np.log([r["D_meV_A2"] for r in rows])
    p = float(np.polyfit(lnm, lnD, 1)[0])
    D22 = float(np.interp(2.2, [r["m_uB"] for r in rows], [r["D_meV_A2"] for r in rows]))
    return dict(points=rows, exponent_dlnD_dlnm=p, D_at_2p2_uB_meV_A2=D22,
                D_at_m_eq_meV_A2=float([r["D_meV_A2"] for r in rows
                                        if r["m_uB"] == float(m_eq)][0]),
                note="D from the three smallest commensurate q along [001] on an "
                     "8^3 conventional cell, extrapolated omega/q^2 -> q^2 -> 0; "
                     "the exponent is a log-log slope over %.1f-%.1f muB"
                     % (rows[0]["m_uB"], rows[-1]["m_uB"]))


def magnetostriction(nep, a0, m, tag, dm=0.15, n=5, npts=41):
    """d ln(omega) / d ln|S|: the exchange-striction channel of spin-phonon coupling.

    The mixed derivative d2E/du dS at a single site is dominated by float32 force
    noise (see coupling_vertex).  This is the noise-robust version: the whole
    dynamical matrix is rebuilt at two spin magnitudes and the fractional
    frequency shift per fractional change of |S| is taken mode by mode.  It is
    the quantity that actually enters a magnetoelastic dispersion correction.
    """
    R.OUT = os.path.join(SER, "runs", tag)
    res = {}
    for sgn, mm in (("-", m - dm), ("+", m + dm)):
        st = R.replicate(R.state_cell("FM", a0), n)
        st["spins"] = st["spins"] * mm
        phi, meta = R.force_constants(st, 0.02, nep=nep, tag=f"ms_{sgn}",
                                      symmetrise=True)
        x, _, _ = path_geometry(a0, npts)
        W = []
        for segment, (pa, pb) in enumerate(R.PATH):
            ka = np.array(R.HSP[pa]) * 2 * np.pi / a0
            kb = np.array(R.HSP[pb]) * 2 * np.pi / a0
            for t in np.linspace(0, 1, npts, endpoint=(segment == len(R.PATH) - 1)):
                W.append(R.frequencies(phi, st, (1 - t) * ka + t * kb)[0])
        res[sgn] = dict(m=mm, W=np.array(W))
    Wp, Wm = res["+"]["W"], res["-"]["W"]
    dlnm = np.log(res["+"]["m"] / res["-"]["m"])
    with np.errstate(divide="ignore", invalid="ignore"):
        # NaN, not 0: where a mode is below 1 THz (the three acoustic branches at
        # Gamma) the logarithmic ratio is meaningless and must break the curve
        # rather than be drawn as a spurious jump to zero.
        dlw = np.where((Wp > 1.0) & (Wm > 1.0),
                       np.log(np.abs(Wp) / np.abs(Wm)) / dlnm, np.nan)
    ok = np.isfinite(dlw)
    # a monatomic bcc cell has three acoustic branches, so omega(Gamma) = 0 for
    # all of them; quoting the shift "at Gamma" is meaningless.  Quote the shift
    # of the highest-frequency mode instead.
    i, j = np.unravel_index(int(np.argmax(Wp)), Wp.shape)
    return dict(dlnomega_dlnS=[[None if not np.isfinite(v) else float(v) for v in row]
                               for row in dlw], x=[float(v) for v in x],
                max_abs=float(np.nanmax(np.abs(dlw))), mean=float(np.nanmean(dlw)),
                at_omega_max=float(dlw[i, j]), omega_max_THz=float(Wp[i, j]),
                n_valid=int(ok.sum()), n_total=int(dlw.size),
                m_center=m, dm=dm, n_atoms=st["n"], n_branches=int(Wp.shape[1]),
                note="fractional phonon shift per fractional change of |S|, "
                     "d ln(omega)/d ln(m) with m = %.2f +- %.2f muB; this is the "
                     "magnetostrictive (exchange-striction) channel, NOT the "
                     "transverse magnon vertex, which vanishes at a collinear state. "
                     "null entries = modes below 1 THz (log ratio undefined)"
                     % (m, dm))


# ------------------------------------------------- volume magnetostriction a0(|S|)
def a0_vs_moment(nep, m_eq, tag, moments=(2.0, 2.2, 2.4)):
    """Equilibrium lattice constant at fixed |S| -- the volume magnetostriction.

    The phonon shift d ln omega/d ln|S| and the lattice shift d ln a0/d ln|S| are
    the two halves of the same magnetoelastic coupling; quoting only one of them
    would overstate or understate the effect.  Same 54-atom EOS cell as
    model_series.eos so that a0(m = 2.4) reproduces that number exactly.
    """
    wd = os.path.join(SER, "runs", tag, "eos_m")
    ms = sorted(set([float(x) for x in moments] + [float(m_eq)]))
    rows = []
    for m in ms:
        def E(a):
            c = U.bcc_cell(3, 3, 3, a)
            return U.static_energy(os.path.join(wd, f"m{m:.2f}_a{a:.3f}"), c,
                                   U.fm_spins(c["n"], m), nep=nep) / c["n"]
        grid = np.arange(2.78, 2.901, 0.02)
        Ev = np.array([E(a) for a in grid])
        i = int(np.argmin(Ev))
        c = np.polyfit(grid[max(0, i - 1):i + 2], Ev[max(0, i - 1):i + 2], 2)
        a0 = float(-c[1] / (2 * c[0]))
        fine = np.linspace(a0 - 0.006, a0 + 0.006, 5)
        Ef = np.array([E(a) for a in fine])
        c2 = np.polyfit(fine, Ef, 2)
        if c2[0] > 0:
            a0 = float(-c2[1] / (2 * c2[0]))
        rows.append(dict(m_uB=m, a0_A=a0,
                         B0_GPa=float(2 * (2 * c2[0]) / (9 * a0) * 160.2176634)))
    lnm = np.log([r["m_uB"] for r in rows])
    lna = np.log([r["a0_A"] for r in rows])
    beta = float(np.polyfit(lnm, lna, 1)[0])
    return dict(points=rows, dln_a0_dln_m=beta,
                a0_at_2p2_A=float([r["a0_A"] for r in rows
                                   if r["m_uB"] == 2.2][0]),
                a0_at_m_eq_A=float([r["a0_A"] for r in rows
                                    if r["m_uB"] == float(m_eq)][0]),
                note="54-atom EOS cell (3x3x3 conventional cubes), spin-polarised "
                     "energy scan with m fixed; beta = dln a0/dln m is the volume "
                     "magnetostriction coefficient of the model. Absolute a0 here "
                     "is the same 54-atom construction as model_series.eos, which "
                     "is 0.003-0.01 A below the 343-atom relaxation of the same "
                     "state -- compare like with like")


# ------------------------------------------------- sound velocities (derived only)
def sound_velocities(phon, a0, npts=41):
    """v_L and v_T along [001] and [111] from the acoustic slopes at small q.

    No new force evaluations: this is the small-q slope of the dispersion that
    was just computed.  Conversions, spelled out because they are easy to get
    wrong by 1000: the ordinate is a frequency in THz (cycles/s, the code's
    sqrt(eV/A^2/amu) * 15.6333 factor is 2*pi*1e12 rad/s), the abscissa is q in
    1/A, so v = 2*pi*1e12 * 1e-10 m/s = 628.3 m/s per (THz * A) = 0.6283 km/s.
    The elastic constants are C = rho v^2 with rho = 2 m_Fe / a^3.
    """
    out = {}
    x = np.array(phon["x"])
    W = np.array(phon["omega"])
    rho = 2 * U.MASS_FE * 1.66053907e-27 / (a0 * 1e-10) ** 3      # kg/m^3
    for name, seg in (("001", 0), ("111", 3)):
        i0 = seg * npts
        q = x[i0:i0 + 6]
        if len(q) < 4:
            continue
        Vs = []
        for b in range(W.shape[1]):
            if abs(W[i0, b]) > 0.05:      # optical branches do not start at zero
                continue
            w = W[i0:i0 + 6, b]
            k = (q > 1e-9) & (np.abs(w) > 1e-6)
            if k.sum() < 3:
                continue
            Vs.append(float(np.polyfit(q[k], w[k], 1)[0] * 0.6283 * 1000))   # m/s
        Vs.sort()
        out[name] = dict(v_T_m_s=Vs[0] if Vs else None,
                         v_L_m_s=Vs[-1] if Vs else None, n_acoustic=len(Vs))
        # a negative v_T is not a bug: it is the soft branch (AFM-G) or the ASR
        # residual of a cell that cannot be cubic-symmetrised (NC-90, AFM-A)
        out[name]["transverse_negative"] = bool(Vs and Vs[0] < 0)
        out[name]["C_T_GPa"] = float(rho * Vs[0] ** 2 / 1e9) if Vs else None
        out[name]["C_L_GPa"] = float(rho * Vs[-1] ** 2 / 1e9) if Vs else None
    out["density_kg_m3"] = float(rho)
    out["note"] = ("acoustic slope over the first 5 finite-q points of Gamma-H "
                   "([001]) and Gamma-P ([111]); only branches with omega(Gamma) < "
                   "0.05 THz are used, otherwise the flat optical branches win the "
                   "sort. v in m/s, C = rho v^2 in GPa, rho = 2 m_Fe / a^3. These "
                   "are finite-q slopes of the finite-displacement dispersion; the "
                   "r18 model gave 8.26 km/s for v_L[001] with a different "
                   "supercell and symmetrisation, so treat 7-8 km/s as the family "
                   "value and not as a precision elastic constant")
    return out


# --------------------------------------------------------------- one checkpoint
def run_physics(cache, nep, tag, kinds=None, npts=41, do_phonons=True):
    """Fill in exchange / magnon / phonon / coupling for one checkpoint (in place)."""
    a0 = cache["eos"]["a0_A"]
    m = cache["moment"]["m_eq_uB"]
    t0 = time.time()
    # a bump here invalidates only the sections whose definition changed
    if cache.get("schema_version") != SCHEMA:
        for k in ("crossings", "magnon_direct", "magnetostriction", "exchange",
              "magnon_path", "stiffness_vs_moment"):
            cache.pop(k, None)
        cache["schema_version"] = SCHEMA
    # backfill metrics that were added to an earlier cached run (no recompute)
    for k, v in (cache.get("phonons") or {}).items():
        if "imaginary_below_0p1_THz" not in v:
            W = np.array(v["omega"])
            v["imaginary_below_0p1_THz"] = int((W < -0.1).sum())
            v["softest_THz"] = float(W.min())
            v["gamma_abs_max_THz"] = float(np.abs(W[0]).max())
    if "exchange" not in cache:
        _log(f"{tag} exchange J1..J6 (frozen spirals, 3 directions)")
        cache["exchange"] = fit_exchange(nep, a0, m, tag)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
        _log(f"{tag} J = {np.round(cache['exchange']['J_meV'], 2)} meV  "
             f"rms = {cache['exchange']['fit_rms_meV_per_atom']*1000:.3f} meV/atom")
    if "exchange_at_m2p2" not in cache:
        _log(f"{tag} exchange refit at m = 2.2 uB")
        cache["exchange_at_m2p2"] = exchange_at_moment(nep, a0, tag)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    if "magnon_path" not in cache:
        _log(f"{tag} magnon dispersion (LSWT on the path + direct spirals)")
        cache["magnon_path"] = magnon_path(cache["exchange"], a0, npts)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    if "magnon_direct" not in cache:
        cache["magnon_direct"] = direct_spiral_magnon(nep, a0, m, tag)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    if do_phonons and "phonons" not in cache:
        _log(f"{tag} phonons of {len(kinds or STATES)} magnetic states")
        cache["phonons"] = state_phonons(nep, m, tag, kinds=kinds, npts=npts)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    if "phonons" in cache and "sound_velocities" not in cache:
        cache["sound_velocities"] = {k: sound_velocities(v, v["a0_A"],
                                                        v.get("npts", 41))
                                     for k, v in cache["phonons"].items()}
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    if "coupling" not in cache:
        _log(f"{tag} magnon-phonon vertex")
        cache["coupling"] = coupling_vertex(nep, a0, m, tag)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    if "stiffness_vs_moment" not in cache:
        _log(f"{tag} D(|S|) scan")
        cache["stiffness_vs_moment"] = stiffness_vs_moment(nep, a0, tag, m)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    if "a0_vs_moment" not in cache:
        _log(f"{tag} volume magnetostriction a0(|S|)")
        cache["a0_vs_moment"] = a0_vs_moment(nep, m, tag)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    if "magnetostriction" not in cache:
        _log(f"{tag} magnetostrictive spin-phonon shift")
        cache["magnetostriction"] = magnetostriction(nep, a0, m, tag)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    if "crossings" not in cache and "phonons" in cache:
        cache["crossings"] = magnon_phonon_crossings(cache["exchange"],
                                                     cache["phonons"]["FM"], a0, npts)
        U.save_json(os.path.join(SER, tag + ".json"), cache)
    _log(f"{tag} physics section done in {time.time() - t0:.0f} s")
    return cache


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[])
    ap.add_argument("--scan", default=None)
    ap.add_argument("--pattern", default="gen_*k.nep.txt")
    ap.add_argument("--no-phonons", action="store_true")
    A = ap.parse_args()
    jobs = []
    for spec in A.add:
        jobs.append(tuple(spec.split("=", 1)))
    if A.scan:
        for p in sorted(glob.glob(os.path.join(A.scan, A.pattern))):
            jobs.append((os.path.basename(p).replace(".nep.txt", ""), p))
    for tag, nep in jobs:
        f = os.path.join(SER, tag + ".json")
        if not os.path.exists(f):
            print(f"{tag}: no base cache yet, run model_series.py first")
            continue
        cache = json.load(open(f))
        run_physics(cache, nep, tag, do_phonons=not A.no_phonons)
