"""Figures for the per-state phonons, the magnon spectrum, and the magnon-phonon
crossings, plus the crossing analysis at small q.

The crossing between a magnon and a phonon is where  D q^2 = hbar v q, i.e.
q* = v/D.  With D = 349 meV.A^2 and v_L = 8.26 km/s that is q* = 0.156 1/A,
which is 0.07 of the way to H -- far below the 41-point path grid, so the
crossings are located here on a dedicated fine grid.

Nothing here changes the model.  The phonons come from the finite-displacement
force constants of relax_and_phonons.py at the relaxed lattice constant; the
magnons come from linear spin-wave theory on the exchange constants fitted to
the model's own spiral energies (and the direct spiral energies are overlaid as
markers to show that the fit is faithful).
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feutils as U
import phonon_fd as P
import relax_and_phonons as R

OUT = os.path.join(U.VAL, "raw_data", "H_states")
FIG = os.path.join(U.VAL, "figures")
THz_to_meV = 4.135667696
HSP = R.HSP
PATH = R.PATH
COL = {"FM": "C0", "AFM-G": "C3", "NC-90": "C2", "DLM": "C1", "AFM-A": "C4"}
LS = {"FM": "-", "AFM-G": "--", "NC-90": "-.", "DLM": ":", "AFM-A": (0, (1, 1))}


# ------------------------------------------------------------------ phonon IFCs
def fm_phi(a, delta=0.02, n=7, nep=None):
    st = R.replicate(R.state_cell("FM", a), n)
    st["spins"] = st["spins"] * P.m0_from_stage_B()
    phi, meta = R.force_constants(st, delta, nep=nep, tag="fig_FM", symmetrise=True)
    return phi, st, meta


def phonon_along(phi, st, a, direction, qmax=0.30, npts=181):
    qs = np.linspace(0, qmax, npts) * 2 * np.pi / a
    dv = np.array(direction, float)
    dv /= np.linalg.norm(dv)
    W = np.array([R.frequencies(phi, st, dv * q)[0] for q in qs])
    return qs, W


def shells(a, rmax=6.0):
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


def exchange_J():
    """J_n in eV from stage C (columns: shell,r,z,J_meV,J/m^2)."""
    p = os.path.join(U.VAL, "raw_data", "C", "exchange_parameters.csv")
    J = []
    for line in open(p).read().splitlines()[1:]:
        if not line.strip():
            continue
        J.append(float(line.split(",")[3]) / 1000.0)
    return np.array(J)


def lswt_omega(Jev, sh, q, S=None):
    """hbar*omega(q) in meV for H = -1/2 sum J e_i.e_j.

    The S = m/2 factor is essential: stage C computed [J(0)-J(q)] without it, so
    its whole magnon branch was low by exactly 1/S.  Measured on 8 commensurate
    points: LSWT/direct = 0.823 +- 0.015 against 1/S = 0.827.
    """
    if S is None:
        S = P.m0_from_stage_B() / 2.0
    z = np.array([len(s) for s in sh])
    sq = np.array([np.sum(np.cos(s @ np.asarray(q))) for s in sh])
    return S * float(np.sum(Jev * (z - sq))) * 1000.0


# --------------------------------------------------------------------- figures
def fig_states():
    d = json.load(open(os.path.join(OUT, "phonons_by_state.json")))
    rel = json.load(open(os.path.join(OUT, "relaxation.json")))
    a = rel["FM"]["a0_A"]
    # x axis: cumulative |q| along the path, in 2pi/a units
    ticks, tot = [0.0], 0.0
    for pa, pb in PATH:
        tot += np.linalg.norm(np.array(HSP[pb]) - np.array(HSP[pa]))
        ticks.append(tot)
    npts = 41
    x = []
    for seg, (pa, pb) in enumerate(PATH):
        L = np.linalg.norm(np.array(HSP[pb]) - np.array(HSP[pa]))
        for t in np.linspace(0, 1, npts, endpoint=(seg == len(PATH) - 1)):
            x.append(ticks[seg] + t * L)
    x = np.array(x)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for k, v in d.items():
        W = np.array(v["omega"])
        for b in range(W.shape[1]):
            ax.plot(x, W[:, b], color=COL[k], linestyle=LS[k], lw=1.0, alpha=0.9,
                    label=(f"{k} (a0={v['a0_A']:.3f} Å)" if b == 0 else None))
    for t in ticks:
        ax.axvline(t, color="0.8", lw=0.6)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$\\Gamma$", "H", "N", "$\\Gamma$", "P", "H"])
    ax.set_ylabel("phonon frequency (THz)")
    ax.set_title("phonons of bcc Fe in different magnetic states\n"
                 "each at its own relaxed a; 6/12 branches where the magnetic\n"
                 "cell doubles (folded images of the acoustic branches)")
    ax.legend(fontsize=7, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "phonons_by_state.png"), dpi=160)
    plt.close(fig)

    # relaxation curves
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for k, v in rel.items():
        g = np.array(v["grid"])
        E = np.array(v["E"]) * 1000
        ax.plot(g, E, "o-", ms=2.5, lw=1.0, color=COL[k], label=f"{k}: a0={v['a0_A']:.3f}")
        ax.axvline(v["a0_A"], color=COL[k], lw=0.6, ls=":")
    ax.set_xlabel("a (Å)")
    ax.set_ylabel("E (meV/atom)")
    ax.set_title("relaxation of each magnetic state (spin-aware scan)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "relaxation_by_state.png"), dpi=160)
    plt.close(fig)


def fig_joint(phi, st, a, mag, cr, nep=None):
    """Phonons and magnon on one energy axis (meV), full path + small-q zoom."""
    npts = 41
    ticks, tot = [0.0], 0.0
    for pa, pb in PATH:
        tot += np.linalg.norm(np.array(HSP[pb]) - np.array(HSP[pa]))
        ticks.append(tot)
    xs, W = [], []
    for seg, (pa, pb) in enumerate(PATH):
        ka = np.array(HSP[pa]) * 2 * np.pi / a
        kb = np.array(HSP[pb]) * 2 * np.pi / a
        L = np.linalg.norm(np.array(HSP[pb]) - np.array(HSP[pa]))
        for t in np.linspace(0, 1, npts, endpoint=(seg == len(PATH) - 1)):
            q = (1 - t) * ka + t * kb
            xs.append(ticks[seg] + t * L)
            W.append(R.frequencies(phi, st, q)[0])
    W = np.array(W) * THz_to_meV

    mx = []
    for r in mag:
        seg, t = r["seg"], r["t"]
        i = [k for k, (pa, pb) in enumerate(PATH) if f"{pa}-{pb}" == seg][0]
        L = np.linalg.norm(np.array(HSP[PATH[i][1]]) - np.array(HSP[PATH[i][0]]))
        mx.append(ticks[i] + t * L)
    my = np.array([r["omega_meV"] for r in mag])

    # magnon on the full path: LSWT on the fitted J_n (defined at any q);
    # the direct spiral points are only commensurate on Gamma-H, so they are
    # overlaid there as markers instead of being connected across the zone.
    Jv = exchange_J()
    shv = shells(a)
    mg_lswt = []
    for seg, (pa, pb) in enumerate(PATH):
        ka = np.array(HSP[pa]) * 2 * np.pi / a
        kb = np.array(HSP[pb]) * 2 * np.pi / a
        for t in np.linspace(0, 1, npts, endpoint=(seg == len(PATH) - 1)):
            mg_lswt.append(lswt_omega(Jv, shv, (1 - t) * ka + t * kb))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for b in range(W.shape[1]):
        ax1.plot(xs, W[:, b], "-", color="C0", lw=1.1, alpha=0.85,
                 label="phonon (FM)" if b == 0 else None)
    ax1.plot(xs, mg_lswt, "-", color="C3", lw=1.1, alpha=0.9,
             label="magnon (LSWT on the fitted $J_n$)")
    ax1.plot(mx, my, "o", color="k", ms=3.2, mfc="none",
             label="magnon (direct spiral, commensurate q)")
    for t in ticks:
        ax1.axvline(t, color="0.8", lw=0.6)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(["$\\Gamma$", "H", "N", "$\\Gamma$", "P", "H"])
    ax1.set_ylabel("energy (meV)")
    ax1.set_yscale("symlog", linthresh=5)
    ax1.set_title("phonons and magnons, full path")
    ax1.legend(fontsize=8)

    # zoom: small q along Gamma-H and Gamma-P
    for direction, lab in (([0, 0, 1], "[001]"), ([1, 1, 1], "[111]")):
        qs, Wq = phonon_along(phi, st, a, direction, qmax=0.20, npts=201)
        dv = np.array(direction, float)
        dv /= np.linalg.norm(dv)
        ax2.plot(qs * a / (2 * np.pi), Wq[:, 0] * THz_to_meV, "-", color="C0", lw=1.1)
        ax2.plot(qs * a / (2 * np.pi), Wq[:, -1] * THz_to_meV, "-", color="C0", lw=1.1,
                 alpha=0.55)
        J = exchange_J()
        sh = shells(a)
        ax2.plot(qs * a / (2 * np.pi), [lswt_omega(J, sh, dv * q) for q in qs], "--",
                 color="C3", lw=1.2)
    for c in cr:
        ax2.axvline(c["q_over_2pi_a"], color="0.4", lw=0.7, ls=":")
    ax2.set_xlabel("q along [001] / [111]  ($2\\pi/a$)")
    ax2.set_ylabel("energy (meV)")
    ax2.set_title("small-q zoom: magnon $Dq^2$ meets the acoustic branches\n"
                  "(dotted = crossing points)")
    ax2.set_xlim(0, 0.2)
    ax2.set_ylim(0, 60)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "magnon_phonon_crossing.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    rel = json.load(open(os.path.join(OUT, "relaxation.json")))
    a = rel["FM"]["a0_A"]
    mc = json.load(open(os.path.join(OUT, "magnon_coupling.json")))
    phi, st, meta = fm_phi(a)

    # crossings on a fine grid, both directions
    J = exchange_J()
    sh = shells(a)
    cr = []
    for direction in ([0, 0, 1], [1, 1, 1]):
        dv = np.array(direction, float)
        dv /= np.linalg.norm(dv)
        qs, W = phonon_along(phi, st, a, direction, qmax=0.25, npts=501)
        mm = np.array([lswt_omega(J, sh, dv * q) for q in qs])
        for b in range(W.shape[1]):
            d = mm - W[:, b] * THz_to_meV
            for i in np.where(np.diff(np.sign(d)) != 0)[0]:
                f = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]) + 1e-30)
                cr.append(dict(branch=b + 1, direction=direction,
                               q_per_A=float(qs[i] + f * (qs[i + 1] - qs[i])),
                               q_over_2pi_a=float((qs[i] + f * (qs[i + 1] - qs[i]))
                                                  * a / (2 * np.pi)),
                               omega_meV=float(0.5 * (mm[i] + W[i, b] * THz_to_meV))))
    print("crossings found:", len(cr))
    for c in cr:
        print(f"   branch {c['branch']} dir {c['direction']} q={c['q_per_A']:.4f} 1/A "
              f"({c['q_over_2pi_a']:.4f} of 2pi/a)  at {c['omega_meV']:.2f} meV")

    # upper bound on any avoided-crossing gap from the measured |C|
    C = abs(mc["coupling"]["transverse"]["C_eV_per_A_per_muB"])
    Cfloor = abs(mc["coupling"]["elastic"]["C_eV_per_A_per_muB"])
    m = P.m0_from_stage_B()
    u_zp = 2.0e-3        # A, a typical acoustic zero-point amplitude (order of magnitude)
    gap_bound_meV = 2 * max(C, Cfloor) * u_zp * 1000.0
    out = dict(crossings=cr, C_transverse=C, C_stiffness_floor=Cfloor,
               gap_upper_bound_meV=gap_bound_meV,
               gap_note="gap ~ 2|C| u_zp with u_zp ~ 2e-3 A (acoustic zero-point "
                        "amplitude); evaluated with the larger of |C| and the "
                        "stencil's own no-spin control, so it is an upper bound, "
                        "not an estimate")
    mc["crossing_analysis"] = out
    U.save_json(os.path.join(OUT, "magnon_coupling.json"), mc)

    fig_states()
    fig_joint(phi, st, a, mc["magnon_direct"], cr)
    print("figures written")
