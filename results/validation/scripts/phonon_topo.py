"""Phonon degeneracies and Berry phases of bcc Fe, from the external D(q).

Everything here is post-processing of `phonon_fd.force_constants`: no change to
the production model, no assumption about what the answer should be.

The honest constraint first: with three acoustic branches in the primitive cell,
the bands touch at Gamma (omega = 0, three-fold) and the model also has a
three-fold degeneracy at H.  A Berry phase is only defined for a set of bands
that is separated by a gap from the rest along the whole loop, so this script
(a) maps where the gaps actually are, and (b) only computes a phase when the
loop clears every degeneracy by a stated margin.  Where it does not, the report
says NOT_DEFINED instead of printing a number.

usage: python phonon_topo.py [--n 7] [--delta 0.02] [--grid 31]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feutils as U
import phonon_fd as P

OUT = P.OUT
GAP_TOL_THz = 0.05          # "degenerate" if two branches are closer than this
LOOP_GAP_MIN_THz = 0.10     # a Berry phase needs at least this gap on the loop


def eig_sorted(phi, q):
    """Eigenvalues (THz, signed) and eigenvectors of D(q), sorted by frequency."""
    D = P.dynamical_matrix(phi, q)
    lam, vec = np.linalg.eigh((D + D.conj().T) / 2)
    order = np.argsort(np.sign(lam) * np.sqrt(np.abs(lam)))
    lam, vec = lam[order], vec[:, order]
    w = np.sign(lam) * np.sqrt(np.abs(lam)) * P.OMEGA_FACTOR
    return w, vec


def gap_scan(phi, a, nk):
    """Minimum gap between adjacent branches over the BZ, and where it closes."""
    ks = np.linspace(-0.5, 0.5, nk, endpoint=False)
    rows = []
    gmin = [1e9, 1e9]
    gmin_at = [None, None]
    for i in ks:
        for j in ks:
            for k in ks:
                q = 2 * np.pi / a * np.array([i, j, k])
                w, _ = eig_sorted(phi, q)
                for b in range(2):
                    g = float(w[b + 1] - w[b])
                    if g < gmin[b]:
                        gmin[b] = g
                        gmin_at[b] = [float(i), float(j), float(k), float(w[b]), float(w[b + 1])]
                    rows.append((float(i), float(j), float(k), b, g))
    return gmin, gmin_at, rows


def wilson_phase(phi, qs, band, gauge_seed=None):
    """Berry phase of one band along a closed loop of q points.

    Product of overlaps <u(k)|u(k+dk)>; the phase is -Im log of the product.
    `gauge_seed` multiplies every eigenvector by a random phase so that the
    gauge invariance of the result can be tested.
    """
    rng = np.random.default_rng(gauge_seed) if gauge_seed is not None else None
    prev = None
    first = None
    prod = 1.0 + 0j
    for q in qs:
        _, vec = eig_sorted(phi, q)
        u = vec[:, band].copy()
        if rng is not None:
            u *= np.exp(1j * rng.uniform(0, 2 * np.pi))
        if prev is not None:
            prod *= np.vdot(prev, u)
        else:
            first = u
        prev = u
    # close the loop: without the last->first overlap the result depends on the
    # relative phase of the two end points and is not gauge invariant (the first
    # version of this function was off by exactly that, measured pi/2)
    prod *= np.vdot(prev, first)
    return float(np.angle(prod)), float(abs(prod))


def loop_gap(phi, qs):
    """Smallest gap between the two lowest (or two highest) branches on a loop."""
    g = 1e9
    for q in qs:
        w, _ = eig_sorted(phi, q)
        g = min(g, float(w[1] - w[0]))
    return g


def circle(center, radius, n, plane=(0, 1), kfix=0.0, a=1.0):
    """Closed loop of q points (Cartesian, 2*pi/a units) in a coordinate plane."""
    pts = []
    ax1, ax2 = plane
    for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
        v = np.zeros(3)
        v[ax1] = center[0] + radius * np.cos(t)
        v[ax2] = center[1] + radius * np.sin(t)
        v[(3 - ax1 - ax2)] = kfix
        pts.append(v * 2 * np.pi / a)
    return pts


def main(n=7, delta=0.02, nk=31):
    a = P.a0_from_stage_B()
    m = P.m0_from_stage_B()
    phi, meta, cell = P.force_constants(n, a, delta, "fm", m, tag="topo")
    rep = dict(supercell=f"{n}x{n}x{n} primitive", n_atoms=meta["n_atoms"],
               delta_A=delta, asr_correction_eV_A2=meta["asr_residual_before_eV_A2"],
               nk=nk, gap_tol_THz=GAP_TOL_THz, loop_gap_min_THz=LOOP_GAP_MIN_THz)

    # ---- 1. where are the gaps?
    gmin, gmin_at, rows = gap_scan(phi, a, nk)
    with open(os.path.join(OUT, "phonon_degeneracies.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kx", "ky", "kz", "gap_index", "gap_THz"])
        w.writerows(rows)
    rep["min_gap_band12_THz"] = gmin[0]
    rep["min_gap_band23_THz"] = gmin[1]
    rep["min_gap_at"] = {"band12": gmin_at[0], "band23": gmin_at[1]}
    rep["gap_closes_band12"] = bool(gmin[0] < GAP_TOL_THz)
    rep["gap_closes_band23"] = bool(gmin[1] < GAP_TOL_THz)

    # count how many grid points are "degenerate" for each pair
    degen = {0: 0, 1: 0}
    total = 0
    for r in rows:
        total += 1
        if r[4] < GAP_TOL_THz:
            degen[r[3]] += 1
    rep["degenerate_grid_fraction"] = {f"band{b+1}{b+2}": degen[b] / max(total, 1)
                                       for b in (0, 1)}

    # ---- 2. Berry phase only where the loop is gapped
    loops = []
    for name, kfix, rad in (("kz=0_offG", 0.0, 0.18), ("kz=0.25", 0.25, 0.18),
                            ("kz=0.5", 0.5, 0.18)):
        npts = 60
        qs = circle((0.25, 0.25), rad, npts, kfix=kfix, a=a)
        g = loop_gap(phi, qs)
        entry = dict(loop=name, center=[0.25, 0.25], radius_2pi_over_a=rad,
                     kfix=kfix, n_points=npts, min_gap_band12_THz=g,
                     berry_phase_rad=None, berry_phase_pi=None,
                     defined=bool(g > LOOP_GAP_MIN_THz))
        if entry["defined"]:
            ph, mag = wilson_phase(phi, qs, 0)
            entry["berry_phase_rad"] = ph
            entry["berry_phase_pi"] = ph / np.pi
            entry["product_magnitude"] = mag
            # gauge invariance: same loop with random phases on every eigenvector
            ph2, _ = wilson_phase(phi, qs, 0, gauge_seed=12345)
            entry["gauge_shift_rad"] = float(np.angle(np.exp(1j * (ph2 - ph))))
            # sampling convergence
            for npts2 in (30, 120):
                ph3, _ = wilson_phase(phi, circle((0.25, 0.25), rad, npts2, kfix=kfix, a=a), 0)
                entry[f"berry_phase_n{npts2}_rad"] = ph3
        loops.append(entry)

    with open(os.path.join(OUT, "phonon_berry_phase.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loop", "center_x", "center_y", "radius", "kfix", "n_points",
                    "min_gap_THz", "defined", "berry_phase_rad", "berry_phase_over_pi",
                    "gauge_shift_rad", "phase_n30", "phase_n120"])
        for e in loops:
            w.writerow([e["loop"], 0.25, 0.25, e["radius_2pi_over_a"], e["kfix"],
                        e["n_points"], f"{e['min_gap_band12_THz']:.4f}",
                        e["defined"],
                        "" if e["berry_phase_rad"] is None else f"{e['berry_phase_rad']:.6f}",
                        "" if e["berry_phase_pi"] is None else f"{e['berry_phase_pi']:.6f}",
                        e.get("gauge_shift_rad", ""),
                        e.get("berry_phase_n30_rad", ""),
                        e.get("berry_phase_n120_rad", "")])

    # Wilson loop along a line: accumulate the phase as a function of k1
    npts_w = 61
    ks = np.linspace(0.5, 1.5, npts_w)
    prev, acc = None, [0.0]
    for x in ks:
        q = np.array([x, 0.25, 0.0]) * 2 * np.pi / a
        _, vec = eig_sorted(phi, q)
        u = vec[:, 0]
        if prev is not None:
            acc.append(acc[-1] + float(np.angle(np.vdot(prev, u))))
        prev = u
    with open(os.path.join(OUT, "phonon_wilson_loop.dat"), "w") as f:
        f.write("# k1 (2pi/a)  accumulated phase (rad)  band 1\n")
        for x, y in zip(ks, acc):
            f.write(f"{x:.5f} {y:.8f}\n")

    rep["loops"] = loops
    rep["wilson_line_total_phase_rad"] = acc[-1]
    rep["wilson_line_note"] = ("accumulated phase of band 1 along k=(0.5..1.5, 0.25, 0) "
                               "in 2pi/a units; the endpoint value is the phase picked "
                               "up over one reciprocal lattice vector for this band")
    rep["chern_numbers"] = None
    rep["chern_note"] = (
        "no Chern number is quoted.  A Chern number needs a set of bands separated by "
        "a gap from all others over the whole 2D torus; the three acoustic branches "
        "here touch at Gamma (three-fold, omega=0) and at H, and the measured minimum "
        f"gap is {gmin[0]:.4f} THz for bands 1-2 and {gmin[1]:.4f} THz for bands 2-3, "
        "so no such set exists in this model and assigning an integer would be "
        "undefined, not merely uncertain.")
    rep["status"] = ("DEGENERACIES_MAPPED_AND_BERRY_PHASE_MEASURED"
                     if any(e["defined"] for e in loops) else
                     "NO_GAPPED_LOOP_FOUND")
    U.save_json(os.path.join(OUT, "phonon_topology_report.json"), rep)
    print(json.dumps(rep, indent=1, default=float)[:2000], flush=True)
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--grid", type=int, default=31)
    A = ap.parse_args()
    main(n=A.n, delta=A.delta, nk=A.grid)
