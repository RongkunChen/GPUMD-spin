"""Independent determination of the magnon frequency: let the model's own SLLG run.

Why this exists
---------------
The magnon energy is obtained two ways in this project, and both convert a *static*
configuration energy into a frequency:

    (A)  hbar*omega(q) = 2 * S * [E(q) - E(FM)] / N          (used everywhere so far)
    (B)  hbar*omega(q) = 2 * [E(q) - E(FM)] / (N * S)        (what LSWT actually gives)

They differ by S^2 = 1.49 at m = 2.44 uB.  The two "independent" routes never caught
this because both carry the same factor.  So neither static route can decide it.

This script decides it *dynamically*: excite a long-wavelength spin wave in a
supercell, integrate the real SLLG equations with alpha = 0, T = 0, and read the
precession period off the trajectory.  No Heisenberg mapping, no factor S, no
convention — just the frequency at which the model's spins actually precess.

The SLLG rate is physical: GPUMD's gamma = 3.0385 is
    3.0385 rad/ps per (meV/uB)  =  g mu_B / hbar * (5.788e-5 eV/T) ...
i.e. the Fe gyromagnetic ratio expressed with energies in meV and moments in uB.
So with time in ps, hbar*omega[meV] = 0.6582 * omega[rad/ps] = 4.1357 / period[ps].

usage: python magnon_dynamics_test.py [--n 5] [--model PATH] [--ps 2.0]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)),
                                "validation", "BCC_Fe_full_physics", "scripts"))
sys.path.insert(0, HERE)
import feutils as U
import series_physics as SPS

OUT = os.path.join(U.VAL, "raw_data", "magnon_dynamics")
HBAR_MEV_PS = 0.6582119569        # hbar in meV*ps (per rad)


def relax_positions(cell, spins, nep, n_iter=120, step=0.002, tol=1e-4):
    """Steepest descent on the atomic positions at fixed spins.

    Without this the lattice is left with the residual force of a 0.001-0.01 A
    lattice mismatch.  Those forces accelerate the atoms immediately and the
    resulting lattice heat (51 K within 25 fs in the first attempt) swamps the
    magnon through magnetoelastic coupling, and the guard then stops the run for
    "energy-conservation drift".  `minimize` cannot be used: it evaluates the
    spin-free energy (stage E).
    """
    pos = cell["pos"].copy()
    fmax = np.inf
    for _ in range(n_iter):
        c = dict(cell)
        c["pos"] = pos
        f = R_forces(c, spins, nep)
        fmax = float(np.abs(f).max())
        if fmax < tol:
            break
        pos = pos + step * f
    c = dict(cell)
    c["pos"] = pos
    return c, fmax


def R_forces(cell, spins, nep):
    import relax_and_phonons as _R
    return _R.forces_at(os.path.join(OUT, "_force_probe"), cell, spins, nep=nep)


def run_one(n, a, m, qmag, eps, ps, dt, model, tag):
    """Excite a flat spin wave of wavevector q = qmag * z_hat and record the spins."""
    cell = U.bcc_cell(n, n, n, a)
    ang = cell["pos"][:, 2] * qmag
    th = eps * np.cos(ang)
    sp = np.zeros((cell["n"], 3))
    sp[:, 0] = m * np.sin(th)
    sp[:, 2] = m * np.cos(th)
    cell, fmax = relax_positions(cell, sp, model)
    print(f"  [{tag}] positions relaxed, residual max|F| = {fmax:.2e} eV/A", flush=True)
    d = os.path.join(OUT, tag)
    steps = int(ps * 1000 / dt)
    lines = ["potential nep.txt", f"time_step {dt} 100", "ensemble nve",
             "spin_llg gamma=3.0385 alpha=0 T=0",
             "dump_xyz 50 traj.xyz spin", "dump_thermo 500", f"run {steps}"]
    # run_case writes model.xyz / nep.txt / run.in itself; do not pre-write run.in
    rc, txt, th_ = U.run_case(d, cell, sp, lines, nep=model, timeout=7200)
    return d, rc


def read_signal(d, dt=0.01, every=50, atom=0):
    """S_x(t) of one atom from the trajectory (positions+spins, every-50 frames)."""
    path = os.path.join(d, "traj.xyz")
    nfr = int(sum(1 for _ in open(path)) and 0)
    t, sx = [], []
    k = 0
    while True:
        try:
            fr = U.read_frame(path, k)
        except IndexError:
            break
        t.append(k * every * dt)
        sx.append(float(fr["spin"][atom, 0]))
        k += 1
    return np.array(t), np.array(sx)


def period_fft(t, y):
    """Dominant period from a zero-padded FFT of the detrended signal."""
    y = y - y.mean()
    dt = t[1] - t[0]
    n = len(y)
    w = np.hanning(n)
    Y = np.abs(np.fft.rfft(y * w, n=8 * n))
    f = np.fft.rfftfreq(8 * n, dt)               # 1/ps
    k = int(np.argmax(Y[1:])) + 1
    # parabolic refinement around the peak
    if 1 <= k < len(Y) - 1:
        a, b, c = Y[k - 1], Y[k], Y[k + 1]
        dk = 0.5 * (a - c) / (a - 2 * b + c)
    else:
        dk = 0.0
    freq = f[k] + dk * (f[1] - f[0])
    return float(freq), float(1.0 / freq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--a", type=float, default=None)
    ap.add_argument("--m", type=float, default=None)
    ap.add_argument("--ps", type=float, default=2.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--model", default=os.path.join(
        U.ROOT, "work_20261101_retrain", "C_5desc_500k_model.nep.txt"))
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    S = (A.m or 2.442) / 2.0
    a = A.a or 2.8346
    m = A.m or 2.442

    # the static spiral energies that the two formulas convert
    ex = None
    rows = []
    for mq in (1, 2):
        q = 2 * np.pi * mq / (A.n * a)
        d, rc = run_one(A.n, a, m, q, A.eps, A.ps, A.dt, A.model, f"n{A.n}_mq{mq}")
        t, sx = read_signal(d, A.dt, 50)
        freq, per = period_fft(t, sx)
        omega_meV = 2 * np.pi * freq * HBAR_MEV_PS
        # the static energy of the same modulation
        e_fm, _ = SPS.spiral_energy(A.model, a, m, np.zeros(3), A.n, f"dyn_FM_{A.n}",
                                    os.path.join(OUT, "static"))
        e_q, _ = SPS.spiral_energy(A.model, a, m, np.array([0, 0, q]), A.n,
                                   f"dyn_q{mq}", os.path.join(OUT, "static"))
        dE = (e_q - e_fm) * 1000.0                     # meV/atom
        cand_A = 2 * S * dE
        cand_B = 2 * dE / S
        rows.append(dict(n=A.n, mq=mq, q_per_A=float(q), period_ps=per,
                         omega_meV=omega_meV,
                         dE_meV_per_atom=float(dE),
                         formula_2S_dE=cand_A, formula_2dE_overS=cand_B,
                         ratio_to_2S_dE=omega_meV / cand_A,
                         ratio_to_2dE_overS=omega_meV / cand_B,
                         D_from_dynamics=omega_meV / q ** 2))
        print(f"q={q:.5f} 1/A  period={per*1000:.1f} fs  hbar*omega={omega_meV:8.3f} meV"
              f"   dE={dE:8.3f} meV/at   2S*dE={cand_A:8.3f}  2dE/S={cand_B:8.3f}"
              f"   ratio={omega_meV/cand_A:6.3f} / {omega_meV/cand_B:6.3f}", flush=True)
    S_true = 1.0 / np.sqrt(rows[0]["ratio_to_2dE_overS"]) if rows else None
    res = dict(rows=rows, S=m, gamma_note="gamma=3.0385 rad/ps per (meV/uB)",
               hbar_meV_ps=HBAR_MEV_PS, eps=A.eps, n=A.n, a0=a,
               verdict=("dynamics matches 2*dE/S (i.e. the code's 2*S*dE is wrong by S^2)"
                        if abs(rows[0]["ratio_to_2dE_overS"] - 1) < 0.05 else
                        ("dynamics matches 2*S*dE (code is right)"
                         if abs(rows[0]["ratio_to_2S_dE"] - 1) < 0.05 else
                         "neither formula matches -- investigate")))
    U.save_json(os.path.join(OUT, "magnon_dynamics_test.json"), res)
    print("\nVERDICT:", res["verdict"])
    for r in rows:
        print(f"  D from dynamics at mq={r['mq']}: {r['D_from_dynamics']:.1f} meV*A^2")


if __name__ == "__main__":
    main()
