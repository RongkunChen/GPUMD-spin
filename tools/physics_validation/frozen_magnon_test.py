"""Decide the magnon conversion factor by measuring the precession frequency.

The static routes (frozen spirals, LSWT from the fitted J_n) both convert an
*energy* into a frequency, and they both used the same factor, so their agreement
could never catch a wrong factor.  This measures the frequency directly:

  * bcc Fe supercell with a small long-wavelength tilt, S_x = m sin(eps cos qz)
  * lattice frozen (`fix 0 1`, group column in the xyz) so the known spin/lattice
    energy-pumping of conservative SLLG runs (F19) cannot contaminate the signal
  * alpha = 0, T = 0, so the magnon precesses without damping
  * hbar*omega = 4.1357 / period[ps] meV  (hbar = 0.6582 meV*ps per rad)

Two candidate conversions are compared:

    A:  hbar*omega = 2 S [E(q)-E(FM)]/N        (what the code has been doing)
    B:  hbar*omega = 2 [E(q)-E(FM)]/(N S)      (what LSWT gives, S = m/2)

usage: python frozen_magnon_test.py [--n 5] [--ps 3] [--mq 1 2]
"""
import argparse
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
HBAR = 0.6582119569        # meV*ps per rad


def write_xyz_group(path, cell, spins):
    """xyz with a group column so that `fix 0 1` can freeze every atom."""
    L, n = cell["lattice"], cell["n"]
    with open(path, "w") as f:
        f.write(f"{n}\n")
        lat = " ".join(f"{x:.10f}" for x in L.reshape(-1))
        f.write(f'Lattice="{lat}" Properties=species:S:1:pos:R:3:spin:R:3:group:I:1\n')
        for i in range(n):
            f.write("Fe " + " ".join(f"{x:.10f}" for x in cell["pos"][i]) + " "
                    + " ".join(f"{s:.10f}" for s in spins[i]) + " 1\n")


def measure(n, a, m, mq, eps, ps, dt, every, model, tag):
    q = 2 * np.pi * mq / (n * a)
    cell = U.bcc_cell(n, n, n, a)
    th = eps * np.cos(cell["pos"][:, 2] * q)
    sp = np.zeros((cell["n"], 3))
    sp[:, 0] = m * np.sin(th)
    sp[:, 2] = m * np.cos(th)
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    write_xyz_group(os.path.join(d, "model.xyz"), cell, sp)
    with open(os.path.join(d, "nep.txt"), "wb") as fh, open(model, "rb") as src:
        fh.write(src.read())
    steps = int(ps * 1000 / dt)
    with open(os.path.join(d, "run.in"), "w") as fh:
        fh.write("potential nep.txt\n"
                 f"time_step {dt} 100\n"
                 "ensemble nve\n"
                 "fix 0 1\n"                     # frozen lattice: spin-only dynamics
                 "spin_llg gamma=3.0385 alpha=0 T=0\n"
                 f"dump_xyz {every} traj.xyz spin\n"
                 "dump_thermo 1000\n"
                 f"run {steps}\n")
    for junk in ("traj.xyz", "thermo.out", "stdout.txt", "SPIN_GUARD_STOP"):
        p = os.path.join(d, junk)
        if os.path.exists(p):
            open(p, "w").close()
    import subprocess
    with open(os.path.join(d, "stdout.txt"), "w") as fh:
        subprocess.run([U.EXE], cwd=d, stdout=fh, stderr=subprocess.STDOUT)
    # read the field of atom 0 -- ONE streaming pass, not read_frame(path, k)
    # per frame (that re-parses the file every time and is O(n^2): 6000 frames
    # took minutes).
    path = os.path.join(d, "traj.xyz")
    t, sx = [], []
    k = 0
    with open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                break
            n = int(line)
            head = fh.readline()
            for _ in range(n):
                fh.readline()
            # rewind-free: parse the frame we just consumed is not possible, so
            # read the header block again from the cached strings instead
            # (cheap: only the atom-0 line is needed, so re-open by offset)
            k += 1
    # the loop above just counts frames; do the actual extraction by offset scan
    t, sx = [], []
    offset = 0
    with open(path, "rb") as fh:
        data = fh.read()
    lines = data.splitlines()
    i = 0
    k = 0
    while i < len(lines):
        n = int(lines[i])
        cols = lines[i + 1]
        # Properties=species:S:1:pos:R:3:spin:R:3  -> spin starts at token 5
        tok = lines[i + 2].split()
        sx.append(float(tok[5]))
        t.append(k * every * dt)
        k += 1
        i += 2 + n
    t = np.array(t)
    sx = np.array(sx)
    y = sx - sx.mean()
    zc = np.where(np.diff(np.sign(y)) != 0)[0]
    tc = []
    for i in zc:
        f = y[i] / (y[i] - y[i + 1]) if y[i] != y[i + 1] else 0.0
        tc.append(t[i] + f * (t[i + 1] - t[i]))
    tc = np.array(tc)
    per_x = 2 * np.polyfit(np.arange(len(tc)), tc, 1)[0] if len(tc) > 3 else np.nan
    # FFT cross-check
    dt_ = t[1] - t[0]
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y)), 8 * len(y)))
    f = np.fft.rfftfreq(8 * len(y), dt_)
    k0 = int(np.argmax(Y[1:])) + 1
    per_f = 1.0 / f[k0]
    return dict(n=n, mq=mq, q_per_A=float(q), eps=eps, ps=ps, dt=dt,
                frames=int(len(t)), zero_crossings=int(len(tc)),
                period_fs_crossings=float(per_x), period_fs_fft=float(per_f),
                omega_meV_crossings=float(4.135667696 / per_x * 1000)
                if np.isfinite(per_x) else None,
                omega_meV_fft=float(4.135667696 / per_f * 1000),
                amplitude=float(np.abs(y).max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--a", type=float, default=2.8346)
    ap.add_argument("--m", type=float, default=2.442)
    ap.add_argument("--ps", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=0.05)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.005)
    ap.add_argument("--mq", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--model", default=os.path.join(
        U.ROOT, "work_20261101_retrain", "C_5desc_500k_model.nep.txt"))
    A = ap.parse_args()
    S = A.m / 2.0
    os.makedirs(OUT, exist_ok=True)
    wd = os.path.join(OUT, "_static")
    e_fm, nat = SPS.spiral_energy(A.model, A.a, A.m, np.zeros(3), A.n, "FM", wd)
    rows = []
    for mq in A.mq:
        r = measure(A.n, A.a, A.m, mq, A.eps, A.ps, A.dt, A.every, A.model,
                    f"frozen_n{A.n}_mq{mq}")
        q = r["q_per_A"]
        e_q, _ = SPS.spiral_energy(A.model, A.a, A.m, np.array([0, 0, q]), A.n,
                                   f"q{mq}", wd)
        dE = (e_q - e_fm) * 1000.0                       # meV/atom, full spiral
        r["dE_meV_per_atom"] = float(dE)
        r["cand_A_2S_dE"] = float(2 * S * dE)
        r["cand_B_2dE_overS"] = float(2 * dE / S)
        r["ratio_A"] = r["omega_meV_fft"] / r["cand_A_2S_dE"]
        r["ratio_B"] = r["omega_meV_fft"] / r["cand_B_2dE_overS"]
        r["D_from_dynamics"] = float(r["omega_meV_fft"] / q ** 2)
        rows.append(r)
        print(f"mq={mq} q={q:.4f} 1/A  period={r['period_fs_fft']:.1f} fs "
              f"(crossings {r['period_fs_crossings']:.1f})  hbar*omega="
              f"{r['omega_meV_fft']:.2f} meV   dE={dE:.2f} meV/at  "
              f"A={r['cand_A_2S_dE']:.2f}  B={r['cand_B_2dE_overS']:.2f}  "
              f"ratio A/B = {r['ratio_A']:.3f} / {r['ratio_B']:.3f}", flush=True)
    res = dict(rows=rows, S=S, m=A.m, a0=A.a, n=A.n, eps=A.eps, dt=A.dt,
               hbar_meV_ps=HBAR, lattice="frozen (fix 0 1, spin-only SLLG)",
               note="period from a zero-padded FFT of S_x(t) of atom 0; the "
                    "crossing count is an independent check")
    U.save_json(os.path.join(OUT, "frozen_magnon_test.json"), res)
    rb = np.mean([r["ratio_B"] for r in rows])
    ra = np.mean([r["ratio_A"] for r in rows])
    print(f"\nmean ratio to A (2S dE) = {ra:.3f}   to B (2dE/S) = {rb:.3f}")
    print("VERDICT:", "B (code's 2S is wrong by S^2)" if abs(rb - 1) < abs(ra - 1)
          else "A (code is right)")


if __name__ == "__main__":
    main()
