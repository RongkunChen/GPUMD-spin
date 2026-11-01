"""Diagnostic: why does the finite-difference Phi violate the acoustic sum rule?

Checks, at fixed geometry and several displacements:
  * total force of the undisplaced cell (must vanish),
  * sum_j dPhi_j over ALL atoms (must vanish if forces are exact gradients),
  * the same sum restricted to the r_c shell (what the dispersion uses),
  * Phi(R) vs Phi(-R)^T.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feutils as U
import phonon_fd as P

a = P.a0_from_stage_B()
m = P.m0_from_stage_B()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
cell = P.primitive_cell(n, a)
L = cell["lattice"]
sp = U.fm_spins(cell["n"], m)
base = os.path.join(P.OUT, "dbg", f"asr_n{n}")
os.makedirs(base, exist_ok=True)

f0 = P.forces_at(os.path.join(base, "f0"), cell, sp)
print(f"n={n} atoms={cell['n']}")
print("sum F (undisplaced) =", np.round(f0.sum(0), 8), " |F|max = %.5f" % np.abs(f0).max())

for delta in (0.005, 0.01, 0.02):
    all_sum, rc_sum, blocks = [], [], []
    for alpha in range(3):
        dF = []
        for sign in (+1.0, -1.0):
            pos = cell["pos"].copy()
            pos[0, alpha] += sign * delta
            c2 = dict(cell)
            c2["pos"] = pos
            dF.append(P.forces_at(os.path.join(base, f"d{alpha}{'p' if sign > 0 else 'm'}_{delta}"),
                                  c2, sp))
        dphi = -(dF[0] - dF[1]) / (2 * delta)          # Phi = -dF/du
        all_sum.append(dphi.sum(0))
        acc = np.zeros(3)
        for j in range(cell["n"]):
            dr = P.min_image(cell["pos"][j] - cell["pos"][0], L)
            if np.linalg.norm(dr) <= P.RC + 1e-6:
                acc += dphi[j]
        rc_sum.append(acc)
    print(f"delta={delta:5.3f}  |sum_all| = {np.abs(np.array(all_sum)).max():.3e}"
          f"  |sum_rc| = {np.abs(np.array(rc_sum)).max():.3e}")

# how much force response lies outside r_c?
pos = cell["pos"].copy()
pos[0, 0] += 0.01
c2 = dict(cell)
c2["pos"] = pos
fp = P.forces_at(os.path.join(base, "outside"), c2, sp)
dphi = -(fp - f0) / 0.01
print("\n|dPhi| summed by distance shell (eV/A^2):")
for lo, hi in ((0, 0.1), (0.1, 2.6), (2.6, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 99)):
    tot = 0.0
    for j in range(cell["n"]):
        r = np.linalg.norm(P.min_image(cell["pos"][j] - cell["pos"][0], L))
        if lo <= r < hi:
            tot += np.abs(dphi[j]).max()
    print(f"  {lo:5.1f}-{hi:5.1f} A : {tot:.4e}")
