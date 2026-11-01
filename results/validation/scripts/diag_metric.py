"""Second diagnostic: is the long-range tail of Phi real, or a metric artefact?

The first version used `f -= round(f)` for the minimum image, which is only
correct for orthogonal cells.  For the rhombohedral bcc primitive cell it can
pick a vector that is up to a lattice vector too long, so a neighbour at 5 A
can be counted at 8 A.  Here the minimum image is found by enumerating the
3x3x3 (or 5x5x5) images explicitly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feutils as U
import phonon_fd as P


def true_min_image(dr, L, span=2):
    f = dr @ np.linalg.inv(L)
    best, bn = None, 1e9
    for i in range(-span, span + 1):
        for j in range(-span, span + 1):
            for k in range(-span, span + 1):
                v = (f + np.array([i, j, k])) @ L
                n = float(v @ v)
                if n < bn:
                    bn, best = n, v
    return best


a = P.a0_from_stage_B()
m = P.m0_from_stage_B()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
cell = P.primitive_cell(n, a)
L = cell["lattice"]
V = abs(np.linalg.det(L))
w = V / np.linalg.norm(np.cross(L[0], L[1]))
print(f"n={n} atoms={cell['n']} box volume={V:.1f} A^3  interplanar width={w:.2f} A")
sp = U.fm_spins(cell["n"], m)
base = os.path.join(P.OUT, "dbg", f"truemi_n{n}")
os.makedirs(base, exist_ok=True)

f0 = P.forces_at(os.path.join(base, "ref"), cell, sp)
dF = []
for sign in (+1.0, -1.0):
    pos = cell["pos"].copy()
    pos[0, 0] += sign * 0.01
    c2 = dict(cell)
    c2["pos"] = pos
    dF.append(P.forces_at(os.path.join(base, f"d{'p' if sign > 0 else 'm'}"), c2, sp))
dphi = -(dF[0] - dF[1]) / 0.02

rows = []
for j in range(cell["n"]):
    r_round = np.linalg.norm(P.min_image(cell["pos"][j] - cell["pos"][0], L))
    r_true = np.linalg.norm(true_min_image(cell["pos"][j] - cell["pos"][0], L))
    rows.append((r_true, r_round, np.abs(dphi[j]).max()))

rows.sort()
print("\n r_true   r_round   |dPhi|max      (the 30 largest distances)")
for r1, r2, v in rows[-30:]:
    print(f" {r1:7.3f} {r2:8.3f}   {v:.3e}")

print("\n|dPhi| summed by TRUE distance shell (eV/A^2):")
for lo, hi in ((0, 0.1), (0.1, 2.6), (2.6, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 99)):
    tot = sum(v for r1, r2, v in rows if lo <= r1 < hi)
    cnt = sum(1 for r1, r2, v in rows if lo <= r1 < hi)
    print(f"  {lo:5.1f}-{hi:5.1f} A : {tot:.4e}   ({cnt} atoms)")

for span, lab in ((0, "round only"), (2, "true min image")):
    acc = np.zeros(3)
    for j in range(cell["n"]):
        if span == 0:
            dr = P.min_image(cell["pos"][j] - cell["pos"][0], L)
        else:
            dr = true_min_image(cell["pos"][j] - cell["pos"][0], L)
        if np.linalg.norm(dr) <= P.RC + 1e-6:
            acc += dphi[j]
    print(f"  r_c-restricted sum with {lab:15s}: {np.abs(acc).max():.4e}")
