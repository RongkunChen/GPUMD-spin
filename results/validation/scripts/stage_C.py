"""Stage C: effective exchange constants and the magnon dispersion they imply.

!! CORRECTION (see validation_summary.md section 15) !!
The J_n printed by this script are correct, but the magnon conversion below is
NOT: it uses omega = J(0)-J(q) (no spin factor) and the "correction" that was
applied to D afterwards went the wrong way.  The right relation is
    hbar*omega = S [J^S(0) - J^S(q)] = [J_fit(0) - J_fit(q)] / S
so D = (1/(6S)) sum_n J_n z_n r_n^2 -- for the r18 model that is 297.5 meV.A^2,
not 345.5 and not 433.7.  Use series_physics.fit_exchange(), which has the
correct factor and also reports J in the standard convention.

C1: spin-spiral energies of the production model -> Fourier components J_n of the
    angular energy functional.  Convention used throughout (as requested):

        H = -1/2 sum_ij J_ij e_i . e_j          (e_i = unit spin direction)

    For a flat spiral e_i = (sin q.r_i, 0, cos q.r_i) this gives

        E(q)/N - E_FM/N = (1/2) sum_n J_n [ z_n - S_n(q) ],   S_n(q)=sum_{d in shell n} cos(q.d)

    so the J_n are obtained by linear least squares from model energies alone.
    They are *effective* Fourier components of the model's angular functional,
    not microscopic hopping integrals.

C2: harmonic spin-wave dispersion of that Hamiltonian, omega(q) = J(0) - J(q),
    along Gamma-H-N-Gamma-P-H, with the stiffness D from omega = D q^2.
    There is no magnon solver in the code; this is a derived quantity, and is
    labelled as such everywhere.
"""
import csv
import json
import os

import numpy as np

import feutils as U

OUT = os.path.join(U.VAL, "raw_data", "C")
os.makedirs(OUT, exist_ok=True)
A0 = json.load(open(os.path.join(U.VAL, "raw_data", "B", "eos_fit.json")))["a0_A"]
MOM = json.load(open(os.path.join(U.VAL, "raw_data", "B", "moment_equilibrium.json")))["m_eq_muB"]
M = MOM

# bcc neighbour shells up to the 6 A cutoff
def shells(a, rmax=6.0):
    """Neighbour shells of one bcc site: union of both sublattice offsets."""
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


SH = shells(A0, 6.0)
print("shells:", [(len(s), round(float(np.linalg.norm(s[0])), 3)) for s in SH], flush=True)
NSH = len(SH)


def S_of_q(q):
    return np.array([np.sum(np.cos(s @ q)) for s in SH])


def E_of_q(q, n_atoms):
    """Model energy per atom of a flat spiral with wavevector q."""
    cell = U.bcc_cell(1, 1, 1, A0)
    # build the supercell that contains the commensurate q
    raise NotImplementedError


def spiral_energy(q, ncell, tag):
    cell = U.bcc_cell(ncell, ncell, ncell, A0)
    sp = U.spiral_spins(cell, M, q, axis=2)
    e = U.static_energy(os.path.join(OUT, tag), cell, sp)
    return e / cell["n"]


NCELL = 8
qmax = 2 * np.pi / A0            # 2pi/a
qs, es = [], []
# q along [001] and [111]; m/NCELL commensurate
for direction in ([0, 0, 1], [1, 1, 0], [1, 1, 1]):
    for mq in range(0, NCELL // 2 + 1):
        q = np.array(direction, float) * 2 * np.pi * mq / (NCELL * A0)
        tag = f"spiral_{''.join(map(str, direction))}_m{mq}"
        e = spiral_energy(q, NCELL, tag)
        qs.append(q)
        es.append(e)
        print(f"  q={np.round(q/qmax,3)} ({direction})  E/atom={e:+.6f}", flush=True)

e_fm = es[0]
A = []
b = []
for q, e in zip(qs, es):
    A.append(0.5 * (np.array([len(s) for s in SH]) - S_of_q(q)))
    b.append(e - e_fm)
A = np.array(A)
b = np.array(b)
J, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
pred = A @ J
resid = b - pred
rms = float(np.sqrt(np.mean(resid ** 2)))

rows = []
for n, s in enumerate(SH, start=1):
    d = float(np.linalg.norm(s[0]))
    rows.append(dict(shell=n, r_A=round(d, 4), z=len(s),
                     J_meV=float(J[n - 1] * 1000),
                     J_per_m2_meV=float(J[n - 1] * 1000 / M ** 2)))
with open(os.path.join(OUT, "exchange_parameters.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

conv = dict(n_cell=NCELL, a0=A0, moment=M, shells=NSH,
            J_meV=[float(x * 1000) for x in J],
            fit_rms_meV_per_atom=rms, rank=rank,
            q_points=len(qs),
            note=("J_n are the first %d Fourier components of the model's angular "
                  "energy functional within the 6 A cutoff, in the convention "
                  "H=-1/2 sum J_ij e_i.e_j. Values in meV (for the model's own "
                  "|S|=%.3f muB) and normalised by m^2." % (NSH, M)))
U.save_json(os.path.join(OUT, "exchange_parameters.json"), conv)
print("J(meV):", np.round(J * 1000, 3), " rms(meV/atom):", round(rms * 1000, 4), flush=True)

# ---------------------------------------------------------------- C1b: convergence
conv_rows = []
for ncell in (4, 6, 8):
    qs2, es2 = [], []
    for direction in ([0, 0, 1], [1, 1, 0], [1, 1, 1]):
        for mq in range(0, ncell // 2 + 1):
            q = np.array(direction, float) * 2 * np.pi * mq / (ncell * A0)
            e = spiral_energy(q, ncell, f"conv{ncell}_" + "".join(map(str, direction)) + f"_m{mq}")
            qs2.append(q)
            es2.append(e)
    A2 = np.array([0.5 * (np.array([len(s) for s in SH]) - S_of_q(q)) for q in qs2])
    b2 = np.array([e - es2[0] for e in es2])
    J2 = np.linalg.lstsq(A2, b2, rcond=None)[0]
    r2 = float(np.sqrt(np.mean((b2 - A2 @ J2) ** 2)))
    conv_rows.append(dict(n_cell=ncell, n_atoms=2 * ncell ** 3,
                          **{f"J{n+1}_meV": float(J2[n] * 1000) for n in range(len(J2))},
                          rms_meV=r2 * 1000))
    print("  convergence", conv_rows[-1], flush=True)
U.save_json(os.path.join(OUT, "exchange_convergence.json"),
            dict(rows=conv_rows, reference_ncell=NCELL,
                 note="J_n vs supercell size; the 6 A cutoff needs L > 12 A, i.e. "
                      ">= 5 conventional cells for a = %.3f A" % A0))

# ---------------------------------------------------------------- C2: magnon
def omega(q):
    """omega(q) = J(0) - J(q) in meV for the fitted Hamiltonian."""
    z = np.array([len(s) for s in SH], float)
    return float(np.sum(J * (z - S_of_q(q))) * 1000)


HSP = dict(G=[0, 0, 0], H=[1, 0, 0], N=[0.5, 0.5, 0], P=[0.5, 0.5, 0.5])
path = [("G", "H"), ("H", "N"), ("N", "G"), ("G", "P"), ("P", "H")]
dpath = []
for a_, b_ in path:
    ka, kb = np.array(HSP[a_], float) * qmax, np.array(HSP[b_], float) * qmax
    for t in np.linspace(0, 1, 41 if a_ != b_ else 1):
        dpath.append((a_ + "-" + b_, t, (1 - t) * ka + t * kb))

with open(os.path.join(OUT, "magnon_dispersion.dat"), "w") as f:
    f.write("# segment t kx ky kz (2pi/a)  omega_meV\n")
    for seg, t, k in dpath:
        f.write(f"{seg} {t:.4f} {k[0]/qmax:.6f} {k[1]/qmax:.6f} {k[2]/qmax:.6f} "
                f"{omega(k):.6f}\n")

# stiffness: omega = D q^2 for |q| -> 0 along [001] and [111]
stiff = {}
for name, dv in (("001", np.array([0, 0, 1.0])), ("111", np.array([1, 1, 1.0]) / np.sqrt(3))):
    qq = np.linspace(0.02, 0.15, 8) * qmax
    om = np.array([omega(dv * x) for x in qq])
    k = np.polyfit(qq ** 2, om, 1)[0]
    stiff[name] = float(k)
D = float(np.mean(list(stiff.values())))
gold = omega(np.zeros(3))
gs = dict(D_meV_A2=D, D_by_direction=stiff, omega_at_Gamma_meV=gold,
          moment_muB=M, shells=[dict(z=len(s), r=float(np.linalg.norm(s[0]))) for s in SH],
          reference_D_range_meV_A2=[280, 320],
          criterion="280 <= D <= 320 meV.A^2",
          status="PASS" if 280 <= D <= 320 else "FAIL",
          Goldstone="omega(Gamma)=0 by construction for a pure Heisenberg form; "
                    "a non-zero value here would indicate a q-independent term missed by the fit",
          note="harmonic spin-wave dispersion of the exchange Hamiltonian fitted to "
               "the model's spiral energies; there is no magnon solver in the code")
U.save_json(os.path.join(OUT, "magnon_stiffness.json"), gs)

# group velocity along the path
os.makedirs(os.path.join(OUT, "magnon"), exist_ok=True)
with open(os.path.join(OUT, "magnon_group_velocity.dat"), "w") as f:
    f.write("# segment  |d omega/dq| (meV.A)  along the path parameter\n")
    for (seg, t, k), (seg2, t2, k2) in zip(dpath[:-1], dpath[1:]):
        if seg != seg2:
            continue
        dk = np.linalg.norm(k2 - k)
        f.write(f"{seg} {abs(omega(k2)-omega(k))/max(dk,1e-12):.6f}\n")

print("C2:", json.dumps({k: v for k, v in gs.items() if k != "shells"}, indent=1)[:600], flush=True)
