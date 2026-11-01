"""Stage G: magneto-elastic coupling.

G1 exchange striction: J_n(a(1+eps)) -> dJ_n/dr in meV/A.  This uses only static
   energies with spins, i.e. the spin-aware part of the potential, so it is a
   genuine model prediction.

G2 FM-vs-DLM phonon shift: BLOCKED.  The Hessian path evaluates the potential in
   the S=0 limit (bit-identical omega2.out for FM and S=0, see stage E), so an
   FM and a DLM dispersion would be identical by construction and the shift is
   not measurable with this build.

G3 magnon-phonon hybridisation: NOT_SUPPORTED.  Documented reasoning plus the
   symmetry evidence from stage A.
"""
import csv
import json
import os

import numpy as np

import feutils as U

OUT = os.path.join(U.VAL, "raw_data", "G")
os.makedirs(OUT, exist_ok=True)
A0 = json.load(open(os.path.join(U.VAL, "raw_data", "B", "eos_fit.json")))["a0_A"]
M0 = json.load(open(os.path.join(U.VAL, "raw_data", "B", "moment_equilibrium.json")))["m_eq_muB"]


def shells(a, rmax=6.0):
    vecs = []
    for off in (np.zeros(3), np.array([0.5, 0.5, 0.5])):
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    d = (np.array([i, j, k], float) + off) * a
                    if np.linalg.norm(d) > 1e-6:
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


def extract_J(a, ncell=6):
    """J_n from spiral energies at lattice constant a (same machinery as stage C)."""
    SH = shells(a)
    z = np.array([len(s) for s in SH], float)
    qs, es = [], []
    for direction in ([0, 0, 1], [1, 1, 0], [1, 1, 1]):
        for mq in range(0, ncell // 2 + 1):
            q = np.array(direction, float) * 2 * np.pi * mq / (ncell * a)
            cell = U.bcc_cell(ncell, ncell, ncell, a)
            e = U.static_energy(os.path.join(OUT, f"J_a{a:.5f}_" +
                                             "".join(map(str, direction)) + f"_m{mq}"),
                                cell, U.spiral_spins(cell, M0, q, axis=2))
            qs.append(q)
            es.append(e / cell["n"])
    A = np.array([0.5 * (z - np.array([np.sum(np.cos(s @ q)) for s in SH])) for q in qs])
    b = np.array([e - es[0] for e in es])
    J, *_ = np.linalg.lstsq(A, b, rcond=None)
    return J * 1000.0, SH, float(np.sqrt(np.mean((b - A @ J) ** 2)) * 1000)


strains = [-0.02, -0.01, 0.0, 0.01, 0.02]
data = {}
for eps in strains:
    a = A0 * (1 + eps)
    J, SH, rms = extract_J(a)
    data[eps] = dict(a=a, J=J, rms=rms)
    print(f"  eps={eps:+.2f} a={a:.4f}  J1={J[0]:.2f} meV  rms={rms:.3f}", flush=True)

eps_arr = np.array(strains)
Jmat = np.array([data[e]["J"] for e in strains])
dJdeps = np.array([np.polyfit(eps_arr, Jmat[:, n], 1)[0] for n in range(Jmat.shape[1])])
SH0 = shells(A0)
r_shell = np.array([np.linalg.norm(s[0]) for s in SH0])   # radii at a0
z_shell = np.array([len(s) for s in SH])
g = dJdeps / r_shell   # dJ/dr = (dJ/deps)/(dr/deps), dr/deps = r -> meV/A

rows = []
for n in range(len(r_shell)):
    rows.append(dict(shell=n + 1, r_A=round(float(r_shell[n]), 4), z=int(z_shell[n]),
                     J0_meV=round(float(data[0.0]["J"][n]), 3),
                     dJ_deps_meV=round(float(dJdeps[n]), 3),
                     dJ_dr_meV_per_A=round(float(g[n]), 3)))
with open(os.path.join(OUT, "exchange_striction.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# displacement convergence surrogate: compare the central difference at +-1% vs +-2%
g_1pct = (data[0.01]["J"] - data[-0.01]["J"]) / (0.02 * r_shell)
g_2pct = (data[0.02]["J"] - data[-0.02]["J"]) / (0.04 * r_shell)
conv = dict(strains=strains, a0=A0, moment=M0, ncell=6,
            rows=rows,
            g_shell1_1pct=float(g_1pct[0]), g_shell1_2pct=float(g_2pct[0]),
            relative_change_percent=float(abs(g_1pct[0] - g_2pct[0]) / abs(g_1pct[0]) * 100),
            fit_rms_meV=[float(data[e]["rms"]) for e in strains],
            criterion="|g(+-1%) - g(+-2%)| / |g(+-1%)| < 20%",
            note=("uniform (hydrostatic) strain only: dJ_n/dr at the shell radius.  "
                  "A site-resolved g_ij,kalpha would need displacements of individual "
                  "atoms for every bond, which is O(N^2) single-point runs; the "
                  "hydrostatic derivative is the isotropic component of that tensor."))
conv["status"] = "PASS" if conv["relative_change_percent"] < 20 else "PARTIAL"
U.save_json(os.path.join(OUT, "exchange_striction_convergence.json"), conv)
print(json.dumps({k: v for k, v in conv.items() if k != "rows"}, indent=1)[:800])

# ---------------------------------------------------------------- G2 / G3
E = json.load(open(os.path.join(U.VAL, "raw_data", "E", "phonon_convergence.json")))
blocked = dict(
    G2=dict(status="BLOCKED",
            reason=("The Hessian is evaluated in the S=0 limit "
                    "(force/force.cu::compute_single_potential drops the spin argument); "
                    "an FM and a DLM cell produce bit-identical omega2.out "
                    f"(md5 {E['spin_aware_evidence']['fm_md5']} vs "
                    f"{E['spin_aware_evidence']['s0_md5']}).  A FM-DLM frequency shift "
                    "is therefore not measurable with this build, and computing one "
                    "would require a spin-aware dynamical matrix that does not exist."),
            consequence="delta_omega(q,nu) and lambda(q,nu) not reported"),
    G3=dict(status="NOT_SUPPORTED",
            reason=("The model's only spin-lattice coupling is the exchange-striction "
                    "type: the bond stiffness depends on S_i.S_j (descriptor Sex/San), "
                    "which is bilinear in the spin deviations and linear in the "
                    "displacement.  In a collinear ferromagnet that term couples two "
                    "magnons to one phonon, not one magnon to one phonon; a linear "
                    "one-magnon-one-phonon vertex needs spin-orbit coupling to make the "
                    "spin direction sense the lattice, and this model has none "
                    "(global spin rotation leaves the energy invariant to 1.3e-6 eV on "
                    "16 atoms, stage A4).  mask31 adds Sdm/Sia, but those are still "
                    "functions of spin and bond direction only, and stage A4's rotation "
                    "test is the direct check."),
            consequence="no avoided crossing, no |g_mp| = gap/2 reported"))
U.save_json(os.path.join(OUT, "magnon_phonon_coupling.json"), blocked)
print(json.dumps(blocked, indent=1)[:700])
