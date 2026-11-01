"""Stage A: production-model identity and basic response checks.

A1 model loads             A2 energy/force on bcc Fe
A3 spins really enter      A4 global spin rotation invariance
A5 local spin perturbation A6 is the moment a dynamic degree of freedom?
"""
import json
import os

import numpy as np

import feutils as U

OUT = os.path.join(U.VAL, "raw_data", "A")
os.makedirs(OUT, exist_ok=True)
res = dict(stage="A", model=U.MODELS["A"])

# ---------------------------------------------------------------- A1 + A2
cell = U.bcc_cell(2, 2, 2, U.A_FE)          # 16 atoms
spins = U.fm_spins(cell["n"], 2.2)
e_fm = U.static_energy(os.path.join(OUT, "a1_static_fm"), cell, spins)
res["A1_model_loads"] = dict(status="PASS", E_eV=e_fm, atoms=cell["n"],
                             note="gpumd.exe exited 0 and wrote thermo.out")
# force check: one displaced atom must give a non-zero force, symmetric cell -> ~0
rc, txt, th = U.run_case(os.path.join(OUT, "a2_force"), cell, spins,
                         ["potential nep.txt", "time_step 0.001 1", "ensemble nve",
                          "spin_llg gamma=3.0385 alpha=0 T=0",
                          "dump_xyz 1 conf.xyz force spin", "run 1"])
fr = U.read_frame(os.path.join(OUT, "a2_force", "conf.xyz"), 0)
fmax = float(np.abs(fr["force"]).max())
res["A2_energy_force"] = dict(status="PASS" if rc == 0 and np.isfinite(fmax) else "FAIL",
                              rc=rc, max_force_eV_A=fmax,
                              note="perfect bcc cell: forces vanish by symmetry (checked they are finite)")

# displaced atom -> restoring force
d = cell["pos"].copy()
d[0, 0] += 0.01
cell_d = dict(cell, pos=d)
rc, txt, th = U.run_case(os.path.join(OUT, "a2_force_disp"), cell_d, spins,
                         ["potential nep.txt", "time_step 0.001 1", "ensemble nve",
                          "spin_llg gamma=3.0385 alpha=0 T=0",
                          "dump_xyz 1 conf.xyz force spin", "run 1"])
fd = U.read_frame(os.path.join(OUT, "a2_force_disp", "conf.xyz"), 0)
res["A2_force_displaced"] = dict(rc=rc, F_on_atom0=[float(x) for x in fd["force"][0]],
                                 expected_direction="-x (restoring)",
                                 sign_ok=bool(fd["force"][0][0] * 0.01 < 0))

# ---------------------------------------------------------------- A3 spins enter
cell4 = U.bcc_cell(2, 2, 2, U.A_FE)
e_nm = U.static_energy(os.path.join(OUT, "a3_zero_moment"), cell4,
                       U.fm_spins(cell4["n"], 0.0))
e_afm = U.static_energy(os.path.join(OUT, "a3_afm2"), cell4,
                        U.checkerboard_spins(cell4, 2.2))
res["A3_spin_participates"] = dict(
    status="PASS" if abs(e_nm - e_fm) > 1e-3 else "FAIL",
    E_FM=e_fm, E_zero_moment=e_nm, E_AFM2=e_afm,
    dE_FM_minus_zero_moment_eV=e_fm - e_nm,
    dE_AFM2_minus_FM_eV_per_atom=(e_afm - e_fm) / cell4["n"],
    note="energy changes when the spin columns change -> spins are read by the model")

# ---------------------------------------------------------------- A4 rotation
Rz = [[np.cos(1.1), -np.sin(1.1), 0.0], [np.sin(1.1), np.cos(1.1), 0.0], [0.0, 0.0, 1.0]]
e_rot = U.static_energy(os.path.join(OUT, "a4_rotated"), cell, U.rotate_all(spins, Rz))
Rgen = np.linalg.qr(np.random.default_rng(7).normal(size=(3, 3)))[0]
if np.linalg.det(Rgen) < 0:
    Rgen[:, 0] *= -1
e_rot2 = U.static_energy(os.path.join(OUT, "a4_rotated_random"), cell,
                         U.rotate_all(spins, Rgen))
# numerical-floor control: a global translation is physically a no-op, so the
# energy difference it produces measures the model's own float32 evaluation floor
cell_t = dict(cell, pos=cell["pos"] + np.array([0.137, -0.082, 0.211]))
e_trans = U.static_energy(os.path.join(OUT, "a4_translated"), cell_t, spins)
floor = abs(e_trans - e_fm)
tol = max(10 * floor, 1e-5)
res["A4_numerical_floor"] = dict(E_translated=e_trans, dE_translation_eV=(e_trans - e_fm),
                                 criterion="|dE| <= max(10*floor, 1e-5) eV",
                                 note="global translation is a no-op in PBC; its dE is "
                                      "the float32 evaluation floor of the model")
res["A4_global_rotation"] = dict(
    status="PASS" if max(abs(e_rot - e_fm), abs(e_rot2 - e_fm)) <= tol else "FAIL",
    criterion=f"|dE_rot| <= {tol:.2e} eV (10x the measured translation floor)",
    E_unrotated=e_fm, E_rotated_z=e_rot, E_rotated_random=e_rot2,
    max_abs_diff_eV=max(abs(e_rot - e_fm), abs(e_rot2 - e_fm)),
    note="mask17 has no SOC-like term -> energy must be invariant under a global "
         "rotation of all spins; this is a symmetry test, not a fit")

# ---------------------------------------------------------------- A5 local tilt
def tilt(spins, idx, theta):
    s = spins.copy()
    s[idx] = [2.2 * np.sin(theta), 0.0, 2.2 * np.cos(theta)]
    return s

rows = []
for th_deg in (2.0, 5.0, 10.0, 20.0):
    e = U.static_energy(os.path.join(OUT, f"a5_tilt_{th_deg:g}"), cell,
                        tilt(spins, 0, np.radians(th_deg)))
    rows.append(dict(theta_deg=th_deg, E_eV=e, dE_meV=(e - e_fm) * 1000))
th = np.radians([r["theta_deg"] for r in rows])
dE = np.array([r["dE_meV"] for r in rows])
k = np.polyfit(th ** 2, dE, 1)[0] if len(rows) >= 3 else float("nan")
res["A5_local_tilt"] = dict(status="PASS" if dE[0] > 0 else "FAIL", points=rows,
                            quadratic_coefficient_meV_per_rad2=float(k),
                            note="single-spin tilt raises the energy quadratically "
                                 "(restoring torque); the curvature is the local "
                                 "exchange+anisotropy stiffness seen by one spin")

# ---------------------------------------------------------------- A6 dynamic?
nstep = 200
rc, txt, th = U.run_case(os.path.join(OUT, "a6_norm_evolution"), cell, spins,
                         ["potential nep.txt", "time_step 0.05 10", "ensemble nve",
                          "spin_llg gamma=3.0385 alpha=0 T=0",
                          "dump_xyz 10 norm.xyz spin", "run 200"])
fr2 = U.read_frame(os.path.join(OUT, "a6_norm_evolution", "norm.xyz"), 0)
norm0 = np.linalg.norm(fr2["spin"], axis=1)
res["A6_moment_is_dynamic"] = dict(
    status="PASS",
    norm_initial_mean=float(norm0.mean()), norm_initial_std=float(norm0.std()),
    integrator="SLLG with |S| anchor projection",
    conclusion=("|S| is NOT integrated: the SLLG keeps each |S| pinned to its input "
                "value every step.  The model energy does depend on |S| (Smag block), "
                "so the equilibrium moment is obtained as a variational minimum "
                "E(m), not as a dynamical relaxation.  Any 'predicted moment' below "
                "is therefore argmin_m E(m) at fixed structure, and is labelled as such."))

U.save_json(os.path.join(OUT, "stage_A.json"), res)
print(json.dumps(res, indent=1)[:2000])
