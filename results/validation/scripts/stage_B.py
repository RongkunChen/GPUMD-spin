"""Stage B: magnetic moment and magnetic ground state.

B1 EOS (a = 2.70-3.00 A, step 0.02) + Birch-Murnaghan fit
B2 moment scan m = 0.0-3.0 muB at a0  (argmin E(m))
B3 magnetic states at 4x4x4 conventional bcc (128 atoms), static and relaxed
"""
import csv
import os

import numpy as np

import feutils as U

OUT = os.path.join(U.VAL, "raw_data", "B")
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ BM3 fit
def birch_murnaghan(V, E0, V0, B0, Bp):
    eta = (V0 / V) ** (2.0 / 3.0) - 1.0
    return E0 + 9.0 * V0 * B0 / 16.0 * (eta ** 3 * Bp + eta ** 2 * (6.0 - 4.0 * eta))


def fit_bm3(V, E):
    """Levenberg-Marquardt with a numeric Jacobian (no scipy in this env)."""
    p = np.array([E.min(), V[np.argmin(E)], 0.15, 4.0])   # eV, A^3, eV/A^3, -
    lam = 1e-3
    for _ in range(400):
        r = E - birch_murnaghan(V, *p)
        J = np.zeros((len(V), 4))
        for k in range(4):
            dp = np.zeros(4)
            dp[k] = max(abs(p[k]) * 1e-5, 1e-7)
            J[:, k] = (birch_murnaghan(V, *(p + dp)) - birch_murnaghan(V, *(p - dp))) / (2 * dp[k])
        A = J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-12)
        step = np.linalg.solve(A, J.T @ r)
        p_new = p + step
        if np.sum((E - birch_murnaghan(V, *p_new)) ** 2) < np.sum(r ** 2):
            p, lam = p_new, max(lam / 3, 1e-12)
        else:
            lam *= 3
        if np.max(np.abs(step)) < 1e-12:
            break
    resid = E - birch_murnaghan(V, *p)
    return dict(E0_eV=float(p[0]), V0_A3=float(p[1]), B0_eV_A3=float(p[2]),
                Bp=float(p[3]), B0_GPa=float(p[2] * 160.2176634),
                rms_residual_eV=float(np.sqrt(np.mean(resid ** 2))),
                max_residual_eV=float(np.max(np.abs(resid))))


# ------------------------------------------------------------------ B1
a_grid = np.round(np.arange(2.70, 3.0001, 0.02), 3)
rows = []
for a in a_grid:
    cell = U.bcc_cell(3, 3, 3, float(a))            # 54 atoms, 16.95 A
    e = U.static_energy(os.path.join(OUT, f"eos_a{a:.2f}"), cell, U.fm_spins(cell["n"], 2.2))
    rows.append(dict(a=float(a), V=float(np.prod(np.diag(cell["lattice"]))),
                     E_tot=e, E_per_atom=e / cell["n"], n=cell["n"]))
    print(f"  a={a:.2f}  E/atom={e/cell['n']:+.6f}", flush=True)

with open(os.path.join(OUT, "energy_volume.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

V = np.array([r["V"] for r in rows])
E = np.array([r["E_tot"] for r in rows])
fit = fit_bm3(V, E)
fit["n_atoms"] = rows[0]["n"]
# the cell is 3x3x3 conventional cubes: V = (3a)^3, so a = V^(1/3)/3
fit["a0_A"] = float((fit["V0_A3"]) ** (1 / 3) / 3.0)
fit["experiment_a0_A"] = 2.866
fit["a0_rel_error_percent"] = float((fit["a0_A"] / 2.866 - 1) * 100)
fit["experiment_B0_GPa"] = 170.0
fit["B0_rel_error_percent"] = float((fit["B0_GPa"] / 170.0 - 1) * 100)
fit["criterion"] = "|a0 - 2.866|/2.866 <= 2%"
fit["status"] = "PASS" if abs(fit["a0_rel_error_percent"]) <= 2.0 else "FAIL"
U.save_json(os.path.join(OUT, "eos_fit.json"), fit)
print("B1:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in fit.items()
              if isinstance(v, (float, int, str))}, flush=True)

A0 = fit["a0_A"]

# ------------------------------------------------------------------ B2
m_grid = np.round(np.arange(0.0, 3.0001, 0.1), 2)
rows_m = []
for m in m_grid:
    cell = U.bcc_cell(2, 2, 2, A0)
    e = U.static_energy(os.path.join(OUT, f"moment_m{m:.1f}"), cell, U.fm_spins(cell["n"], float(m)))
    rows_m.append(dict(m=float(m), E_tot=e, E_per_atom=e / cell["n"]))
    print(f"  m={m:.1f}  E/atom={e/cell['n']:+.6f}", flush=True)

with open(os.path.join(OUT, "energy_moment.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_m[0]))
    w.writeheader()
    w.writerows(rows_m)

m = np.array([r["m"] for r in rows_m])
em = np.array([r["E_per_atom"] for r in rows_m])
i = int(np.argmin(em))
# local parabola on 3 points around the minimum
c = np.polyfit(m[i - 1:i + 2], em[i - 1:i + 2], 2)
m_eq = float(-c[1] / (2 * c[0]))
mom = dict(m_eq_muB=m_eq, E0_eV_per_atom=float(np.polyval(c, m_eq)),
           curvature_eV_per_muB2=float(2 * c[0]),
           grid_min_muB=float(m[i]), grid_min_E_eV_per_atom=float(em[i]),
           experiment_muB=2.2,
           rel_error_percent=float((m_eq / 2.2 - 1) * 100),
           criterion="|m_eq - 2.2|/2.2 <= 10%",
           status="PASS" if abs(m_eq / 2.2 - 1) <= 0.10 else "FAIL",
           a0_A=A0,
           method=("argmin_m E(m) at fixed bcc structure: the energy is evaluated by the "
                   "production model for each input |S|; |S| is not integrated by the "
                   "SLLG (see A6), so this is a variational prediction of the local moment"))
U.save_json(os.path.join(OUT, "moment_equilibrium.json"), mom)
print("B2:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in mom.items()
              if isinstance(v, (float, int, str))}, flush=True)

# ------------------------------------------------------------------ B3
REP = 4                                  # 4x4x4 conventional = 128 atoms
cell = U.bcc_cell(REP, REP, REP, A0)
n = cell["n"]
frac = cell["pos"] / np.diag(cell["lattice"])


def afm1(mm):
    """(001) planes alternate every a/2."""
    s = np.zeros((n, 3))
    plane = np.floor(frac[:, 2] * 2 + 1e-6).astype(int) % 2
    s[:, 2] = np.where(plane == 0, mm, -mm)
    return s


def afm2(mm):
    s = np.zeros((n, 3))
    is_b = np.all(np.abs(frac % 1 - 0.5) < 0.25, axis=1)
    s[~is_b, 2] = mm
    s[is_b, 2] = -mm
    return s


def cant(mm, deg):
    """Two sublattices in the x-z plane with a relative canting angle."""
    s = np.zeros((n, 3))
    is_b = np.all(np.abs(frac % 1 - 0.5) < 0.25, axis=1)
    t = np.radians(deg)
    s[~is_b] = [0.0, 0.0, mm]
    s[is_b] = [mm * np.sin(t), 0.0, mm * np.cos(t)]
    return s


def wall180(mm):
    """180 degree domain wall: spin rotates by 180 deg across the cell along z."""
    s = np.zeros((n, 3))
    ang = np.pi * frac[:, 2]
    s[:, 0] = mm * np.sin(ang)
    s[:, 2] = mm * np.cos(ang)
    return s


states = {
    "FM": U.fm_spins(n, mom["m_eq_muB"]),
    "NM_zero_moment": U.fm_spins(n, 0.0),
    "AFM_I_001_layers": afm1(mom["m_eq_muB"]),
    "AFM_II_bcc_sublattice": afm2(mom["m_eq_muB"]),
    "NC_cant_90deg": cant(mom["m_eq_muB"], 90.0),
    "NC_domain_wall_180": wall180(mom["m_eq_muB"]),
    "DLM_random_1": U.random_spins(n, mom["m_eq_muB"], 11),
    "DLM_random_2": U.random_spins(n, mom["m_eq_muB"], 12),
    "DLM_random_3": U.random_spins(n, mom["m_eq_muB"], 13),
}

rows_s = []
for name, sp in states.items():
    e = U.static_energy(os.path.join(OUT, f"state_{name}"), cell, sp)
    rows_s.append(dict(state=name, E_tot_static=e, E_per_atom_static=e / n,
                       m_input=mom["m_eq_muB"] if name != "NM_zero_moment" else 0.0))
    print(f"  {name:26s} E/atom={e/n:+.6f}", flush=True)

# relaxed structures: minimise the lattice at fixed spin configuration
for r in rows_s:
    name = r["state"]
    lines = ["potential nep.txt", "time_step 0.001 1",
             "minimize fire 1e-3 3000 1",
             "spin_llg gamma=3.0385 alpha=0 T=0"]
    rc, txt, _ = U.run_case(os.path.join(OUT, f"state_{name}_relax"), cell,
                            states[name], lines, timeout=3600)
    # FIRE prints "    step N: total_potential = X eV, f_max = Y eV/A."
    import re as _re
    hits = _re.findall(r"total_potential = ([0-9eE+.\-]+) eV", txt)
    ok = rc == 0 and len(hits) > 0
    r["E_tot_relaxed"] = float(hits[-1]) if ok else None
    r["E_per_atom_relaxed"] = float(hits[-1]) / n if ok else None
    r["relax_ok"] = bool(ok)
    print(f"  {name:26s} relaxed E/atom="
          f"{r['E_per_atom_relaxed'] if r['E_per_atom_relaxed'] is None else round(r['E_per_atom_relaxed'], 6)}",
          flush=True)

e_fm = next(r["E_per_atom_static"] for r in rows_s if r["state"] == "FM")
e_fm_rel = next(r["E_per_atom_relaxed"] for r in rows_s if r["state"] == "FM")
for r in rows_s:
    r["dE_meV_static"] = (r["E_per_atom_static"] - e_fm) * 1000
    r["dE_meV_relaxed"] = ((r["E_per_atom_relaxed"] - e_fm_rel) * 1000
                           if r["E_per_atom_relaxed"] is not None else None)

with open(os.path.join(OUT, "magnetic_ground_states.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_s[0]))
    w.writeheader()
    w.writerows(rows_s)

order = sorted(rows_s, key=lambda r: r["E_per_atom_static"])
gs = dict(states=rows_s, n_atoms=n, a0=A0,
          lowest_state=order[0]["state"],
          ranking=[(r["state"], round(r["dE_meV_static"], 3)) for r in order],
          criterion="FM must have the lowest energy among the collinear states",
          status="PASS" if order[0]["state"] == "FM" else "FAIL")
U.save_json(os.path.join(OUT, "magnetic_ground_state.json"), gs)
print("B3 lowest:", gs["lowest_state"], gs["ranking"], flush=True)
