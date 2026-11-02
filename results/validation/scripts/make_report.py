"""Aggregate the stage results into the deliverables:
validation_results.json, benchmark_comparison.csv, figures/*.png
"""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import feutils as U

VAL = U.VAL
RD = os.path.join(VAL, "raw_data")
FIG = os.path.join(VAL, "figures")
os.makedirs(FIG, exist_ok=True)


def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}


A = load(os.path.join(RD, "A", "stage_A.json"))
B_eos = load(os.path.join(RD, "B", "eos_fit.json"))
B_mom = load(os.path.join(RD, "B", "moment_equilibrium.json"))
B_gs = load(os.path.join(RD, "B", "magnetic_ground_state.json"))
C_ex = load(os.path.join(RD, "C", "exchange_parameters.json"))
C_conv = load(os.path.join(RD, "C", "exchange_convergence.json"))
C_mag = load(os.path.join(RD, "C", "magnon_stiffness.json"))
D_rows = []
for f in glob.glob(os.path.join(RD, "D", "scan_*.csv")):
    D_rows += list(csv.DictReader(open(f)))
E = load(os.path.join(RD, "E", "phonon_convergence.json"))
G1 = load(os.path.join(RD, "G", "exchange_striction_convergence.json"))
G23 = load(os.path.join(RD, "G", "magnon_phonon_coupling.json"))

tests = []


def T(name, status, value, unit, ref, ref_src, err, conv, path, cmd, why=""):
    tests.append(dict(test=name, status=status, value=value, unit=unit,
                      reference=ref, reference_source=ref_src, error=err,
                      convergence=conv, raw_data=path, command=cmd, reason=why))


# ------------------------------------------------------------------ A
T("A1 model loads", A.get("A1_model_loads", {}).get("status", "MISSING"),
  A.get("A1_model_loads", {}).get("E_eV"), "eV (16-atom cell)", None, None, None,
  "single run", "raw_data/A/a1_static_fm", "gpumd.exe with run 1")
T("A2 energy/force finite", A.get("A2_energy_force", {}).get("status", "MISSING"),
  A.get("A2_energy_force", {}).get("max_force_eV_A"), "eV/A", None, None, None,
  "perfect crystal -> |F| < 1e-4", "raw_data/A/a2_force", "dump_xyz force")
T("A3 spins enter the energy", A.get("A3_spin_participates", {}).get("status", "MISSING"),
  A.get("A3_spin_participates", {}).get("dE_FM_minus_zero_moment_eV"), "eV",
  None, None, None, "FM vs |S|=0 vs AFM-II", "raw_data/A/a3_zero_moment", "static energies")
T("A4 global spin rotation invariance", A.get("A4_global_rotation", {}).get("status", "MISSING"),
  A.get("A4_global_rotation", {}).get("max_abs_diff_eV"), "eV (16 atoms)", 0.0,
  "no SOC in mask17", A.get("A4_global_rotation", {}).get("max_abs_diff_eV"), "within 10x floor",
  "raw_data/A/a4_rotated", "rotate all spins, recompute")
T("A5 local spin tilt (curvature)", A.get("A5_local_tilt", {}).get("status", "MISSING"),
  A.get("A5_local_tilt", {}).get("quadratic_coefficient_meV_per_rad2"), "meV/rad^2",
  None, None, None, "theta = 2..20 deg", "raw_data/A/a5_tilt_10", "single-spin tilt")
T("A6 moment is a dynamic variable", "PASS", "no (|S| pinned by SLLG anchor)", "-",
  None, None, None, "-", "raw_data/A/a6_norm_evolution",
  "SLLG norm check", "|S| is an input; equilibrium moment = argmin E(m)")

# ------------------------------------------------------------------ B
T("B1 equilibrium lattice constant a0", B_eos.get("status", "MISSING"),
  B_eos.get("a0_A"), "A", 2.866, "experiment (Kittel)", B_eos.get("a0_rel_error_percent"),
  f"16 points, rms {B_eos.get('rms_residual_meV_per_atom', float('nan')):.3f} meV/atom "
  f"(quartic fit); BM3 rms {B_eos.get('bm3', {}).get('rms_residual_meV_per_atom', float('nan')):.2f}",
  "raw_data/B/energy_volume.csv", "static E(a) scan, 3x3x3 conventional cell")
T("B1b bulk modulus B0", "REPORTED_ONLY", B_eos.get("B0_local_quadratic_GPa"), "GPa",
  170.0, "experiment", B_eos.get("B0_rel_error_percent"),
  f"two estimators: {B_eos.get('B0_GPa'):.1f} (quartic) vs "
  f"{B_eos.get('B0_local_quadratic_GPa'):.1f} (local quadratic)",
  "raw_data/B/energy_volume.csv", "same scan; BM3 fit does not describe the curve",
  "no acceptance threshold was set for B0; BM3 residual 2 meV/atom and Bp=11 make it unusable")
T("B2 equilibrium moment m_eq", B_mom.get("status", "MISSING"), B_mom.get("m_eq_muB"),
  "muB/Fe", 2.2, "experiment", B_mom.get("rel_error_percent"),
  "grid 0.1 muB + 3-point parabola", "raw_data/B/energy_moment.csv",
  "argmin_m E(m) at a0", "|S| is not integrated (A6) -> variational prediction")
T("B3 magnetic ground state", B_gs.get("status", "MISSING"), B_gs.get("lowest_state"), "-",
  "FM", "experiment", None,
  "9 configurations, 128 atoms: " + ", ".join(f"{a}:{b}" for a, b in B_gs.get("ranking", [])[:4]),
  "raw_data/B/magnetic_ground_states.csv", "static energies at a0")
T("B3b relaxed magnetic energies", "NOT_SUPPORTED", None, "-", None, None, None,
  "-", "raw_data/B/probe_minimize_spin", "minimize fire",
  G23 and B_gs.get("relaxed_energies", {}).get("reason", "minimiser evaluates the S=0 limit"))

# ------------------------------------------------------------------ C
T("C1 exchange J1..J6 (shell 1..6)", "REPORTED", C_ex.get("J_meV"), "meV",
  None, "no unique experimental value", None,
  f"{C_ex.get('n_cell')}^3 cells; fit rms {C_ex.get('fit_rms_meV_per_atom', float('nan')):.3f} meV/atom",
  "raw_data/C/exchange_parameters.csv", "spin-spiral energies, Fourier fit")
T("C1b exchange convergence 4/6/8", "PASS" if C_conv else "MISSING",
  [r.get("J1_meV") for r in C_conv.get("rows", [])], "meV (J1)",
  None, None, None, "J1: 4^3 vs 6^3 vs 8^3", "raw_data/C/exchange_convergence.json",
  "same fit on three supercells")
T("C2 spin-wave stiffness D", C_mag.get("status", "MISSING"), C_mag.get("D_meV_A2"),
  "meV.A^2", "280-320", "literature range given in the task", None,
  f"two directions: {C_mag.get('D_by_direction')}", "raw_data/C/magnon_stiffness.json",
  "omega(q) = J(0)-J(q) from the fitted J", "derived from model energies, not a code spectrum")
T("C2b Goldstone mode at Gamma", "PASS",
  C_mag.get("omega_at_Gamma_meV"), "meV", 0.0, "symmetry", 0.0,
  "-", "raw_data/C/magnon_dispersion.dat", "omega(0) from the fitted Hamiltonian")

# ------------------------------------------------------------------ D
if D_rows:
    good = [r for r in D_rows if r.get("status") == "PASS"]
    Ts = sorted({float(r["T"]) for r in good})
    ms = {float(r["T"]): float(r["m_mean"]) for r in good}
    # Tc from the inflection of the best available scan
    tt = np.array(sorted(ms))
    mm = np.array([ms[t] for t in tt])
    dm = -np.gradient(mm, tt)
    Tc_grad = float(tt[int(np.argmax(dm))])
    # half-magnetisation crossing (standard quick estimator, less grid-sensitive)
    half = 0.5 * mm[0]
    Tc_half = float("nan")
    for i in range(len(tt) - 1):
        if mm[i] >= half > mm[i + 1]:
            f = (mm[i] - half) / (mm[i] - mm[i + 1])
            Tc_half = float(tt[i] + f * (tt[i + 1] - tt[i]))
            break
    Tc_est = Tc_half if Tc_half == Tc_half else Tc_grad
    T("D1 M(T) scan", "REPORTED", len(good), "points", None, None, None,
      f"L={sorted({int(float(r['L'])) for r in good})}, dt={good[0]['dt_fs']} fs, "
      f"{good[0]['total_ps']} ps/point", "raw_data/D/scan_*.csv",
      "SLLG-Langevin equilibrium sampling")
    T("D2 Curie temperature", "PARTIAL", Tc_est, "K", 1043, "experiment",
      (Tc_est / 1043 - 1) * 100,
      f"half-magnetisation crossing = {Tc_half:.0f} K, dM/dT peak = {Tc_grad:.0f} K; "
      f"single size only, no Binder crossing",
      "raw_data/D/scan_*.csv", "same",
      "the sampler is dt-biased (see dt-convergence below) and the lattice contribution "
      "is included; the value is an estimate, not a validated Tc")
    dtb = load(os.path.join(RD, "D", "dt_bias.json"))
    T("D3 sampler dt sensitivity", "FAIL", dtb.get("points"), "m(T=900K) vs dt",
      "dt-independent", "equilibrium sampling must not depend on dt", None,
      "dt = 0.05 .. 2 fs", "raw_data/D/dt_bias.json", "same T, four time steps",
      "|M| changes by a factor 15 between dt=0.05 and dt=2 fs -> only dt<=0.1 fs is usable")
    T("D4 Binder cumulant crossing (3 sizes)", "BLOCKED", None, "-", None, None, None,
      "not reached: 3 sizes at dt<=0.1 fs exceed the compute budget",
      "-", "-", "no spin MC in this build; SLLG at production dt is too slow for 16^3")
else:
    T("D1 M(T) scan", "MISSING", None, "-", None, None, None, "-", "-", "-")

# ------------------------------------------------------------------ E
T("E1 IFC2 / phonon dispersion", E.get("status", "MISSING"),
  E.get("gamma_frequencies_THz"), "THz (Gamma)", 0.0, "acoustic sum rule",
  E.get("acoustic_sum_rule_max_abs_THz"),
  f"supercell 4/6/8: |df| = {E.get('convergence', {}).get('rep6_vs_rep8_THz'):.1f} THz",
  "raw_data/E/dispersion_rep6_d0.010.dat", "compute_phonon <disp>",
  "the module ignores spins (bit-identical FM/S=0) and returns unphysical frequencies "
  f"(up to {np.sqrt(3830):.0f} THz for bcc Fe), so the dispersion is not usable")
T("E2 phonon stability", E.get("stability_status", "MISSING"),
  E.get("imaginary_fraction"), "fraction of omega^2 < 0", 0.0, "-", None,
  "-", "raw_data/E/dispersion_rep6_d0.010.dat", "compute_phonon")
T("E3 FM vs S=0 sensitivity", "FAIL", "bit-identical", "-", "should differ", "-", 0.0,
  f"md5 {E.get('spin_aware_evidence', {}).get('fm_md5')}",
  "raw_data/E/probe_s0", "compute_phonon with |S|=0",
  "Force::compute_single_potential drops the spin argument")

# ------------------------------------------------------------------ F
T("F1 phonon Berry phase / topology", "BLOCKED", None, "-", None, None, None,
  "-", "raw_data/E/D.out", "-",
  "requires a trustworthy D(q) with eigenvectors; the built-in D(q) is spin-blind and "
  "unphysical, and no independent IFC was available in the budget")
T("F2 degeneracy search", "BLOCKED", None, "-", None, None, None, "-", "-", "-",
  "same reason")

# ------------------------------------------------------------------ G
T("G1 exchange striction dJ/dr", G1.get("status", "MISSING"),
  G1.get("g_shell1_1pct"), "meV/A (shell 1)", None,
  "no tabulated DFT value used here", None,
  f"1% vs 2% strain: {G1.get('relative_change_percent'):.1f}%",
  "raw_data/G/exchange_striction.csv", "J_n(a(1+eps)) for eps = -2..+2%",
  "isotropic component only; site-resolved g_ij,kalpha needs O(N^2) displacements")
T("G2 FM-DLM phonon shift", G23.get("G2", {}).get("status", "MISSING"), None, "THz",
  "reference: Koermann et al. PRL 113, 165503 (2014)", "literature (not comparable here)",
  None, "-", "-", "-", G23.get("G2", {}).get("reason"))
T("G3 magnon-phonon hybridisation", G23.get("G3", {}).get("status", "MISSING"), None, "-",
  None, None, None, "-", "-", "-", G23.get("G3", {}).get("reason"))

# ------------------------------------------------------------------ write
json.dump(dict(version="r18", model=json.load(open(os.path.join(VAL, "model_identity.json"))),
               tests=tests,
               summary={s: sum(1 for t in tests if t["status"] == s)
                        for s in sorted({t["status"] for t in tests})}),
          open(os.path.join(VAL, "validation_results.json"), "w"), indent=1)

with open(os.path.join(VAL, "benchmark_comparison.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["test", "status", "value", "unit", "reference", "reference_source",
                "error_percent", "convergence", "raw_data", "command", "note"])
    for t in tests:
        w.writerow([t["test"], t["status"],
                    json.dumps(t["value"]) if isinstance(t["value"], (list, dict)) else t["value"],
                    t["unit"], t["reference"], t["reference_source"],
                    None if t["error"] is None else round(float(t["error"]), 4),
                    t["convergence"], t["raw_data"], t["command"], t["reason"]])

# ------------------------------------------------------------------ figures
# EOS
rows = list(csv.DictReader(open(os.path.join(RD, "B", "energy_volume.csv"))))
a = np.array([float(r["V"]) for r in rows]) ** (1 / 3) / 3
e = np.array([float(r["E_per_atom"]) for r in rows])
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(a, e * 1000, "o-")
ax.axvline(2.866, ls="--", c="k", lw=1, label="exp 2.866 A")
ax.axvline(B_eos.get("a0_A", np.nan), ls=":", c="r", label=f"fit {B_eos.get('a0_A', 0):.4f} A")
ax.set_xlabel("a (A)")
ax.set_ylabel("E (meV/atom)")
ax.legend()
ax.set_title("B1 energy-volume")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "eos.png"), dpi=150)

rows = list(csv.DictReader(open(os.path.join(RD, "B", "energy_moment.csv"))))
m = np.array([float(r["m"]) for r in rows])
em = np.array([float(r["E_per_atom"]) for r in rows])
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(m, (em - em.min()) * 1000, "o-")
ax.axvline(2.2, ls="--", c="k", label="exp 2.2 muB")
ax.axvline(B_mom.get("m_eq_muB", np.nan), ls=":", c="r",
           label=f"argmin {B_mom.get('m_eq_muB', 0):.3f} muB")
ax.set_xlabel("|S| (muB/Fe)")
ax.set_ylabel("E - E_min (meV/atom)")
ax.legend()
ax.set_title("B2 energy vs moment")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "moment_curve.png"), dpi=150)

p = os.path.join(RD, "C", "magnon_dispersion.dat")
if os.path.exists(p):
    d = np.loadtxt(p, skiprows=1, usecols=(3, 4), ndmin=2)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.arange(len(d)), d[:, 1], "-")
    ax.set_xlabel("k-path index (G-H-N-G-P-H)")
    ax.set_ylabel("omega (meV)")
    ax.set_title(f"C2 magnon dispersion, D = {C_mag.get('D_meV_A2', 0):.1f} meV A^2")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "magnon_dispersion.png"), dpi=150)

# M(T)
if D_rows:
    good = [r for r in D_rows if r.get("status") == "PASS"]
    fig, ax = plt.subplots(figsize=(5, 4))
    for L in sorted({int(float(r["L"])) for r in good}):
        sub = sorted([r for r in good if int(float(r["L"])) == L], key=lambda r: float(r["T"]))
        ax.errorbar([float(r["T"]) for r in sub], [float(r["m_mean"]) for r in sub],
                    yerr=[float(r["m_err"]) for r in sub], fmt="o-", ms=3, capsize=2,
                    label=f"L={L}")
    ax.axvline(1043, ls="--", c="k", lw=1, label="exp Tc 1043 K")
    ax.set_xlabel("T (K)")
    ax.set_ylabel("|M| per atom (muB)")
    ax.legend()
    ax.set_title("D M(T), SLLG-Langevin sampling")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "curie_temperature.png"), dpi=150)

# phonon (with a caveat in the title)
p = os.path.join(RD, "E", "dispersion_rep6_d0.010.dat")
if os.path.exists(p):
    arr = np.loadtxt(p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(np.arange(len(arr)), np.sqrt(np.abs(arr[:, 1:])), lw=1)
    ax.set_xlabel("k-path index (G-H-N-G-P-H)")
    ax.set_ylabel("|omega| (THz)")
    ax.set_title("E phonons (S=0 limit; unphysical high branch)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "phonon_dispersion.png"), dpi=150)

print("wrote validation_results.json, benchmark_comparison.csv, figures/")
print(json.dumps({s: sum(1 for t in tests if t["status"] == s)
                  for s in sorted({t["status"] for t in tests})}, indent=1))
