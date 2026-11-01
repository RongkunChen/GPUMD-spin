"""Fold the external phonon/magnon results into validation_results.json.

Replaces the E1/E2/F1/F2/G2 entries that were FAIL/BLOCKED because they relied on
the built-in spin-blind Hessian, and keeps the FAIL for the built-in module
itself (that finding does not go away).
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feutils as U

VAL = U.VAL
E = os.path.join(VAL, "raw_data", "E_fd")
res = json.load(open(os.path.join(VAL, "validation_results.json")))
ph = json.load(open(os.path.join(E, "phonon_fd.json")))
cv = json.load(open(os.path.join(E, "phonon_convergence.json")))
tp = json.load(open(os.path.join(E, "phonon_topology_report.json")))
sp = json.load(open(os.path.join(E, "spin_phonon_shift.json")))
gs = json.load(open(os.path.join(VAL, "raw_data", "C", "magnon_stiffness.json")))

CMD = ("cd validation/BCC_Fe_full_physics/scripts && "
       "python phonon_fd.py --mode fm --n 7 --delta 0.02")
NEW = {}

NEW["E1 IFC2 / phonon dispersion"] = dict(
    test="E1 IFC2 and phonon dispersion (external finite displacement)",
    status="PASS",
    value=dict(omega_max_THz=ph["omega_max_THz"],
               omega_at_Gamma_THz=ph["omega_at_Gamma_THz"],
               omega_H_THz=10.273, omega_N_THz=[5.760, 7.117, 11.235],
               omega_P_THz=9.189,
               sound_velocity_km_s=ph["sound_velocity_km_s"],
               dos_peak_THz=ph["dos_peak_THz"]),
    unit="THz / km s^-1",
    reference=dict(omega_max_THz="~9.4-9.6 (literature, not re-verified here)",
                   sound_velocity_km_s={"L[001]": 5.44, "T[001]": 3.87,
                                        "L[111]": 6.43, "T[111]": 3.03,
                                        "source": "C11=2.33, C12=1.35, C44=1.18 "
                                                  "(1e11 Pa), rho=7874 kg/m^3, "
                                                  "textbook values, not re-verified"}),
    error=dict(omega_max="+18% vs the literature anchor",
               sound_velocity_vs_experiment=dict(L001="+52%", T001="+26%",
                                                 L111="+34%", T111="+50%")),
    convergence=dict(supercell_dev_THz=cv["max_supercell_dev_THz"],
                     displacement_dev_THz=cv["max_amplitude_dev_THz"],
                     widths_A={k: v["width_A"] for k, v in cv["supercell"].items()}),
    method=ph["method"],
    cell=ph["cell"], supercell=ph["supercell"], delta_A=ph["delta_A"],
    asr_correction_eV_A2=ph["asr_correction_eV_A2"],
    asr_correction_fraction=ph["asr_correction_fraction_of_self_term"],
    criteria=dict(criterion_asr=ph["criterion_asr"],
                  criterion_stability=ph["criterion_stability"],
                  gamma_measured=ph["acoustic_sum_rule_gamma_max_THz"]),
    verdict_note="the acceptance criterion for this item is stability plus "
                 "convergence, and both hold; the absolute frequencies are 18-52% "
                 "above experiment, which is the same 'too stiff' trend as B0 and D "
                 "and is reported as an accuracy failure, not hidden",
    raw_data=[f"{E}/phonon_dispersion.dat", f"{E}/phonon_dos.dat",
              f"{E}/phonon_fd.json", f"{E}/phonon_convergence.json"],
    command=CMD,
    failure_reason=None,
)

NEW["E2 phonon stability"] = dict(
    test="E2 phonon stability (no imaginary branch)",
    status="PASS",
    value=dict(imaginary_mode_count=ph["imaginary_mode_count"],
               n_q_path=ph["n_q"], q_grid="16x16x16",
               omega_min_THz=ph["omega_min_THz"]),
    unit="THz",
    reference=dict(criterion="no branch below -1e-6 THz anywhere on the path or grid"),
    error=dict(min_frequency=ph["omega_min_THz"]),
    convergence="two independent grids (41 points/segment path, 16^3 MP grid) "
                "both show no negative eigenvalue",
    raw_data=[f"{E}/phonon_dispersion.dat", f"{E}/phonon_dos.dat"],
    command=CMD, failure_reason=None,
)

NEW["E3 FM vs S=0 sensitivity"] = dict(
    test="E3 built-in compute_phonon is spin-blind (defect, still present)",
    status="FAIL",
    value=dict(fm_vs_s0_identical=True,
               note="omega2.out is byte-identical (same md5) for a FM cell and a "
                    "|S|=0 cell, and the built-in module returns 61.9 THz for bcc Fe"),
    unit="—",
    reference=dict(criterion="FM and |S|=0 must differ"),
    error=dict(note="the Hessian path calls the spin-free Potential::compute overload"),
    convergence="not applicable",
    raw_data=[f"{VAL}/raw_data/E/probe/omega2.out", f"{VAL}/raw_data/E/probe_s0/omega2.out"],
    command="cd raw_data/E/probe && gpumd.exe   (and the same in probe_s0)",
    failure_reason="force.cu::compute_single_potential does not pass atom.spin; this is "
                   "why the external finite-displacement path was built",
)

NEW["F1 phonon Berry phase / topology"] = dict(
    test="F1 phonon Berry phase (Wilson loop) and degeneracy structure",
    status="PASS",
    value=dict(berry_phase_rad={e["loop"]: e["berry_phase_rad"] for e in tp["loops"]},
               berry_phase_over_pi={e["loop"]: e["berry_phase_pi"] for e in tp["loops"]},
               min_gap_on_loop_THz={e["loop"]: e["min_gap_band12_THz"] for e in tp["loops"]}),
    unit="rad (mod 2 pi)",
    reference=dict(criterion="gauge invariant, sampling converged, only quoted where "
                             "the loop is gapped"),
    error=dict(gauge_shift_rad=max(e.get("gauge_shift_rad", 0) for e in tp["loops"]),
               sampling="30 and 120 points agree modulo 2 pi"),
    convergence="same phase for 30, 60 and 120 points per loop; identical for random "
                "gauge phases on every eigenvector (1e-16)",
    interpretation="the kz=0.25 loop (centre 0.25,0.25,0.25) links the Gamma-P [111] "
                   "axis, where the two TA branches are degenerate by C3v symmetry; "
                   "the single-band phase is therefore pi, and the two-band total is "
                   "2 pi (=0 mod 2 pi).  Loops at kz=0 and 0.5 do not link it and give 0",
    raw_data=[f"{E}/phonon_berry_phase.csv", f"{E}/phonon_wilson_loop.dat",
              f"{E}/phonon_topology_report.json"],
    command="cd validation/BCC_Fe_full_physics/scripts && python phonon_topo.py --grid 31",
    failure_reason=None,
)

NEW["F2 degeneracy search"] = dict(
    test="F2 phonon degeneracy search on a 31^3 grid",
    status="PASS",
    value=dict(min_gap_band12_THz=tp["min_gap_band12_THz"],
               min_gap_band23_THz=tp["min_gap_band23_THz"],
               degenerate_fraction=tp["degenerate_grid_fraction"],
               omega_H_THz=10.273, omega_P_THz=9.189),
    unit="THz",
    reference=dict(criterion="map where the gaps close; no assumption about the answer"),
    error=dict(note="gap < 0.05 THz counted as degenerate"),
    convergence="31^3 grid (29791 points) plus a 101-point scan along Gamma-P",
    interpretation="H and P are exactly three-fold degenerate for any force-constant "
                   "matrix with cubic symmetry and inversion (at those wavevectors every "
                   "shell contributes an isotropic matrix), the two TA branches are "
                   "degenerate along Gamma-P, and N splits 2+1",
    raw_data=[f"{E}/phonon_degeneracies.csv", f"{E}/phonon_topology_report.json"],
    command="cd validation/BCC_Fe_full_physics/scripts && python phonon_topo.py --grid 31",
    failure_reason=None,
)

NEW["G2 FM-DLM phonon shift"] = dict(
    test="G2 FM vs DLM phonon frequency shift",
    status="PASS",
    value=dict(high_symmetry_points={
        k: dict(FM=v["FM"], DLM=v["DLM"], domega=v["domega"], sd=v["sd"])
        for k, v in sp["high_symmetry_points"].items()},
        mean_abs_domega_THz=sp["mean_abs_domega_THz"],
        max_abs_domega_THz=sp["max_abs_domega_THz"]),
    unit="THz",
    reference=dict(literature="Kormann et al., PRL 113, 165503 (2014): strong softening "
                              "of the lowest branch at N in the DLM state (qualitative "
                              "comparison only, no digitised data)"),
    error=dict(N_lowest="2.244 THz = 4.8 sigma over 4 random spin states",
               H="0.463 THz = 5.1 sigma", P="0.204 THz = 1.4 sigma (not significant)"),
    convergence=dict(n_seeds=sp["n_seeds"], mean_sd_THz=sp["mean_sd_across_seeds_THz"],
                     supercell=sp["supercell"]),
    note=sp["note"],
    raw_data=[f"{E}/spin_phonon_frequency_shift.csv", f"{E}/spin_phonon_shift.json"],
    command="cd validation/BCC_Fe_full_physics/scripts && python phonon_fd.py --mode dlm --n 6 --delta 0.02",
    failure_reason=None,
)

tests = []
for t in res["tests"]:
    tests.append(NEW.pop(t["test"], t))
for k in ("E1 IFC2 / phonon dispersion", "E2 phonon stability"):
    pass
# re-insert the new ones in a stable order after D4
order = [t["test"] for t in tests]
insert_at = order.index("E1 IFC2 / phonon dispersion") if "E1 IFC2 / phonon dispersion" in order else len(tests)
seq = ["E1 IFC2 / phonon dispersion", "E2 phonon stability",
       "E3 FM vs S=0 sensitivity", "E4 phonon convergence (supercell/displacement)",
       "F1 phonon Berry phase / topology", "F2 degeneracy search",
       "G1 exchange striction dJ/dr", "G2 FM-DLM phonon shift",
       "G3 magnon-phonon hybridisation"]

NEW["E4 phonon convergence (supercell/displacement)"] = dict(
    test="E4 phonon convergence with supercell and displacement",
    status=cv["status"],
    value=dict(supercell_dev_THz={k: v["max_dev_from_7_THz"] for k, v in cv["supercell"].items()},
               displacement_dev_THz={k: v["max_dev_from_0p01_THz"] for k, v in cv["amplitude"].items()}),
    unit="THz",
    reference=dict(criterion="supercell < 0.10 THz, displacement < 0.05 THz"),
    error=dict(max_supercell=cv["max_supercell_dev_THz"], max_displacement=cv["max_amplitude_dev_THz"]),
    convergence="6x6x6 (12.0 A), 7x7x7 (14.0 A), 8x8x8 (16.0 A) at fixed delta; "
                "delta = 0.005, 0.01, 0.02 A at fixed 7x7x7",
    raw_data=[f"{E}/phonon_convergence.json"],
    command="cd validation/BCC_Fe_full_physics/scripts && python phonon_fd.py --mode conv",
    failure_reason=None,
)

rest = [t for t in tests if t["test"] not in seq]
final = []
for t in rest:
    final.append(t)
    if t["test"] == "D4 Binder cumulant crossing (3 sizes)":
        for s in seq:
            if s in NEW:
                final.append(NEW[s])
for s in seq:
    if s in NEW and not any(t["test"] == s for t in final):
        final.append(NEW[s])

res["tests"] = final
counts = {}
for t in res["tests"]:
    counts[t["status"]] = counts.get(t["status"], 0) + 1
res["summary"] = dict(n_tests=len(res["tests"]), counts=counts,
                      note="E1/E2/E4/F1/F2/G2 were redone with the external "
                           "finite-displacement solver (spin-aware force dump); "
                           "E3 is the built-in module defect that forced it and "
                           "stays FAIL")
json.dump(res, open(os.path.join(VAL, "validation_results.json"), "w"), indent=1,
          default=float)
print(json.dumps(counts, indent=1))

# benchmark table
with open(os.path.join(VAL, "benchmark_comparison.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item", "reference", "criterion", "measured", "status"])
    w.writerow(["a0 (A)", "2.866", "<=2%", "2.8254 (-1.42%)", "PASS"])
    w.writerow(["m_eq (muB)", "2.2", "<=10%", "2.417 (+9.9%)", "PASS"])
    w.writerow(["magnetic ground state", "FM", "FM lowest", "FM lowest (+19.7 meV next)",
                "PASS"])
    w.writerow(["spin-wave stiffness D (meV A^2)", "280-320", "inside range", "345.5",
                "FAIL"])
    w.writerow(["Tc (K)", "1043", "<=15%", "1101 (half-M crossing, single size)", "PARTIAL"])
    w.writerow(["phonon stability", "no imaginary mode", "stability + convergence",
                f"{ph['imaginary_mode_count']} imaginary modes, omega_max "
                f"{ph['omega_max_THz']:.2f} THz, conv {cv['max_supercell_dev_THz']:.4f}/"
                f"{cv['max_amplitude_dev_THz']:.4f} THz", "PASS"])
    w.writerow(["phonon frequencies vs experiment", "omega_max ~9.5 THz", "informational",
                f"{ph['omega_max_THz']:.2f} THz (+18%); v_L[001] "
                f"{ph['sound_velocity_km_s']['001'][2]:.2f} vs 5.44 km/s (+52%)", "FAIL"])
    w.writerow(["phonon degeneracies", "no assumption", "mapped",
                "H and P 3-fold, TA doublet along Gamma-P, N 2+1", "PASS"])
    w.writerow(["phonon Berry phase", "no assumption", "gauge invariant + converged",
                "kz=0: 0; kz=0.25: pi; kz=0.5: 0 (links Gamma-P)", "PASS"])
    w.writerow(["exchange striction g1 (meV/A)", "DFT", "quantitative", "72.9 (spread 23%)",
                "PARTIAL"])
    w.writerow(["FM-DLM phonon shift", "Kormann 2014", "mode resolved",
                f"N: {sp['high_symmetry_points']['N']['domega'][0]:+.2f} THz (4.8 sigma)",
                "PASS"])
    w.writerow(["magnon-phonon hybridisation", "no assumption", "mechanism + spectrum",
                "no SOC: bilinear coupling only", "NOT_SUPPORTED"])
print("benchmark written")
