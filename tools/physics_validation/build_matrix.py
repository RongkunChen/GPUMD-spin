"""Build the v3.1 validation matrix.

Rule: the rows of the shipped `FINAL_TEST_MATRIX.csv` are copied **verbatim**,
including every historical FAIL.  New rows are appended with their own ids
(`R2-*`, `PV1-*`) and point at the evidence file that holds the number.  Nothing
is re-graded: a FAIL stays a FAIL, and the new PASS rows state the scope in the
criterion column instead of the status column.
"""

import csv
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OLD = os.path.join(ROOT, "final_delivery_20261031_174555", "FINAL_TEST_MATRIX.csv")
STAGE = os.path.join(ROOT, "evidence", "stage_PV1")
LT = os.path.join(STAGE, "fe_lt", "fe_lt_results.json")
HEAT = os.path.join(STAGE, "heat_audit", "heat_audit_results.json")


def load(path):
    if os.path.exists(path):
        return json.load(open(path))
    return None


def main():
    rows = list(csv.reader(open(OLD)))
    header = rows[0]
    body = rows[1:]

    new = []

    # ---- R2 (precision) ------------------------------------------------------
    new.append(["R2-01", "double-precision energy mode (P1) on the same trajectory",
                "1e-5 eV/atom", "1.0105e-05 (dt=0.2), 3.1107e-06 (0.1), 1.2199e-06 (0.05)",
                "FAIL at 0.2 fs (kept)", "numerical_accuracy_R2/20261031_214208/R2_ENERGY_CONSERVATION.json"])
    new.append(["R2-02", "same gate at dt=0.1 fs, 0.1 ps, 4-atom Fe cell", "1e-5 eV/atom",
                "3.1557e-06 (P0) / 3.1107e-06 (P1)", "PASS (short-time, scoped)",
                "same file, case B_dt0.1"])
    new.append(["R2-03", "same gate at dt=0.05 fs, 0.1 ps, 4-atom Fe cell", "1e-5 eV/atom",
                "1.6099e-06 (P0) / 1.2199e-06 (P1)", "PASS (short-time, scoped)",
                "same file, case B_dt0.05"])
    new.append(["R2-04", "error attribution: truncation vs float32", "-",
                "truncation ~85%, float32 ~9.4%, spin sqrt(N) ~5.2%", "INFORMATIVE",
                "R2_ERROR_ATTRIBUTION.md"])
    new.append(["R2-05", "R2 raw evidence untouched by later stages", "203/203 sha256",
                "203/203 match", "PASS", "stage_PV1/r2_closure/R2_INTEGRITY.json"])
    new.append(["R2-06", "P2/P3/P4 double refactor", "-", "frozen by decision",
                "DEFERRED (not started)", "r2_closure/R2_CLOSURE_AND_TIMESTEP_POLICY.md"])

    lt = load(LT)
    if lt:
        by = {r["case"]: r for r in lt}
        def g(case, key, fmt="{:.3e}"):
            v = by.get(case, {}).get(key)
            return fmt.format(v) if isinstance(v, (int, float)) else "n/a"

        new.append(["PV1-01", "Fe 1024 atoms, NVE 250 K, 1 ps, dt sweep 0.2/0.1/0.05/0.025",
                    "1e-5 eV/atom (historical gate, unchanged)",
                    "; ".join(f"{c}: {g(c,'dE_max_per_atom')}" for c in
                              ("dt_0.2", "dt_0.1", "dt_0.05", "dt_0.025")),
                    "PASS at all four dt (1 ps only)", "fe_lt/fe_lt_results.json"])
        new.append(["PV1-02", "same as PV1-01 but sampled at 0.1 ps (scale of the R2 runs)",
                    "1e-5 eV/atom", "not re-measured at 0.1 ps in this stage",
                    "NOT_MEASURED", "fe_lt/fe_lt_results.json"])
        new.append(["PV1-03", "system-size convergence of the drift floor, dt=0.05, 1 ps",
                    "monotone in N", "; ".join(f"{c.replace('size_','')}: {g(c,'dE_max_per_atom')}"
                                               for c in ("size_fe128_ds_fm", "size_fe432_ds_fm",
                                                         "size_fe1024_ds_fm", "size_fe3456_ds_fm")),
                    "checked in FE_LT_REPORT.md", "fe_lt/fe_lt_results.json"])
        new.append(["PV1-04", "duration convergence, dt=0.05, 1/5/10 ps", "sub-linear growth",
                    "; ".join(f"{c}: {g(c,'dE_max_per_atom')}" for c in
                              ("dur_1ps", "dur_5ps", "dur_10ps")),
                    "checked in FE_LT_REPORT.md", "fe_lt/fe_lt_results.json"])
        new.append(["PV1-05", "initial-state spread, 4 independent microstates (the `seed` keyword is a no-op in deterministic NVE: 5 runs, one sha256)", "spread < 3x",
                    "; ".join(f"{c}: {g(c,'dE_max_per_atom')}" for c in
                              ("seed2_1", "seed2_2", "seed2_3", "seed2_4")),
                    "checked in FE_LT_REPORT.md", "fe_lt/fe_lt_results.json"])
        new.append(["PV1-06", "spin-length conservation in the LT runs", "1e-5 mu_B",
                    g("dur_5ps", "S_mag_drift"), "PASS/see report",
                    "fe_lt/fe_lt_results.json"])
        new.append(["PV1-07", "R19 guard blocks NVE above 300 K", "reproducible",
                    "limit prints 300 K at start T = 200 K and 600 K; two independent builds",
                    "DEFECT (release, unfixed)", "r2_closure/R2_CLOSURE_AND_TIMESTEP_POLICY.md §5"])

    h = load(HEAT)
    if h:
        c = h.get("continuity", {})
        if "moment_direct" in c:
            new.append(["PV1-08", "energy-moment continuity d/dt sum r e = J", "rel RMS < 10%, corr > 0.99",
                        f"{c['moment_direct']['rel_rms']:.3f} / corr {c['moment_direct']['corr']:.4f}",
                        "NOT_VALIDATED_BY_DESIGN (criterion ill-posed in PBC, see audit 4)",
                        "heat_audit/HEAT_CURRENT_AUDIT.md"])
            new.append(["PV1-09", "same with the convective sum v e removed", "rel RMS < 10%",
                        f"{c['moment_conduction_only']['rel_rms']:.3f} / corr "
                        f"{c['moment_conduction_only']['corr']:.4f}",
                        "NOT_VALIDATED_BY_DESIGN",
                        "heat_audit/HEAT_CURRENT_AUDIT.md"])
            new.append(["PV1-13", "no-spin control of the continuity test", "should close",
                        "relRMS 18 (also fails) -> the failure is in the criterion, not the spin channel",
                        "CONTROL", "heat_audit/cases/cont_nospin_N128"])
        s = h.get("translation", {})
        if "rel_rms_deviation" in s:
            new.append(["PV1-10", "origin independence of J", "rel dev < 1e-6",
                        f"{s['rel_rms_deviation']:.3e} over {s['compared_steps']} steps",
                        "PASS", "heat_audit/heat_audit_results.json"])
        gk = h.get("green_kubo", {})
        if "kappa_W_per_mK" in gk:
            new.append(["PV1-11", "Green-Kubo plateau and block error", "rel error < 30%",
                        f"kappa {gk['kappa_W_per_mK']:.4g} +- {gk['kappa_block_error']:.2g} W/mK, "
                        f"rel {gk['relative_error']:.2f}, tau {gk['tau_zero_crossing_ps']:.3f} ps, "
                        f"N_indep {gk['independent_samples']:.1f}",
                        "FAIL (no plateau; 14 independent samples in 5 ps)",
                        "heat_audit/heat_audit_results.json"])
            new.append(["PV1-14", "absolute normalisation of the heat current",
                        "three paths must agree",
                        "6e5 (independent SI) vs -6e-2 (compute_hac) vs ~80 W/mK (physical)",
                        "NOT_VALIDATED (spread 1e7)", "heat_audit/HAC_CROSSCHECK.json"])

    new.append(["PV1-12", "thermal conductivity of Fe", "-", "not reported", "BLOCKED (S09)",
                "HEAT_CURRENT_AUDIT.md §6"])

    out = os.path.join(STAGE, "VALIDATION_MATRIX_v3.1.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(header)
        for r in body:
            w.writerow(r)
        for r in new:
            w.writerow(r)
    print(f"{len(body)} historical rows + {len(new)} new rows -> {out}")


if __name__ == "__main__":
    main()
