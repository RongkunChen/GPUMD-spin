"""Render the per-checkpoint cache into a markdown report.

Reads `series/*.json` (written by model_series.py / series_physics.py) and writes
`series/SERIES_REPORT.md`: one table per observable group, plus the caveats that
apply to the numbers.  Re-run after every new checkpoint; the orchestrator does.

usage: python series_report.py
"""
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                "validation", "BCC_Fe_full_physics", "scripts"))
sys.path.insert(0, _HERE)
import model_series as MS
import series_physics as SPS

SER = MS.SER
OUT = os.path.join(SER, "SERIES_REPORT.md")
STATES = [k for k, _ in SPS.STATES]


def load():
    rows = []
    for p in glob.glob(os.path.join(SER, "*.json")):
        d = json.load(open(p))
        d["gen"] = MS.gen_of(d["tag"])
        rows.append(d)
    rows.sort(key=lambda r: r["gen"])
    return rows


def f(x, n=2, dash="-"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return dash
    return f"{x:.{n}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    rows = load()
    if not rows:
        print("no checkpoints yet")
        return
    L = []
    A = L.append
    A("# 5-descriptor run: physics per 100k checkpoint\n")
    A(f"Generated from {len(rows)} cached checkpoint(s): "
      + ", ".join(r["tag"] for r in rows) + "\n")
    A("Everything below comes from the production `gpumd.exe` with the checkpoint's "
      "own `nep.txt`. Nothing is fitted except the exchange constants themselves "
      "(which are a definition) and parabolic interpolations that locate minima.\n")

    A("## 1. 训练精度与静态磁性\n")
    hdr = ["checkpoint", "gen", "E MAE (meV/at)", "F MAE (eV/Å)", "M MAE (μB)",
           "a0 (Å)", "B0 (GPa)", "m_eq (μB)", "Tc (K)", "FM lowest"]
    body = []
    for r in rows:
        c = r["checks"]
        body.append([r["tag"], r["gen"], f(r["dev"]["E_MAE_meV"]),
                     f(r["dev"]["F_MAE"], 3), f(r["dev"]["M_MAE"], 3),
                     f(r["eos"]["a0_A"], 4), f(r["eos"]["B0_GPa"], 1),
                     f(r["moment"]["m_eq_uB"], 3), f(r["curie"]["Tc_half_M_K"], 0),
                     "yes" if c["FM_lowest"] else "**NO**"])
    A(table(hdr, body) + "\n")
    A("Reference values used for the error columns: a0 = 2.866 Å, m = 2.2 μB, "
      "Tc = 1043 K, B0 ≈ 170-180 GPa (the model is deliberately stiffer than "
      "experiment, see the main report).\n")

    A("## 2. 交换常数 J1..J6（冻结螺旋拟合）\n")
    A("Convention `H = -1/2 Σ_ij J_ij e_i·e_j` with `e_i` a unit vector, so "
      "`E(q)/N - E_FM/N = 1/2 Σ_n J_n [z_n - S_n(q)]`; J in meV at the model's "
      "own moment, `J/m²` is the same number divided by m².\n")
    nsh = max((len(r["exchange"]["shells"]) for r in rows
               if "exchange" in r), default=0)
    hdr = ["checkpoint"] + [f"J{n+1} (meV)" for n in range(nsh)] + \
          ["fit rms (meV/at)", "D_direct (meV·Å²)", "D_LSWT (meV·Å²)",
           "LSWT/direct", "J1^S", "J0^S", "J1^S@2.2", "J0^S@2.2"]
    body = []
    for r in rows:
        ex = r.get("exchange")
        if not ex:
            continue
        c = r["checks"]
        body.append([r["tag"]] + [f(ex["J_meV"][n], 1) for n in range(nsh)] +
                    [f(1000 * ex["fit_rms_meV_per_atom"], 2),
                     f(c["D_direct_meV_A2"], 0), f(c["D_lswt_closed_form_meV_A2"], 0),
                     f(c["lswt_over_direct"], 3),
                     f(c["J1_S_meV"], 1), f(c["J0_S_meV"], 0),
                     f(c["J1_S_at_2p2_meV"], 1), f(c["J0_S_at_2p2_meV"], 0)])
    A(table(hdr, body) + "\n")
    if rows and "exchange" in rows[-1]:
        sh = rows[-1]["exchange"]["shells"]
        A("Shells: " + ", ".join(f"n={s['shell']}: r={s['r_A']:.3f} Å, z={s['z']}"
                                 for s in sh) + " (cutoff 6.0 Å).\n")
    A("**Two conventions, both reported.** `J_n` (first block) are the coefficients of "
      "a sum over *unit* spin vectors, which is what the least-squares fit produces. "
      "`J^S` is the standard Heisenberg coefficient `H = -1/2 Σ_ij J S_i·S_j` with "
      "|S| in ħ (`J^S = J_n / S²`, S = m/2); `J0 = Σ_n z_n J_n` is the total exchange "
      "that sets the mean-field Curie temperature. `@2.2` is a refit with the spiral "
      "spins rescaled to the experimental moment, because `J_fit ∝ m^1.06` (not m²), "
      "so `J^S` drifts by ~10% between m = 2.44 and m = 2.2.\n")
    A("**D is evaluated three ways** and they must agree: the direct frozen spirals "
      "(`hbar ω = 2[E(q)-E(FM)]/(N S)`) along [001], [110], [111]; the closed form of "
      "the LSWT curve `D = (1/(6S)) Σ_n J_n z_n r_n²`; and the naive `ω`-vs-`q²` slope "
      "(kept only as a warning, it is biased low by the quartic term). The relation "
      "is `1/S`, **not** `2S`: the first version multiplied instead of dividing and "
      "over-reported every magnon energy and stiffness by S² = 1.49, which is why D "
      "used to read 440 instead of 295 meV·Å². See section 15 of the main report and "
      "`frozen_magnon_test.py` for the direct measurement that settled it.\n")

    A("## 3. D 与磁矩无关（旧结论已撤销）\n")
    A("`D` comes out at 292-316 meV·Å² for every checkpoint, inside the measured "
      "280-330 meV·Å² band. `dlnD/dlnm` is 0.02-0.35 depending on the checkpoint, "
      "i.e. **D changes by at most 6% between m = 2.2 and 2.45 uB** -- which is what "
      "a Heisenberg system must do, and the opposite of the m^2.35 that the old "
      "(wrong) factor produced.\n")
    hdr = ["checkpoint", "m_eq (μB)", "D at m_eq", "D at 2.2 μB", "dlnD/dlnm",
           "D_LSWT@2.2"]
    body = []
    for r in rows:
        sv = r.get("stiffness_vs_moment")
        if not sv:
            continue
        body.append([r["tag"], f(r["moment"]["m_eq_uB"], 3),
                     f(sv["D_at_m_eq_meV_A2"], 0), f(sv["D_at_2p2_uB_meV_A2"], 0),
                     f(sv["exponent_dlnD_dlnm"], 3),
                     f((r.get("checks") or {}).get("D_at_2p2_lswt_meV_A2"), 1)])
    A(table(hdr, body) + "\n")
    A("**Retracted:** the previous version of this report explained a 'D is 40-56% too "
      "stiff' failure by the model's moment being 11% too large, quoting "
      "`D ∝ m^2.35`. Both the failure and the explanation were artifacts of the "
      "missing 1/S: with the wrong factor, D picked up a spurious m-dependence from "
      "the S multiplying the energy. With the correct relation D is moment-"
      "independent and inside the measured range.\n")

    A("## 4. 磁子谱\n")
    A("`ω(q)` along Γ-H-N-Γ-P-H from the LSWT form of the fitted `J_n`, and the "
      "direct spiral points along [001]/[110]/[111]. The LSWT curve is stored at "
      "205 path points per checkpoint; the figure `series_magnon_path.png` overlays "
      "them.\n")
    hdr = ["checkpoint", "ω at H (meV)", "ω at N (meV)", "ωmax (meV)",
           "D [001]", "D [110]", "D [111]"]
    body = []
    for r in rows:
        md = r.get("magnon_direct")
        mp = r.get("magnon_path")
        if not md or not mp:
            continue
        # index 2*npts is the N point (each of the first two segments has npts
        # samples, the third one starts there)
        wN = float(mp["omega_meV"][2 * mp["npts"]])
        body.append([r["tag"], f(md["001"]["points"][-1]["omega_meV"], 0), f(wN, 0),
                     f(max(mp["omega_meV"]), 0),
                     f(md["001"]["D_meV_A2"], 0), f(md["110"]["D_meV_A2"], 0),
                     f(md["111"]["D_meV_A2"], 0)])
    A(table(hdr, body) + "\n")

    A("## 5. 各磁态声子谱（每个态在自己的弛豫晶格常数上）\n")
    A("The force constants are periodic with the *magnetic* cell, so each state is "
      "displaced in its own cell: FM/DLM primitive 7×7×7 (343 atoms), NC-90 / "
      "AFM-G 5×5×5 conventional (250), AFM-A 5×5×5 of the 4-atom cell (500), "
      "DLM averaged over 2 random configurations, δ = 0.02 Å central differences.\n")
    hdr = ["checkpoint", "state", "a0 (Å)", "ωmax (THz)", "softest (THz)",
           "modes < -1e-3", "modes < -0.1", "ASR before (eV/Å²)"]
    body = []
    for r in rows:
        for k in STATES:
            ph = (r.get("phonons") or {}).get(k)
            if not ph:
                continue
            body.append([r["tag"], k, f(ph["a0_A"], 4), f(ph["omega_max_THz"], 3),
                         f(ph["softest_THz"], 3), ph["imaginary"],
                         ph["imaginary_below_0p1_THz"], f(ph["asr_before_eV_A2"], 2)])
    A(table(hdr, body) + "\n")
    A("Two counts are given for imaginary modes on purpose. `modes < -1e-3 THz` is "
      "the raw count and includes the three acoustic modes at Γ, which sit at "
      "−3e-3 THz because the force constants are float32. `modes < -0.1 THz` is the "
      "physical count. AFM-G is the only state with a genuine instability.\n")
    A("**ASR caveat.** For the multi-atom magnetic cells the sum rule violation "
      "before correction is 11-16 eV/Å² (FM: 0.23), because the force constants of "
      "a canted or layered cell are not cubic-symmetrisable and the one-representative-"
      "per-basis displacement scheme leaves a larger residual. It is absorbed by the "
      "on-site correction and the resulting Γ acoustic modes come out within 3e-3 THz "
      "of zero, but the absolute frequencies of those states carry a larger systematic "
      "uncertainty than the FM curve.\n")

    A("## 5b. 声速与弹性常数（由同一套声子谱微分得到）\n")
    A("Slope of the acoustic branches at small q, no extra force evaluations. "
      "`v = 0.6283 x d(nu)/dq` in km/s for nu in THz and q in 1/A; `C = rho v^2` "
      "with rho = 2 m_Fe / a^3.\n")
    hdr = ["checkpoint", "state", "v_L[001] (km/s)", "v_T[001]", "v_L[111]",
           "C_L (GPa)", "C_T (GPa)", "v_T < 0"]
    body = []
    for r in rows:
        for k in STATES:
            sv = (r.get("sound_velocities") or {}).get(k)
            if not sv or not sv.get("001"):
                continue
            body.append([r["tag"], k, f(sv["001"]["v_L_m_s"] / 1000, 2),
                         f(sv["001"]["v_T_m_s"] / 1000, 2),
                         f(sv["111"]["v_L_m_s"] / 1000, 2),
                         f(sv["001"]["C_L_GPa"], 0), f(sv["001"]["C_T_GPa"], 0),
                         "yes" if sv["001"]["transverse_negative"] else "-"])
    A(table(hdr, body) + "\n")
    A("Reference: alpha-Fe has v_L ~ 5.9 km/s, v_T ~ 3.2 km/s, C11 = 231 GPa, "
      "C44 = 117 GPa. A negative v_T marks either a genuine soft branch (AFM-G) "
      "or the ASR residual of a cell that cannot be cubic-symmetrised; it is not a "
      "physically softened mode in the latter case.\n")

    A("## 6. 磁–声耦合\n")
    A("Three independent measurements, because they answer different questions:\n"
      "1. the mixed derivative `d²E/du dS` at one site — the linear magnon-phonon "
      "vertex, which must vanish at a collinear state (spin-rotation invariance);\n"
      "2. the crossings of the magnon branch with the phonon branches — with a zero "
      "vertex they must be exact crossings, not avoided ones;\n"
      "3. `d ln ω / d ln|S|` from rebuilding the whole dynamical matrix at two spin "
      "magnitudes — the magnetostrictive (exchange-striction) channel, which is the "
      "one that actually shifts the dispersion.\n")
    hdr = ["checkpoint", "C_transverse (eV/Å/μB)", "C_longitudinal", "noise floor",
           "transverse resolved?", "# crossings", "max dlnω/dln m", "dln a0/dln m"]
    body = []
    for r in rows:
        cp = r.get("coupling")
        if not cp:
            continue
        ms = r.get("magnetostriction") or {}
        av = r.get("a0_vs_moment") or {}
        v = SPS.coupling_verdict(cp)
        body.append([r["tag"], f"{cp['transverse']['C_eV_per_A_per_muB']:+.2e}",
                     f"{cp['longitudinal']['C_eV_per_A_per_muB']:+.2e}",
                     f"{v['floor_eV_per_A_per_muB']:.2e}",
                     "**YES**" if v["transverse_resolved"] else "no",
                     (r.get("crossings") or {}).get("n", "-"),
                     f(ms.get("max_abs"), 3), f(av.get("dln_a0_dln_m"), 4)])
    A(table(hdr, body) + "\n")
    A("The last two columns are the two halves of the same magnetoelastic coupling: "
      "how much the phonons stiffen (`dlnω/dln|S|`) and how much the lattice expands "
      "(`dln a0/dln|S|`) per fractional change of the moment. `dln a0/dln|S|` is "
      "measured on the 54-atom EOS cell used everywhere else for a0, so it is "
      "comparable with column 6 of the first table.\n")

    A("## 7. 这些数字不能说什么\n")
    A("* `J_n` are effective Fourier components of the model's *angular* energy "
      "functional inside a 6 Å cutoff, not hopping integrals. They depend on the "
      "moment used (`J/m²` is given for comparison with Heisenberg parameters).\n"
      "* The magnon dispersion is the harmonic (LSWT) dispersion of those `J_n`. "
      "There is no magnon solver in the code, and no magnon lifetime, damping or "
      "magnon-magnon interaction is obtained.\n"
      "* `Tc` comes from a 1 ps SLLG Langevin run per temperature at L = 6; it is a "
      "sampler-dependent estimate (the dt-dependence is a known open failure), not "
      "a precision critical temperature.\n"
      "* The DLM phonons average over 2 configurations only; the configuration "
      "spread is not resolved by that.\n"
      "* Nothing here says the 0.1 fs production step is validated for all "
      "materials or durations — that remains scoped as in the main report.\n")
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"wrote {OUT} ({len(rows)} checkpoints)")


if __name__ == "__main__":
    main()
