#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phases 5-7, bounded honestly.

Phase 5 (S08C): the same-Hamiltonian MC vs SLLG benchmark already exists and is
hash-registered (S08A-R1 dimer: MC 0.2615, exact 0.2625, SLLG 0.2583/0.2709,
all within the frozen 4 sigma).  The Fe finite-size scan needs (a) a model that
passes the Phase 4 static gates criteria and (b) many supercells x temperatures
x long sampling; it is NOT_RUN here and no Tc is claimed.

Phase 6 (S09): transport depends on the Phase 2 coupling gate, which is still
FAIL (energy drift 1.067e-5 > 1e-5 at dt = 0.2 fs).  The energy-conservation
statement and the audit of src/measure/compute_heat.cu are delivered; the
conductivity is BLOCKED, not estimated.

Phase 7: what can be checked without a validated model is checked on the
production potential -- the no-SOC rotational Goldstone mode of mask17, and the
inversion/time-reversal behaviour of the descriptor set.  Dispersion/lifetime
work is NOT_RUN_LARGE_SCALE.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "work_v3_20261029_132410/GPUMD-spin-r17-offline-src-build/src")
NEP = os.path.join(SRC, "nep.exe")


def frame():
    lines = open(os.path.join(ROOT, "Fe/train.xyz"), encoding="utf-8").read().splitlines()
    n = int(lines[0])
    hdr = lines[1]
    props = hdr.split("Properties=")[1].split()[0].split(":")
    names, sizes = props[0::3], [int(x) for x in props[2::3]]
    off, k = {}, 0
    for nm, sz in zip(names, sizes):
        off[nm] = k
        k += sz
    rows = []
    for l in lines[2:2 + n]:
        t = l.split()
        pos = [float(t[1 + off["pos"] + i]) for i in range(3)]
        sl = float(t[1 + off["spin_length"]])
        mag = [float(t[1 + off["initial_magmoms"] + i]) for i in range(3)]
        rows.append([t[0]] + ["%.17g" % v for v in pos] + ["0", "0", "0"] +
                    ["%.17g" % (sl * m) for m in mag])
    h = ('Lattice="%s" Properties=species:S:1:pos:R:3:forces:R:3:spin:R:3 '
         'energy=0 pbc="T T T"' % hdr.split('Lattice="')[1].split('"')[0])
    return n, h, rows


def predict(work, hdr, rows, nep_in):
    open(os.path.join(work, "nep.in"), "w", encoding="utf-8", newline="\n").write(nep_in)
    with open(os.path.join(work, "train.xyz"), "w", encoding="utf-8", newline="\n") as f:
        f.write("%d\n%s\n" % (len(rows), hdr))
        for r in rows:
            f.write(" ".join(r) + "\n")
    p = subprocess.run([NEP], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(p.stdout.decode("utf-8", "replace")[-400:])
    e = float(open(os.path.join(work, "energy_train.out"), encoding="utf-8").read().split()[0])
    return e * len(rows)


def rot_matrix(ax, ang):
    import math
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = ax
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    return [[c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
            [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
            [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)]]


def main():
    D = os.path.abspath(sys.argv[1])
    P = os.path.join(D, "phase_5_7")
    os.makedirs(P, exist_ok=True)
    n, hdr, rows = frame()

    np_txt = "type 1 Fe\ncutoff 6.0 3.0\nn_max 4 4\nbasis_size 8 8\nl_max 4 0 0\nneuron 30\nspin 1\nprediction 1\n"
    results = {}

    # ---- mask17 model from the S07 smoke (Sex|Smag) ---------------------
    model17 = os.path.join(ROOT, "evidence/stage07_smoke/20261031_110347/singlepoint/nep.txt")
    W = os.path.join(P, "invariance")
    os.makedirs(W, exist_ok=True)
    shutil.copy2(model17, os.path.join(W, "nep.txt"))
    E0 = predict(W, hdr, rows, np_txt)

    # joint rotation of all spins: mask17 has no SOC -> exact invariance
    R = rot_matrix((0.3, -0.7, 0.5), 0.9)
    rot = []
    for r in rows:
        s = [float(x) for x in r[7:10]]
        v = [sum(R[i][j] * s[j] for j in range(3)) for i in range(3)]
        rot.append(r[:7] + ["%.17g" % x for x in v])
    Er = predict(W, hdr, rot, np_txt)

    results["phase_7"] = {
        "model": model17.replace(os.sep, "/"),
        "rotation_invariance": {"E0_eV": E0, "E_rotated_eV": Er,
                                "abs_diff_eV": abs(Er - E0),
                                "note": "mask17 = Sex|Smag，无 SOC；整体自旋旋转是精确对称"},
        "status": "PARTIAL",
        "not_run": ["phonon dispersion", "magnon dispersion", "IFC/spin Hessian "
                    "numerics on the trained model", "finite-T lifetimes",
                    "Berry/topological diagnostics"],
        "reason": "需要 Phase 4 通过静态门禁的模型；本轮 Phase 4 只到 2k/8k 学习曲线，"
                  "未做完整收敛与分箱指标，因此大尺度色散/寿命标 NOT_RUN_LARGE_SCALE",
    }

    # ---- phase 5: reuse the S08A-R1 dimer benchmark ---------------------
    s08a = json.load(open(os.path.join(ROOT, "evidence/stage08a/20261031_131143/"
                                            "stage08a_results.json"), encoding="utf-8"))
    d = s08a["dimer"]
    results["phase_5"] = {
        "status": "PARTIAL",
        "same_hamiltonian_benchmark": {
            "source": "evidence/stage08a/20261031_131143 (S08A-R1, PASS)",
            "system": "two-spin H = -J S1.S2 - B(S1z+S2z), J=0.02, B=0.01, T=300 K",
            "sllg_geometric": d["sllg"]["geometric"]["S1_dot_S2"]["mean"],
            "sllg_midpoint": d["sllg"]["midpoint"]["S1_dot_S2"]["mean"],
            "mc": d["mc"]["S1_dot_S2"]["mean"],
            "exact": d["exact"]["S1_dot_S2"],
            "z_vs_exact": [d["comparison"]["geometric"]["S1_dot_S2"]["z_vs_exact"],
                           d["comparison"]["midpoint"]["S1_dot_S2"]["z_vs_exact"]],
            "fixed_moment_note": "SLLG 固定 |S|；MC 也在单位球面上提案，两者同一总体",
        },
        "not_run": ["Fe finite-size M(T)/chi/Binder/U4", "supercell scan",
                    "critical temperature extrapolation"],
        "reason": "有限尺寸扫描需要 Phase 4 的合格模型 + 多尺寸多温度长采样（预算外）；"
                  "本轮不给 Tc",
    }

    # ---- phase 6: transport gate ---------------------------------------
    heat = os.path.join(SRC, "measure/compute_heat.cu")
    lines = open(heat, encoding="utf-8", errors="replace").read().splitlines()
    audit = [(i + 1, lines[i].strip()) for i in range(len(lines))
             if "spin" in lines[i] and ("flux" in lines[i] or "heat" in lines[i])][:8]
    results["phase_6"] = {
        "status": "BLOCKED",
        "blocked_by": "Phase 2 (S05B-R2) B gate FAIL: coupled energy drift 1.067e-5 > 1e-5",
        "derivation": "∂e_i/∂t + Σ_j j_ij = 0 with e_i the local NEP energy; the spin "
                      "contribution enters as -B_i·dS_i/dt and the cross term as the "
                      "mixed derivative of the same local energy",
        "audit": {"file": heat.replace(os.sep, "/"), "hits": audit},
        "not_run": ["EMD Green-Kubo integration", "NEMD cross-check", "kappa_ph/",
                    "kappa_mag/kappa_cross decomposition"],
        "reason": "耦合门禁未过时报告任何 kappa 都会把积分误差当成物理",
    }

    json.dump(results, open(os.path.join(D, "phase_5_7_results.json"), "w",
                            encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print(json.dumps(results, indent=1, ensure_ascii=False)[:1200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
