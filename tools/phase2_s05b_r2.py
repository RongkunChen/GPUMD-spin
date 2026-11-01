#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 / S05B-R2: audit the conservative spin-lattice coupling.

Two things this script actually measures (the rest is quoted from the delivered
S05B-R1 evidence, which is hash-registered and not re-run):

  1. the production Strang split is read back from run.cu by text search with
     line numbers, so the audit refers to the code that is in the tree;
  2. the mixed-derivative identity of the potential the coupling relies on:
        F = -dE/dR,  g = -dE/dS,  dF_j/dS_i = dg_i/dR_j
     is checked with finite differences on the production NEP via nep.exe
     predictions (same model, same frame, one coordinate perturbed at a time).

The B gate (energy drift of the coupled integration) is NOT re-run here: it needs
the S05B harness, and its frozen numbers are quoted from S05B-R1.  This phase
therefore stays FAIL as a gate even if the derivative check passes.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "work_v3_20261029_132410/GPUMD-spin-r17-offline-src-build/src")
NEP = os.path.join(SRC, "nep.exe")
MODEL = os.path.join(ROOT, "evidence/stage07_smoke/20261031_110347/singlepoint/model.xyz")
NEP_TXT = os.path.join(ROOT, "evidence/stage07_smoke/20261031_110347/singlepoint/nep.txt")
NEP_IN = os.path.join(ROOT, "evidence/stage07_smoke/20261031_110347/singlepoint/run.in")

FROZEN = {"B_max_abs_dE_per_atom_1e-5": 1.05e-5,
          "measured_dt02": 1.06731513e-5, "measured_dt01": 4.33211882e-6,
          "measured_dt005": 1.67583382e-6}


def read_frame():
    """A frame with the columns the trainer's reader insists on in prediction:
    species, pos, forces (reference only), spin, and energy= in the header."""
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
    hdr = ('Lattice="%s" Properties=species:S:1:pos:R:3:forces:R:3:spin:R:3 '
           'energy=0 pbc="T T T"'
           % hdr.split('Lattice="')[1].split('"')[0])
    return n, hdr, rows


def write_xyz(path, hdr, rows):
    # the trainer's frame reader insists on an energy= attribute; the value is
    # only used as the reference in prediction mode, so a placeholder is fine
    if "energy=" not in hdr:
        hdr = "energy=0 " + hdr
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("%d\n%s\n" % (len(rows), hdr))
        for r in rows:
            f.write(" ".join(r) + "\n")


def predict(work, hdr, rows):
    """Run nep.exe in prediction mode; return (E_total, F[n][3], g[n][3])."""
    write_xyz(os.path.join(work, "train.xyz"), hdr, rows)
    p = subprocess.run([NEP], cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        raise RuntimeError(p.stdout.decode("utf-8", "replace")[-500:])
    e = open(os.path.join(work, "energy_train.out"), encoding="utf-8").read().split()
    n = len(rows)
    etot = float(e[0]) * n
    f = [float(x) for x in open(os.path.join(work, "force_train.out"),
                                encoding="utf-8").read().split()]
    g = [float(x) for x in open(os.path.join(work, "spin_train.out"),
                                encoding="utf-8").read().split()]
    # columns are (pred_x,pred_y,pred_z, ref_x,ref_y,ref_z): the model output
    # is the first triple; the reference triple is only for scoring.
    F = [[f[6 * i + c] for c in range(3)] for i in range(n)]
    G = [[g[6 * i + c] for c in range(3)] for i in range(n)]
    return etot, F, G


def main():
    D = os.path.abspath(sys.argv[1])
    P = os.path.join(D, "phase_2")
    W = os.path.join(P, "maxwell")
    os.makedirs(W, exist_ok=True)
    shutil.copy2(NEP_TXT, os.path.join(W, "nep.txt"))
    txt = open(os.path.join(NEP_TXT), encoding="utf-8").read().splitlines()
    # a minimal nep.in for prediction: reuse the model header + prediction flag
    open(os.path.join(W, "nep.in"), "w", encoding="utf-8", newline="\n").write(
        "type 1 Fe\ncutoff 6.0 3.0\nn_max 4 4\nbasis_size 8 8\nl_max 4 0 0\n"
        "neuron 30\nspin 1\nprediction 1\n")

    n, hdr, rows = read_frame()
    # rows were rebuilt as [species, pos(3), forces(3), spin(3)]:
    # value index 0..2 = pos, 3..5 = forces, 6..8 = spin.
    POS, SPIN = 1, 7

    h = 1.0e-3
    E0, F0, G0 = predict(W, hdr, rows)

    # dE/dR and dE/dS by central differences (atom 0, component 0)
    def perturb(kind, atom, comp, delta):
        new = [list(r) for r in rows]
        base = POS if kind == "pos" else SPIN
        v = float(new[atom][base + comp]) + delta
        new[atom][base + comp] = "%.17g" % v
        return [["%s" % x for x in r] for r in new]

    Ep, Fp, Gp = predict(W, hdr, perturb("pos", 0, 0, h))
    Em, Fm, Gm = predict(W, hdr, perturb("pos", 0, 0, -h))
    Es, Fs, Gs = predict(W, hdr, perturb("spin", 0, 0, h))
    Esm, Fsm, Gsm = predict(W, hdr, perturb("spin", 0, 0, -h))
    dE_dR = (Ep - Em) / (2 * h)
    dE_dS = (Es - Esm) / (2 * h)
    # Maxwell: F = -dE/dR, g = -dE/dS  =>  dF/dS = dg/dR.
    dF_dS = (Fs[0][0] - Fsm[0][0]) / (2 * h)      # F at S +/- h
    dG_dR = (Gp[0][0] - Gm[0][0]) / (2 * h)       # g at R +/- h

    # dF/dS and dg/dR are the same mixed derivative up to the sign convention
    # F = -dE/dR, g = -dE/dS  =>  dF/dS = -d2E/dRdS = dg/dR.
    scale = max(abs(dF_dS), abs(dG_dR), 1e-12)
    sym_err = abs(dF_dS - dG_dR) / scale

    # ---- audit: read the Strang split back from run.cu -------------------
    audit = {}
    rp = os.path.join(SRC, "main_gpumd/run.cu")
    lines = open(rp, encoding="utf-8", errors="replace").read().splitlines()
    for tag, needle in (("strang_header", "S05B: symmetric (Strang) spin-lattice splitting"),
                        ("strang_step_A", "A: first spin half step"),
                        ("strang_state_G", "the isochronous output state"),
                        ("strang_force_recompute", "compute_force();")):
        hits = [(i + 1, lines[i].strip()) for i in range(len(lines))
                if needle in lines[i]]
        audit[tag] = hits[:6]
    out = {
        "phase": "2",
        "derivative_check": {
            "frame": MODEL.replace(os.sep, "/"),
            "n_atoms": n, "step": h,
            "E0_eV": E0,
            "dE_dR_atom0_x": dE_dR, "minus_F0_atom0_x": -F0[0][0],
            "dE_dS_atom0_x": dE_dS, "minus_g0_atom0_x": -G0[0][0],
            "dF_dS": dF_dS, "dg_dR": dG_dR,
            "symmetry_rel_error": sym_err,
            "forces_vs_minus_gradR_rel": abs(dE_dR + F0[0][0]) / max(abs(F0[0][0]), 1e-12),
            "torque_vs_minus_gradS_rel": abs(dE_dS + G0[0][0]) / max(abs(G0[0][0]), 1e-12),
            "pass": bool(sym_err < 0.05),
        },
        "code_audit": audit,
        "b_gate": {"status": "FAIL (not re-run)",
                   "frozen": FROZEN,
                   "note": "S05B-R1 的 B 组数字：dt=0.2 fs 时 max|dE|/atom 1.067e-5 > 1e-5；"
                           "本轮未重跑该 harness，门禁结论不变"},
        "verdict": "PARTIAL: derivative consistency measured; B gate NOT re-run and still FAIL",
    }
    json.dump(out, open(os.path.join(D, "phase_2_results.json"), "w", encoding="utf-8",
                        newline="\n"), indent=1, ensure_ascii=False)

    L = ["=" * 78, "阶段2：S05B-R2 保守自旋—晶格耦合审计", "=" * 78, ""]
    L.append("1. 生产 Strang 分裂（从 run.cu 回读，行号即源码行号）")
    for tag, hits in audit.items():
        for ln, txt in hits[:2]:
            L.append("   %-26s L%-5d %s" % (tag, ln, txt[:70]))
    L.append("")
    L.append("2. 导数一致性（生产 NEP，16 Fe 原子，中心差分 h=%g）" % h)
    L.append("   F = -dE/dR：dE/dR %+.9f vs -F_x %+.9f（相对 %.2e）"
             % (dE_dR, -F0[0][0], out["derivative_check"]["forces_vs_minus_gradR_rel"]))
    L.append("   g = -dE/dS：dE/dS %+.9f vs -g_x %+.9f（相对 %.2e）"
             % (dE_dS, -G0[0][0], out["derivative_check"]["torque_vs_minus_gradS_rel"]))
    L.append("   混合导数 dF_x/dS_x %+.6f vs dg_x/dR_x %+.6f（相对差 %.2e）"
             % (dF_dS, dG_dR, sym_err))
    L.append("")
    L.append("3. B 门禁（能量漂移）：未重跑，沿用 S05B-R1，仍 FAIL")
    L.append("   dt=0.2 fs：1.067e-5 > 1e-5；dt=0.1：4.332e-6；dt=0.05：1.676e-6")
    L.append("")
    L.append("4. 结论：%s" % out["verdict"])
    open(os.path.join(D, "phase_2_summary.txt"), "w", encoding="utf-8",
         newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
