#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task I (G6) - current-build, per-atom where the binaries allow it.

What can and cannot be measured with the shipped executables:

  * the MD path exposes the per-atom local energy e_i (`dump_xyz ... potential`
    writes `energy_atom`), the force and the torque;
  * the trainer path (`nep.exe prediction 1`) exposes the frame total energy,
    the forces and the spin gradient, but **not** e_i.  There is no shipped
    option that prints the trainer's per-atom decomposition, and the historical
    Stage03A probes that did are separate binaries built in that stage.

So this script measures, for the 5 historical cases:
  1. sum_i e_i^MD vs E^MD          (internal consistency of the local model)
  2. E^MD vs E^trainer             (total-energy parity, current build)
  3. max_i |e_i^MD - e_i^trainer|  -> NOT MEASURABLE, reported as such

The G6 gate is a per-atom statement, so it stays FAIL until the per-atom
trainer value is available again; nothing here turns it into PASS.
"""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "work_v3_20261029_132410/GPUMD-spin-r17-offline-src-build/src")
GPUMD = os.path.join(SRC, "gpumd.exe")
NEP = os.path.join(SRC, "nep.exe")
CASES = os.path.join(ROOT, "evidence/stage03a_2/20261030_103721/numerical_cases.json")
TOL = 1.05e-6
WANT = ["fe24_mask31_f03_a4", "fe24_mask4_f03_a4", "fe24_mask16_f11_a0",
        "fe24_mask31_f02_a4", "fe24_mask16_f10_a0"]


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frame(path):
    lines = open(path, encoding="utf-8").read().splitlines()
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
        # off is computed over ALL columns including species, so the row index
        # is off[...] + c (species is column 0).
        pos = [float(t[off["pos"] + i]) for i in range(3)]
        sl = float(t[off["spin_length"]])
        mag = [float(t[off["initial_magmoms"] + i]) for i in range(3)]
        rows.append([t[0]] + ["%.17g" % v for v in pos] + ["0", "0", "0"] +
                    ["%.17g" % (sl * m) for m in mag])
    return n, hdr, rows


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def parse_traj(path):
    """last frame: per-atom energy, |F|, |g| (columns by Properties)."""
    lines = open(path, encoding="utf-8").read().splitlines()
    i = 0
    frames = []
    while i < len(lines):
        if not lines[i].strip():
            break
        n = int(lines[i])
        hdr = lines[i + 1]
        props = hdr.split("Properties=")[1].split()[0].split(":")
        names, sizes = props[0::3], [int(x) for x in props[2::3]]
        off, k = {}, 0
        for nm, sz in zip(names, sizes):
            off[nm] = k
            k += sz
        rows = [l.split() for l in lines[i + 2:i + 2 + n]]
        frames.append((hdr, off, rows))
        i += 2 + n
    return frames[-1]


def main():
    D = os.path.abspath(sys.argv[1])
    P = os.path.join(D, "phase_1", "g6_per_atom")
    os.makedirs(P, exist_ok=True)
    cases = {c["case_id"]: c for c in json.load(open(CASES, encoding="utf-8"))["cases"]}
    out = {"task": "I / G6", "frozen_tol_eV_per_atom": TOL,
           "binaries": {"gpumd": sha(GPUMD), "nep": sha(NEP)},
           "cases": []}
    for cid in WANT:
        c = cases[cid]
        w = os.path.join(P, cid)
        if os.path.isdir(w):
            shutil.rmtree(w)
        os.makedirs(w)
        shutil.copy2(c["input_files"]["nep.txt"]["path"], os.path.join(w, "nep.txt"))
        shutil.copy2(c["input_files"]["train.xyz"]["path"], os.path.join(w, "train.xyz"))
        n, hdr, rows = frame(c["input_files"]["train.xyz"]["path"])
        with open(os.path.join(w, "model.xyz"), "w", encoding="utf-8", newline="\n") as f:
            lat = hdr.split('Lattice="')[1].split('"')[0]
            f.write("%d\nLattice=\"%s\" Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3 "
                    "pbc=\"T T T\"\n" % (n, lat))
            for r in rows:
                f.write(" ".join(r) + "\n")
        # MD
        open(os.path.join(w, "run.in"), "w", encoding="utf-8", newline="\n").write(
            "potential nep.txt\ntime_step 0.0001\nensemble nve\ndump_thermo 1\n"
            "dump_xyz 1 t.xyz potential force spin\ndump_restart 1000\nrun 1\n")
        rc1, log1 = run([GPUMD], w)
        pe = None
        tr = [l.split() for l in open(os.path.join(w, "thermo.out"), encoding="utf-8")
              if l.strip() and not l.startswith("#")]
        if tr:
            pe = float(tr[-1][2])
        e_atom = None
        hdr2 = off2 = None
        if rc1 == 0:
            hdr2, off2, rows2 = parse_traj(os.path.join(w, "t.xyz"))
            oa = off2["energy_atom"]
            e_atom = [float(r[oa]) for r in rows2]
        # trainer
        nep_in = open(c["input_files"]["nep.in"]["path"], encoding="utf-8").read()
        open(os.path.join(w, "nep.in"), "w", encoding="utf-8", newline="\n").write(
            nep_in.rstrip() + "\nprediction 1\n")
        rc2, log2 = run([NEP], w)
        e_tr_per_atom = None
        if rc2 == 0:
            e_tr_per_atom = float(open(os.path.join(w, "energy_train.out"),
                                       encoding="utf-8").read().split()[0])
        e_tr = e_tr_per_atom * n if e_tr_per_atom is not None else None
        rec = {"case_id": cid, "role": c["role"], "n_atoms": n,
               "E_md_total_eV": pe,
               "sum_e_atom_eV": sum(e_atom) if e_atom else None,
               "E_trainer_total_eV": e_tr,
               "sum_vs_E_md_eV": (sum(e_atom) - pe) if e_atom and pe else None,
               "md_vs_trainer_total_eV": (pe - e_tr) if pe and e_tr else None,
               "md_vs_trainer_per_atom_eV": ((pe - e_tr) / n) if pe and e_tr else None,
               "per_atom_max_abs_eV": None,
               "per_atom_measurable": False,
               "rc": {"gpumd": rc1, "nep": rc2},
               "calls": {"gpumd": w.replace(os.sep, "/"), "nep": w.replace(os.sep, "/")}}
        rec["pass_frozen_gate_per_atom"] = None
        out["cases"].append(rec)
        print("%-22s sum e_i - E_md %+.2e   E_md - E_tr %+.2e (%.2e /atom)  rc=%d/%d"
              % (cid, rec["sum_vs_E_md_eV"] or 0.0,
                 rec["md_vs_trainer_total_eV"] or 0.0,
                 rec["md_vs_trainer_per_atom_eV"] or 0.0, rc1, rc2))

    out["status"] = "FAIL"
    out["reason"] = (
        "G6 的判据是逐原子 e_i 的 trainer↔MD 差（冻结 1.05e-6 eV）。当前发布的二进制里，"
        "trainer 侧只有帧总能量（nep.exe prediction 1），没有 e_i 输出；Stage03A.2 当年用的是"
        "另行编译的 probe 对。本轮因此只能给出：Σe_i^MD - E^MD 的内部一致性、以及 E^MD - E^trainer "
        "的总量差（≤1.3e-7 eV/atom）。总量小并不证明逐原子最大差小（误差可抵消），"
        "所以 G6 维持 FAIL，需重建 probe 对才能复测。")
    out["what_would_close_it"] = [
        "重建 tools/spin_validation/{md_singlepoint_probe.cu,trainer_probe.cu} 与 s3a 头文件",
        "在同一 bundle 上取 e_i 两侧数值，按原检查器逐原子比较",
    ]
    json.dump(out, open(os.path.join(D, "phase_1_results.json"), "w", encoding="utf-8",
                        newline="\n"), indent=1, ensure_ascii=False)
    L = ["=" * 78, "任务 I：G6 逐原子静态一致性（当前构建）", "=" * 78, ""]
    L.append("冻结门禁：1.05e-6 eV/atom（逐原子），历史最差 1.19e-6。")
    L.append("当前二进制：gpumd %s / nep %s" % (out["binaries"]["gpumd"][:16],
                                               out["binaries"]["nep"][:16]))
    L.append("")
    L.append("%-22s %-14s %-16s %-12s" % ("case", "Σe_i - E_md", "E_md - E_tr", "per atom"))
    for c in out["cases"]:
        L.append("%-22s %+.2e    %+.2e      %+.2e"
                 % (c["case_id"], c["sum_vs_E_md_eV"], c["md_vs_trainer_total_eV"],
                    c["md_vs_trainer_per_atom_eV"]))
    L.append("")
    L.append("结论：FAIL（维持）。" + out["reason"])
    open(os.path.join(D, "phase_1_summary.txt"), "w", encoding="utf-8",
         newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L[-4:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
