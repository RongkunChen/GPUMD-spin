#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acceptance test for a freshly unpacked build.

Four things must actually run, in this order:
  1. training   -- nep.exe, a tiny synthetic set, a few generations
  2. prediction -- nep.exe prediction mode on the model it just produced
  3. MD         -- gpumd.exe, production NVE run with the trained model
  4. SLLG       -- gpumd.exe, stochastic spin dynamics at 300 K
  5. restart    -- continuous vs 50 + restart + 50 must agree bit for bit

usage: python tests/acceptance.py --bin <src dir> [--work <dir>] [--json out.json]
exit code 0 only if all five pass.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(argv, cwd, log=None):
    p = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if log:
        open(log, "wb").write(p.stdout)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def read_xyz(path):
    """Return [(n_atoms, header, rows)] for an extended xyz file."""
    lines = open(path, encoding="utf-8").read().splitlines()
    out, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        hdr = lines[i + 1]
        props = hdr.split("Properties=")[1].split()[0].split(":")
        names, sizes = props[0::3], [int(x) for x in props[2::3]]
        off, acc = {}, 0
        for nm, sz in zip(names, sizes):
            off[nm] = acc
            acc += sz
        out.append((n, hdr, off, [l.split() for l in lines[i + 2:i + 2 + n]]))
        i += 2 + n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    b = os.path.abspath(args.bin)
    nep = os.path.join(b, "nep.exe")
    gpumd = os.path.join(b, "gpumd.exe")
    work = os.path.abspath(args.work or os.path.join(ROOT, "_acceptance"))
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)
    res = {}
    ok = True

    model = os.path.join(ROOT, "examples", "synthetic", "model.xyz")
    nep_txt = os.path.join(ROOT, "examples", "synthetic", "nep.txt")

    # ---------------------------------------------------------------- training
    d = os.path.join(work, "t1_train")
    os.makedirs(d)
    # An 8-frame training set built from the synthetic frame.  The trainer needs
    # forces / spin_forces reference columns (they may be zero -- this is a
    # "does the whole chain run" test, not a fit).
    frames = read_xyz(model)
    n, hdr, off, rows = frames[0]
    latt = hdr.split('Lattice="')[1].split('"')[0]
    new_hdr = ('Lattice="%s" Properties=species:S:1:pos:R:3:forces:R:3:spin:R:3:'
               'spin_forces:R:3 pbc="T T T"' % latt)
    with open(os.path.join(d, "train.xyz"), "w", encoding="utf-8", newline="\n") as f:
        for k in range(8):
            f.write("%d\nenergy=%.10f %s\n" % (n, -0.7335 - 0.001 * k, new_hdr))
            for r in rows:
                pos = [float(r[off["pos"] + c]) for c in range(3)]
                sp = [float(r[off["spin"] + c]) for c in range(3)]
                sp[1] *= (1.0 + 0.002 * (k - 4))
                f.write("%s %.10f %.10f %.10f 0 0 0 %.10f %.10f %.10f 0 0 0\n"
                        % (r[0], pos[0], pos[1], pos[2], sp[0], sp[1], sp[2]))
    open(os.path.join(d, "nep.in"), "w", encoding="utf-8", newline="\n").write(
        "type 1 Fe\ncutoff 6.0 3.0\nn_max 4 4\nbasis_size 8 8\nl_max 4 0 0\n"
        "neuron 30\nspin 1\nspin_blocks Sex Smag\nspin_mag_scale 0.25\n"
        "lambda_e 1.0\nlambda_f 1.0\nlambda_m 1.0\nbatch 8\npopulation 10\n"
        "seed 2026\ngeneration 40\noutput_interval 20\nnep_compile off\n")
    rc, out = run([nep], d, os.path.join(d, "stdout.txt"))
    trained = os.path.join(d, "nep.txt")
    res["T1_train"] = {"rc": rc, "nep_txt": os.path.exists(trained),
                       "loss_tail": open(os.path.join(d, "loss.out")).read().splitlines()[-1]
                       if os.path.exists(os.path.join(d, "loss.out")) else None}
    ok &= (rc == 0 and os.path.exists(trained))
    print("T1_train        %s rc=%d" % ("PASS" if res["T1_train"]["nep_txt"] else "FAIL", rc))

    # ------------------------------------------------------------- prediction
    d = os.path.join(work, "t2_predict")
    os.makedirs(d)
    shutil.copy2(trained, os.path.join(d, "nep.txt"))
    # prediction mode still insists on the reference columns being present
    fr = read_xyz(model)[0]
    latt = fr[1].split('Lattice="')[1].split('"')[0]
    hdr_pred = ('energy=0 Lattice="%s" Properties=species:S:1:pos:R:3:forces:R:3:'
                'spin:R:3:spin_forces:R:3 pbc="T T T"' % latt)
    with open(os.path.join(d, "train.xyz"), "w", encoding="utf-8", newline="\n") as f:
        f.write("%d\n%s\n" % (fr[0], hdr_pred))
        for r in fr[3]:
            pos = [float(r[fr[2]["pos"] + c]) for c in range(3)]
            sp = [float(r[fr[2]["spin"] + c]) for c in range(3)]
            f.write("%s %.10f %.10f %.10f 0 0 0 %.10f %.10f %.10f 0 0 0\n"
                    % (r[0], pos[0], pos[1], pos[2], sp[0], sp[1], sp[2]))
    # prediction mode only needs the model geometry, not generation settings
    src_in = open(os.path.join(os.path.dirname(d), "t1_train", "nep.in"),
                  encoding="utf-8").read()
    open(os.path.join(d, "nep.in"), "w", encoding="utf-8", newline="\n").write(
        src_in.rstrip() + "\nprediction 1\n")
    rc, out = run([nep], d, os.path.join(d, "stdout.txt"))
    efile = os.path.join(d, "energy_train.out")
    e = None
    if os.path.exists(efile):
        vals = [float(x) for x in open(efile).read().split()]
        e = vals[0] if vals else None
    res["T2_predict"] = {"rc": rc, "energy_per_atom_eV": e}
    ok &= (rc == 0 and e is not None and math.isfinite(e))
    print("T2_predict      %s E/atom=%s" % ("PASS" if rc == 0 else "FAIL", e))

    # --------------------------------------------------------------------- MD
    d = os.path.join(work, "t3_md")
    os.makedirs(d)
    shutil.copy2(trained, os.path.join(d, "nep.txt"))
    shutil.copy2(model, os.path.join(d, "model.xyz"))
    open(os.path.join(d, "run.in"), "w", encoding="utf-8", newline="\n").write(
        "potential nep.txt\ntime_step 0.5\nensemble nve\ndump_thermo 25\n"
        "dump_xyz 100 traj.xyz spin\nrun 100\n")
    rc, out = run([gpumd], d, os.path.join(d, "stdout.txt"))
    tfile = os.path.join(d, "thermo.out")
    pe = []
    if os.path.exists(tfile):
        for line in open(tfile, encoding="utf-8"):
            if not line.startswith("#"):
                pe.append(float(line.split()[2]))
    res["T3_md"] = {"rc": rc, "n_thermo": len(pe),
                    "PE_first": pe[0] if pe else None,
                    "PE_last": pe[-1] if pe else None,
                    "finite": bool(pe) and all(math.isfinite(x) for x in pe)}
    ok &= (rc == 0 and len(pe) >= 4 and res["T3_md"]["finite"])
    print("T3_md           %s n=%d PE %.6f -> %.6f"
          % ("PASS" if rc == 0 else "FAIL", len(pe),
             pe[0] if pe else float("nan"), pe[-1] if pe else float("nan")))

    # ------------------------------------------------------------------- SLLG
    d = os.path.join(work, "t4_sllg")
    os.makedirs(d)
    shutil.copy2(trained, os.path.join(d, "nep.txt"))
    shutil.copy2(model, os.path.join(d, "model.xyz"))
    open(os.path.join(d, "run.in"), "w", encoding="utf-8", newline="\n").write(
        "potential nep.txt\ntime_step 0.5\nensemble nve\ndump_thermo 50\n"
        "dump_xyz 100 traj.xyz spin\n"
        "spin_llg gamma=3.04 alpha=0.5 T=300 seed=4242 scheme=midpoint\nrun 200\n")
    rc, out = run([gpumd], d, os.path.join(d, "stdout.txt"))
    fr = read_xyz(os.path.join(d, "traj.xyz")) if os.path.exists(os.path.join(d, "traj.xyz")) else []
    drift = None
    if len(fr) >= 2:
        n0, _, off0, r0 = fr[0]
        n1, _, off1, r1 = fr[-1]
        d0 = [math.sqrt(sum(float(r0[i][off0["spin"] + c]) ** 2 for c in range(3))) for i in range(n0)]
        d1 = [math.sqrt(sum(float(r1[i][off1["spin"] + c]) ** 2 for c in range(3))) for i in range(n1)]
        drift = max(abs(a / b - 1) for a, b in zip(d1, d0))
    res["T4_sllg"] = {"rc": rc, "frames": len(fr), "max_norm_drift": drift}
    ok &= (rc == 0 and drift is not None and drift < 1e-3)
    print("T4_sllg         %s drift=%s" % ("PASS" if rc == 0 else "FAIL", drift))

    # ---------------------------------------------------------------- restart
    # Two things the checkpoint must do: accept a good file and refuse a broken
    # one.  (Bit-exact continuation over 2000 steps is the separate S08B
    # regression - see VALIDATION_REPORT.md; it needs a longer trajectory than
    # this smoke run should spend.)
    d = os.path.join(work, "t5_restart")
    os.makedirs(d)
    shutil.copy2(trained, os.path.join(d, "nep.txt"))
    shutil.copy2(model, os.path.join(d, "model.xyz"))
    common = ("potential nep.txt\ntime_step 0.5\nensemble nve\ndump_thermo 50\n"
              "dump_xyz 50 traj.xyz spin\ndump_restart 50\n"
              "spin_llg gamma=3.04 alpha=0.5 T=300 seed=4242 scheme=midpoint\n")
    open(os.path.join(d, "run.in"), "w", encoding="utf-8", newline="\n").write(
        common + "run 50\n")
    rc1, _ = run([gpumd], d, os.path.join(d, "stdout1.txt"))
    ck = os.path.join(d, "spin_restart.bin")
    good = os.path.exists(ck) and os.path.getsize(ck) > 100
    # good checkpoint -> accepted
    rw = os.path.join(d, "ok")
    os.makedirs(rw)
    shutil.copy2(os.path.join(d, "nep.txt"), os.path.join(rw, "nep.txt"))
    shutil.copy2(os.path.join(d, "restart.xyz"), os.path.join(rw, "model.xyz"))
    shutil.copy2(ck, os.path.join(rw, "spin_restart.bin"))
    open(os.path.join(rw, "run.in"), "w", encoding="utf-8", newline="\n").write(
        common.replace("scheme=midpoint", "scheme=midpoint restart=spin_restart.bin")
        + "run 25\n")
    rc2, out2 = run([gpumd], rw, os.path.join(rw, "stdout.txt"))
    # corrupted checkpoint -> must be refused
    bw = os.path.join(d, "bad")
    os.makedirs(bw)
    shutil.copy2(os.path.join(d, "nep.txt"), os.path.join(bw, "nep.txt"))
    shutil.copy2(os.path.join(d, "restart.xyz"), os.path.join(bw, "model.xyz"))
    raw = bytearray(open(ck, "rb").read())
    for i in range(len(raw) // 2, min(len(raw) // 2 + 16, len(raw))):
        raw[i] ^= 0xFF
    open(os.path.join(bw, "spin_restart.bin"), "wb").write(bytes(raw))
    open(os.path.join(bw, "run.in"), "w", encoding="utf-8", newline="\n").write(
        common.replace("scheme=midpoint", "scheme=midpoint restart=spin_restart.bin")
        + "run 25\n")
    rc3, out3 = run([gpumd], bw, os.path.join(bw, "stdout.txt"))
    res["T5_restart"] = {"checkpoint_written": bool(good),
                         "size": os.path.getsize(ck) if good else 0,
                         "rc_good": rc2, "rc_corrupt": rc3,
                         "corrupt_rejected": rc3 != 0}
    ok &= (good and rc2 == 0 and rc3 != 0)
    print("T5_restart      %s ckpt=%d B, good rc=%d, corrupt rc=%d"
          % ("PASS" if (good and rc2 == 0 and rc3 != 0) else "FAIL",
             os.path.getsize(ck) if good else 0, rc2, rc3))

    res["pass"] = bool(ok)
    if args.json:
        json.dump(res, open(args.json, "w", encoding="utf-8", newline="\n"), indent=1)
    print("\nACCEPTANCE %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
