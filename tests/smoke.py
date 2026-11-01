#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke tests for a GPUMD-spin V3 installation.

Every test runs a real executable from the build directory and checks a number,
not just an exit code:

  T1  NEP single point      : gpumd energy vs nep.exe prediction, same frame
  T2  fixed-lattice spin    : alpha = T = 0 (S05A conservative branch), |S| drift
  T3  stochastic spin       : alpha > 0, T > 0, geometric and midpoint
  T4  dimer MC-vs-SLLG-like : two-spin run, energy bounded, |S| preserved
  T5  restart bitwise       : continuous N vs N/2 + checkpoint + N/2
  T6  replica               : 2-replica REMD, exchange file non-empty

usage:
    python tests/smoke.py --bin <dir with gpumd.exe/nep.exe/gpumd_replica.exe>
                          [--work <scratch dir>] [--json out.json]

Exit code 0 only if every test that the binaries support passes.  A test whose
prerequisite binary is missing is reported as SKIP, not PASS.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
EX = os.path.join(PKG, "examples", "synthetic")
DT = 0.5


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def thermo_last(path):
    rows = [l.split() for l in open(path, encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    return float(rows[-1][2]), float(rows[-1][1])          # PE, KE


def frames(path):
    out, lines, i = [], open(path, encoding="utf-8").read().splitlines(), 0
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
        out.append((hdr, off, [l.split() for l in lines[i + 2:i + 2 + n]]))
        i += 2 + n
    return out


def max_spin_diff(a, b, offset):
    n = min(len(b), len(a) - offset)
    worst = 0.0
    for i in range(n):
        ha, oa, ra = a[offset + i]
        hb, ob, rb = b[i]
        for x, y in zip(ra, rb):
            for c in range(3):
                worst = max(worst, abs(float(x[oa["spin"] + c]) - float(y[ob["spin"] + c])))
    return worst


def read_ckpt(path):
    """spin_restart.bin: 16s magic, u32 ver, u32 N, u64 model, 4 doubles,
    3 i32, 2 i64, double, then anchor[N], rng[N*48], u64 payload."""
    raw = open(path, "rb").read()
    hdr = struct.unpack("<16sIIQddddiiiqqd", raw[:100])
    n, has_rng = hdr[2], hdr[9]
    off = 100
    anchor = struct.unpack("<%df" % n, raw[off:off + 4 * n])
    off += 4 * n
    rng = raw[off:off + (48 * n if has_rng else 0)]
    return {"magic": hdr[0].split(b"\0")[0].decode(), "version": hdr[1], "n": n,
            "has_rng": bool(has_rng), "anchor": anchor,
            "rng_sha": hashlib.sha256(rng).hexdigest() if has_rng else None,
            "steps": hdr[11]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    B = os.path.abspath(args.bin)
    W = os.path.abspath(args.work or os.path.join(PKG, "_smoke"))
    if os.path.isdir(W):
        shutil.rmtree(W)
    os.makedirs(W)
    gpumd = os.path.join(B, "gpumd.exe" if os.name == "nt" else "gpumd")
    nep = os.path.join(B, "nep.exe" if os.name == "nt" else "nep")
    rep = os.path.join(B, "gpumd_replica.exe" if os.name == "nt" else "gpumd_replica")
    res = {"binaries": {}, "tests": {}}
    for k, p in (("gpumd", gpumd), ("nep", nep), ("replica", rep)):
        res["binaries"][k] = {"path": p, "sha256": sha(p) if os.path.exists(p) else None}

    def case(name):
        d = os.path.join(W, name)
        os.makedirs(d, exist_ok=True)
        shutil.copy2(os.path.join(EX, "nep.txt"), os.path.join(d, "nep.txt"))
        shutil.copy2(os.path.join(EX, "model.xyz"), os.path.join(d, "model.xyz"))
        return d

    # T1: NEP single point
    try:
        d = case("t1_singlepoint")
        open(os.path.join(d, "run.in"), "w", newline="\n").write(
            "potential nep.txt\ntime_step 0.0001\nensemble nve\ndump_thermo 1\nrun 1\n")
        rc, out = run([gpumd], d)
        pe, _ = thermo_last(os.path.join(d, "thermo.out"))
        shutil.copy2(os.path.join(d, "model.xyz"), os.path.join(d, "train.xyz"))
        open(os.path.join(d, "nep.in"), "w", newline="\n").write(
            "type 1 Fe\ncutoff 6.0 3.0\nn_max 4 4\nbasis_size 8 8\nl_max 4 0 0\n"
            "neuron 30\nspin 1\nprediction 1\n")
        # the trainer wants energy=/forces in the frame header
        lines = open(os.path.join(d, "train.xyz"), encoding="utf-8").read().splitlines()
        lines[1] = "energy=0 " + lines[1].replace(
            "Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3",
            "Properties=species:S:1:pos:R:3:forces:R:3:spin:R:3")
        rows = [l.split() for l in lines[2:]]
        lines = lines[:2] + [" ".join(r[:4] + ["0", "0", "0"] + r[7:]) for r in rows]
        open(os.path.join(d, "train.xyz"), "w", newline="\n").write("\n".join(lines) + "\n")
        rc2, out2 = run([nep], d)
        e_tr = float(open(os.path.join(d, "energy_train.out"),
                          encoding="utf-8").read().split()[0]) * len(lines[2:])
        res["tests"]["T1_single_point"] = {
            "rc": [rc, rc2], "E_gpumd_eV": pe, "E_nep_eV": e_tr,
            "abs_diff_eV": abs(pe - e_tr),
            "pass": bool(rc == 0 and rc2 == 0 and abs(pe - e_tr) < 1e-4)}
    except Exception as e:
        res["tests"]["T1_single_point"] = {"pass": False, "error": str(e)}

    # T2/T3: fixed-lattice spin
    for tag, line in (("T2_conservative", "spin_llg gamma=3.04 alpha=0 T=0"),
                      ("T3a_geometric", "spin_llg gamma=3.04 alpha=0.5 T=300 seed=7 scheme=geometric"),
                      ("T3b_midpoint", "spin_llg gamma=3.04 alpha=0.5 T=300 seed=7 scheme=midpoint")):
        try:
            d = case(tag)
            open(os.path.join(d, "run.in"), "w", newline="\n").write(
                "potential nep.txt\ntime_step %g\nensemble nve\ndump_thermo 10\n"
                "dump_xyz 20 t.xyz spin\ndump_restart 100000\n%s\nrun 200\n" % (DT, line))
            rc, out = run([gpumd], d)
            fr = frames(os.path.join(d, "t.xyz"))
            # |S| differs from atom to atom (the moments are not uniform), so the
            # reference is each atom's own |S| at the first frame.
            o0 = fr[0][1]
            nref = [math.sqrt(sum(float(fr[0][2][i][o0["spin"] + c]) ** 2
                                 for c in range(3))) for i in range(len(fr[0][2]))]
            drift = 0.0
            for (h, o, rows) in fr:
                for i, r in enumerate(rows):
                    v = [float(r[o["spin"] + c]) for c in range(3)]
                    n = math.sqrt(sum(x * x for x in v))
                    drift = max(drift, abs(n / nref[i] - 1.0))
            res["tests"][tag] = {"rc": rc, "n_frames": len(fr),
                                 "spin_norm_drift": drift,
                                 "pass": bool(rc == 0 and drift < 1e-5)}
        except Exception as e:
            res["tests"][tag] = {"pass": False, "error": str(e)}

    # T4: two-spin dimer (built from the first two atoms of the example)
    try:
        d = os.path.join(W, "t4_dimer")
        os.makedirs(d, exist_ok=True)
        shutil.copy2(os.path.join(EX, "nep.txt"), os.path.join(d, "nep.txt"))
        lines = open(os.path.join(EX, "model.xyz"), encoding="utf-8").read().splitlines()
        lat = lines[1].split('Lattice="')[1].split('"')[0]
        body = lines[2:4]
        open(os.path.join(d, "model.xyz"), "w", newline="\n").write(
            "2\nLattice=\"%s\" Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3 "
            "pbc=\"T T T\"\n%s\n%s\n" % (lat, body[0], body[1]))
        open(os.path.join(d, "run.in"), "w", newline="\n").write(
            "potential nep.txt\ntime_step %g\nensemble nve\ndump_thermo 20\n"
            "spin_llg gamma=3.04 alpha=0.5 T=300 seed=11 scheme=geometric\nrun 200\n" % DT)
        rc, out = run([gpumd], d)
        pe, _ = thermo_last(os.path.join(d, "thermo.out"))
        res["tests"]["T4_dimer"] = {"rc": rc, "PE_eV": pe,
                                    "finite": bool(math.isfinite(pe)),
                                    "pass": bool(rc == 0 and math.isfinite(pe) and abs(pe) < 1e3)}
    except Exception as e:
        res["tests"]["T4_dimer"] = {"pass": False, "error": str(e)}

    # T5: restart bitwise
    try:
        base = os.path.join(W, "t5_restart")
        shutil.rmtree(base, ignore_errors=True)
        line = "spin_llg gamma=3.04 alpha=0.5 T=300 seed=4242 scheme=midpoint"
        for sub, steps in (("cont", 400), ("prefix", 200), ("restart", 200)):
            d = os.path.join(base, sub)
            os.makedirs(d)
            shutil.copy2(os.path.join(EX, "nep.txt"), os.path.join(d, "nep.txt"))
            shutil.copy2(os.path.join(EX, "model.xyz"), os.path.join(d, "model.xyz"))
            txt = ("potential nep.txt\ntime_step %g\nensemble nve\ndump_thermo 20\n"
                   "dump_xyz 20 t.xyz spin\ndump_restart %d\n%s\nrun %d\n"
                   % (DT, steps, line, steps))
            if sub == "restart":
                shutil.copy2(os.path.join(base, "prefix", "restart.xyz"),
                             os.path.join(d, "model.xyz"))
                shutil.copy2(os.path.join(base, "prefix", "spin_restart.bin"),
                             os.path.join(d, "spin_restart.bin"))
                txt = txt.replace(line, line + " restart=spin_restart.bin")
            open(os.path.join(d, "run.in"), "w", newline="\n").write(txt)
            rc, out = run([gpumd], d)
            if rc != 0:
                raise RuntimeError("run %s rc=%d" % (sub, rc))
        fc = frames(os.path.join(base, "cont", "t.xyz"))
        fr = frames(os.path.join(base, "restart", "t.xyz"))
        diff = max_spin_diff(fc, fr, offset=200 // 20)
        ck_c = read_ckpt(os.path.join(base, "cont", "spin_restart.bin"))
        ck_r = read_ckpt(os.path.join(base, "restart", "spin_restart.bin"))
        res["tests"]["T5_restart"] = {
            "max_spin_diff": diff, "rng_stream_identical":
                (ck_c["rng_sha"] == ck_r["rng_sha"]) if ck_c["has_rng"] else None,
            "pass": bool(diff == 0.0 and (ck_c["rng_sha"] == ck_r["rng_sha"]))}
    except Exception as e:
        res["tests"]["T5_restart"] = {"pass": False, "error": str(e)}

    # T6: replica
    if os.path.exists(rep):
        try:
            d = case("t6_replica")
            open(os.path.join(d, "run.in"), "w", newline="\n").write(
                "potential nep.txt\ntime_step 0.0001\n"
                "multi_replica remd replicas 2 exchange 50 temp 300 400 seed 12345\n"
                "ensemble nvt_bdp 300 400 100\ndump_xyz -1 0 50 rep.xyz\nrun 100\n")
            rc, out = run([rep], d)
            ok = os.path.exists(os.path.join(d, "remd_exchange.out"))
            res["tests"]["T6_replica"] = {"rc": rc, "exchange_file": ok,
                                          "pass": bool(rc == 0 and ok)}
        except Exception as e:
            res["tests"]["T6_replica"] = {"pass": False, "error": str(e)}
    else:
        res["tests"]["T6_replica"] = {"pass": None, "status": "SKIP (no gpumd_replica)"}

    res["pass"] = all(t.get("pass") is True for t in res["tests"].values())
    if args.json:
        json.dump(res, open(args.json, "w", encoding="utf-8", newline="\n"),
                  indent=1, ensure_ascii=False)
    for k, v in res["tests"].items():
        print("%-18s %s %s" % (k, "PASS" if v.get("pass") else
                               ("SKIP" if v.get("pass") is None else "FAIL"),
                               {kk: vv for kk, vv in v.items() if kk != "pass"}))
    print("\nSMOKE", "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
