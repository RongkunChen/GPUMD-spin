#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0: freeze the working tree, record the environment, write the checkpoint.

Read-only with respect to the source tree: everything is copied out, nothing is
modified.  The one thing written outside the evidence directory is the
checkpoint file, which lives in the evidence directory too.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_REL = "work_v3_20261029_132410/GPUMD-spin-r17-offline-src-build"
SRC = os.path.join(ROOT, SRC_REL)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd=None):
    try:
        p = subprocess.run(cmd, cwd=cwd, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        return p.returncode, p.stdout.decode("utf-8", "replace").strip()
    except Exception as e:  # pragma: no cover
        return -1, str(e)


def source_files():
    out = []
    for base, dirs, files in os.walk(SRC):
        dirs[:] = [d for d in dirs if d not in (".git", "build", "examples",
                                                "golden", "docs", "tests_regression",
                                                "tests_spin", "scripts", "postprocessing")]
        for f in files:
            if f.endswith((".cu", ".cuh")) or f == "Makefile":
                out.append(os.path.join(base, f))
    return sorted(out)


def main():
    if len(sys.argv) != 2:
        print("usage: phase0_freeze.py <one_shot_dir>")
        return 2
    D = os.path.abspath(sys.argv[1])
    base = os.path.join(D, "baseline")
    os.makedirs(base, exist_ok=True)
    os.makedirs(os.path.join(D, "checkpoints"), exist_ok=True)

    # ---- source snapshot (relative layout preserved) --------------------
    snap = os.path.join(base, "src_snapshot")
    if os.path.isdir(snap):
        shutil.rmtree(snap)
    hashes = {}
    for p in source_files():
        rel = os.path.relpath(p, SRC)
        dst = os.path.join(snap, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(p, dst)
        hashes[rel.replace(os.sep, "/")] = sha256(p)
    with open(os.path.join(base, "src_sha256.txt"), "w", encoding="utf-8",
              newline="\n") as f:
        for k in sorted(hashes):
            f.write("%s  %s\n" % (hashes[k], k))

    # ---- data and models ------------------------------------------------
    assets = {}
    for rel in ("Fe/train.xyz", "Fe/test.xyz"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            assets[rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    for rel in (SRC_REL + "/src/gpumd.exe", SRC_REL + "/src/nep.exe",
                SRC_REL + "/src/gpumd_replica.exe"):
        p = os.path.join(ROOT, rel)
        assets[rel] = ({"sha256": sha256(p), "bytes": os.path.getsize(p)}
                       if os.path.exists(p) else {"present": False})

    # ---- environment ----------------------------------------------------
    env = {}
    for key, cmd in (("nvcc", "nvcc --version"), ("gpu", "nvidia-smi"),
                     ("cmake", "cmake --version"), ("make", "make --version")):
        rc, out = run(cmd)
        env[key] = {"rc": rc, "out": out[:1500]}
    env["os"] = platform.platform()
    env["python"] = sys.version.split()[0]
    env["cpu_count"] = os.cpu_count()
    with open(os.path.join(base, "env.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(env, f, indent=1, ensure_ascii=False)

    # ---- already-delivered patches (identity of the baseline) -----------
    patches = {}
    for rel in ("evidence/stage08a/20261031_131143/stage08a_changes.patch",
                "evidence/stage07_smoke/20261031_110347/stage07_smoke_changes.patch",
                "evidence/stage06/20261030_224130/stage06_changes.patch"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            patches[rel] = sha256(p)

    summary = {
        "generated": __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "root": ROOT.replace(os.sep, "/"),
        "src": SRC.replace(os.sep, "/"),
        "git": run("git status", cwd=ROOT)[1][:300],
        "n_source_files": len(hashes),
        "src_sha256_all": hashlib.sha256(
            "".join("%s%s" % (k, hashes[k]) for k in sorted(hashes)).encode()
        ).hexdigest(),
        "assets": assets,
        "patches": patches,
    }
    with open(os.path.join(base, "baseline_manifest.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    ckpt = {
        "one_shot_dir": D.replace(os.sep, "/"),
        "generated": summary["generated"],
        "phases": {
            "0": {"status": "PASS", "evidence": "baseline/, phase_0_*"},
            "1": {"status": "PENDING"}, "2": {"status": "PENDING"},
            "3": {"status": "PENDING"}, "4": {"status": "PENDING"},
            "5": {"status": "PENDING"}, "6": {"status": "PENDING"},
            "7": {"status": "PENDING"}, "8": {"status": "PENDING"},
        },
        "notes": [],
    }
    with open(os.path.join(D, "checkpoints", "checkpoint.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(ckpt, f, indent=1, ensure_ascii=False)

    print("source files", len(hashes))
    print("tree sha", summary["src_sha256_all"][:16])
    for k, v in assets.items():
        print(" ", k, v.get("sha256", "MISSING")[:16], v.get("bytes"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
