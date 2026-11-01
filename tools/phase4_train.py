#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 4: train the Fe model with the production nep.exe and evaluate it.

Scope control (budget, stated up front):
  * the official 800 frames are never used for training, early stopping or model
    selection -- they are only scored after the fact;
  * the train/validation split groups frames by identical input state (same
    species+positions+spins), so a duplicate cannot sit on both sides.  The
    parent-trajectory identity of the frames is unknown, so this is "input
    state isolation", not trajectory-level independence;
  * two pre-declared hypotheses: mask17 (Sex|Smag, the no-SOC assumption) and
    mask31 (all five blocks, SOC-like representation).  Neither is called
    "correct" because of its block count.

usage:
  phase4_train.py <one_shot_dir> prepare
  phase4_train.py <one_shot_dir> train <mask> <seed> <size:2k|8k|full> <generations>
  phase4_train.py <one_shot_dir> eval  <run_dir>
  phase4_train.py <one_shot_dir> report
"""
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "work_v3_20261029_132410/GPUMD-spin-r17-offline-src-build/src")
NEP = os.path.join(SRC, "nep.exe")
TRAIN = os.path.join(ROOT, "Fe/train.xyz")
TEST = os.path.join(ROOT, "Fe/test.xyz")
KB = 8.617333262e-5

BASE_IN = """type 1 Fe
cutoff 6.0 3.0
n_max 4 4
basis_size 8 8
l_max 4 0 0
neuron 30
spin 1
spin_blocks {blocks}
spin_mag_scale 0.25
lambda_e 1.0
lambda_f 1.0
lambda_m 1.0
batch 32
population 20
seed {seed}
generation {gen}
output_interval {interval}
nep_compile off
"""


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frames(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            break
        n = int(lines[i])
        blk = lines[i:i + 2 + n]
        out.append(blk)
        i += 2 + n
    return out


def state_key(blk):
    """Identity of the *input state*: species + positions + spins, text-exact."""
    hdr = blk[1]
    props = hdr.split("Properties=")[1].split()[0].split(":")
    names, sizes = props[0::3], [int(x) for x in props[2::3]]
    keep = {"species", "pos", "spin"}
    cols, k = [], 0
    for nm, sz in zip(names, sizes):
        if nm in keep:
            cols.append((k, sz, nm))
        k += sz
    h = hashlib.sha256()
    for t in blk[2:]:
        f = t.split()
        for (k, sz, nm) in cols:
            h.update(" ".join(f[k + 1:k + 1 + sz]).encode())
    return h.hexdigest()


def prepare(D):
    P = os.path.join(D, "phase_4")
    os.makedirs(os.path.join(P, "data"), exist_ok=True)
    meta = {"train_sha256": sha(TRAIN), "test_sha256": sha(TEST)}
    fr = frames(TRAIN)
    keys = [state_key(b) for b in fr]
    uniq = {}
    for b, k in zip(fr, keys):
        uniq.setdefault(k, b)
    ordered = sorted(uniq.items(), key=lambda kv: kv[0])
    meta["n_frames"] = len(fr)
    meta["n_unique_states"] = len(ordered)
    meta["n_duplicate_frames"] = len(fr) - len(ordered)
    # deterministic, state-grouped split
    rng = random.Random(20261031)
    idx = list(range(len(ordered)))
    rng.shuffle(idx)
    dev_idx = idx[:256]
    tr_idx = idx[256:]
    def dump(idxs, name):
        with open(os.path.join(P, "data", name), "w", encoding="utf-8", newline="\n") as f:
            for i in idxs:
                f.write("\n".join(ordered[i][1]) + "\n")
        return len(idxs)
    # dev must carry reference columns for scoring; ordered[] keeps the frames
    meta["n_train_pool"] = dump(tr_idx, "train_pool.xyz")
    meta["n_dev"] = dump(dev_idx, "dev.xyz")
    # 2k / 8k subsets of the training pool (prefix of the shuffled order)
    for size, n in (("2k", 2048), ("8k", 8192)):
        with open(os.path.join(P, "data", "train_%s.xyz" % size), "w",
                  encoding="utf-8", newline="\n") as f:
            for i in tr_idx[:n]:
                f.write("\n".join(ordered[i][1]) + "\n")
        meta["n_%s" % size] = n
    shutil.copy2(TRAIN, os.path.join(P, "data", "full.xyz"))
    # overlap audit
    tr_keys = {ordered[i][0] for i in tr_idx}
    dev_keys = {ordered[i][0] for i in dev_idx}
    test_keys = {state_key(b) for b in frames(TEST)}
    meta["overlap"] = {
        "train_dev": len(tr_keys & dev_keys),
        "train_test_official": len(tr_keys & test_keys),
        "dev_test_official": len(dev_keys & test_keys),
        "test_official_frames": len(frames(TEST)),
        "test_official_matching_train": sum(1 for b in frames(TEST)
                                            if state_key(b) in tr_keys),
    }
    json.dump(meta, open(os.path.join(P, "data", "split_meta.json"), "w",
                         encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    return 0


def train(D, mask, seed, size, gen):
    gen = int(gen)
    P = os.path.join(D, "phase_4")
    blocks = "Sex Smag" if int(mask) == 17 else "Sex San Sdm Sia Smag"
    name = "m%s_%s_s%s_g%s" % (mask, size, seed, gen)
    d = os.path.join(P, "runs", name)
    os.makedirs(d, exist_ok=True)
    src = os.path.join(P, "data", ("full.xyz" if size == "full"
                                   else "train_%s.xyz" % size))
    shutil.copy2(src, os.path.join(d, "train.xyz"))
    open(os.path.join(d, "nep.in"), "w", encoding="utf-8", newline="\n").write(
        BASE_IN.format(blocks=blocks, seed=seed, gen=gen, interval=max(100, gen // 20)))
    p = subprocess.run([NEP], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    open(os.path.join(d, "stdout.txt"), "wb").write(p.stdout)
    info = {"mask": int(mask), "seed": int(seed), "size": size, "generations": int(gen),
            "rc": p.returncode, "nep_txt_sha256": sha(os.path.join(d, "nep.txt"))
            if p.returncode == 0 else None,
            "train_frames": len(frames(os.path.join(d, "train.xyz"))),
            "blocks": blocks}
    json.dump(info, open(os.path.join(d, "run.json"), "w", encoding="utf-8",
                         newline="\n"), indent=1, ensure_ascii=False)
    print(json.dumps(info))
    return 0 if p.returncode == 0 else 1


def predict(d, xyz, tag):
    w = os.path.join(d, "eval_%s" % tag)
    if os.path.isdir(w):
        shutil.rmtree(w)
    os.makedirs(w)
    shutil.copy2(os.path.join(d, "nep.txt"), os.path.join(w, "nep.txt"))
    shutil.copy2(xyz, os.path.join(w, "train.xyz"))
    nep_in = open(os.path.join(d, "nep.in"), encoding="utf-8").read()
    open(os.path.join(w, "nep.in"), "w", encoding="utf-8", newline="\n").write(
        nep_in.rstrip() + "\nprediction 1\n")
    p = subprocess.run([NEP], cwd=w, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    open(os.path.join(w, "stdout.txt"), "wb").write(p.stdout)
    return w, p.returncode


def metrics(d, xyz, tag):
    import numpy as np
    w, rc = predict(d, xyz, tag)
    if rc != 0:
        return {"rc": rc}
    e = np.loadtxt(os.path.join(w, "energy_train.out"))
    if e.ndim == 1:
        e = e[None, :]
    de = e[:, 0] - e[:, 1]                       # eV/atom
    f = np.loadtxt(os.path.join(w, "force_train.out"))
    if f.ndim == 1:
        f = f[None, :]
    df = (f[:, :3] - f[:, 3:]).reshape(-1)         # eV/A per component
    out = {"n_frames": int(len(e)), "rc": rc,
           "E_MAE_meV_atom": float(np.abs(de).mean() * 1000),
           "E_RMSE_meV_atom": float(np.sqrt((de ** 2).mean()) * 1000),
           "F_MAE_eV_A": float(np.abs(df).mean()),
           "F_RMSE_eV_A": float(np.sqrt((df ** 2).mean())),
           "F_ref_rms_eV_A": float(f[:, 3:].std()),
           "ref_E_std_meV_atom": float(e[:, 1].std() * 1000)}
    sp = os.path.join(w, "spin_train.out")
    if os.path.exists(sp):
        g = np.loadtxt(sp)
        if g.ndim == 1:
            g = g[None, :]
        dg = (g[:, :3] - g[:, 3:]).reshape(-1)
        out["g_MAE_eV_muB"] = float(np.abs(dg).mean())
        out["g_RMSE_eV_muB"] = float(np.sqrt((dg ** 2).mean()))
        out["g_ref_rms_eV_muB"] = float(g[:, 3:].std())
        # spins, flattened in the same atom order as the prediction columns
        lines = open(xyz, encoding="utf-8").read().splitlines()
        spins, i = [], 0
        while i < len(lines):
            if not lines[i].strip():
                break
            n = int(lines[i])
            props = lines[i + 1].split("Properties=")[1].split()[0].split(":")
            names, sizes = props[0::3], [int(x) for x in props[2::3]]
            off, k = 0, 0
            for nm, sz in zip(names, sizes):
                if nm == "spin":
                    off = k
                k += sz
            for t in lines[i + 2:i + 2 + n]:
                q = t.split()
                spins.append([float(x) for x in q[1 + off:1 + off + 3]])
            i += 2 + n
        s_arr = np.array(spins)
        gv = g[:, :3] - g[:, 3:]      # error vector, not the predicted value - g[:, 3:]      # error vector, not the predicted value
        if len(s_arr) == len(gv):
            sn = np.linalg.norm(s_arr, axis=1, keepdims=True)
            u = np.divide(s_arr, sn, out=np.zeros_like(s_arr), where=sn > 0)
            g_par = (gv * u).sum(axis=1)
            g_perp = np.linalg.norm(gv - g_par[:, None] * u, axis=1)
            out["g_parallel_MAE_eV_muB"] = float(np.abs(g_par).mean())
            out["g_perp_MAE_eV_muB"] = float(np.abs(g_perp).mean())
    return out


def eval_run(run_dir):
    res = {}
    P = os.path.dirname(os.path.dirname(run_dir))
    for tag, xyz in (("dev", os.path.join(P, "data", "dev.xyz")),
                     ("official800", TEST)):
        res[tag] = metrics(run_dir, xyz, tag)
    json.dump(res, open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8",
                        newline="\n"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


def main():
    D = os.path.abspath(sys.argv[1])
    cmd = sys.argv[2]
    if cmd == "prepare":
        return prepare(D)
    if cmd == "train":
        return train(D, sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
    if cmd == "eval":
        return eval_run(os.path.abspath(sys.argv[3]))
    if cmd == "report":
        return report(D)
    return 2


def report(D):
    P = os.path.join(D, "phase_4")
    runs = []
    for name in sorted(os.listdir(os.path.join(P, "runs"))):
        d = os.path.join(P, "runs", name)
        m = os.path.join(d, "metrics.json")
        if os.path.exists(m):
            r = json.load(open(os.path.join(d, "run.json"), encoding="utf-8"))
            r["metrics"] = json.load(open(m, encoding="utf-8"))
            r["dir"] = d.replace(os.sep, "/")
            runs.append(r)
    out = {"phase": "4", "runs": runs,
           "data": json.load(open(os.path.join(P, "data", "split_meta.json"),
                                  encoding="utf-8"))}
    json.dump(out, open(os.path.join(D, "phase_4_results.json"), "w",
                        encoding="utf-8", newline="\n"), indent=1, ensure_ascii=False)
    print("runs", len(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
