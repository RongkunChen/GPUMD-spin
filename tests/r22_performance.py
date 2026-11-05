#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R22-NonHeat production GPU performance comparison.

Six configurations, measured with 5 ALTERNATING repeats on the SAME GPU,
model, seed, atom count, time step and compiler flags:

  r21 baseline dynamics | r22 plain dynamics | r22 ME C2 smooth cutoff |
  r22 time-dependent field work ledger | r22 spin-lattice diagnostics |
  r22 magnon trajectory + CPU DSF analysis

The GPU timing comes from GPUMD's own reported wall time AND an independent
Python monotonic clock around the process.  GPU peak memory is sampled with
nvidia-smi during each run.  CPU post-processing time is reported separately.

Usage:
  python tests/r22_performance.py --r21-bin <dir> --r22-bin <dir> \
     --work <dir> --csv <out.csv> --json <out.json> [--repeats 5] [--steps 400]
"""
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
NEP = os.path.join(DATA, "fege.nep.txt")
B20 = os.path.join(DATA, "fege_b20.xyz")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_b20():
    lines = open(B20, encoding="utf-8").read().splitlines()
    n = int(lines[0])
    hdr = lines[1]
    m = re.search(r'Lattice="([^"]+)"', hdr)
    cell = np.array([float(x) for x in m.group(1).split()]).reshape(3, 3)
    rows = [l.split() for l in lines[2:2 + n]]
    species = [r[0] for r in rows]
    pos = np.array([[float(x) for x in r[1:4]] for r in rows])
    spin = np.array([[float(x) for x in r[4:7]] for r in rows])
    return species, pos, spin, cell


def tiled_model(path, target_n):
    """Tile the FeGe B20 cell to >= target_n atoms, then truncate to exactly N.

    Both Fe and Ge remain present (required by compact_nep.cu for a spin NEP).
    """
    species, pos, spin, cell = load_b20()
    n0 = len(pos)
    reps = int(np.ceil((target_n / n0) ** (1.0 / 3.0)))
    out_s, out_p, out_S = [], [], []
    for i in range(reps):
        for j in range(reps):
            for k in range(reps):
                shift = i * cell[0] + j * cell[1] + k * cell[2]
                out_s.extend(species)
                out_p.extend(pos + shift)
                out_S.extend(spin)
                if len(out_s) >= target_n:
                    break
            if len(out_s) >= target_n:
                break
        if len(out_s) >= target_n:
            break
    out_s = out_s[:target_n]
    out_p = np.array(out_p[:target_n], float)
    out_S = np.array(out_S[:target_n], float)
    big = cell * reps
    lat = " ".join("%.10f" % v for v in big.reshape(9))
    with open(path, "w", encoding="utf-8") as f:
        f.write("%d\n" % target_n)
        f.write('Lattice="%s" Properties=species:S:1:pos:R:3:spin:R:3 '
                'pbc="T T T"\n' % lat)
        for s, p, S in zip(out_s, out_p, out_S):
            f.write("%s %.8f %.8f %.8f %.8f %.8f %.8f\n" %
                    (s, p[0], p[1], p[2], S[0], S[1], S[2]))
    return target_n


def gpu_mem_sampler(stop, peak, interval=0.08):
    while not stop.is_set():
        try:
            p = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=2)
            v = float(p.stdout.decode().strip().splitlines()[0])
            peak[0] = max(peak[0], v)
        except Exception:
            pass
        stop.wait(interval)


def base_lines(steps):
    return ["potential nep.txt", "time_step 0.05", "ensemble nve",
            "velocity 300 seed 4242", "dump_thermo 100",
            "spin_llg gamma=3.0385 alpha=0 T=0"]


def config_lines(key, steps, r21=False):
    """All six configurations use the SAME model, seed, dt, steps and GPU."""
    b = base_lines(steps)
    if key == "r21_baseline":
        return b + ["run %d" % steps]
    if key == "r22_plain":
        return b + ["run %d" % steps]
    if key == "r22_me_smooth":
        return b + ["magelec 4.5 0.001 0.3 -0.7 0.5 cutoff=smooth r_on=3.6",
                    "run %d" % steps]
    if key == "r22_field_work":
        return b + ["magfield 0.01 0 0 1 unit=ev wave=sin freq=2.0",
                    "run %d" % steps]
    if key == "r22_spin_lattice_diag":
        return b + ["dump_xyz %d diag.xyz spin torque force potential virial precision double" % steps,
                    "run %d" % steps]
    if key == "r22_magnon_analysis":
        return b + ["dump_xyz 1 traj.xyz spin precision double",
                    "run %d" % steps]
    raise KeyError(key)


CONFIGS = ["r21_baseline", "r22_plain", "r22_me_smooth",
           "r22_field_work", "r22_spin_lattice_diag", "r22_magnon_analysis"]


def cpu_dsf_time(d, steps):
    """CPU-only post-processing: parse the dumped spin trajectory and compute
    a q=0 dynamic structure factor.  Timing is reported SEPARATELY from GPU."""
    t0 = time.perf_counter()
    path = os.path.join(d, "traj.xyz")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return 0.0, 0
    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    frames = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        fields = lines[i + 1].split("Properties=", 1)[1].split()[0].split(":")
        off, k = {}, 0
        for j in range(0, len(fields) - 2, 3):
            off[fields[j]] = (k, int(fields[j + 2])); k += int(fields[j + 2])
        o, nc = off["spin"]
        a = np.array([[float(x) for x in l.split()[1:]]
                      for l in lines[i + 2:i + 2 + n]])
        frames.append(a[:, o:o + nc])
        i += 2 + n
    if len(frames) < 8:
        return time.perf_counter() - t0, len(frames)
    S = np.array(frames, float)
    S = S - S.mean(axis=0, keepdims=True)
    w = np.hanning(len(S))
    spec = np.abs(np.fft.fftshift(np.fft.fft(S * w[:, None, None], axis=0),
                                  axes=0)) ** 2
    _ = spec.sum()
    return time.perf_counter() - t0, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r21-bin", required=True)
    ap.add_argument("--r22-bin", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--json", default="")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sizes", default="216,432,1024,2048")
    a = ap.parse_args()
    r21 = os.path.join(a.r21_bin, "gpumd.exe")
    r22 = os.path.join(a.r22_bin, "gpumd.exe")
    for p in (r21, r22):
        if not os.path.isfile(p):
            print("FATAL missing binary", p)
            return 90
    exes = {"r21_baseline": r21}
    for k in CONFIGS[1:]:
        exes[k] = r22
    binsha = {p: sha256(p) for p in set(exes.values())}
    os.makedirs(a.work, exist_ok=True)
    sizes = [int(x) for x in a.sizes.split(",")]
    rows = []
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                            "--format=csv,noheader"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        gpu = p.stdout.decode().strip()
    except Exception:
        gpu = "UNKNOWN"
    print("R22 performance: gpu=%s steps=%d repeats=%d sizes=%s" %
          (gpu, a.steps, a.repeats, sizes))
    print("r21 sha=%s  r22 sha=%s" % (binsha[r21][:16], binsha[r22][:16]))

    for N in sizes:
        # Prepare ONE model per size, shared by all configurations/repeats.
        prep = os.path.join(a.work, "model_%d" % N)
        os.makedirs(prep, exist_ok=True)
        tiled_model(os.path.join(prep, "model.xyz"), N)
        shutil.copy(NEP, os.path.join(prep, "nep.txt"))
        print("\n== N=%d ==" % N, flush=True)
        for rep in range(a.repeats):
            # ALTERNATING order: reverse configuration order on odd repeats.
            order = CONFIGS if rep % 2 == 0 else list(reversed(CONFIGS))
            for key in order:
                d = os.path.join(a.work, "run_%d_%s_%d" % (N, key, rep))
                os.makedirs(d, exist_ok=True)
                for f in ("model.xyz", "nep.txt"):
                    shutil.copy(os.path.join(prep, f), os.path.join(d, f))
                for out in ("thermo.out", "traj.xyz", "diag.xyz",
                            "spin_work.out", "spin_energy_ledger.out", "stdout.txt"):
                    with open(os.path.join(d, out), "w", encoding="utf-8"):
                        pass
                lines = config_lines(key, a.steps)
                with open(os.path.join(d, "run.in"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                stop = threading.Event()
                peak = [0.0]
                th = threading.Thread(target=gpu_mem_sampler, args=(stop, peak),
                                      daemon=True)
                th.start()
                t0 = time.perf_counter()
                proc = subprocess.run([exes[key]], cwd=d,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
                wall = time.perf_counter() - t0
                stop.set(); th.join(timeout=2)
                txt = proc.stdout.decode("utf-8", "replace")
                with open(os.path.join(d, "stdout.txt"), "w", encoding="utf-8") as f:
                    f.write(txt)
                m = re.findall(r"Speed of this run =\s*([0-9.eE+]+)", txt)
                reported = float(m[-1]) if m else float("nan")
                cpu_t, nframes = 0.0, 0
                if key == "r22_magnon_analysis":
                    cpu_t, nframes = cpu_dsf_time(d, a.steps)
                rows.append(dict(
                    version=key, binary_sha256=binsha[exes[key]], gpu=gpu,
                    model="fege.nep.txt (FeGe B20, spin NEP)", atom_count=N,
                    steps=a.steps, dt_fs=0.05, repeat=rep,
                    enabled_features=key, wall_time_s="%.6f" % wall,
                    atoms_steps_per_s_measured="%.4f" % (N * a.steps / wall),
                    atoms_steps_per_s_reported="%.4f" % reported,
                    gpu_peak_memory_mib="%.1f" % peak[0],
                    cpu_postprocess_s="%.6f" % cpu_t,
                    magnon_frames=nframes, exit_code=proc.returncode))
                print("  rep%d %-24s wall=%7.3fs speed=%10.0f peak=%6.0fMiB rc=%d" %
                      (rep, key, wall, N * a.steps / wall, peak[0],
                       proc.returncode), flush=True)

    # Aggregate: mean/std/speed ratio to r21 for each (N, version).
    agg = {}
    for r in rows:
        k = (r["atom_count"], r["version"])
        agg.setdefault(k, []).append(float(r["atoms_steps_per_s_measured"]))
    for r in rows:
        base = np.mean(agg[(r["atom_count"], "r21_baseline")])
        v = np.array(agg[(r["atom_count"], r["version"])])
        r["mean_speed"] = "%.4f" % v.mean()
        r["speed_std"] = "%.4f" % v.std(ddof=0)
        r["speed_ratio_to_r21"] = "%.6f" % (v.mean() / base if base else float("nan"))
    cols = ["version", "binary_sha256", "gpu", "model", "atom_count", "steps",
            "dt_fs", "repeat", "enabled_features", "wall_time_s",
            "atoms_steps_per_s_measured", "atoms_steps_per_s_reported",
            "gpu_peak_memory_mib", "cpu_postprocess_s", "magnon_frames",
            "exit_code", "mean_speed", "speed_std", "speed_ratio_to_r21"]
    with open(a.csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote", a.csv, "rows", len(rows), flush=True)
    if a.json:
        summary = {}
        for (N, key), vals in agg.items():
            v = np.array(vals)
            base = np.mean(agg[(N, "r21_baseline")])
            summary["%d/%s" % (N, key)] = dict(
                mean_speed=float(v.mean()), std=float(v.std(ddof=0)),
                ratio_to_r21=float(v.mean() / base if base else float("nan")),
                repeats=len(v))
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(dict(gpu=gpu, steps=a.steps, repeats=a.repeats,
                           binary_sha256=binsha, summary=summary), f, indent=2)
        print("wrote", a.json, flush=True)
    bad = [r for r in rows if r["exit_code"] != 0]
    print("NONZERO_EXIT_RUNS=%d" % len(bad), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
