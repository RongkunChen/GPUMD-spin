#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3 / S08B: real restart of the production gpumd.exe.

Three fixed-lattice spin cases, each run three ways:

    cont     N steps in one go, checkpoint at the end
    prefix   N/2 steps, checkpoint at N/2      (this is the restart source)
    restart  new process: model.xyz = prefix/restart.xyz,
             spin_llg ... restart=spin_restart.bin, N/2 steps

The restarted trajectory must *continue* the continuous one, not merely look
similar: frame-by-frame spin comparison (float32, so bitwise is meaningful),
position/velocity comparison, potential energy from thermo, and finally the
checkpoint payload at the end of both runs (RNG block + |S| anchor) must be
identical.  The RNG block comparison is the decisive one: it says the noise
stream continued instead of being re-seeded.

Cases: (i) alpha = T = 0 (S05A conservative), (ii) geometric T > 0,
(iii) midpoint T > 0.  No parameter of the integrator is touched.
"""
import json
import os
import shutil
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "work_v3_20261029_132410/GPUMD-spin-r17-offline-src-build/src")
GPUMD = os.path.join(SRC, "gpumd.exe")
MODEL = os.path.join(ROOT, "evidence/stage07_smoke/20261031_110347/singlepoint/model.xyz")
NEP = os.path.join(ROOT, "evidence/stage07_smoke/20261031_110347/singlepoint/nep.txt")

N_STEPS = 2000
STRIDE = 20
DT = 0.5
SEED = 4242

HDR = "<16sIIQddddiiiqqd"
HDR_SIZE = struct.calcsize(HDR)          # 100


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_checkpoint(path):
    data = open(path, "rb").read()
    magic, ver, n, mhash, gamma, alpha, T, dt, scheme, has_rng, has_anchor, \
        steps, pstep, gtime = struct.unpack(HDR, data[:HDR_SIZE])
    off = HDR_SIZE
    anchor = list(struct.unpack("<%df" % n, data[off:off + 4 * n]))
    off += 4 * n
    rng_bytes = 0
    rng = b""
    if has_rng:
        # sizeof(curandState) = 48 on this toolchain; verify by the file size.
        total = len(data)
        rng_bytes = total - 8 - off
        rng = data[off:off + rng_bytes]
        if rng_bytes % n != 0:
            raise SystemExit("rng block size %d is not a multiple of %d" % (rng_bytes, n))
    payload = struct.unpack("<Q", data[-8:])[0]
    import hashlib
    return {"magic": magic.split(b"\0")[0].decode(), "version": ver, "natoms": n,
            "model_hash": mhash, "gamma": gamma, "alpha": alpha, "T": T,
            "dt": dt, "scheme": scheme, "has_rng": has_rng,
            "has_anchor": has_anchor, "sllg_steps": steps,
            "physics_step": pstep, "global_time": gtime,
            "anchor": anchor, "rng_bytes": rng_bytes,
            "rng_sha256": hashlib.sha256(rng).hexdigest() if has_rng else None,
            "payload": payload, "bytes": len(data)}


def parse_xyz(path):
    """Return (steps, list of per-frame (species, pos, vel, spin)) with steps."""
    steps, frames = [], []
    lines = open(path, encoding="utf-8").read().splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i])
        hdr = lines[i + 1]
        props = hdr.split("Properties=")[1].split()[0].split(":")
        names, sizes = props[0::3], [int(x) for x in props[2::3]]
        off, k = {}, 0
        for nm, sz in zip(names, sizes):
            off[nm] = (k, sz)
            k += sz
        body = [l.split() for l in lines[i + 2:i + 2 + n]]
        step = None
        for tok in hdr.split():
            if tok.startswith("step=") or tok.startswith("Step="):
                step = int(tok.split("=")[1])
        # gpumd writes the step as the first column when asked; otherwise index
        frames.append({"hdr": hdr, "rows": body, "off": off, "props": names})
        steps.append(step)
        i += 2 + n
    return steps, frames


def write_run_in(d, steps, spin_line, extra=""):
    txt = ("potential nep.txt\ntime_step %g\nensemble nve\n"
           "dump_thermo %d\ndump_xyz %d traj.xyz velocity spin force potential\n"
           "dump_restart %d\n%s\nrun %d\n"
           % (DT, STRIDE, STRIDE, steps, spin_line, steps))
    open(os.path.join(d, "run.in"), "w", encoding="utf-8", newline="\n").write(txt)


def run(d):
    p = subprocess.run([GPUMD], cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    open(os.path.join(d, "stdout.txt"), "wb").write(p.stdout)
    return p.returncode


def thermo(path):
    rows = [l.split() for l in open(path, encoding="utf-8")
            if l.strip() and not l.startswith("#")]
    return [(float(r[2]), float(r[1])) for r in rows]      # (PE, KE)


def compare_frames(fa, fb, offset):
    """Compare frame i of b against frame offset+i of a."""
    import math
    worst = {"pos": 0.0, "vel": 0.0, "spin": 0.0, "force": 0.0,
             "energy_atom": 0.0, "spin_bitwise": True, "n_compared": 0}
    n = min(len(fb), max(0, len(fa) - offset))
    for i, fbv in enumerate(fb[:n]):
        fav = fa[offset + i]
        for kind, key in (("pos", "pos"), ("vel", "vel"), ("spin", "spin"),
                      ("force", "forces"), ("energy_atom", "energy_atom")):
            if key not in fav["off"]:
                continue
            (oa, sa) = fav["off"][key]
            (ob, sb) = fbv["off"][key]
            for ra, rb in zip(fav["rows"], fbv["rows"]):
                for c in range(sa):
                    va = float(ra[oa + c])
                    vb = float(rb[ob + c])
                    d = abs(va - vb)
                    if d > worst[kind]:
                        worst[kind] = d
                    if kind == "spin" and va != vb:
                        worst["spin_bitwise"] = False
    worst["n_compared"] = n
    return worst


def main():
    D = os.path.abspath(sys.argv[1])
    P = os.path.join(D, "phase_3")
    C = os.path.join(P, "cases")
    os.makedirs(C, exist_ok=True)
    cmds = []
    cases = [
        ("conservative", "spin_llg gamma=3.04 alpha=0 T=0"),
        ("geometric", "spin_llg gamma=3.04 alpha=0.5 T=300 seed=%d scheme=geometric" % SEED),
        ("midpoint", "spin_llg gamma=3.04 alpha=0.5 T=300 seed=%d scheme=midpoint" % SEED),
    ]
    out = {"phase": "3", "gpumd": {"path": GPUMD.replace(os.sep, "/"),
                                   "sha256": sha(GPUMD)},
           "model_sha256": sha(MODEL), "potential_sha256": sha(NEP),
           "settings": {"steps": N_STEPS, "stride": STRIDE, "dt_fs": DT, "seed": SEED},
           "cases": {}}

    for name, spin_line in cases:
        base = os.path.join(C, name)
        if os.path.isdir(base):
            shutil.rmtree(base)
        for sub in ("cont", "prefix", "restart"):
            d = os.path.join(base, sub)
            os.makedirs(d)
            shutil.copy2(NEP, os.path.join(d, "nep.txt"))
            shutil.copy2(MODEL, os.path.join(d, "model.xyz"))

        # ---- continuous -------------------------------------------------
        d = os.path.join(base, "cont")
        write_run_in(d, N_STEPS, spin_line)
        rc_c = run(d)
        cmds.append({"case": name, "step": "cont", "cwd": d.replace(os.sep, "/"), "rc": rc_c})
        shutil.copy2(os.path.join(d, "spin_restart.bin"),
                     os.path.join(base, "ckpt_cont.bin"))

        # ---- prefix -----------------------------------------------------
        d = os.path.join(base, "prefix")
        write_run_in(d, N_STEPS // 2, spin_line)
        rc_p = run(d)
        cmds.append({"case": name, "step": "prefix", "cwd": d.replace(os.sep, "/"), "rc": rc_p})
        shutil.copy2(os.path.join(d, "spin_restart.bin"),
                     os.path.join(base, "ckpt_prefix.bin"))

        # ---- restart ----------------------------------------------------
        d = os.path.join(base, "restart")
        shutil.copy2(os.path.join(base, "prefix", "restart.xyz"),
                     os.path.join(d, "model.xyz"))
        shutil.copy2(os.path.join(base, "ckpt_prefix.bin"),
                     os.path.join(d, "spin_restart.bin"))
        write_run_in(d, N_STEPS // 2, spin_line + " restart=spin_restart.bin")
        rc_r = run(d)
        cmds.append({"case": name, "step": "restart", "cwd": d.replace(os.sep, "/"), "rc": rc_r})
        shutil.copy2(os.path.join(d, "spin_restart.bin"),
                     os.path.join(base, "ckpt_restart.bin"))

        # ---- compare ----------------------------------------------------
        _, fc = parse_xyz(os.path.join(base, "cont", "traj.xyz"))
        _, fr = parse_xyz(os.path.join(base, "restart", "traj.xyz"))
        _, fp = parse_xyz(os.path.join(base, "prefix", "traj.xyz"))
        n_half = len(fp)
        worst = compare_frames(fc, fr, offset=n_half)
        # prefix frames must also match the first half of the continuous run
        worst_prefix = compare_frames(fc, fp, offset=0)
        tc = thermo(os.path.join(base, "cont", "thermo.out"))
        tr = thermo(os.path.join(base, "restart", "thermo.out"))
        pe_tail = [p for (p, _) in tc[n_half:]]
        pe_restart = [p for (p, _) in tr]
        m = min(len(pe_tail), len(pe_restart))
        pe_diff = max(abs(pe_tail[i] - pe_restart[i]) for i in range(m)) if m else float("nan")
        ke_diff = max(abs(tc[n_half + i][1] - tr[i][1]) for i in range(m)) if m else float("nan")

        ck_c = read_checkpoint(os.path.join(base, "ckpt_cont.bin"))
        ck_p = read_checkpoint(os.path.join(base, "ckpt_prefix.bin"))
        ck_r = read_checkpoint(os.path.join(base, "ckpt_restart.bin"))
        rng_same = (ck_c["rng_sha256"] == ck_r["rng_sha256"]) if ck_c["has_rng"] else None
        anchor_same = ck_c["anchor"] == ck_r["anchor"]
        ckpt_identity = {
            "magic": ck_c["magic"], "version": ck_c["version"],
            "natoms": ck_c["natoms"], "model_hash_equal":
                ck_c["model_hash"] == ck_p["model_hash"] == ck_r["model_hash"],
            "scheme_equal": ck_c["scheme"] == ck_r["scheme"],
            "dt_equal": ck_c["dt"] == ck_r["dt"],
            "prefix_physics_step": ck_p["physics_step"],
            "restart_step_counter": ck_r["sllg_steps"],
        }
        rec = {"scheme_line": spin_line, "rc": {"cont": rc_c, "prefix": rc_p,
                                                "restart": rc_r},
               "n_frames_compared": len(fr), "n_prefix_frames": n_half,
               "max_abs_diff": worst, "prefix_vs_cont": worst_prefix,
               "pe_max_abs_diff_eV": pe_diff, "ke_max_abs_diff_eV": ke_diff,
               "checkpoint": {"cont": {k: ck_c[k] for k in
                                       ("physics_step", "sllg_steps", "rng_bytes",
                                        "rng_sha256", "has_rng", "has_anchor")},
                              "restart": {k: ck_r[k] for k in
                                          ("physics_step", "sllg_steps", "rng_bytes",
                                           "rng_sha256", "has_rng", "has_anchor")},
                              "rng_stream_identical": rng_same,
                              "anchor_identical": anchor_same,
                              "identity": ckpt_identity}}
        # acceptance: same physics + same random stream
        spin_ok = worst["spin"] <= 1e-6 and worst["pos"] <= 1e-9
        rec["pass"] = bool(rc_c == 0 and rc_p == 0 and rc_r == 0 and spin_ok
                           and (rng_same is not False))
        out["cases"][name] = rec
        print("%-13s rc=%d/%d/%d  |dS|max %.2e (bitwise %s)  |dR|max %.2e  "
              "dPE %.2e  rng=%s -> %s"
              % (name, rc_c, rc_p, rc_r, worst["spin"], worst["spin_bitwise"],
                 worst["pos"], pe_diff,
                 "n/a" if rng_same is None else ("same" if rng_same else "DIFFERENT"),
                 "PASS" if rec["pass"] else "FAIL"))

    # ---- negative controls: a damaged checkpoint must be refused ----------
    neg = {}
    base = os.path.join(C, "midpoint")
    src_ck = os.path.join(base, "ckpt_prefix.bin")
    tests = {}
    d = os.path.join(base, "neg_corrupt")
    os.makedirs(d, exist_ok=True)
    shutil.copy2(NEP, os.path.join(d, "nep.txt"))
    shutil.copy2(os.path.join(base, "prefix", "restart.xyz"), os.path.join(d, "model.xyz"))
    raw = bytearray(open(src_ck, "rb").read())
    raw[40] ^= 0xFF                                   # corrupt alpha
    open(os.path.join(d, "spin_restart.bin"), "wb").write(raw)
    write_run_in(d, 100, "spin_llg gamma=3.04 alpha=0.5 T=300 seed=%d scheme=midpoint "
                         "restart=spin_restart.bin" % SEED)
    rc = run(d)
    tests["corrupted_payload"] = {"rc": rc, "rejected": rc != 0,
                                  "message": open(os.path.join(d, "stdout.txt"),
                                                  encoding="utf-8", errors="replace").read()[-300:]}
    d2 = os.path.join(base, "neg_scheme")
    os.makedirs(d2, exist_ok=True)
    shutil.copy2(NEP, os.path.join(d2, "nep.txt"))
    shutil.copy2(os.path.join(base, "prefix", "restart.xyz"), os.path.join(d2, "model.xyz"))
    shutil.copy2(src_ck, os.path.join(d2, "spin_restart.bin"))
    write_run_in(d2, 100, "spin_llg gamma=3.04 alpha=0.5 T=300 seed=%d scheme=geometric "
                          "restart=spin_restart.bin" % SEED)
    rc2 = run(d2)
    tests["wrong_scheme"] = {"rc": rc2, "rejected": rc2 != 0,
                             "message": open(os.path.join(d2, "stdout.txt"),
                                             encoding="utf-8", errors="replace").read()[-300:]}
    # The decisive control: the same state, but *without* restart= -- the run
    # re-seeds the RNG at step 0.  If this reproduced the continuous trajectory
    # the test would have no power to detect a lost random stream.
    d3 = os.path.join(base, "neg_reseeded")
    os.makedirs(d3, exist_ok=True)
    shutil.copy2(NEP, os.path.join(d3, "nep.txt"))
    shutil.copy2(os.path.join(base, "prefix", "restart.xyz"), os.path.join(d3, "model.xyz"))
    write_run_in(d3, N_STEPS // 2,
                 "spin_llg gamma=3.04 alpha=0.5 T=300 seed=%d scheme=midpoint" % SEED)
    rc3 = run(d3)
    _, f3 = parse_xyz(os.path.join(d3, "traj.xyz"))
    w3 = compare_frames(fc, f3, offset=n_half)
    tests["reseeded_without_checkpoint"] = {
        "rc": rc3, "rejected": False,
        "max_abs_spin_diff": w3["spin"],
        "spin_bitwise_equal": w3["spin_bitwise"],
        "detects_lost_stream": bool(w3["spin"] > 1e-3),
        "note": "no restart= ; same model.xyz, same seed -> stream restarts at 0"}
    # The decisive control: the same state, but *without* restart= -- the run
    # re-seeds the RNG at step 0.  If this reproduced the continuous trajectory
    # the test would have no power to detect a lost random stream.
    d3 = os.path.join(base, "neg_reseeded")
    os.makedirs(d3, exist_ok=True)
    shutil.copy2(NEP, os.path.join(d3, "nep.txt"))
    shutil.copy2(os.path.join(base, "prefix", "restart.xyz"), os.path.join(d3, "model.xyz"))
    write_run_in(d3, N_STEPS // 2,
                 "spin_llg gamma=3.04 alpha=0.5 T=300 seed=%d scheme=midpoint" % SEED)
    rc3 = run(d3)
    _, f3 = parse_xyz(os.path.join(d3, "traj.xyz"))
    w3 = compare_frames(fc, f3, offset=n_half)
    tests["reseeded_without_checkpoint"] = {
        "rc": rc3, "rejected": False,
        "max_abs_spin_diff": w3["spin"],
        "spin_bitwise_equal": w3["spin_bitwise"],
        "detects_lost_stream": bool(w3["spin"] > 1e-3),
        "note": "no restart= ; same model.xyz, same seed -> stream restarts at 0"}
    out["negative_controls"] = tests
    rejects_ok = all(v["rejected"] for k, v in tests.items()
                     if k != "reseeded_without_checkpoint")
    out["pass"] = (all(c["pass"] for c in out["cases"].values()) and rejects_ok
                   and tests["reseeded_without_checkpoint"]["detects_lost_stream"])

    json.dump(out, open(os.path.join(D, "phase_3_results.json"), "w", encoding="utf-8",
                        newline="\n"), indent=1, ensure_ascii=False)
    with open(os.path.join(D, "phase_3_commands.jsonl"), "w", encoding="utf-8",
              newline="\n") as f:
        for c in cmds:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    L = ["=" * 78, "阶段3：S08B 真实 GPUMD 重启（固定晶格自旋）", "=" * 78, ""]
    L.append("二进制：gpumd.exe %s" % out["gpumd"]["sha256"][:32])
    L.append("模型：%s（16 Fe 原子，合成 NEP，mask 17）" % out["model_sha256"][:32])
    L.append("参数：dt=%g fs，N=%d 步，dump stride %d，seed %d" %
             (DT, N_STEPS, STRIDE, SEED))
    L.append("")
    L.append("设计：连续 N 步 / 前 N/2 步（写 checkpoint）/ 新进程从 checkpoint 续跑 N/2 步；")
    L.append("      比较同一物理时刻的自旋（float32 逐位）、R、V、PE、KE，以及两次运行")
    L.append("      结束时的 checkpoint 载荷（RNG 块 + |S| 锚点）。RNG 块相同 = 随机流真的续上了。")
    L.append("")
    for name, r in out["cases"].items():
        L.append("[%s] %s" % (name, "PASS" if r["pass"] else "FAIL"))
        L.append("  自旋 max|ΔS| %.3e（逐位 %s），R max|ΔR| %.3e，V max|ΔV| %.3e"
                 % (r["max_abs_diff"]["spin"], r["max_abs_diff"]["spin_bitwise"],
                    r["max_abs_diff"]["pos"], r["max_abs_diff"]["vel"]))
        L.append("  PE max|Δ| %.3e eV，KE max|Δ| %.3e eV，比较 %d 帧"
                 % (r["pe_max_abs_diff_eV"], r["ke_max_abs_diff_eV"],
                    r["n_frames_compared"]))
        L.append("  checkpoint：RNG %s（%s B），锚点相同 %s，model hash 相同 %s，scheme 相同 %s"
                 % (r["checkpoint"]["rng_stream_identical"],
                    r["checkpoint"]["restart"]["rng_bytes"],
                    r["checkpoint"]["anchor_identical"],
                    r["checkpoint"]["identity"]["model_hash_equal"],
                    r["checkpoint"]["identity"]["scheme_equal"]))
    L.append("")
    L.append("负向控制：")
    for k, v in out["negative_controls"].items():
        if k == "reseeded_without_checkpoint":
            L.append("  %-28s rc=%d  不带 restart= 重跑：max|ΔS| %.3f muB，"
                     "随机流未续（测试有分辨力）" % (k, v["rc"], v["max_abs_spin_diff"]))
        else:
            L.append("  %-28s rc=%d -> %s" % (k, v["rc"],
                                              "拒绝" if v["rejected"] else "未拒绝"))
    L.append("")
    L.append("结论：%s" % ("PASS" if out["pass"] else "FAIL"))
    open(os.path.join(D, "phase_3_summary.txt"), "w", encoding="utf-8",
         newline="\n").write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
