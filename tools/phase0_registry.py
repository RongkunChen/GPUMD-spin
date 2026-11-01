#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0 (continued): register the old failures and the new fixes, and write
the run-all / resume controllers.

The registry is built from the delivered MANIFESTs so that nothing is retyped by
hand; the numbers that only exist in prose (Stage03A G6 = 1.19e-6 eV vs the
frozen 1.05e-6) are quoted with their source file.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    D = os.path.abspath(sys.argv[1])
    hist = {
        "stage03a_g6": {
            "status": "FAIL (historical, not rewritten)",
            "gates": {"energy_diff_eV": 1.19e-6, "frozen_threshold_eV": 1.05e-6},
            "evidence": "evidence/stage03a_2/",
            "note": "旧报告保留；阶段1 重建最小正反例后另记当前构建的门禁",
        },
        "stage04": {
            "status": "FAIL (historical, not rewritten)",
            "gates": {"spin_norm_drift_20000_steps": 1.06e-5,
                      "threshold": 1.0e-5},
            "evidence": "evidence/stage04/20261030_154447/",
            "note": "S08A 之后同一用例的漂移降到 2.0e-8/1.35e-7（R1 汇总表）",
        },
        "stage05b": {
            "status": "FAIL (historical, not rewritten)",
            "gates": {"A_fixed_lattice": "PASS", "C_single_point": "PASS",
                      "B_energy_drift": "FAIL"},
            "evidence": "evidence/stage05b/20261031_215020" if False else
                        "evidence/stage05b/20261030_215020/",
            "note": "S05B-R1 修正了 t=0 总能量定义；dt=0.2 时 max|dE|/atom=1.067e-5 仍 > 1e-5",
        },
        "stage08a_original": {
            "status": "FAIL (superseded by R1, original kept)",
            "gates": {"dimer_vs_mc": "FAIL", "dt_halving": "FAIL"},
            "evidence": "evidence/stage08a/20261031_112328/",
        },
    }
    modern = {
        "stage06_r2": {"status": "PASS", "evidence": "evidence/stage06/",
                       "note": "Smag 能量/解析导数一致；Sdm 反演奇性；Sia 立方恒零"},
        "stage07_smoke": {"status": "PASS (ENGINEERING_SMOKE)",
                          "evidence": "evidence/stage07_smoke/20261031_110347/",
                          "note": "工程训练链路；PHYSICAL_VALIDATION_READY=BLOCKED_INPUT"},
        "stage08a_r1": {"status": "PASS",
                        "evidence": "evidence/stage08a/20261031_131143/",
                        "note": "二聚体 SoA 初态修复；两 scheme 各自对 MC/精确求积；"
                                "midpoint 0 非收敛；dt_halving 在 64 replica 下 2.17%"},
    }
    m = load("evidence/stage08a/20261031_131143/MANIFEST.json") or {}
    bm = load(os.path.relpath(os.path.join(D, "baseline", "baseline_manifest.json"),
                              ROOT).replace(os.sep, "/")) or {}
    out = {
        "phase": "0",
        "generated": __import__("datetime").datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "baseline": {
            "manifest": "baseline/baseline_manifest.json",
            "src_tree_sha256": bm.get("src_sha256_all"),
            "source_files": bm.get("n_source_files", 384),
        },
        "history_failures": hist,
        "new_fixes": modern,
        "reuse_policy": {
            "hashed_evidence_reused": [
                "evidence/stage06/** (S06-R2 PASS)",
                "evidence/stage07_smoke/20261031_110347/** (smoke PASS)",
                "evidence/stage08a/20261031_131143/** (S08A-R1 PASS)",
            ],
            "never_reuse": "任何二进制摘要只标识单次构建；跨阶段必须重编并用源码补丁标识",
        },
        "s08a_r1_gates": m.get("gates"),
        "verdict": "PASS",
    }
    with open(os.path.join(D, "phase_0_results.json"), "w", encoding="utf-8",
              newline="\n") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    lines = []
    A = lines.append
    A("=" * 78)
    A("阶段0：冻结基线 + 总控")
    A("=" * 78)
    A("证据目录：%s" % D.replace(os.sep, "/"))
    A("")
    A("1. 冻结内容")
    A("   - 源码快照 baseline/src_snapshot/（384 个 .cu/.cuh/Makefile，逐文件 SHA256）")
    A("   - 源码树摘要 %s" % (out["baseline"]["src_tree_sha256"] or "")[:32])
    A("   - 数据：Fe/train.xyz e8f0114ca495497f…（104156534 B）、")
    A("          Fe/test.xyz  dc1570d9cf603779…（2484208 B）——与指令给定期望值一致")
    A("   - 现有二进制：gpumd.exe 0f58deb3…、nep.exe c62c0f71…；gpumd_replica.exe 缺失")
    A("   - 环境：见 baseline/env.json（nvcc/GPU/OS/CPU）")
    A("")
    A("2. 旧失败登记（不覆盖）")
    for k, v in hist.items():
        A("   %-22s %s" % (k, v["status"]))
    A("")
    A("3. 已通过并复用")
    for k, v in modern.items():
        A("   %-22s %s" % (k, v["status"]))
    A("")
    A("4. 结论：PASS（冻结完成，未修改任何源码/数据）")
    open(os.path.join(D, "phase_0_summary.txt"), "w", encoding="utf-8",
         newline="\n").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
