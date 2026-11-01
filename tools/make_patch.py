#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write a unified diff between the phase-0 source snapshot and the working tree.

    make_patch.py <one_shot_dir> <out.patch> [relpath ...]

Paths in the patch are relative to the source tree root, so it applies with
`git -c core.autocrlf=false apply -p1` from .../GPUMD-spin-r17-offline-src-build/src.
Only files that actually differ are emitted; an empty result is an error here
(callers pass changes.note instead).
"""
import difflib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "work_v3_20261029_132410/GPUMD-spin-r17-offline-src-build/src")


def main():
    D = os.path.abspath(sys.argv[1])
    out_path = sys.argv[2]
    only = sys.argv[3:]
    snap = os.path.join(D, "baseline", "src_snapshot")
    files = []
    for base, _, names in os.walk(SRC):
        for n in names:
            if not n.endswith((".cu", ".cuh")) and n != "makefile" and n != "Makefile":
                continue
            rel = os.path.relpath(os.path.join(base, n), SRC)
            if only and rel.replace(os.sep, "/") not in only:
                continue
            files.append(rel)
    chunks = []
    for rel in sorted(files):
        cur = os.path.join(SRC, rel)
        old = os.path.join(snap, "src", rel)
        if not os.path.exists(old):
            continue
        a = open(old, encoding="utf-8", errors="replace", newline="").read().splitlines(keepends=True)
        b = open(cur, encoding="utf-8", errors="replace", newline="").read().splitlines(keepends=True)
        if a == b:
            continue
        d = list(difflib.unified_diff(a, b, fromfile="a/" + rel.replace(os.sep, "/"),
                                      tofile="b/" + rel.replace(os.sep, "/"), n=3))
        chunks.extend(d)
    text = "".join(chunks)
    if not text:
        print("no differences")
        return 1
    open(out_path, "w", encoding="utf-8", newline="\n").write(text)
    print("wrote %s (%d bytes, %d hunks, %d files)" %
          (out_path, len(text.encode("utf-8")), text.count("\n@@"),
           sum(1 for l in chunks if l.startswith("--- "))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
