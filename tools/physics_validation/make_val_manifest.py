"""Re-hash the BCC Fe validation tree into MANIFEST.sha256.

The manifest next to the results is what makes "which numbers came from which
run" checkable; it has to be regenerated every time a battery adds files, so it
is a script and not a one-off command.  Excludes the manifest itself and any
in-progress temp files.

usage: python make_val_manifest.py [--dir validation/BCC_Fe_full_physics]
"""
import argparse
import hashlib
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("validation", "BCC_Fe_full_physics"))
    A = ap.parse_args()
    stage = os.path.join(ROOT, A.dir)
    rows = []
    for dirpath, _, filenames in os.walk(stage):
        for fn in sorted(filenames):
            if fn == "MANIFEST.sha256" or fn.endswith(".tmp.png"):
                continue
            fp = os.path.join(dirpath, fn)
            h = hashlib.sha256()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            rel = os.path.relpath(fp, stage).replace("\\", "/")
            rows.append((h.hexdigest(), os.path.getsize(fp), rel))
    rows.sort(key=lambda r: r[2])
    out = os.path.join(stage, "MANIFEST.sha256")
    with open(out, "w") as f:
        for h, n, rel in rows:
            f.write(f"{h}  {n:>10d}  {rel}\n")
    print(f"{len(rows)} files -> {out}")


if __name__ == "__main__":
    main()
