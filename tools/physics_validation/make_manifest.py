"""Hash every file of the PV1 stage into MANIFEST.sha256 (hash, size, path)."""
import hashlib, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STAGE = os.path.join(ROOT, "evidence", "stage_PV1")
rows = []
for dirpath, dirnames, filenames in os.walk(STAGE):
    for fn in sorted(filenames):
        if fn == "MANIFEST.sha256":
            continue
        fp = os.path.join(dirpath, fn)
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        rel = os.path.relpath(fp, STAGE).replace("\\", "/")
        rows.append((h.hexdigest(), os.path.getsize(fp), rel))
rows.sort(key=lambda r: r[2])
out = os.path.join(STAGE, "MANIFEST.sha256")
with open(out, "w") as f:
    for h, n, rel in rows:
        f.write(f"{h}  {n:>10d}  {rel}\n")
print(f"{len(rows)} files -> {out}")
