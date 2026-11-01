"""Generate bcc Fe cells for the E(a) or E(|S|) scan."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "tools", "physics_validation"))
import make_fe_system as G

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["lattice", "moment"], default="lattice")
ap.add_argument("--out", default="cells")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

if a.mode == "lattice":
    grid = [(round(x, 3), 2.2) for x in
            [2.750, 2.775, 2.800, 2.825, 2.850, 2.875, 2.900]]
else:
    grid = [(2.8258, s) for s in [2.0, 2.2, 2.4, 2.6, 2.8]]

for i, (lat, s) in enumerate(grid):
    fr = G.bcc(2, 2, 2, lat)          # 16 atoms
    spins, _ = G.set_spins(fr, "fm", s, 1)
    G.write_xyz(os.path.join(a.out, f"cell_{i:02d}.xyz"), fr, spins)
    open(os.path.join(a.out, f"cell_{i:02d}.meta"), "w").write(f"{lat} {s}\n")
print(f"{len(grid)} cells -> {a.out}")
