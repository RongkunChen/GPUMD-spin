"""F-25 verification: does a single run.in with two `run` blocks work?

Checks, for N = 250 / 432 / 1024:
  * NVT -> NVE in one file finishes (no hang) and writes the full number of
    thermo lines (the old binary only wrote the first block's share),
  * the trajectory keeps growing in the second block,
  * the record is continuous across the switch (no jump in T / PE).
Controls: NVT->NVT, NVE->NVE, single-block NVT and NVE.
"""
import os, subprocess, sys, time, shutil

EXE = sys.argv[1]
TAG = sys.argv[2]
HEAD = "potential nep.txt\ntime_step 0.05 100\nvelocity 300 seed 1\n"
SPIN = "spin_llg gamma=3.0385 alpha=0.3 T=300 seed=1\n"
NVT = "ensemble nvt_lan 300 300 100\n"
NVE = "ensemble nve\n"
BLOCKS = {  # name -> (steps_block1, steps_block2, ensemble1, ensemble2)
    "A_nvt_nve": (3000, 12000, NVT, NVE),
    "B_nvt_nvt": (3000, 12000, NVT, NVT),
    "D_nvt_only": (15000, 0, NVT, ""),
    "E_nve_only": (15000, 0, NVE, ""),
}


def run(sysdir, name, nb1, nb2, e1, e2, dump_xyz=False, timeout=300):
    d = os.path.join(TAG, sysdir, name)
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(os.path.join(sysdir, "model.xyz"), os.path.join(d, "model.xyz"))
    shutil.copyfile(os.path.join(sysdir, "nep.txt"), os.path.join(d, "nep.txt"))
    txt = HEAD + SPIN + "dump_thermo 100\n"
    if dump_xyz:
        txt += "dump_xyz 500 traj.xyz\n"
    txt += e1 + f"run {nb1}\n"
    if nb2:
        txt += e2 + f"run {nb2}\n"
    open(os.path.join(d, "run.in"), "w").write(txt)
    for j in ("thermo.out", "stdout.txt", "traj.xyz", "SPIN_GUARD_STOP"):
        p = os.path.join(d, j)
        if os.path.exists(p):
            os.remove(p)
    t0 = time.time()
    with open(os.path.join(d, "stdout.txt"), "w") as fh:
        try:
            rc = subprocess.run([EXE], cwd=d, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
    el = time.time() - t0
    th = os.path.join(d, "thermo.out")
    data = []
    if os.path.exists(th):
        for line in open(th):
            if not line.startswith("#") and line.strip():
                data.append([float(x) for x in line.split()])
    nframes = 0
    if dump_xyz and os.path.exists(os.path.join(d, "traj.xyz")):
        nframes = sum(1 for l in open(os.path.join(d, "traj.xyz"))
                      if l.startswith("432") or l.startswith("250") or l.startswith("1024"))
    return dict(rc=rc, el=round(el, 1), ndump=len(data), nframes=nframes,
                T=data[-1][0] if data else None, PE=data[-1][2] if data else None)


def main():
    for sysdir in ("N250", "N432", "N1024"):
        print(f"=== {sysdir} ({EXE.split('/')[-1]}, {TAG}) ===", flush=True)
        for name, (b1, b2, e1, e2) in BLOCKS.items():
            r = run(sysdir, name, b1, b2, e1, e2)
            expect = (b1 + b2) // 100
            ok = "OK " if (r["rc"] == 0 and abs(r["ndump"] - expect) <= 2) else "BAD"
            print(f"  {ok} {name:12s} rc={str(r['rc']):8s} {r['el']:6.1f}s "
                  f"dumps={r['ndump']:4d}/{expect:4d}  T_end={r['T']}", flush=True)
        # trajectory continuity across the switch
        r = run(sysdir, "A_traj", 3000, 6000, NVT, NVE, dump_xyz=True)
        print(f"  traj: rc={r['rc']} frames={r['nframes']} (expect 18)", flush=True)


if __name__ == "__main__":
    main()
