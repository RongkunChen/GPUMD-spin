import os, subprocess, sys, time, shutil

EXE = r"D:/researche_test/rongkunchen/GPUMD-spin/V3/final_delivery_20261101_r18/win_staging/bin/gpumd.exe"
HEAD = "potential nep.txt\ntime_step 0.05 100\nvelocity 300 seed 1\nspin_llg gamma=3.0385 alpha=0.3 T=300 seed=1\n"
NVT = "ensemble nvt_lan 300 300 100\n"
NVE = "ensemble nve\n"
CASES = {
    "A_nvt_nve": HEAD + NVT + "run 3000\n" + NVE + "run 3000\n",
    "B_nvt_nvt": HEAD + NVT + "run 3000\n" + NVT + "run 3000\n",
    "C_nve_nve": HEAD + NVE + "run 3000\n" + NVE + "run 3000\n",
    "D_nvt_only": HEAD + NVT + "run 6000\n",
    "E_nve_only": HEAD + NVE + "run 6000\n",
    "F_nve_nvt": HEAD + NVE + "run 3000\n" + NVT + "run 3000\n",
}

def run(sysdir, name, txt, timeout=60):
    d = os.path.join(sysdir, name)
    os.makedirs(d, exist_ok=True)
    shutil.copyfile(os.path.join(sysdir, "model.xyz"), os.path.join(d, "model.xyz"))
    shutil.copyfile(os.path.join(sysdir, "nep.txt"), os.path.join(d, "nep.txt"))
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
    nlines = sum(1 for _ in open(th)) if os.path.exists(th) else 0
    return rc, round(el, 1), nlines

for sysdir in sys.argv[1:]:
    print(f"=== {sysdir} ===", flush=True)
    for name, txt in CASES.items():
        rc, el, nl = run(sysdir, name, txt)
        print(f"  {name:12s} rc={str(rc):8s} {el:6.1f}s  thermo_lines={nl}", flush=True)
