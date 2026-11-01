import os, subprocess, time, shutil
EXE = r"D:/researche_test/rongkunchen/GPUMD-spin/V3/final_delivery_20261101_r18/win_staging/bin/gpumd.exe"
HEAD = "potential nep.txt\ntime_step 0.05 100\nvelocity 300 seed 1\n"
CASES = {
 "G_nve_spinT0": HEAD+"spin_llg gamma=3.0385 alpha=0.3 T=300 seed=1\ndump_thermo 100\nensemble nvt_lan 300 300 100\nrun 3000\nspin_llg gamma=3.0385 alpha=0.3 T=0 seed=1\nensemble nve\nrun 12000\n",
 "I_nvt_ber":    HEAD+"spin_llg gamma=3.0385 alpha=0.3 T=300 seed=1\ndump_thermo 100\nensemble nvt_ber 300 300 100\nrun 3000\nensemble nve\nrun 12000\n",
 "J_long_nve":   HEAD+"spin_llg gamma=3.0385 alpha=0.3 T=300 seed=1\ndump_thermo 100\nensemble nvt_lan 300 300 100\nrun 3000\nensemble nve\nrun 60000\n",
 "K_dt01":       "potential nep.txt\ntime_step 0.1 100\nvelocity 300 seed 1\nspin_llg gamma=3.0385 alpha=0.3 T=300 seed=1\ndump_thermo 100\nensemble nvt_lan 300 300 100\nrun 3000\nensemble nve\nrun 12000\n",
}
for name, txt in CASES.items():
    d = name; os.makedirs(d, exist_ok=True)
    shutil.copyfile("N432/model.xyz", os.path.join(d,"model.xyz"))
    shutil.copyfile("N432/nep.txt", os.path.join(d,"nep.txt"))
    open(os.path.join(d,"run.in"),"w").write(txt)
    for j in ("thermo.out","stdout.txt","SPIN_GUARD_STOP"):
        p=os.path.join(d,j)
        if os.path.exists(p): os.remove(p)
    t0=time.time()
    with open(os.path.join(d,"stdout.txt"),"w") as fh:
        try:
            rc=subprocess.run([EXE],cwd=d,stdout=fh,stderr=subprocess.STDOUT,timeout=240).returncode
        except subprocess.TimeoutExpired: rc="TIMEOUT"
    el=time.time()-t0
    th=os.path.join(d,"thermo.out")
    nl=len(open(th).readlines()) if os.path.exists(th) else 0
    print(f"{name:14s} rc={str(rc):8s} {el:6.1f}s thermo_lines={nl}", flush=True)
