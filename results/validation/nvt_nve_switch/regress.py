import hashlib, os, shutil, subprocess, sys, time
OLD = r"D:/researche_test/rongkunchen/GPUMD-spin/V3/final_delivery_20261101_r18/win_staging/bin/gpumd.exe"
NEW = r"D:/researche_test/rongkunchen/GPUMD-spin/V3/work_f25/base/src/gpumd.exe"
CASES = {
  "single_nvt": "potential nep.txt\ntime_step 0.05 100\nvelocity 300 seed 1\nspin_llg gamma=3.0385 alpha=0.3 T=300 seed=1\ndump_thermo 100\nensemble nvt_lan 300 300 100\nrun 2000\n",
  "single_nve": "potential nep.txt\ntime_step 0.05 100\nvelocity 300 seed 1\nspin_llg gamma=3.0385 alpha=0.3 T=300 seed=1\ndump_thermo 100\nensemble nve\nrun 2000\n",
}
def one(exe, tag, name, txt):
    d = f"reg_{tag}_{name}"; os.makedirs(d, exist_ok=True)
    shutil.copyfile("N432/model.xyz", os.path.join(d,"model.xyz"))
    shutil.copyfile("N432/nep.txt", os.path.join(d,"nep.txt"))
    open(os.path.join(d,"run.in"),"w").write(txt)
    for j in ("thermo.out","stdout.txt"):
        p=os.path.join(d,j)
        if os.path.exists(p): os.remove(p)
    with open(os.path.join(d,"stdout.txt"),"w") as fh:
        subprocess.run([exe], cwd=d, stdout=fh, stderr=subprocess.STDOUT)
    return hashlib.sha256(open(os.path.join(d,"thermo.out"),"rb").read()).hexdigest()
for name, txt in CASES.items():
    a = one(OLD, "old", name, txt); b = one(NEW, "new", name, txt)
    print(f"{name:12s} identical={a==b}  {a[:16]} / {b[:16]}")
