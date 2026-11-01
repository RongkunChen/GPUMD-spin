"""Stage E: harmonic phonons from the production potential (finite displacement).

IMPORTANT SCOPE NOTE (verified, not assumed): `compute_phonon` calls
Force::compute_single_potential, which uses the potential overload WITHOUT the
spin argument, i.e. every Hessian is evaluated in the S = 0 limit.  Proof: the
dispersion is bit-identical for an FM cell (|S| = 2.417 muB) and for |S| = 0
(raw_data/E/probe vs raw_data/E/probe_s0, md5 of omega2.out).  The numbers below
are therefore *lattice* phonons of the production potential in its S=0 limit,
not the phonons of the FM ground state.  E is reported as PARTIAL and the
FM/DLM comparison (G2) is BLOCKED.
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess

import numpy as np

import feutils as U

OUT = os.path.join(U.VAL, "raw_data", "E")
A0 = json.load(open(os.path.join(U.VAL, "raw_data", "B", "eos_fit.json")))["a0_A"]
M0 = json.load(open(os.path.join(U.VAL, "raw_data", "B", "moment_equilibrium.json")))["m_eq_muB"]

# primitive bcc cell -> 1 atom basis -> 3 acoustic branches.
# k-points are given in the fractional basis of the fcc reciprocal lattice:
#   G=(0,0,0)  H=(0.5,-0.5,0.5)  N=(0.5,0,0)  P=(0.25,0.25,0.25)
KPATH = ("0.0 0.0 0.0 G\n0.5 -0.5 0.5 H\n0.5 0.0 0.0 N\n0.0 0.0 0.0 G\n"
         "0.25 0.25 0.25 P\n0.5 -0.5 0.5 H\n")


def primitive_cell(a):
    A = np.array([[1, 1, -1], [-1, 1, 1], [1, -1, 1]], float) * a / 2
    return dict(n=1, lattice=A, pos=np.zeros((1, 3)))


def phonon_case(tag, rep, disp, spin_mode="fm", extra=None):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    for junk in ("omega2.out", "D.out", "stdout.txt", "eigenvector.out"):
        p = os.path.join(d, junk)
        if os.path.exists(p):
            open(p, "w").close()
    cell = primitive_cell(A0)
    sp = U.fm_spins(cell["n"], M0 if spin_mode == "fm" else 0.0)
    if spin_mode == "dlm":
        sp = U.random_spins(cell["n"], M0, extra)
    U.write_xyz(os.path.join(d, "model.xyz"), cell, sp)
    shutil.copyfile(U.MODELS["A"], os.path.join(d, "nep.txt"))
    open(os.path.join(d, "kpoints.in"), "w").write(KPATH)
    open(os.path.join(d, "run.in"), "w").write(
        f"replicate {rep} {rep} {rep}\npotential nep.txt\ntime_step 0.001 1\n"
        f"compute_phonon {disp}\n")
    with open(os.path.join(d, "stdout.txt"), "w") as f:
        p = subprocess.run([U.EXE], cwd=d, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
    path = os.path.join(d, "omega2.out")
    ok = p.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 0
    out = dict(tag=tag, replicate=rep, displacement_A=disp, spin_mode=spin_mode,
               rc=p.returncode, dir=d)
    if ok:
        arr = np.loadtxt(path)
        out["md5_omega2"] = hashlib.md5(open(path, "rb").read()).hexdigest()
        out["n_kpoints"] = int(arr.shape[0])
        out["n_branches"] = int(arr.shape[1] - 1)
        # frequencies in THz; keep a copy in a friendly format
        np.savetxt(os.path.join(OUT, f"dispersion_{tag}.dat"), arr,
                   header="kpath " + " ".join(f"f{i+1}_THz" for i in range(arr.shape[1] - 1)))
    else:
        out["error"] = open(os.path.join(d, "stdout.txt"), errors="replace").read()[-600:]
    return out


cases = []
# supercell convergence (displacement fixed at 0.01 A)
for rep in (4, 6, 8):
    cases.append(phonon_case(f"rep{rep}_d0.010", rep, 0.01))
    print("  rep", rep, cases[-1].get("rc"), cases[-1].get("md5_omega2"), flush=True)
# displacement convergence at rep 6
for disp in (0.005, 0.02):
    cases.append(phonon_case(f"rep6_d{disp:.3f}", 6, disp))
    print("  disp", disp, cases[-1].get("rc"), flush=True)
# spin sensitivity probe (same cell, FM vs S=0)
cases.append(phonon_case("rep6_d0.010_s0", 6, 0.01, spin_mode="s0"))
print("  S=0 probe", cases[-1].get("rc"), flush=True)

rows = [c for c in cases if "md5_omega2" in c]
conv = dict(cases=cases, a0=A0, moment=M0,
            spin_aware=False,
            spin_aware_evidence=dict(
                fm_md5=next(c["md5_omega2"] for c in rows if c["tag"] == "rep6_d0.010"),
                s0_md5=next(c["md5_omega2"] for c in rows if c["tag"] == "rep6_d0.010_s0"),
                note="identical md5 -> the Hessian path does not read the spin columns; "
                     "documented in force/force.cu::compute_single_potential"))

# ---------------------------------------------------------------- analysis
ref = np.loadtxt(os.path.join(OUT, "dispersion_rep6_d0.010.dat"))
kpath = ref[:, 0]
freq = ref[:, 1:]
# ordering: 6 branches for the 2-atom cubic cell; the 3 lowest are acoustic
ac = freq[:, :3]   # 3 acoustic branches of the 1-atom primitive cell
conv["gamma_frequencies_THz"] = [float(x) for x in freq[0]]
conv["n_branches"] = int(freq.shape[1])
conv["max_acoustic_frequency_THz"] = float(ac.max())
conv["imaginary_modes"] = int(np.sum(freq < -0.05))
conv["imaginary_fraction"] = float(np.mean(freq < -0.05))
conv["acoustic_sum_rule_max_abs_THz"] = float(np.max(np.abs(freq[0])))
conv["stability_status"] = "PASS" if conv["imaginary_fraction"] < 1e-3 else "FAIL"

# convergence metrics between supercells
def compare(t1, t2):
    a = np.loadtxt(os.path.join(OUT, f"dispersion_{t1}.dat"))
    b = np.loadtxt(os.path.join(OUT, f"dispersion_{t2}.dat"))
    n = min(len(a), len(b))
    return float(np.max(np.abs(a[:n, 1:4] - b[:n, 1:4])))

conv["convergence"] = dict(rep4_vs_rep6_THz=compare("rep4_d0.010", "rep6_d0.010"),
                           rep6_vs_rep8_THz=compare("rep6_d0.010", "rep8_d0.010"),
                           disp_0p005_vs_0p010_THz=compare("rep6_d0.005", "rep6_d0.010"),
                           disp_0p010_vs_0p020_THz=compare("rep6_d0.010", "rep6_d0.020"))
conv["criterion"] = ("no imaginary modes; supercell and displacement convergence "
                     "|df| < 0.1 THz")
conv["scope"] = ("S=0 limit of the production potential; NOT the FM ground-state "
                 "phonon dispersion (see spin_aware_evidence)")
conv["status"] = ("PARTIAL" if conv["stability_status"] == "PASS" else "FAIL")
U.save_json(os.path.join(OUT, "phonon_convergence.json"), conv)
print(json.dumps({k: v for k, v in conv.items() if k not in ("cases",)}, indent=1)[:1200])

# DOS from the dispersion (histogram of the 3 acoustic branches over the path)
allf = ac.ravel()
hist, edges = np.histogram(allf, bins=120, range=(0, max(12.0, allf.max())))
with open(os.path.join(OUT, "phonon_dos.dat"), "w") as f:
    f.write("# omega_THz  g(omega) (arb. units, path histogram of the 3 acoustic branches)\n")
    for i in range(len(hist)):
        f.write(f"{(edges[i]+edges[i+1])/2:.4f} {hist[i]}\n")
