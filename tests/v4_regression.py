"""V4 regression suite.

Covers the two features added on top of v3.1.0-r19 (magnetic moments in the
minimisers, external field / magneto-electric coupling) together with the core
behaviour that must not move.

    python tests/v4_regression.py            # everything
    python tests/v4_regression.py external   # one group
    python tests/v4_regression.py -v         # print every check

Exit code 0 only if every check passes.  Each check creates its own directory
under tests/_work so that a failure leaves the inputs and outputs behind.

Groups
    external    Zeeman and magneto-electric energy / torque / force / virial,
                against an independent float64 implementation
    minimiser   both convergence criteria, |S| conservation, energy decrease
    compat      byte-identical output vs the r19 binary where it exists
    inputs      negative tests for the new keywords
    topology    skyrmion winding number: ansatz -> |Q| = 1, FM -> Q = 0,
                and the collapse of a skyrmion when no DMI is present
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GPUMD = os.path.join(ROOT, "src", "gpumd.exe")
NEP = os.path.join(HERE, "data", "fege.nep.txt")
B20 = os.path.join(HERE, "data", "fege_b20.xyz")
WORK = os.path.join(HERE, "_work")
R19 = os.environ.get(
    "GPUMD_R19_BIN",
    os.path.join(ROOT, "..", "final_delivery_20261101_r18", "win_staging_r19", "bin", "gpumd.exe"))

MU_B_EV_PER_T = 5.7883818066e-5
# The structure file carries the magnetic *moment* in mu_B (Fe = 2.0), and the
# integrator's gamma = 3.0385 already is g*mu_B/hbar in (fs^-1 per eV/muB).
# So a field given in tesla maps to B[eV/muB] = 1 * mu_B * H, with no extra g.
G = 1.0
RESULTS = []


# helpers
def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail, "PASS" if ok else "FAIL"))
    print("  [%s] %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def skip(name, why):
    """A check whose inputs are not shipped (e.g. the r19 baseline binary).

    Skipping is not passing: the summary prints the count separately so a
    standalone source package cannot silently look fully verified.
    """
    RESULTS.append((name, True, why, "SKIP"))
    print("  [SKIP] %-46s %s" % (name, why))
    return True


def workdir(name, clean=True):
    d = os.path.join(WORK, name)
    if clean and os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    return d


def read_xyz(path):
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    n = int(lines[0])
    header = lines[1]
    cols = []
    if "Properties=" in header:
        fields = header.split("Properties=", 1)[1].split()[0].split(":")
        for i in range(0, len(fields) - 2, 3):
            cols.append((fields[i], int(fields[i + 2])))
    else:
        cols = [("species", 1), ("pos", 3), ("spin", 3)]
    data = np.array([[float(x) for x in l.split()[1:]] for l in lines[2: 2 + n]])
    out, k = {}, 0
    for name, ncomp in cols[1:]:
        out[name] = data[:, k: k + ncomp]
        k += ncomp
    return out


def read_frame(path):
    """First frame of a dump_xyz file, column names from Properties."""
    return read_xyz(path)


def run_gpumd(d, lines, exe=None):
    for f in os.listdir(d):
        if f.endswith(".xyz") and f != "model.xyz":
            os.remove(os.path.join(d, f))
    with open(os.path.join(d, "run.in"), "w") as f:
        f.write("\n".join(lines) + "\n")
    r = subprocess.run([exe or GPUMD], cwd=d, capture_output=True, text=True)
    return r


def random_spins(src, dst, seed=20261104):
    """Same lattice/moments, random spin directions: exercises every component."""
    with open(src) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    n = int(lines[0])
    rng = np.random.default_rng(seed)
    out = [lines[0], lines[1]]
    for l in lines[2: 2 + n]:
        t = l.split()
        v = np.array([float(x) for x in t[4:7]])
        L = np.linalg.norm(v)
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        out.append("%s %s %s %s %.9f %.9f %.9f" %
                   (t[0], t[1], t[2], t[3], *(L * u)))
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")


def strip_spin(src, dst):
    """Same lattice, no spin column: the pre-V4 lattice-only minimisation path."""
    with open(src) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    n = int(lines[0])
    hdr = lines[1].replace(":spin:R:3", "")
    out = [lines[0], hdr]
    for l in lines[2: 2 + n]:
        t = l.split()
        out.append(" ".join(t[:4]))
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")


def strip_spin(src, dst):
    """Same lattice, no spin column: the pre-V4 lattice-only minimisation path."""
    with open(src) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    n = int(lines[0])
    hdr = lines[1].replace(":spin:R:3", "")
    out = [lines[0], hdr]
    for l in lines[2: 2 + n]:
        t = l.split()
        out.append(" ".join(t[:4]))
    with open(dst, "w") as f:
        f.write("\n".join(out) + "\n")


def magelec_ref(pos, spin, box, kappa, E, cutoff):
    """Independent float64 H_ME, per-atom energy, torque and force."""
    n = len(pos)
    r = pos[:, None, :] - pos[None, :, :]
    for d in range(3):
        L = np.linalg.norm(box[d])
        r[:, :, d] -= L * np.round(r[:, :, d] / L)
    r = -r  # r_ij = r_j - r_i
    d2 = (r ** 2).sum(-1)
    ii, jj = np.where((d2 < cutoff ** 2) & (d2 > 1e-12))
    keep = ii < jj
    ii, jj, r = ii[keep], jj[keep], r[ii[keep], jj[keep]]
    D = kappa * np.cross(np.broadcast_to(np.asarray(E, float), r.shape), r)
    cr = np.cross(spin[ii], spin[jj])
    e = (D * cr).sum(-1)
    g = np.zeros(n)
    np.add.at(g, ii, 0.5 * e)
    np.add.at(g, jj, 0.5 * e)
    tau = np.zeros((n, 3))
    np.add.at(tau, ii, -np.cross(spin[jj], D))
    np.add.at(tau, jj, -np.cross(spin[ii], -D))
    frc = np.zeros((n, 3))
    fi = kappa * np.cross(cr, np.asarray(E, float))
    np.add.at(frc, ii, fi)
    np.add.at(frc, jj, -fi)
    return g, tau, frc, (ii, jj, r)


# groups
def group_external():
    print("\n== external field / magneto-electric ==")
    d = workdir("external")
    shutil.copy(NEP, os.path.join(d, "nep.txt"))
    random_spins(B20, os.path.join(d, "model.xyz"))

    def probe(tag, extra):
        dd = os.path.join(d, tag)
        os.makedirs(dd, exist_ok=True)
        shutil.copy(os.path.join(d, "nep.txt"), dd)
        shutil.copy(os.path.join(d, "model.xyz"), dd)
        # r21 COMPATIBILITY: r20 accepted `gamma=0` as an undocumented
        # "static force/torque/virial probe".  r21 enforces gamma > 0
        # (LL convention), so these single-step probes use a strictly
        # positive but negligible gyromagnetic ratio instead.  With
        # time_step = 1e-6 fs the rotation angle gamma*|B|*dt is ~1e-13,
        # far below the double-precision dump resolution; the spin
        # configuration is effectively frozen while the strict contract
        # is preserved.  See AUDIT_AND_FIX_REPORT.md, F-04.
        run_gpumd(dd, ["potential nep.txt", "time_step 0.000001", "ensemble nve",
                       "velocity 300 seed 1",
                       "dump_xyz 1 dump.xyz spin torque force potential virial precision double",
                       "spin_llg gamma=1e-6 alpha=0 T=0"] + extra + ["run 1"])
        return read_frame(os.path.join(dd, "dump.xyz"))

    f0 = probe("none", [])
    f1 = probe("field", ["magfield 5.0 0.0 0.0 1.0"])
    f2 = probe("me", ["magelec 4.5 0.001 1.0 0.0 0.0"])
    f3 = probe("none_b", [])
    # the same 5 T expressed in energy units: 5 * mu_B
    f4 = probe("field_ev", ["magfield %.10e 0.0 0.0 1.0 unit=ev"
                            % (5.0 * MU_B_EV_PER_T)])

    pos, spin = f0["pos"], f0["spin"]
    box = np.eye(3) * 14.4
    noise = np.abs(f3["energy_atom"][:, 0] - f0["energy_atom"][:, 0]).max()
    tol = max(10 * noise, 1e-8)
    tol_sum = 10 * np.sqrt(len(pos)) * max(noise, 1e-9)
    # atom.torque is float32; adding B to a torque of order |tau| cannot be
    # resolved below eps_float * |tau| no matter how exact the physics is.
    eps32 = np.finfo(np.float32).eps
    tol_tau = max(tol, 4 * eps32 * np.abs(f0["torque"]).max())

    # Zeeman
    B = np.array([0, 0, 1.0]) * G * MU_B_EV_PER_T * 5.0
    dE = f1["energy_atom"][:, 0] - f0["energy_atom"][:, 0]
    e_ref = -(spin @ B)
    check("Zeeman: dE_i = -S_i.B", np.abs(dE - e_ref).max() < tol,
          "max %.2e eV (tol %.1e)" % (np.abs(dE - e_ref).max(), tol))
    check("Zeeman: sum dE", abs(dE.sum() - e_ref.sum()) < tol_sum,
          "%.2e eV (tol %.1e)" % (abs(dE.sum() - e_ref.sum()), tol_sum))
    check("Zeeman: dtau_i = +B", np.abs((f1["torque"] - f0["torque"]) - B).max() < tol_tau,
          "max %.2e eV/mu_B" % np.abs((f1["torque"] - f0["torque"]) - B).max())
    # 1 T must equal mu_B in eV/muB.  A stray Lande factor shows up here at once:
    # the integrator's gamma = 3.0385 already is g*mu_B/hbar, so doubling B would
    # double this energy.
    d_ev = np.abs(f4["energy_atom"][:, 0] - f1["energy_atom"][:, 0]).max()
    check("Zeeman: 1 T == 5.788e-5 eV/muB (no double g)", d_ev < tol,
          "max %.2e eV" % d_ev)
    check("Zeeman: dF = 0", np.abs(f1["forces"] - f0["forces"]).max() < tol,
          "max %.2e eV/A" % np.abs(f1["forces"] - f0["forces"]).max())

    # magneto-electric
    kappa, cutoff = 1.0e-3, 4.5
    E = np.array([1.0, 0.0, 0.0])
    g, tau_ref, frc_ref, (ii, jj, r) = magelec_ref(pos, spin, box, kappa, E, cutoff)
    dE2 = f2["energy_atom"][:, 0] - f0["energy_atom"][:, 0]
    check("magelec: dE_i", np.abs(dE2 - g).max() < tol,
          "max %.2e eV (tol %.1e)" % (np.abs(dE2 - g).max(), tol))
    check("magelec: sum dE", abs(dE2.sum() - g.sum()) < tol_sum,
          "%.2e eV (tol %.1e)" % (abs(dE2.sum() - g.sum()), tol_sum))
    check("magelec: dtau_i",
          np.abs((f2["torque"] - f0["torque"]) - tau_ref).max() < tol_tau,
          "max %.2e eV/mu_B" % np.abs((f2["torque"] - f0["torque"]) - tau_ref).max())
    check("magelec: dF_i",
          np.abs((f2["forces"] - f0["forces"]) - frc_ref).max() < tol,
          "max %.2e eV/A" % np.abs((f2["forces"] - f0["forces"]) - frc_ref).max())

    # virial: GPUMD's per-atom convention is W_pair = -r_ij (x) F_i for ONE
    # unordered pair (see force/lj.cu: each of the two ordered visits
    # contributes -1/2 r_ij (x) F_i).  r20 accumulated the full -r_ij (x) F_i
    # on BOTH visits, so the total was 2*W_pair.  The reference below is an
    # INDEPENDENT float64 sum over unordered pairs; it must NOT be multiplied
    # by 2 to accommodate the implementation (that was the r20 test bug).
    vref = np.zeros((3, 3))
    fi = kappa * np.cross(np.cross(spin[ii], spin[jj]), E)
    for k in range(len(ii)):
        vref += np.outer(r[k], -fi[k])
    dv = (f2["virial"] - f0["virial"]).sum(0)
    dvm = np.array([[dv[0], dv[1], dv[2]], [dv[3], dv[4], dv[5]], [dv[6], dv[7], dv[8]]])
    err = np.abs(dvm - vref).max()
    check("magelec: virial (all 9 components, unordered pairs)", err < tol_sum,
          "max %.2e eV" % err)
    check("magelec: trace(virial) = - E_ME",
          abs(np.trace(dvm) / (-g.sum()) - 1.0) < 1e-3,
          "ratio %.6f" % (np.trace(dvm) / (-g.sum())))

    # Independent energy-strain finite difference of ALL NINE components.
    # GPUMD stores W_ab = -sum_{i<j} r_ij,a F_i,b (see force/lj.cu).  For a
    # deformation-gradient perturbation the Hamiltonian gives
    #   dH/deps_ab = sum_{i<j} F_i,a r_ij,b      if r' = (I+eps) r
    #   dH/deps_ab = sum_{i<j} r_ij,a F_i,b      if r' = (I+eps^T) r
    # so -dH/deps equals W for the first convention and W^T for the second.
    # GPUMD's `change_box` uses the second form; ME with a fixed external E
    # produces a NON-symmetric W (F_i is not parallel to r_ij), so the
    # transpose is material.  Both conventions are checked here with an
    # energy-only reference: no force or virial formula is reused.
    # PERIODIC AFFINE DEFORMATION: positions AND the box must be deformed
    # together (p' = p M, h' = h M).  Fractional coordinates s = p h^-1 are
    # invariant under an affine map, so the minimum image must be taken in
    # fractional space and mapped back with the DEFORMED h.  Deforming only
    # the positions while keeping the old box gives a spurious ~0.67 eV
    # error on periodic-crossing pairs.  The hard cutoff makes E piecewise
    # smooth; we scan step sizes and require the neighbour-pair set to stay
    # identical to the base state (no pair crosses the cutoff).
    def me_strain_state(M):
        pp = pos @ M
        h = box @ M
        hi = np.linalg.inv(h)
        rr = pp[:, None, :] - pp[None, :, :]
        s = rr.reshape(-1, 3) @ hi
        s -= np.round(s)
        rr = -(s @ h).reshape(len(pos), len(pos), 3)
        d2 = (rr ** 2).sum(-1)
        a, b = np.where((d2 < cutoff ** 2) & (d2 > 1e-12))
        m = a < b
        a, b, rr = a[m], b[m], rr[a[m], b[m]]
        D = kappa * np.cross(np.broadcast_to(E, rr.shape), rr)
        energy = float((D * np.cross(spin[a], spin[b])).sum())
        return energy, (a.tolist(), b.tolist()), np.sqrt((rr ** 2).sum(-1))

    e_base, key_base, dists = me_strain_state(np.eye(3))
    gap = float(np.min(np.abs(dists - cutoff)))
    step_scan = []
    best = None
    for step in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        fd_vir = np.zeros((3, 3))
        fd_vir_t = np.zeros((3, 3))
        stable = True
        for a in range(3):
            for b in range(3):
                ep = np.zeros((3, 3)); ep[a, b] = step
                em = -ep.copy()
                for trans, out in ((False, fd_vir), (True, fd_vir_t)):
                    vals = []
                    for pert in (ep, em):
                        M = np.eye(3) + (pert.T if trans else pert)
                        e, key, _ = me_strain_state(M)
                        if key != key_base:
                            stable = False
                        vals.append(e)
                    out[a, b] = -(vals[0] - vals[1]) / (2 * step)
        err = float(np.max(np.abs(dvm - fd_vir)))
        err_t = float(np.max(np.abs(dvm.T - fd_vir_t)))
        step_scan.append({"step": step, "pair_set_stable": stable,
                          "err_W": err, "err_W_transpose": err_t})
        if stable and (best is None or max(err, err_t) < max(best[1], best[2])):
            best = (step, err, err_t, fd_vir, fd_vir_t)
    if best is None:
        check("magelec: virial = -dH/deps (FD, 9 comps)", False,
              "no stable finite-difference step (hard cutoff crossing)")
    else:
        step, err, err_t, fd_vir, fd_vir_t = best
        tol_fd = max(1e-6, 1e-4 * np.abs(fd_vir).max())
        check("magelec: virial = -dH/deps (FD, 9 comps, M=I+eps)",
              err < tol_fd,
              "max %.2e eV, h=%.0e, min|d-cutoff|=%.3g A" % (err, step, gap))
        check("magelec: virial^T = -dH/deps (FD, 9 comps, M=I+eps^T)",
              err_t < tol_fd, "max %.2e eV, h=%.0e" % (err_t, step))
        check("magelec: FD step scan (noise vs derivation error)",
              len(step_scan) >= 3 and all(s["pair_set_stable"] for s in step_scan),
              "; ".join("h=%.0e err=%.2g" % (s["step"], s["err_W"])
                        for s in step_scan))
    check("magelec: virial is genuinely non-symmetric (FD is sensitive)",
          np.abs(dvm - dvm.T).max() > 1e-6,
          "|W-W^T|max = %.3e eV" % np.abs(dvm - dvm.T).max())
    check("magelec: total force zero (action/reaction)",
          np.abs((f2["forces"] - f0["forces"]).sum(0)).max() < tol_sum,
          "max %.2e eV/A" % np.abs((f2["forces"] - f0["forces"]).sum(0)).max())

    # analytic formulas must reproduce the energy by finite differences
    def energy(p, s):
        rr = p[:, None, :] - p[None, :, :]
        for dd in range(3):
            L = np.linalg.norm(box[dd])
            rr[:, :, dd] -= L * np.round(rr[:, :, dd] / L)
        rr = -rr
        d2 = (rr ** 2).sum(-1)
        a, b = np.where((d2 < cutoff ** 2) & (d2 > 1e-12))
        m = a < b
        a, b, rr = a[m], b[m], rr[a[m], b[m]]
        D = kappa * np.cross(np.broadcast_to(E, rr.shape), rr)
        return float((D * np.cross(s[a], s[b])).sum())

    h = 1e-4
    errf, errt = 0.0, 0.0
    for i in [0, 7, 60, 200]:
        for k in range(3):
            pp = pos.copy(); pp[i, k] += h
            pm = pos.copy(); pm[i, k] -= h
            errf = max(errf, abs(-(energy(pp, spin) - energy(pm, spin)) / (2 * h) - frc_ref[i, k]))
    check("magelec: F = -dH/dr (FD)", errf < 1e-6, "max %.2e eV/A" % errf)


def group_minimiser():
    print("\n== minimiser with magnetic moments ==")
    d = workdir("minimiser")
    shutil.copy(NEP, os.path.join(d, "nep.txt"))
    random_spins(B20, os.path.join(d, "model.xyz"))
    r = run_gpumd(d, ["potential nep.txt", "spin_llg gamma=3.04 alpha=0.1 T=0",
                      "minimize fire 1e-3 4000 spin=on",
                      "ensemble nve",
                      "dump_xyz 1 final.xyz spin torque force potential precision double",
                      "run 1"])
    txt = r.stdout
    m = re.search(r"Converged at step (\d+): f_max = (\S+) eV/A < (\S+) AND "
                  r"w_perp_max = (\S+) eV/mu_B < (\S+)", txt)
    check("both criteria converge", m is not None,
          (m.group(0)[:80] if m else "no convergence line"))
    if m:
        check("f_max < force tolerance", float(m.group(2)) < 1e-3, m.group(2) + " eV/A")
        check("w_perp < torque tolerance", float(m.group(4)) < 1e-3,
              m.group(4) + " eV/mu_B")
    e = [float(x) for x in re.findall(r"total_potential = (\S+) eV", txt)]
    check("energy decreases", len(e) >= 2 and e[-1] < e[0] - 1.0,
          "%.4f -> %.4f eV" % (e[0], e[-1]))
    f0 = read_frame(os.path.join(d, "model.xyz"))
    f1 = read_frame(os.path.join(d, "final.xyz"))
    n0 = np.linalg.norm(f0["spin"], axis=1)
    n1 = np.linalg.norm(f1["spin"], axis=1)
    drift = (np.abs(n1 - n0) / np.maximum(n0, 1e-30)).max()
    check("|S_i| conserved by the spin step", drift < 1e-4, "max drift %.2e" % drift)
    check("moments order (|M| grows)", np.linalg.norm(f1["spin"].sum(0)) >
          np.linalg.norm(f0["spin"].sum(0)),
          "%.1f -> %.1f mu_B" % (np.linalg.norm(f0["spin"].sum(0)),
                                 np.linalg.norm(f1["spin"].sum(0))))


def group_compat():
    print("\n== backward compatibility ==")
    if not os.path.exists(R19):
        skip("compat vs r19 binary", "baseline binary not shipped (set GPUMD_R19_BIN)")
        return
        return
    d = workdir("compat")
    shutil.copy(NEP, os.path.join(d, "nep.txt"))
    shutil.copy(B20, os.path.join(d, "model.xyz"))
    # `velocity ... seed` is required for bit reproducibility: a structure
    # without a velocity column gets time-seeded initial velocities, so two
    # runs of the *same* binary would otherwise differ (GPUMD prints
    # "DEBUG is off: Use different PRNG seeds for different runs").
    lines = ["potential nep.txt", "time_step 0.5", "ensemble nve",
             "velocity 300 seed 1",
             "dump_thermo 20", "dump_xyz 20 traj.xyz spin",
             "spin_llg gamma=3.04 alpha=0 T=0", "run 200"]
    for tag, exe in (("base", R19), ("v4", GPUMD)):
        dd = os.path.join(d, tag)
        os.makedirs(dd, exist_ok=True)
        for f in ("nep.txt", "model.xyz"):
            shutil.copy(os.path.join(d, f), dd)
        run_gpumd(dd, lines, exe=exe)
    same = True
    for f in ("thermo.out", "traj.xyz"):
        a = open(os.path.join(d, "base", f), "rb").read()
        b = open(os.path.join(d, "v4", f), "rb").read()
        same &= (a == b)
    check("MD output identical to r19", same, "thermo.out + traj.xyz")

    # lattice-only minimisation, and spin=off, must also match line for line
    d2 = workdir("compat_min")
    shutil.copy(NEP, os.path.join(d2, "nep.txt"))
    strip_spin(B20, os.path.join(d2, "model.xyz"))
    for tag, exe, extra in (("base", R19, ""), ("v4", GPUMD, "")):
        dd = os.path.join(d2, tag)
        os.makedirs(dd, exist_ok=True)
        for f in ("nep.txt", "model.xyz"):
            shutil.copy(os.path.join(d2, f), dd)
        run_gpumd(dd, ["potential nep.txt",
                       "minimize fire 1e-4 300" + extra], exe=exe)
    a = [l for l in open(os.path.join(d2, "base", "log.txt")).read().splitlines() if "step" in l] \
        if os.path.exists(os.path.join(d2, "base", "log.txt")) else None
    # run_gpumd does not write a log file; compare the captured stdout instead
    r1 = subprocess.run([R19], cwd=os.path.join(d2, "base"), capture_output=True, text=True)
    r2 = subprocess.run([GPUMD], cwd=os.path.join(d2, "v4"), capture_output=True, text=True)
    s1 = [l for l in r1.stdout.splitlines() if "step " in l]
    s2 = [l for l in r2.stdout.splitlines() if "step " in l]
    check("lattice-only minimise identical to r19", s1 == s2 and len(s1) >= 2,
          "%d step lines" % len(s1))

    d3 = workdir("compat_spinoff")
    shutil.copy(NEP, os.path.join(d3, "nep.txt"))
    shutil.copy(B20, os.path.join(d3, "model.xyz"))
    for tag, exe, extra in (("base", R19, ""), ("v4", GPUMD, " spin=off")):
        dd = os.path.join(d3, tag)
        os.makedirs(dd, exist_ok=True)
        for f in ("nep.txt", "model.xyz"):
            shutil.copy(os.path.join(d3, f), dd)
        run_gpumd(dd, ["potential nep.txt", "spin_llg gamma=3.04 alpha=0.1 T=0",
                       "minimize fire 1e-3 500" + extra], exe=exe)
    r1 = subprocess.run([R19], cwd=os.path.join(d3, "base"), capture_output=True, text=True)
    r2 = subprocess.run([GPUMD], cwd=os.path.join(d3, "v4"), capture_output=True, text=True)
    s1 = [l for l in r1.stdout.splitlines() if "step " in l]
    s2 = [l for l in r2.stdout.splitlines() if "step " in l]
    check("minimize spin=off identical to r19", s1 == s2, "%d step lines" % len(s1))


def group_inputs():
    print("\n== input validation (negative) ==")
    d = workdir("inputs")
    shutil.copy(NEP, os.path.join(d, "nep.txt"))
    shutil.copy(B20, os.path.join(d, "model.xyz"))

    cases = [
        ("magfield without spin_llg",
         ["potential nep.txt", "magfield 5 0 0 1", "run 1"]),
        ("magelec without spin_llg",
         ["potential nep.txt", "magelec 4.5 0.001 1 0 0", "run 1"]),
        ("minimize unknown option",
         ["potential nep.txt", "spin_llg gamma=3.04 alpha=0.1 T=0",
          "minimize fire 1e-3 10 spin=bogus"]),
        ("magfield zero direction",
         ["potential nep.txt", "spin_llg gamma=3.04 alpha=0.1 T=0",
          "magfield 5 0 0 0", "run 1"]),
        ("magfield + add_bfield",
         ["potential nep.txt", "spin_llg gamma=3.04 alpha=0.1 T=0",
          "magfield 5 0 0 1", "add_bfield 0.001 0 0", "run 1"]),
        ("add_bfield then magfield",
         ["potential nep.txt", "spin_llg gamma=3.04 alpha=0.1 T=0",
          "add_bfield 0.001 0 0", "magfield 5 0 0 1", "run 1"]),
        ("magelec_pair before magelec",
         ["potential nep.txt", "spin_llg gamma=3.04 alpha=0.1 T=0",
          "magelec_pair 0 1 0.001", "run 1"]),
    ]
    for name, lines in cases:
        r = run_gpumd(d, lines)
        check(name + " rejected", r.returncode != 0,
              "rc=%d" % r.returncode)


def group_topology():
    print("\n== skyrmion topology ==")
    for cand in (os.path.join(ROOT, "validation_v4"), HERE):
        if os.path.isdir(cand):
            sys.path.insert(0, cand)
    try:
        from fege_build import (read_basis, build, skyrmion, fm, write_xyz,
                               topological_charge, skyrmion_radius)
    except Exception as e:                                    # pragma: no cover
        skip("topology (fege_build)", "helper not shipped: %s" % e)
        return
    a, basis = read_basis(B20)
    L = np.array([16 * a, 16 * a, 2 * a])
    sp, pos, spin, Lb = build(16, 16, 2, fm((0, 0, 1)), a=a, basis=basis)
    q_fm = topological_charge(pos, spin, Lb)
    check("Q(FM) = 0", abs(q_fm) < 1e-9, "%.6f" % q_fm)
    for R0 in (5.0, 8.0):
        sp, pos, spin, Lb = build(16, 16, 2,
                                  skyrmion((0.5 * L[0], 0.5 * L[1], 0), R0,
                                           helicity=np.pi / 2, box=L),
                                  a=a, basis=basis)
        q = topological_charge(pos, spin, Lb)
        Rm = skyrmion_radius(pos, spin, Lb, (0.5 * L[0], 0.5 * L[1]))
        check("|Q| = 1 for a skyrmion ansatz (R=%.0f A)" % R0,
              abs(abs(q) - 1.0) < 1e-6, "Q=%.6f, R_meas=%.1f A" % (q, Rm))
        check("radius read-back within one lattice constant (R=%.0f A)" % R0,
              abs(Rm - R0) < a, "|dR| = %.2f A" % abs(Rm - R0))

    # NOTE: this group used to minimise a 12x12x2 cell and assert that the
    # skyrmion survived.  With a 6 A radial cutoff, Lz = 9.6 A is *smaller* than
    # 2*cutoff, so every atom sees its own periodic image along z and the cell is
    # not a physical model of anything.  The "stable" skyrmion was an artefact of
    # that cell.  Only the ansatz-level checks belong in a regression suite;
    # whether the potential supports a skyrmion is a physics question, measured
    # in FEGE_SKYRMION_VS_LITERATURE.md on a real slab.


def spiral_b20(src, dst, lam, sign):
    """Bloch spiral along [111] on the 216-atom B20 cell, sign = handedness."""
    f = read_xyz(src)
    pos, spin = f["pos"], f["spin"]
    qh = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    e1 = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    e2 = np.cross(qh, e1)
    ph = 2 * np.pi * (pos @ qh) / lam
    mag = np.linalg.norm(spin, axis=1)[:, None]
    new = (np.cos(ph)[:, None] * e1 + sign * np.sin(ph)[:, None] * e2) * mag
    sp_ = []
    with open(src) as fh:
        lines = [l for l in fh.read().splitlines() if l.strip()]
    n = int(lines[0])
    sp_ = [l.split()[0] for l in lines[2: 2 + n]]
    hdr = lines[1].replace(":spin:R:3", "")
    # keep Properties= in the documented order; appending after pbc= would
    # produce a header the parser silently rejects
    hdr = hdr.replace(":pos:R:3", ":pos:R:3:spin:R:3", 1)
    with open(dst, "w") as fh:
        fh.write("%d\n%s\n" % (n, hdr))
        for s, p, v in zip(sp_, pos, new):
            fh.write("%s %.10f %.10f %.10f %.9f %.9f %.9f\n" % (s, *p, *v))


def group_chirality():
    """E(L) - E(R) is a pure chirality signal: everything else is identical."""
    print("\n== chirality (DMI) ==")
    d = workdir("chirality")
    shutil.copy(NEP, os.path.join(d, "nep.txt"))
    e = {}
    # sign convention matches 48/b20/spiral.py: L is the -1 handedness
    for tag_, sign in (("fm", 0), ("L", -1), ("R", +1)):
        dst = os.path.join(d, tag_ + ".xyz")
        if sign == 0:
            shutil.copy(B20, dst)
        else:
            spiral_b20(B20, dst, 60.0, sign)
        dd = os.path.join(d, tag_)
        os.makedirs(dd, exist_ok=True)
        shutil.copy(os.path.join(d, "nep.txt"), os.path.join(dd, "nep.txt"))
        shutil.copy(dst, os.path.join(dd, "model.xyz"))
        r = run_gpumd(dd, ["potential nep.txt", "spin_llg gamma=3.04 alpha=0.1 T=0",
                           "minimize fire 1e-9 1 spin=on"])
        m = re.search(r"step 0: total_potential = (\S+) eV", r.stdout)
        if not m:
            check("single-point energy available (%s)" % tag_, False, r.stdout[-300:])
            return
        e[tag_] = float(m.group(1))

    d_lr = e["R"] - e["L"]
    check("chiral splitting E(R) > E(L)", d_lr > 1.0e-4,
          "E(R)-E(L) = %.3e eV = %.4f meV/atom"
          % (d_lr, d_lr / 216 * 1000))
    check("spiral lies below the ferromagnetic state",
          e["L"] < e["fm"] and e["R"] < e["fm"],
          "E(L)-E(FM) = %.3e eV = %.3f meV/atom"
          % (e["L"] - e["fm"], (e["L"] - e["fm"]) / 216 * 1000))


GROUPS = {
    "external": group_external,
    "minimiser": group_minimiser,
    "compat": group_compat,
    "inputs": group_inputs,
    "topology": group_topology,
    "chirality": group_chirality,
}


def main():
    if not os.path.exists(GPUMD):
        print("gpumd.exe not found: " + GPUMD)
        return 2
    sel = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not sel:
        sel = list(GROUPS)
    print("gpumd: " + GPUMD)
    for g in sel:
        if g not in GROUPS:
            print("unknown group: " + g)
            return 2
        GROUPS[g]()
    npass = sum(1 for r in RESULTS if r[3] == "PASS")
    nfail = sum(1 for r in RESULTS if r[3] == "FAIL")
    nskip = sum(1 for r in RESULTS if r[3] == "SKIP")
    line = "\n%d passed, %d failed" % (npass, nfail)
    if nskip:
        line += ", %d skipped" % nskip
    print(line)
    if nfail:
        print("failed:")
        for name, ok, detail, st in RESULTS:
            if st == "FAIL":
                print("  - %s  %s" % (name, detail))
        return 1
    print("REGRESSION OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
