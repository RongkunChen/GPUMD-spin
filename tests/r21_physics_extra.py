#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r21 supplementary physics validation.

Called by tests/r21_validation.py so that every result lands in the SAME
TEST_RESULTS.json / TEST_MATRIX.csv.  All references here are analytic or
float64 CPU formulas derived independently of the GPUMD kernels; the only
GPUMD involvement is generating the trajectories under test.

Groups:
  E_sllg     finite-temperature SLLG equilibrium distribution (analytic
             single-spin Boltzmann law, autocorrelation, statistical error)
  E_lattice  frozen lattice vs coupled spin-lattice, for no field / static
             Zeeman / magneto-electric
  E_size     two system sizes x normal / deliberately oversized dt
  E_work     external-field work and energy balance (static and driven)
  E_perf     before/after performance on the SAME model/input/GPU
"""
import csv
import math
import os
import re
import shutil
import subprocess
import threading
import time
import traceback

import numpy as np

KB_EV_K = 8.617333262e-5
HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def err_text(e):
    return "%r | %s" % (e, traceback.format_exc().replace("\n", " / "))


# model.xyz writers (no dependency on personal absolute paths)
def write_two_atom_model(path, frozen=True, spin0=(0.0, 0.0, 2.0)):
    """Two isolated atoms (Fe + Ge), separated by more than the NEP cutoff.

    Both species are needed because this NEP file declares two types; a
    one-species model makes GPUMD try to compact the potential and fail on
    the spin-NEP header.  With no neighbours either spin is an independent
    single spin in the external field -- the analytic case we want.
    """
    props = ("species:S:1:pos:R:3:group:I:1:spin:R:3" if frozen
             else "species:S:1:pos:R:3:spin:R:3")
    lines = [
        "2",
        'Lattice="20.0 0 0 0 20.0 0 0 0 20.0" Properties=%s pbc="T T T"' % props,
    ]
    rows = [("Fe", (1.0, 1.0, 1.0)), ("Ge", (11.0, 11.0, 11.0))]
    for sym, p in rows:
        if frozen:
            lines.append("%s %.6f %.6f %.6f 0 %.9f %.9f %.9f" %
                         (sym, p[0], p[1], p[2], spin0[0], spin0[1], spin0[2]))
        else:
            lines.append("%s %.6f %.6f %.6f %.9f %.9f %.9f" %
                         (sym, p[0], p[1], p[2], spin0[0], spin0[1], spin0[2]))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_b20_model(path, src, nx=1, seed=20261104, frozen=True):
    """Tile the shipped FeGe B20 cell nx times along x, randomise spin
    directions (magnitudes unchanged), optionally add a group column with
    every atom in group 0 so `fix 0 0` freezes the lattice."""
    with open(src) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    n = int(lines[0])
    rng = np.random.default_rng(seed)
    props = ("species:S:1:pos:R:3:group:I:1:spin:R:3" if frozen
             else "species:S:1:pos:R:3:spin:R:3")
    lx = 14.4 * nx
    out = [str(n * nx),
           'Lattice="%.8f 0 0 0 14.40000000 0 0 0 14.40000000" '
           'Properties=%s pbc="T T T"' % (lx, props)]
    for rep in range(nx):
        for l in lines[2:2 + n]:
            t = l.split()
            p = [float(x) for x in t[1:4]]
            v = np.array([float(x) for x in t[4:7]])
            L = float(np.linalg.norm(v))
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            s = L * u
            cols = "%s %.6f %.6f %.6f" % (t[0], p[0] + rep * 14.4, p[1], p[2])
            if frozen:
                cols += " 0"
            cols += " %.9f %.9f %.9f" % (s[0], s[1], s[2])
            out.append(cols)
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return n * nx


def read_all_spin_frames(path, natoms):
    lines = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
    frames = len(lines) // (natoms + 2)
    S = []
    for k in range(frames):
        block = lines[k * (natoms + 2):(k + 1) * (natoms + 2)][2:]
        S.append([[float(x) for x in l.split()[-3:]] for l in block])
    return np.asarray(S, dtype=np.float64)


def clean_outputs(d):
    for name in ("traj.xyz", "dump.xyz", "thermo.out", "spin_work.out",
                 "restart.xyz", "spin_restart.bin", "hac.out", "heat_flux.out"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            os.remove(p)


def collect(rec, bin_dir, work, baseline_bin=None, perf_csv=None):
    import sys
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    import v4_regression as V

    gpumd = os.path.join(bin_dir, "gpumd.exe" if os.name == "nt" else "gpumd")
    V.GPUMD = gpumd
    data = os.path.join(PKG, "tests", "data")
    nep = os.path.join(data, "fege.nep.txt")
    b20 = os.path.join(data, "fege_b20.xyz")
    os.makedirs(work, exist_ok=True)

    def fresh(tag, model_writer, *args):
        d = os.path.join(work, tag)
        # Do NOT rmtree: a work directory can hold >50 trajectory frames and
        # some execution environments abort bulk deletions.  Reuse the
        # directory and remove only the known output files instead.
        os.makedirs(d, exist_ok=True)
        clean_outputs(d)
        shutil.copy(nep, os.path.join(d, "nep.txt"))
        model_writer(os.path.join(d, "model.xyz"), *args)
        return d

    def run(d, lines, timeout=600):
        clean_outputs(d)
        return V.run_gpumd(d, lines)

    # ---------------------------------------------------------------- E_sllg
    # A frozen lattice is required for a clean equilibrium sample: with a
    # moving lattice the R19 runaway guard (designed for MD trajectories)
    # stops the run on the noise/rotation work ratio.  run.cu F20 switches
    # the guard to spin-only mode when every atom is fixed.
    for tag, B_ev, T_kelvin, steps in (
            ("B003_T300", 0.03, 300.0, 12000),
            ("B001_T300", 0.01, 300.0, 6000)):
        try:
            d = fresh("sllg_" + tag, write_two_atom_model, True)
            lines = ["potential nep.txt", "time_step 1.0", "ensemble nve",
                     "fix 0 0", "dump_xyz 5 traj.xyz spin",
                     "spin_llg gamma=3.04 alpha=1.0 T=%.1f seed=7 scheme=midpoint"
                     % T_kelvin,
                     "magfield %.10g 0 0 1 unit=ev" % B_ev,
                     "run %d" % steps]
            r = run(d, lines)
            if r.returncode != 0:
                rec("EXTRA.SLLG." + tag, "E_sllg", "FAIL",
                    "gpumd exit=%d" % r.returncode, rc=r.returncode)
                continue
            S = read_all_spin_frames(os.path.join(d, "traj.xyz"), 2)
            u = (S[:, :, 2] / np.linalg.norm(S, axis=2)).reshape(-1)
            burn = len(u) // 4
            x = u[burn:] - u[burn:].mean()
            kT = KB_EV_K * T_kelvin
            a = B_ev * 2.0 / kT
            mean_analytic = 1.0 / math.tanh(a) - 1.0 / a
            var_analytic = 1.0 / a ** 2 - 1.0 / math.sinh(a) ** 2
            # integrated autocorrelation time, initial positive sequence
            c0 = float(np.dot(x, x) / len(x))
            tau = 0.5
            for lag in range(1, min(len(x) // 4, 3000)):
                c = float(np.dot(x[:-lag], x[lag:]) / (len(x) - lag))
                if c <= 0:
                    break
                tau += c / c0
                if lag > 6 * tau:
                    break
            se = math.sqrt(var_analytic * 2.0 * tau / len(x))
            measured = float(u[burn:].mean())
            z = (measured - mean_analytic) / se
            # histogram against the analytic density, with binomial s.e.
            bins = np.linspace(-1.0, 1.0, 11)
            hist, _ = np.histogram(u[burn:], bins=bins)
            width = bins[1] - bins[0]
            p = a / (2.0 * math.sinh(a)) * np.exp(
                a * 0.5 * (bins[1:] + bins[:-1]))
            expected = p * width * len(x)
            hist_sigma = np.sqrt(np.maximum(expected, 1.0)) / (width * len(x))
            hist_dev = float(np.max(np.abs(hist / (width * len(x)) - p) /
                                    np.maximum(hist_sigma, 1e-12)))
            norm_err = float(np.max(np.abs(np.linalg.norm(S, axis=2) - 2.0)))
            ok = (abs(z) < 3.0 and hist_dev < 4.0 and norm_err < 1e-4 and
                  tau < 20.0 and "GUARD STOP" not in r.stdout)
            rec("EXTRA.SLLG." + tag, "E_sllg", "PASS" if ok else "FAIL",
                "B=%.3g eV/muB T=%.0f K a=%.4f; <u>=%.5f analytic=%.5f "
                "se=%.5f z=%.2f; tau_int=%.2f samples, max histogram "
                "deviation=%.2f sigma, |S| error=%.2g, N=%d" %
                (B_ev, T_kelvin, a, measured, mean_analytic, se, z, tau,
                 hist_dev, norm_err, len(x)),
                expected=mean_analytic, measured=measured,
                tol="3 sigma = %.5f" % (3 * se), rc=r.returncode,
                note="Analytic Boltzmann law p(u)=a/(2 sinh a) exp(a u) for "
                     "an isolated fixed-|S| spin; se uses the measured "
                     "integrated autocorrelation time (burn-in 25%).")
        except Exception as e:
            rec("EXTRA.SLLG." + tag, "E_sllg", "BLOCKED", err_text(e))

    # ------------------------------------------------------------- E_lattice
    cases = [
        ("none", []),
        ("static", ["magfield 5.0 0 0 1"]),
        ("me", ["magelec 4.5 0.001 1.0 0.0 0.0"]),
    ]
    for tag, extra in cases:
        for mode in ("frozen", "coupled"):
            tid = "EXTRA.LATTICE.%s.%s" % (tag, mode)
            try:
                d = fresh("lattice_%s_%s" % (tag, mode), write_b20_model,
                          b20, 1, 8123, mode == "frozen")
                lines = ["potential nep.txt", "time_step 0.05", "ensemble nve"]
                if mode == "frozen":
                    lines.append("fix 0 0")
                else:
                    lines.append("velocity 300 seed 1")
                lines += ["dump_thermo 1", "dump_xyz 1 traj.xyz spin",
                          "spin_llg gamma=3.04 alpha=0 T=0"] + extra + ["run 50"]
                r = run(d, lines)
                p = os.path.join(d, "traj.xyz")
                if r.returncode != 0 or not os.path.exists(p):
                    rec(tid, "E_lattice", "FAIL",
                        "exit=%d traj=%s" % (r.returncode,
                                             os.path.exists(p)), rc=r.returncode)
                    continue
                S = read_all_spin_frames(p, 216)
                # The shipped B20 cell has |S| = 2.0 on Fe and 0.11 on Ge;
                # compare each atom against its OWN initial anchor.
                anchor = np.linalg.norm(
                    V.read_xyz(os.path.join(d, "model.xyz"))["spin"], axis=1)
                raw = [l for l in open(p).read().splitlines() if l.strip()]
                stride = 216 + 2
                pos = []
                for k in range(S.shape[0]):
                    pos.append([[float(x) for x in l.split()[1:4]]
                                for l in raw[k * stride + 2:(k + 1) * stride]])
                pos = np.array(pos)
                disp = float(np.max(np.abs(pos - pos[0])))
                norm_err = float(np.max(
                    np.abs(np.linalg.norm(S, axis=2) - anchor[None, :])))
                rows = [l.split() for l in open(os.path.join(d, "thermo.out"))
                        if l.strip() and not l.startswith("#")]
                pe = np.array([float(x[2]) for x in rows])
                drift = float(np.max(np.abs(pe - pe[0])))
                finite = bool(np.all(np.isfinite(S)) and np.isfinite(drift))
                if mode == "frozen":
                    ok = finite and disp == 0.0 and norm_err < 1e-4
                    detail = ("max|dr-r0|=%.3g A (must be 0), |S|err=%.2g, "
                              "max|dPE|=%.3g eV" % (disp, norm_err, drift))
                else:
                    ok = finite and disp > 0.0 and norm_err < 1e-4
                    detail = ("max|dr-r0|=%.3g A (must be >0), |S|err=%.2g, "
                              "max|dPE|=%.3g eV" % (disp, norm_err, drift))
                rec(tid, "E_lattice", "PASS" if ok else "FAIL", detail,
                    rc=r.returncode)
            except Exception as e:
                rec(tid, "E_lattice", "BLOCKED", err_text(e))

    # ---------------------------------------------------------------- E_size
    for nx in (1, 2):
        for dt_kind, dt_fs, nsteps in (("normal", 0.05, 100),
                                       ("oversize", 5.0, 20)):
            tid = "EXTRA.SIZE.N%d.%s" % (216 * nx, dt_kind)
            try:
                d = fresh("size_%d_%s" % (nx, dt_kind), write_b20_model,
                          b20, nx, 8123, False)
                lines = ["potential nep.txt", "time_step %g" % dt_fs,
                         "ensemble nve", "velocity 300 seed 1",
                         "dump_xyz %d traj.xyz spin" % nsteps,
                         "spin_midpoint 1e-6 1e-6 3",
                         "spin_llg gamma=3.04 alpha=0 T=0",
                         "run %d" % nsteps]
                r = run(d, lines)
                out = (r.stdout or "") + (r.stderr or "")
                p = os.path.join(d, "traj.xyz")
                if dt_kind == "normal":
                    S = read_all_spin_frames(p, 216 * nx) \
                        if os.path.exists(p) else None
                    anchor = np.linalg.norm(
                        V.read_xyz(os.path.join(d, "model.xyz"))["spin"],
                        axis=1)
                    ne = float("nan") if S is None else float(np.max(np.abs(
                        np.linalg.norm(S, axis=2) - anchor[None, :])))
                    ok = (r.returncode == 0 and S is not None and
                          np.all(np.isfinite(S)) and ne < 1e-4)
                    detail = ("N=%d dt=%g fs %d steps exit=%d finite=%s "
                              "|S|err=%.2g" %
                              (216 * nx, dt_fs, nsteps, r.returncode,
                               S is not None, ne))
                else:
                    clean_fail = (r.returncode != 0 and
                                  "did not converge" in out.lower())
                    clean_finite = (r.returncode == 0 and
                                    "nan" not in out.lower())
                    ok = clean_fail or clean_finite
                    detail = ("N=%d dt=%g fs %d steps exit=%d "
                              "clean_hard_fail=%s clean_finite=%s" %
                              (216 * nx, dt_fs, nsteps, r.returncode,
                               clean_fail, clean_finite))
                rec(tid, "E_size", "PASS" if ok else "FAIL", detail,
                    rc=r.returncode,
                    note="Normal step must complete with finite spins and "
                         "conserved |S|; the deliberately oversized step must "
                         "either abort cleanly on the residual test or finish "
                         "with finite values.")
            except Exception as e:
                rec(tid, "E_size", "BLOCKED", err_text(e))

    # ---------------------------------------------------------------- E_work
    for wave in ("static", "sin"):
        try:
            # non-collinear initial spin: both precession and the explicit
            # dB/dt work are present, so a trivially constant trajectory
            # cannot make the static-field test pass.
            s0 = 2.0 / math.sqrt(2.0)
            d = fresh("work_" + wave, write_two_atom_model, True,
                      (s0, 0.0, s0))
            field = ("magfield 0.03 0 0 1 unit=ev" if wave == "static" else
                     "magfield 0.03 0 0 1 unit=ev wave=sin freq=2.0 phase=0.3")
            lines = ["potential nep.txt", "time_step 0.1", "ensemble nve",
                     "fix 0 0", "dump_xyz 1 traj.xyz spin", "dump_thermo 1",
                     "spin_llg gamma=3.04 alpha=0 T=0", field, "run 200"]
            r = run(d, lines)
            S = read_all_spin_frames(os.path.join(d, "traj.xyz"), 2)
            rows = [l.split() for l in open(os.path.join(d, "thermo.out"))
                    if l.strip() and not l.startswith("#")]
            pe = np.array([float(x[2]) for x in rows])
            m = min(S.shape[0], len(pe))
            dt_fs, freq, phase, b0 = 0.1, 2.0, 0.3, 0.03

            def B(k):
                if wave == "static":
                    return b0
                return b0 * math.sin(2 * math.pi * freq *
                                     ((k - 0.5) * dt_fs * 1e-3) + phase)

            dpe = pe[1:m] - pe[:m - 1]
            # NOTE: do NOT name this array `work`: `work` is the working
            # directory of collect(), and shadowing it with an ndarray makes
            # every subsequent os.path.join() raise a TypeError.
            w_drive = np.array(
                [-(B(k) - B(k - 1)) * 0.5 *
                 (S[k - 1, :, 2].sum() + S[k, :, 2].sum())
                 for k in range(1, m)])
            residual = float(np.abs(dpe - w_drive).max())
            total_change = float(pe[m - 1] - pe[0])
            w_total = float(w_drive.sum())
            norm_err = float(np.max(np.abs(np.linalg.norm(S, axis=2) - 2.0)))
            if wave == "static":
                ok = (r.returncode == 0 and abs(total_change) < 1e-4 and
                      residual < 1e-6 and norm_err < 1e-4)
            else:
                ok = (r.returncode == 0 and abs(total_change) > 1e-3 and
                      abs(total_change - w_total) < 1e-4 and
                      residual < 1e-6 and norm_err < 1e-4)
            rec("EXTRA.WORK." + wave.upper(), "E_work",
                "PASS" if ok else "FAIL",
                "dE_bare=%+.8g eV, W_ext(drive)=%+.8g eV, residual=%+.3g eV, "
                "max per-step |dE-W| = %.3g eV, |S|err=%.2g, exit=%d" %
                (total_change, w_total, total_change - w_total, residual,
                 norm_err, r.returncode),
                expected=w_total, measured=total_change, tol=1e-4,
                rc=r.returncode,
                note="Work of the EXPLICITLY time-dependent field is "
                     "W = -integral (dB/dt).S dt, evaluated independently from "
                     "the trajectory with the documented midpoint envelope "
                     "B(t)=B0 sin(2 pi f t + phase).  A static field must "
                     "conserve the bare energy; a driven field must NOT.")
        except Exception as e:
            rec("EXTRA.WORK." + wave.upper(), "E_work", "BLOCKED", err_text(e))

    # ---------------------------------------------------------------- E_perf
    try:
        _performance(rec, V, gpumd, baseline_bin, work, perf_csv, b20, nep)
    except Exception as e:
        rec("EXTRA.PERFORMANCE", "E_perf", "BLOCKED", err_text(e))


def _performance(rec, V, gpumd, baseline_bin, work, perf_csv, b20, nep):
    rows = []
    if not baseline_bin:
        rec("EXTRA.PERFORMANCE", "E_perf", "NOT_RUN",
            "no --baseline-bin given; the r20 binary is not shipped with the "
            "source package")
    else:
        base = os.path.join(baseline_bin,
                            "gpumd.exe" if os.name == "nt" else "gpumd")
        if not os.path.exists(base):
            rec("EXTRA.PERFORMANCE", "E_perf", "NOT_RUN",
                "baseline gpumd not found: " + base)
        else:
            steps, nx, dt = 2000, 1, 0.05
            for label, exe in (("r20", base), ("r21", gpumd)):
                d = os.path.join(work, "perf_" + label)
                os.makedirs(d, exist_ok=True)
                clean_outputs(d)
                shutil.copy(nep, os.path.join(d, "nep.txt"))
                write_b20_model(os.path.join(d, "model.xyz"), b20, nx, 8123,
                                False)
                lines = ["potential nep.txt", "time_step %g" % dt,
                         "ensemble nve", "velocity 300 seed 1",
                         "spin_llg gamma=3.04 alpha=0 T=0",
                         "run %d" % steps]
                V.run_gpumd  # ensure module is loaded
                clean_outputs(d)
                with open(os.path.join(d, "run.in"), "w") as f:
                    f.write("\n".join(lines) + "\n")
                peak = [0]
                stop = threading.Event()

                def sampler():
                    while not stop.is_set():
                        try:
                            q = subprocess.run(
                                ["nvidia-smi",
                                 "--query-gpu=memory.used",
                                 "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3)
                            vals = [int(v.strip()) for v in q.stdout.splitlines()
                                    if v.strip().isdigit()]
                            if vals:
                                peak[0] = max(peak[0], max(vals))
                        except Exception:
                            pass
                        stop.wait(0.25)

                th = threading.Thread(target=sampler, daemon=True)
                t0 = time.perf_counter()
                th.start()
                r = subprocess.run([exe], cwd=d, capture_output=True, text=True)
                elapsed = time.perf_counter() - t0
                stop.set()
                th.join(timeout=2)
                m = re.search(r"Speed of this run\s*=\s*([\d.eE+-]+)", r.stdout)
                reported = float(m.group(1)) if m else float("nan")
                diag = re.search(
                    r"conservative steps=(\d+) max_iterations=(\d+) "
                    r"max_residual=([\d.eE+-]+) mu_B field_evaluations=(\d+)",
                    r.stdout)
                iters = int(diag.group(4)) if diag else None
                measured_speed = 216 * nx * steps / elapsed
                # Never persist a machine-specific absolute path: the
                # delivered PERFORMANCE.csv must not depend on the local
                # checkout location.  Use a stable relative identifier.
                binary_label = ("baseline_r20/src/" + os.path.basename(exe)
                                if label == "r20"
                                else "src/" + os.path.basename(exe))
                rows.append({
                    "version": label, "binary": binary_label,
                    "natoms": 216 * nx,
                    "steps": steps, "dt_fs": dt, "wall_s": elapsed,
                    "atoms_steps_per_s_measured": measured_speed,
                    "atoms_steps_per_s_reported": reported,
                    "peak_gpu_mem_MiB": peak[0],
                    "midpoint_field_evaluations":
                        "" if iters is None else iters,
                    "residual_readbacks": "" if iters is None else 2 * iters,
                    "exit_code": r.returncode,
                })
            ok = len(rows) == 2 and all(x["exit_code"] == 0 for x in rows)
            ratio = (rows[1]["atoms_steps_per_s_measured"] /
                     rows[0]["atoms_steps_per_s_measured"]) if ok else float("nan")
            rec("EXTRA.PERFORMANCE", "E_perf", "PASS" if ok else "FAIL",
                "; ".join("%s wall=%.2fs speed=%.0f atom*step/s peak=%d MiB "
                          "field_evals=%s" %
                          (x["version"], x["wall_s"],
                           x["atoms_steps_per_s_measured"],
                           x["peak_gpu_mem_MiB"],
                           x["midpoint_field_evaluations"]) for x in rows) +
                "; r21/r20 speed ratio=%.3f" % ratio,
                note="Identical model, input seed, step count and GPU.  r20 "
                     "has no residual diagnostics, so its iteration/readback "
                     "columns are NOT_AVAILABLE rather than zero.  Each r21 "
                     "iteration performs two GPU->CPU readbacks (double[2] "
                     "residual and int[1] bad counter).")
            if perf_csv:
                with open(perf_csv, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    for x in rows:
                        w.writerow(x)
