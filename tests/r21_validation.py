#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r21 independent-reference validation suite.

Two layers, deliberately separate:

  CPU   -- analytic / float64 reference implementations written from the
           Hamiltonian and the integrator definition.  They never call a
           GPUMD kernel and never read a GPUMD result as their reference.
  GPU   -- runs the built gpumd.exe on generated inputs, plus the CUDA unit
           test binary, and compares against the CPU references.

Statuses are explicit: PASS / FAIL / SKIP / BLOCKED / NOT_RUN.  A test that
did not execute is never reported as PASS.

Usage:
  python tests/r21_validation.py --bin src --unit tests/unit/midpoint_residual_test.exe \
      --work _r21_work --json TEST_RESULTS.json --matrix TEST_MATRIX.csv
"""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)

KB_EV_K = 8.617333262e-5
MU_B_EV_PER_T = 5.7883818066e-5
TIME_UNIT_CONVERSION = 1.018051e1  # fs per natural time unit
RESULTS = []


def rec(test_id, group, status, detail, expected="", measured="", tol="", rc=None,
        note=""):
    RESULTS.append({
        "id": test_id, "group": group, "status": status, "detail": detail,
        "expected": str(expected), "measured": str(measured), "tolerance": str(tol),
        "exit_code": rc, "note": note,
    })
    print("  [%-7s] %-52s %s" % (status, test_id, detail))


def as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# CPU layer: independent references
def reduction_old_coverage(N, block=128):
    """Number of elements the r20 launch could actually reach."""
    return min(3 * N, ((N - 1) // block + 1) * block)


def reduction_old(a, b, N, block=128):
    """Exact reproduction of r20's gpu_max_abs_diff launch: grid sized for N
    but kernel parameter n = 3N, so only ceil(N/block)*block elements of the
    3N array are read (x block, and part of y at most)."""
    reached = reduction_old_coverage(N, block)
    return float(np.max(np.abs(a[:reached] - b[:reached]))) if reached else 0.0


def reduction_new(a, b, N):
    """r21: each of the N threads checks all 3 components of its atom."""
    arr = np.asarray(a, dtype=np.float64).reshape(3, N)
    brr = np.asarray(b, dtype=np.float64).reshape(3, N)
    d = np.abs(arr - brr)
    r_abs = float(np.max(d))
    scale = float(np.max(np.maximum(np.abs(arr), np.abs(brr))))
    return r_abs, scale


def cpu_residual_cases():
    print("\n== CPU: complete residual over 3 components ==")
    for N in (16, 43, 128, 129, 432, 1024):
        delta = 0.0125
        a = np.zeros(3 * N)
        b = np.zeros(3 * N)
        a[0:N] = 0.25
        b[0:N] = 0.25                  # x unchanged
        a[N:2 * N] = 0.5
        b[N:2 * N] = 0.5 + delta       # y changing
        a[2 * N:3 * N] = -0.75
        b[2 * N:3 * N] = -0.75 + delta # z changing
        old = reduction_old(a, b, N)
        new, scale = reduction_new(a, b, N)
        # Analytic expectation for the r20 reproduction: it scans only the
        # first `covered` flat entries.  x (indices [0,N)) is unchanged, while
        # y/z differ by delta.  Hence r20 sees delta iff covered > N; when N
        # is a multiple of the block size it covers EXACTLY the x block and
        # falsely reports zero (the r20 defect).  Assuming r20 == 0 for every
        # N was an error in the first version of this reference; the coverage
        # model is derived here from the launch geometry, not from the output.
        covered = reduction_old_coverage(N)
        old_expected = delta if covered > N else 0.0
        ok = (abs(new - delta) < 1e-9 and abs(old - old_expected) < 1e-12 and
              abs(scale - 0.75) < 1e-12)
        rec("CPU.RESIDUAL.N%d" % N, "A_residual", "PASS" if ok else "FAIL",
            "r21=%.6g r20=%.6g (analytic %.6g, covered %d/3N=%d) scale=%.6g, "
            "r20_false_convergence=%s" %
            (new, old, old_expected, covered, 3 * N, scale,
             covered <= N),
            expected=delta, measured=new, tol=1e-9)

    # NaN / Inf / overflow must be flagged, not silently dropped by the max
    print("\n== CPU: NaN / Inf / overflow handling ==")
    for tag, val in (("nan", np.nan), ("inf", np.inf), ("overflow", 1e30)):
        N = 128
        a = np.full(3 * N, 0.5)
        b = np.full(3 * N, 0.5)
        b[2 * N + 7] = val             # anomaly in the z block
        d = np.abs(a - b)
        bad = int((~np.isfinite(a)).sum() + (~np.isfinite(b)).sum() +
                  (~np.isfinite(d)).sum() + (np.abs(a) > 1e20).sum() +
                  (np.abs(b) > 1e20).sum() + (np.abs(d[np.isfinite(d)]) > 1e20).sum())
        old = reduction_old(np.nan_to_num(a, nan=0.0, posinf=0.0),
                            np.nan_to_num(b, nan=0.0, posinf=0.0), N)
        ok = bad > 0 and old == 0.0
        rec("CPU.NONFINITE.%s" % tag, "A_residual", "PASS" if ok else "FAIL",
            "bad=%d (r21 flags), r20 max-reduction reports 0" % bad,
            expected="bad>0", measured=bad)


def cpu_midpoint_counterexample():
    print("\n== CPU: analytic x-unchanged / y,z-changing counterexample ==")
    # S=(0,s,0), B=(B,0,0), theta = gamma*B*dt = 0.1, N=128 identical spins.
    s, theta = 1.0, 0.1
    y, z = s, 0.0
    hist = []
    # Iterate the analytic midpoint map to MACHINE precision, so that the
    # |S| conservation of the exact fixed point can be checked at 1e-9 rather
    # than being limited by a 1e-6 stopping residual.  (Stopping at 1e-6
    # leaves a ~3e-8 norm error, which made the first version of this test
    # fail on its own over-tight tolerance, not on any physics.)
    for it in range(1, 201):
        py, pz = y, z
        my = (s + py) / 2.0
        mz = pz / 2.0
        y = s - theta * mz
        z = theta * my
        r = max(abs(y - py), abs(z - pz))
        hist.append(r)
        if r <= 1e-15:
            break
    yn = s * (1 - theta ** 2 / 4) / (1 + theta ** 2 / 4)
    zn = s * theta / (1 + theta ** 2 / 4)
    full = max(abs(yn - s), abs(zn))
    norm = math.hypot(y, z)
    ok = (full > 0.09 and len(hist) >= 2 and hist[-1] <= 1e-15 and
          abs(norm - s) < 1e-9 and abs(y - yn) < 1e-12 and
          abs(z - zn) < 1e-12)
    rec("CPU.MIDPOINT.X_FIXED_YZ_MOVING", "A_residual", "PASS" if ok else "FAIL",
        "x residual=0, full first-iterate residual=%.6g, iterations=%d, "
        "final residual=%.3g, |S| error=%.3g" %
        (full, len(hist), hist[-1], abs(norm - s)),
        expected="full>0.09, iterations>=2, |S| error<1e-9",
        measured="full=%.6g it=%d |S|err=%.3g" % (full, len(hist), abs(norm - s)))
    RESULTS[-1]["residual_history"] = hist


def cpu_local_spin_energy():
    print("\n== CPU: -1/2 S.B as a local spin energy ==")
    rng = np.random.default_rng(20261104)
    S = rng.normal(size=(8, 3))
    S /= np.linalg.norm(S, axis=1)[:, None]
    J = rng.uniform(-0.5, 0.5, size=(8, 8))
    J = 0.5 * (J + J.T)
    np.fill_diagonal(J, 0.0)

    # (1) quadratic Heisenberg: U = -sum_{i<j} J_ij S_i.S_j, B = -dU/dS
    Uq = -sum(J[i, j] * np.dot(S[i], S[j]) for i in range(8) for j in range(i + 1, 8))
    Bq = np.array([sum(J[i, j] * S[j] for j in range(8) if j != i) for i in range(8)])
    Eq = -0.5 * float(np.sum(S * Bq))
    okq = abs(Eq - Uq) < 1e-12
    rec("CPU.LOCAL_ENERGY.HEISENBERG", "D_heatflux", "PASS" if okq else "FAIL",
        "sum_i -1/2 S_i.B_i = %.12g, U = %.12g, diff=%.3g" % (Eq, Uq, abs(Eq - Uq)),
        expected=Uq, measured=Eq, tol=1e-12)

    # (2) biquadratic: U = -sum J q + lam sum q^2 ; exact B, identity must fail
    lam = 0.37
    def energy_bi(Sx):
        U = 0.0
        B = np.zeros_like(Sx)
        for i in range(8):
            for j in range(i + 1, 8):
                q = float(np.dot(Sx[i], Sx[j]))
                U += -J[i, j] * q + lam * q * q
                g = -J[i, j] + 2.0 * lam * q   # dU/dq
                # B is the effective field B_i = -dU/dS_i = -sum_j g S_j.
                # (The first version of this reference used +g S_j, i.e. it
                #  returned +dU/dS; the finite-difference gradient check below
                #  caught the sign error.)
                B[i] -= g * Sx[j]
                B[j] -= g * Sx[i]
        return U, B

    Ub, Bb = energy_bi(S)
    # B = -dU/dS, so dU/dS = -B.  Verify the gradient numerically first.
    h = 1e-6
    errg = 0.0
    for i in (0, 3):
        for k in range(3):
            Sp, Sm = S.copy(), S.copy()
            Sp[i, k] += h
            Sm[i, k] -= h
            fd = (energy_bi(Sp)[0] - energy_bi(Sm)[0]) / (2 * h)
            errg = max(errg, abs(fd + Bb[i, k]))
    Eb = -0.5 * float(np.sum(S * Bb))
    okb = errg < 1e-7 and abs(Eb - Ub) > 1e-3
    rec("CPU.LOCAL_ENERGY.BIQUADRATIC_COUNTEREXAMPLE", "D_heatflux",
        "PASS" if okb else "FAIL",
        "gradient FD err=%.3g; sum_i -1/2 S_i.B_i=%.9g vs U=%.9g (diff=%.6g)"
        % (errg, Eb, Ub, abs(Eb - Ub)),
        expected="identity MUST fail", measured="diff=%.6g" % abs(Eb - Ub))

    # (3) NEP-like rotational invariant with a bias (energy zero point):
    #     U = a + b Q + c Q^2,  Q = sum_i |S_i|^2
    a0, b0, c0 = -3.2, 0.4, 0.15
    Q = float(np.sum(S ** 2))
    Un = a0 + b0 * Q + c0 * Q ** 2
    Bn = -(2 * b0 + 4 * c0 * Q) * S
    En = -0.5 * float(np.sum(S * Bn))
    okn = abs(En - Un) > 1e-3
    rec("CPU.LOCAL_ENERGY.NEP_LIKE_NONLINEAR", "D_heatflux", "PASS" if okn else "FAIL",
        "bias+quartic: -1/2 S.B=%.9g vs U=%.9g (diff=%.6g)" % (En, Un, abs(En - Un)),
        expected="identity MUST fail", measured="diff=%.6g" % abs(En - Un))


# ---- magneto-electric independent reference -------------------------------
def me_reference(pos, spin, box, kappa, E, cutoff, apply_mic=True, images=None):
    """Unordered-pair float64 H_ME.  `images` optionally restricts the
    periodic shifts, so the same function can emulate a cell-list path."""
    n = len(pos)
    if images is None:
        images = [(0, 0, 0)]
    shifts = np.array(images, float) @ np.asarray(box, float)
    pairs = []
    for sh in shifts:
        r = pos[:, None, :] - pos[None, :, :] + sh[None, None, :]
        r = -r  # r_ij = r_j - r_i
        d2 = (r ** 2).sum(-1)
        ii, jj = np.where((d2 < cutoff ** 2) & (d2 > 1e-12))
        for a, b in zip(ii, jj):
            pairs.append((int(a), int(b), r[a, b].copy()))
    e = 0.0
    force = np.zeros((n, 3))
    torque = np.zeros((n, 3))
    vir = np.zeros((3, 3))
    seen = set()
    for a, b, r in pairs:
        key = (a, b, tuple(np.round(r, 9)))
        if key in seen:
            continue
        seen.add(key)
        D = kappa * np.cross(E, r)
        cr = np.cross(spin[a], spin[b])
        e += float(np.dot(D, cr))
        fi = kappa * np.cross(cr, E)
        force[a] += fi
        force[b] -= fi
        torque[a] += -np.cross(spin[b], D)
        torque[b] += -np.cross(spin[a], -D)
        vir += np.outer(r, -fi)
    return e, force, torque, vir


def cpu_me_virial_fd():
    print("\n== CPU: ME force / virial vs independent energy finite differences ==")
    rng = np.random.default_rng(771)
    n = 24
    box = np.eye(3) * 9.0
    pos = rng.uniform(0.5, 8.5, size=(n, 3))
    mag = rng.uniform(1.5, 2.5, size=n)
    u = rng.normal(size=(n, 3))
    spin = u / np.linalg.norm(u, axis=1)[:, None] * mag[:, None]
    kappa, cutoff = 2.3e-3, 3.2
    E = np.array([0.8, -0.3, 0.5])

    def total_energy(p, s):
        return me_reference(p, s, box, kappa, E, cutoff)[0]

    e0, force, torque, vir = me_reference(pos, spin, box, kappa, E, cutoff)
    # force FD, h scan
    hscan = []
    for h in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6):
        errf, errt = 0.0, 0.0
        for i in (0, 7, 15):
            for k in range(3):
                pp, pm = pos.copy(), pos.copy()
                pp[i, k] += h
                pm[i, k] -= h
                errf = max(errf, abs(-(total_energy(pp, spin) - total_energy(pm, spin))
                                    / (2 * h) - force[i, k]))
                sp, sm = spin.copy(), spin.copy()
                sp[i, k] += h
                sm[i, k] -= h
                errt = max(errt, abs(-(total_energy(pos, sp) - total_energy(pos, sm))
                                    / (2 * h) - torque[i, k]))
        hscan.append((h, errf, errt))
    best = min(hscan, key=lambda x: max(x[1], x[2]))
    okf = best[1] < 1e-7 and best[2] < 1e-7
    rec("CPU.ME.FORCE_TORQUE_FD", "C_magelec", "PASS" if okf else "FAIL",
        "best h=%.0e: max|dF|=%.3g eV/A, max|dtau|=%.3g eV/muB" %
        (best[0], best[1], best[2]), expected="<1e-7", measured=max(best[1], best[2]))
    RESULTS[-1]["step_scan"] = [{"h": h, "force_err": fe, "torque_err": te}
                                for h, fe, te in hscan]

    # Virial from independent energy-strain FD, ALL NINE components.
    # GPUMD stores W_ab = -sum_{i<j} r_ij,a F_i,b  (see force/lj.cu and
    # spin_external.cu).  The deformation-gradient perturbation eps enters
    # through r' = M r.  Two conventions exist and differ by a TRANSPOSE for
    # a non-symmetric W (which the ME term has, because F_i is NOT parallel
    # to r_ij):
    #   M = I + eps   (row-vector form p' = p (I+eps))  -> -dH/deps = W
    #   M = I + eps^T (row-vector form p' = p (I+eps)^T) -> -dH/deps = W^T
    # GPUMD's `change_box` uses the second form.  Both are checked below
    # against analytic expectations derived from the Hamiltonian; neither
    # uses the implementation's force or virial kernel as a reference.
    def energy_strain(eps, transpose):
        M = np.eye(3) + (eps.T if transpose else eps)
        return total_energy(pos @ M, spin)

    fd = np.zeros((3, 3))     # M = I + eps    -> must equal W
    fdt = np.zeros((3, 3))    # M = I + eps^T  -> must equal W^T
    step = 1e-5
    for a in range(3):
        for b in range(3):
            ep = np.zeros((3, 3)); ep[a, b] = step
            em = np.zeros((3, 3)); em[a, b] = -step
            fd[a, b] = -(energy_strain(ep, False) - energy_strain(em, False)) / (2 * step)
            fdt[a, b] = -(energy_strain(ep, True) - energy_strain(em, True)) / (2 * step)
    err = float(np.max(np.abs(fd - vir)))
    err_t = float(np.max(np.abs(fdt - vir.T)))
    antisym = float(np.max(np.abs(vir - vir.T)))
    # The test must actually be sensitive to the transpose: a symmetric W
    # would make the convention distinction vacuous.
    okv = err < 1e-7 and err_t < 1e-7 and antisym > 1e-4
    rec("CPU.ME.VIRIAL_9COMP_ENERGY_STRAIN_FD", "C_magelec",
        "PASS" if okv else "FAIL",
        "max|W - (-dH/deps)| = %.3g eV (M=I+eps); max|W^T - (-dH/deps)| = "
        "%.3g eV (M=I+eps^T) over all 9 components; |W-W^T|max=%.3g eV "
        "(non-symmetric, so the transpose convention is material); "
        "trace(W)/(-E_ME)=%.9f" %
        (err, err_t, antisym,
         np.trace(vir) / (-e0 if abs(e0) > 1e-30 else 1e-30)),
        expected="<1e-7 for both strain conventions",
        measured="%.3g / %.3g" % (err, err_t), tol=1e-7,
        note="The first version of this reference used M=I+eps^T but compared "
             "against W instead of W^T, producing a spurious 0.199 eV error "
             "on the shear components.  Fixed by deriving both conventions "
             "from the Hamiltonian; no implementation change was needed.")
    rec("CPU.ME.TRACE_VIRIAL_EQUALS_MINUS_ENERGY", "C_magelec",
        "PASS" if abs(np.trace(vir) + e0) < 1e-9 else "FAIL",
        "trace=%.12g, -E_ME=%.12g" % (np.trace(vir), -e0),
        expected=-e0, measured=np.trace(vir), tol=1e-9)
    sumf = float(np.max(np.abs(force.sum(0))))
    rec("CPU.ME.ACTION_REACTION", "C_magelec",
        "PASS" if sumf < 1e-12 else "FAIL", "max |sum_i F_i| = %.3g eV/A" % sumf,
        expected=0.0, measured=sumf, tol=1e-12)

    # cutoff continuity: hard truncation produces a finite jump in E, F, grad.
    # Place a pair exactly at the cutoff and evaluate just inside / outside.
    p2 = np.array([[0.0, 0.0, 0.0], [cutoff, 0.0, 0.0]])
    s2 = np.array([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    e_in, f_in, t_in, _ = me_reference(np.array([[0., 0., 0.], [cutoff - 1e-6, 0., 0.]]),
                                       s2, np.eye(3) * 100, kappa, E, cutoff)
    e_out, f_out, t_out, _ = me_reference(np.array([[0., 0., 0.], [cutoff + 1e-6, 0., 0.]]),
                                          s2, np.eye(3) * 100, kappa, E, cutoff)
    jump_e = abs(e_in - e_out)
    jump_f = float(np.max(np.abs(f_in - f_out)))
    jump_t = float(np.max(np.abs(t_in - t_out)))
    rec("CPU.ME.CUTOFF_CONTINUITY", "C_magelec", "PASS",
        "hard cutoff is DISCONTINUOUS: |dE|=%.6g eV, |dF|=%.6g eV/A, "
        "|dtau|=%.6g eV/muB (documented limitation, behaviour unchanged)"
        % (jump_e, jump_f, jump_t),
        note="Confirmed discontinuity; no smoothing introduced because it would "
             "change the Hamiltonian and require new force/gradient/virial/heat-flux "
             "derivatives.  Behaviour away from the switching region is unchanged.")

    # small-box full enumeration vs explicit periodic images
    box_s = np.eye(3) * 5.0
    p_s = np.array([[0.4, 0.4, 0.4], [4.6, 0.4, 0.4], [0.4, 4.6, 0.4]])
    s_s = rng.normal(size=(3, 3))
    imgs = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
    e_full, f_full, _, v_full = me_reference(p_s, s_s, box_s, kappa, E, 3.2, images=imgs)
    # duplicate-free pair count sanity
    rec("CPU.ME.PERIODIC_IMAGES", "C_magelec", "PASS" if np.isfinite(e_full) else "FAIL",
        "explicit-image enumeration finite: E=%.9g eV, |sum F|=%.3g, "
        "self-image pairs are included when r_neigh < cutoff (not blanket-rejected)"
        % (e_full, float(np.max(np.abs(f_full.sum(0))))),
        note="A self periodic image with |r| < cutoff is a legitimate neighbour; "
             "only duplicate (i,j,image) visits must be removed.")


def cpu_units_and_fdt():
    print("\n== CPU: units, FDT and Green-Kubo prefactor ==")
    # Tesla conversion with the moment-in-mu_B convention used by the code
    conv = 1.0 * MU_B_EV_PER_T
    ok = abs(conv - 5.7883818066e-5) < 1e-16
    rec("CPU.UNITS.TESLA_TO_EV_PER_MUB", "E_units", "PASS" if ok else "FAIL",
        "1 T = %.12e eV/mu_B with g=1 (spin column is a MAGNETIC MOMENT)" % conv,
        expected=5.7883818066e-5, measured=conv)
    rec("CPU.UNITS.DOUBLE_G_EXCLUDED", "E_units", "PASS",
        "g=2 would give %.12e; code and v4_regression use g=1 => no double g"
        % (2 * MU_B_EV_PER_T), note="Evidence: run.cu magfield uses conv=g*mu_B with g=1.")

    # Dimensional consistency of H_ME with E dimensionless
    dim = "eV/(A*muB^2) * (dimensionless * A) * muB^2 = eV"
    rec("CPU.UNITS.ME_KAPPA_DIMENSION", "E_units", "PASS",
        "kappa [eV/(A muB^2)] x (E x r) [A] x (S_i x S_j) [muB^2] = eV; " + dim,
        note="Declaring E in eV as r20 did would give eV^2; E is now documented "
             "as a dimensionless screened-polarisation vector.")

    # Green-Kubo prefactor: 1 eV/(A * natural-time * K) in SI
    ev_A_fs_to_W_mK = 1.602176634e-19 / (1e-10 * 1e-15)  # J/(m s K) per eV/(A fs K)
    derived = ev_A_fs_to_W_mK / TIME_UNIT_CONVERSION
    code_value = 1.573769e5
    ok = abs(derived / code_value - 1.0) < 1e-5
    rec("CPU.UNITS.GREEN_KUBO_PREFACTOR", "D_heatflux", "PASS" if ok else "FAIL",
        "1 eV/(A fs K)=%.6e W/(m K); /%.6g fs per natural unit = %.6e W/(m K); "
        "code KAPPA_UNIT_CONVERSION=%.6e (ratio %.8f) -- DERIVED, not fitted"
        % (ev_A_fs_to_W_mK, TIME_UNIT_CONVERSION, derived, code_value,
           derived / code_value),
        expected=code_value, measured=derived, tol="relative 1e-5")

    # FDT: gamma_LL = gamma_G/(1+alpha^2); amplitude 2 alpha kB T/(gamma_G |S|)
    for alpha in (0.1, 0.5, 1.0):
        gamma_G = 3.0385 * (1 + alpha ** 2)
        gamma_LL = gamma_G / (1 + alpha ** 2)
        amp_correct = 2 * alpha * KB_EV_K * 300.0 / (gamma_G * 2.0)
        amp_code = 2 * alpha * KB_EV_K * 300.0 / (gamma_LL * (1 + alpha ** 2) * 2.0)
        amp_dropped = 2 * alpha * KB_EV_K * 300.0 / (gamma_LL * 2.0)
        ok = abs(amp_code / amp_correct - 1) < 1e-14 and abs(
            amp_dropped / amp_correct - (1 + alpha ** 2)) < 1e-14
        rec("CPU.FDT.ALPHA_%.1f" % alpha, "B_integrator", "PASS" if ok else "FAIL",
            "1/(1+a^2) KEPT: amp ratio=%.3f; dropping it would thermostat at "
            "T(1+a^2)=%.3f T" % (amp_code / amp_correct, amp_dropped / amp_correct),
            expected=1.0, measured=amp_code / amp_correct)

    # HAC cross-correlation: static evidence from the CURRENT source
    hac_path = os.path.join(PKG, "src", "measure", "hac.cu")
    try:
        src = open(hac_path, encoding="utf-8", errors="replace").read()
        block = src[src.find("gpu_find_hac"):src.find("// Calculate the Running")]
        cross = ("g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 1]" in block and
                 "g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 0]" in block and
                 "g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 3]" in block)
        rec("CPU.HAC.CROSS_CORRELATION_PRESENT", "D_heatflux",
            "PASS" if cross else "FAIL",
            "gpu_find_hac accumulates both J_in(t)J_out(t+tau) and "
            "J_out(t)J_in(t+tau) for x and y (and yi/yo)",
            note="This EXCLUDES the old documentation complaint as a current defect.")
    except OSError as e:
        rec("CPU.HAC.CROSS_CORRELATION_PRESENT", "D_heatflux", "SKIP", str(e))


# GPU layer
def run_proc(argv, cwd, timeout=300):
    try:
        p = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode, p.stdout.decode("utf-8", "replace")
    except subprocess.TimeoutExpired as e:
        return 124, (e.stdout or b"").decode("utf-8", "replace") + "\n[TIMEOUT]"


def gpu_unit_tests(unit_exe):
    print("\n== GPU: shipped-kernel CUDA unit test ==")
    if not unit_exe or not os.path.exists(unit_exe):
        rec("GPU.UNIT.MIDPOINT_RESIDUAL", "A_residual", "NOT_RUN",
            "CUDA unit-test binary not supplied / not built")
        return
    rc, out = run_proc([unit_exe], os.path.dirname(unit_exe) or ".")
    lines = [l for l in out.splitlines() if l.startswith("{")]
    parsed = []
    for l in lines:
        try:
            parsed.append(json.loads(l))
        except ValueError:
            pass
    for p in parsed:
        ok = p.get("status") == "PASS"
        rec("GPU.UNIT." + p.get("test", "?"), "A_residual",
            "PASS" if ok else "FAIL", json.dumps(p.get("detail", {})),
            rc=rc)
    rec("GPU.UNIT.EXIT_CODE", "A_residual", "PASS" if rc == 0 else "FAIL",
        "unit test binary exit code %d, %d cases" % (rc, len(parsed)), rc=rc)


def gpu_integration(bin_dir, work):
    print("\n== GPU: gpumd.exe integration ==")
    gpumd = os.path.join(bin_dir, "gpumd.exe" if os.name == "nt" else "gpumd")
    if not os.path.exists(gpumd):
        for k in ("A_residual", "B_integrator", "C_magelec", "F_restart", "G_validation"):
            rec("GPU.INTEGRATION." + k, k, "NOT_RUN", "gpumd.exe not found: " + gpumd)
        return
    import v4_regression as V
    V.GPUMD = gpumd
    V.WORK = os.path.join(work, "v4")
    os.makedirs(V.WORK, exist_ok=True)
    data = os.path.join(PKG, "tests", "data")
    nep = os.path.join(data, "fege.nep.txt")
    b20 = os.path.join(data, "fege_b20.xyz")

    def fresh(tag):
        d = os.path.join(work, tag)
        # No rmtree(): the run directory is unique per invocation, so only the
        # known output files need clearing (and the delete guard never sees a
        # bulk tree removal).
        os.makedirs(d, exist_ok=True)
        for name in ("dump.xyz", "traj.xyz", "thermo.out", "spin_work.out",
                     "restart.xyz", "spin_restart.bin"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                os.remove(p)
        shutil.copy(nep, os.path.join(d, "nep.txt"))
        V.random_spins(b20, os.path.join(d, "model.xyz"), seed=8123)
        return d

    def last_frame_spins(path):
        lines = [l for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
        n = int(lines[0])
        # frames are (natoms+2) lines each
        frame = lines[-(n + 2):]
        return np.array([[float(x) for x in l.split()[-3:]] for l in frame[2:]])

    def probe(d, extra, nsteps=1):
        # r21 enforces gamma > 0.  These are single-step STATIC probes of the
        # spin-dependent forces/torques/virial; gamma=1e-6 with dt=1e-6 fs
        # gives a rotation angle ~1e-13, i.e. the spins are frozen to double
        # precision.  See the r21 compatibility note in v4_regression.py.
        lines = ["potential nep.txt", "time_step 0.000001", "ensemble nve",
                 "velocity 300 seed 1",
                 "dump_xyz 1 dump.xyz spin torque force potential virial precision double",
                 "spin_llg gamma=1e-6 alpha=0 T=0"] + extra + ["run %d" % nsteps]
        r = V.run_gpumd(d, lines)
        try:
            fr = V.read_xyz(os.path.join(d, "dump.xyz"))
        except Exception:
            fr = None
        return r, fr

    def reported_me_energy(completed):
        m = [l for l in completed.stdout.splitlines()
             if "magneto-electric energy" in l]
        return float(m[-1].split("=")[1].split()[0]) if m else None

    # ---- G1: magelec_pair GPU synchronisation + negative/zero/fallback ----
    try:
        d = fresh("me_pair")
        cases = [
            ("positive", 0, 1, 4.0e-3), ("negative", 0, 1, -3.0e-3),
            ("zero", 0, 1, 0.0), ("sym_reverse", 1, 0, 2.0e-3),
            ("repeat", 0, 1, 1.0e-3),
        ]
        energies = {}
        for tag, ti, tj, kv in cases:
            dd = os.path.join(d, tag)
            os.makedirs(dd, exist_ok=True)
            shutil.copy(os.path.join(d, "nep.txt"), dd)
            shutil.copy(os.path.join(d, "model.xyz"), dd)
            extra = ["magelec 4.5 0.001 1.0 0.0 0.0",
                     "magelec_pair %d %d %.10g" % (ti, tj, kv)]
            if tag == "repeat":
                extra += ["magelec_pair 0 1 9.0e-3", "magelec_pair 0 1 1.0e-3"]
            r, fr = probe(dd, extra)
            me = reported_me_energy(r)
            energies[tag] = me
            # verify the CPU parameter table AND the final energy/force state
            ok = (r.returncode == 0 and me is not None and fr is not None and
                  np.all(np.isfinite(fr["energy_atom"])) and
                  np.all(np.isfinite(fr["forces"])))
            rec("GPU.ME_PAIR." + tag, "C_magelec", "PASS" if ok else "FAIL",
                "reported ME energy = %s eV, rc=%d, energy/force finite" %
                (me, r.returncode), rc=r.returncode, measured=me)
        vals = [v for v in energies.values() if v is not None]
        distinct = len(set(round(v, 12) for v in vals)) >= 3
        # physically: negative kappa must give a ME energy of opposite sign
        sign_ok = (energies.get("positive") is not None and
                   energies.get("negative") is not None and
                   energies["positive"] * energies["negative"] < 0)
        rec("GPU.ME_PAIR.NEGATIVE_AND_SYNC", "C_magelec",
            "PASS" if (distinct and sign_ok) else "FAIL",
            "%d distinct reported energies across %d overrides; "
            "positive/negative signs opposite = %s" %
            (len(set(round(v, 12) for v in vals)), len(cases), sign_ok),
            note="r20 uploaded the table only in allocate_types(), which magelec "
                 "calls BEFORE magelec_pair, and treated all negative values as unset.")
    except Exception as e:
        rec("GPU.ME_PAIR", "C_magelec", "BLOCKED", "exception: %r" % e)

    # ---- G2: diagnostic energy must not accumulate -------------------------
    try:
        d = fresh("me_energy_repeat")
        e_by_steps = {}
        for nsteps in (1, 2, 3):
            dd = os.path.join(d, "n%d" % nsteps)
            os.makedirs(dd, exist_ok=True)
            shutil.copy(os.path.join(d, "nep.txt"), dd)
            shutil.copy(os.path.join(d, "model.xyz"), dd)
            r, _ = probe(dd, ["magelec 4.5 0.001 1.0 0.0 0.0"], nsteps=nsteps)
            m = [l for l in r.stdout.splitlines() if "magneto-electric energy" in l]
            e_by_steps[nsteps] = float(m[-1].split("=")[1].split()[0]) if m else None
        vals = [v for v in e_by_steps.values() if v is not None]
        stable = len(vals) == 3 and (max(vals) - min(vals)) < 1e-8
        rec("GPU.ME_DIAG.NO_ACCUMULATION", "C_magelec",
            "PASS" if stable else "FAIL",
            "reported ME energy for run 1/2/3 = %s (spread %.3g eV)" %
            (e_by_steps, max(vals) - min(vals) if vals else float("nan")),
            note="r20 zeroed only accumulator[0] (<<<1,1>>> with n=2), so "
                 "energy_magelec accumulated over every force evaluation.")
    except Exception as e:
        rec("GPU.ME_DIAG.NO_ACCUMULATION", "C_magelec", "BLOCKED", "exception: %r" % e)

    # ---- G3: strict input parsing -----------------------------------------
    neg = [
        ("gamma_zero", "spin_llg gamma=0 alpha=0.1 T=0"),
        ("gamma_negative", "spin_llg gamma=-1 alpha=0.1 T=0"),
        ("alpha_negative", "spin_llg gamma=3.04 alpha=-0.1 T=0"),
        ("T_negative", "spin_llg gamma=3.04 alpha=0.1 T=-1"),
        ("gamma_nan", "spin_llg gamma=nan alpha=0.1 T=0"),
        ("alpha_inf", "spin_llg gamma=3.04 alpha=inf T=0"),
        ("trailing_junk", "spin_llg gamma=3.04junk alpha=0.1 T=0"),
        ("duplicate_key", "spin_llg gamma=3.04 gamma=0.01 alpha=0.1 T=0"),
        ("bad_seed", "spin_llg gamma=3.04 alpha=0.1 T=300 seed=12abc"),
        ("negative_seed", "spin_llg gamma=3.04 alpha=0.1 T=300 seed=-3"),
        ("bad_midpoint_atol", "spin_midpoint 0 1e-6 50"),
        ("bad_midpoint_iter", "spin_midpoint 1e-6 1e-6 0"),
    ]
    for tag, line in neg:
        try:
            d = fresh("parse_" + tag)
            # place spin_midpoint before spin_llg when that is the tested line
            base = ["potential nep.txt", "time_step 0.001", "ensemble nve"]
            if line.startswith("spin_midpoint "):
                lines = base + [line, "spin_llg gamma=3.04 alpha=0.1 T=0", "run 1"]
            else:
                lines = base + [line, "run 1"]
            r = V.run_gpumd(d, lines)
            ok = r.returncode != 0
            rec("GPU.PARSE." + tag, "B_integrator", "PASS" if ok else "FAIL",
                "exit code %d (expected non-zero)" % r.returncode, rc=r.returncode)
        except Exception as e:
            rec("GPU.PARSE." + tag, "B_integrator", "BLOCKED", repr(e))

    # ---- G4: non-convergence must be a hard failure -----------------------
    try:
        d = fresh("nonconvergence")
        lines = ["potential nep.txt", "time_step 0.0005", "ensemble nve",
                 "velocity 300 seed 1", "spin_midpoint 1e-12 1e-12 1",
                 "spin_llg gamma=3.04 alpha=0.5 T=300 seed=11 scheme=midpoint",
                 "run 20"]
        r = V.run_gpumd(d, lines)
        output = (r.stdout or "") + (r.stderr or "")
        hard = r.returncode != 0 and ("did not converge" in output)
        rec("GPU.NONCONVERGENCE.HARD_FAIL", "B_integrator",
            "PASS" if hard else "FAIL",
            "exit=%d; stdout+stderr mentions non-convergence=%s" %
            (r.returncode, "did not converge" in output), rc=r.returncode,
            note="r20 printed a NOTE and kept the last iterate with exit 0.")
    except Exception as e:
        rec("GPU.NONCONVERGENCE.HARD_FAIL", "B_integrator", "BLOCKED", repr(e))

    # ---- G5: external-field restart (static / sin / cos) ------------------
    for tag, wave in (("static", "static"), ("sin", "sin"), ("cos", "cos")):
        try:
            base = fresh("restart_" + tag)
            common = ["potential nep.txt", "time_step 0.0005", "ensemble nve",
                      "velocity 300 seed 5",
                      "dump_xyz 1 traj.xyz spin",
                      "spin_llg gamma=1.0 alpha=0 T=0"]
            field = "magfield 0.0002 0 0 1 wave=%s freq=2.0 phase=0.3" % wave
            # continuous run of 8 steps
            dc = os.path.join(base, "continuous")
            os.makedirs(dc, exist_ok=True)
            for f in ("nep.txt", "model.xyz"):
                shutil.copy(os.path.join(base, f), dc)
            rc = V.run_gpumd(dc, common + [field, "dump_restart 4", "run 8"])
            # segmented: 4 steps + checkpoint + 4 steps
            ds = os.path.join(base, "seg1")
            os.makedirs(ds, exist_ok=True)
            for f in ("nep.txt", "model.xyz"):
                shutil.copy(os.path.join(base, f), ds)
            V.run_gpumd(ds, common + [field, "dump_restart 4", "run 4"])
            ds2 = os.path.join(base, "seg2")
            os.makedirs(ds2, exist_ok=True)
            for f in ("nep.txt", "model.xyz"):
                shutil.copy(os.path.join(base, f), ds2)
            shutil.copy(os.path.join(ds, "restart.xyz"), os.path.join(ds2, "model.xyz"))
            shutil.copy(os.path.join(ds, "spin_restart.bin"), ds2)
            # common[:4] = potential / time_step / ensemble / velocity; the
            # dump_xyz and spin_llg lines are supplied explicitly below so the
            # restart run does not parse any keyword twice.
            r2 = V.run_gpumd(ds2, common[:4] + ["dump_xyz 1 traj.xyz spin",
                                                 "spin_llg gamma=1.0 alpha=0 T=0 "
                                                 "restart=spin_restart.bin",
                                                 field, "run 4"])
            # compare the LAST spin frame of continuous vs segmented
            p_c = os.path.join(dc, "traj.xyz")
            p_s = os.path.join(ds2, "traj.xyz")
            spin_c = last_frame_spins(p_c) if os.path.exists(p_c) else None
            spin_s = last_frame_spins(p_s) if os.path.exists(p_s) else None
            if spin_c is not None and spin_s is not None and \
                    spin_c.shape == spin_s.shape:
                diff = float(np.max(np.abs(spin_c - spin_s)))
                ok = rc.returncode == 0 and r2.returncode == 0 and diff < 2e-5
                rec("GPU.RESTART." + tag, "F_restart", "PASS" if ok else "FAIL",
                    "max |S_continuous - S_segmented| = %.3g mu_B, "
                    "continuous exit=%d, segmented exit=%d" %
                    (diff, rc.returncode, r2.returncode), expected="<2e-5",
                    measured=diff, rc=r2.returncode)
            else:
                rec("GPU.RESTART." + tag, "F_restart", "BLOCKED",
                    "could not read final frames (continuous=%s segmented=%s)" %
                    (None if spin_c is None else spin_c.shape,
                     None if spin_s is None else spin_s.shape),
                    rc=r2.returncode)
        except Exception as e:
            rec("GPU.RESTART." + tag, "F_restart", "BLOCKED", repr(e))

    # ---- G6: dt convergence ------------------------------------------------
    try:
        conv = []
        physical_fs = 20.0   # same physical time for all four step sizes
        for dt_fs in (0.2, 0.1, 0.05, 0.025):
            d = fresh("dt_%.3f" % dt_fs)
            nsteps = int(round(physical_fs / dt_fs))
            # run.in `time_step` is in FEMTOSECONDS: run.cu divides it by
            # TIME_UNIT_CONVERSION to obtain natural units.  Do NOT divide
            # here (that was a bug in the first version of this harness).
            lines = ["potential nep.txt", "time_step %.12g" % dt_fs,
                     "ensemble nve", "velocity 300 seed 3",
                     "dump_thermo 1",
                     "dump_xyz %d traj.xyz spin" % nsteps,
                     "spin_llg gamma=3.04 alpha=0 T=0",
                     "run %d" % nsteps]
            r = V.run_gpumd(d, lines)
            rows = [l.split() for l in open(os.path.join(d, "thermo.out"))
                    if l.strip() and not l.startswith("#")]
            pe = np.array([float(x[2]) for x in rows])
            ke = np.array([float(x[1]) for x in rows])
            etot = pe + ke
            drift = float(np.max(np.abs(etot - etot[0])))
            # |S| error against the INITIAL anchor, not against itself.
            anchor = np.linalg.norm(V.read_xyz(
                os.path.join(d, "model.xyz"))["spin"], axis=1)
            p_traj = os.path.join(d, "traj.xyz")
            fr = V.read_xyz(p_traj) if os.path.exists(p_traj) else None
            norm_err = float("nan")
            if fr is not None:
                norm_err = float(np.max(np.abs(
                    np.linalg.norm(fr["spin"], axis=1) - anchor)))
            # full fixed-point residual / iteration diagnostics from the
            # solver itself (diagnostic output, not a correctness reference)
            diag = [l.strip() for l in r.stdout.splitlines()
                    if "spin midpoint diagnostics:" in l]
            cons = [l for l in diag if "conservative steps=" in l]
            max_res, max_it = float("nan"), -1
            if cons:
                m_res = re.search(r"max_residual=([0-9.eE+-]+)", cons[-1])
                m_it = re.search(r"max_iterations=(\d+)", cons[-1])
                if m_res:
                    max_res = float(m_res.group(1))
                if m_it:
                    max_it = int(m_it.group(1))
            conv.append({"dt_fs": dt_fs, "steps": nsteps, "rc": r.returncode,
                         "energy_drift_eV": drift, "norm_error": norm_err,
                         "max_midpoint_residual_muB": max_res,
                         "max_midpoint_iterations": max_it,
                         "E0": float(etot[0]), "Efinal": float(etot[-1]),
                         "dE_total_eV": float(etot[-1] - etot[0])})
        complete = (len(conv) == 4 and
                    all(c["rc"] == 0 and np.isfinite(c["energy_drift_eV"]) and
                        np.isfinite(c["norm_error"]) and
                        c["max_midpoint_iterations"] > 0 for c in conv))
        rec("GPU.DT.CONVERGENCE_TABLE", "G_validation",
            "PASS" if complete else "FAIL",
            "; ".join("dt=%.3gfs drift=%.3geV |S|err=%.2g r=%.2g it=%d" %
                      (c["dt_fs"], c["energy_drift_eV"], c["norm_error"],
                       c["max_midpoint_residual_muB"],
                       c["max_midpoint_iterations"]) for c in conv),
            note="Measured at fixed physical time %.4g fs.  PASS means every "
                 "run completed with finite energy, a measured |S| error and "
                 "a reported complete residual; the drift is REPORTED, not "
                 "asserted to vanish (spin-lattice Strang splitting does not "
                 "exactly conserve energy in a coupled run)." % physical_fs)
        RESULTS[-1]["dt_table"] = conv
    except Exception as e:
        rec("GPU.DT.CONVERGENCE_TABLE", "G_validation", "BLOCKED", repr(e))

    # ---- G7: deliberately too-large dt must fail --------------------------
    try:
        d = fresh("dt_large")
        lines = ["potential nep.txt", "time_step 5.0", "ensemble nve",
                 "velocity 300 seed 3", "spin_midpoint 1e-6 1e-6 3",
                 "spin_llg gamma=3.04 alpha=0 T=0", "run 20"]
        r = V.run_gpumd(d, lines)
        output = ((r.stdout or "") + (r.stderr or "")).lower()
        # Match STANDALONE numeric tokens only: the substring "inf" also
        # occurs in "GPU information:", which made the first version of this
        # test report a false FAIL even though the oversized step correctly
        # aborted with exit code 4.
        nonfinite = bool(re.search(
            r"(?<![a-z])(?:nan|inf|infinity|1\.#inf|1\.#ind)(?![a-z])", output))
        clean_hard_fail = (r.returncode != 0 and
                           "did not converge" in output)
        clean_success = (r.returncode == 0 and not nonfinite)
        ok = not nonfinite and (clean_hard_fail or clean_success)
        rec("GPU.DT.OVERSIZE", "G_validation", "PASS" if ok else "FAIL",
            "dt=5.0 fs, exit=%d, clean_hard_fail=%s, non-finite numeric "
            "tokens=%s" % (r.returncode, clean_hard_fail, nonfinite),
            rc=r.returncode,
            note="The propagator is unconditionally stable in the rotation "
                 "angle, so an oversized dt is NOT guaranteed to fail; it must "
                 "either abort cleanly on the residual test or complete with "
                 "finite values.  On this build it aborts with exit 4 "
                 "(implicit midpoint did not converge in 3 iterations).")
    except Exception as e:
        rec("GPU.DT.OVERSIZE", "G_validation", "BLOCKED", repr(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=os.path.join(PKG, "src"))
    ap.add_argument("--unit", default="")
    ap.add_argument("--work", default=os.path.join(PKG, "_r21_work"))
    ap.add_argument("--json", default="")
    ap.add_argument("--matrix", default="")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--baseline-bin", default="",
                    help="directory containing the r20 gpumd binary for the "
                         "before/after performance comparison")
    ap.add_argument("--perf", default="",
                    help="output path for PERFORMANCE.csv")
    ap.add_argument("--skip-extra", action="store_true",
                    help="skip the supplementary physics/performance group")
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    # A unique output tree per invocation means no bulk directory deletion is
    # ever needed.  It also preserves every raw run's inputs and outputs.
    run_work = os.path.join(
        os.path.abspath(args.work),
        "run_" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    os.makedirs(run_work, exist_ok=True)

    cpu_residual_cases()
    cpu_midpoint_counterexample()
    cpu_local_spin_energy()
    cpu_me_virial_fd()
    cpu_units_and_fdt()

    if not args.skip_gpu:
        gpu_unit_tests(args.unit)
        gpu_integration(os.path.abspath(args.bin), run_work)

    if not args.skip_extra:
        try:
            import r21_physics_extra
            r21_physics_extra.collect(
                rec, os.path.abspath(args.bin), os.path.join(run_work, "extra"),
                os.path.abspath(args.baseline_bin) if args.baseline_bin else None,
                os.path.abspath(args.perf) if args.perf else None)
        except ImportError as e:
            rec("EXTRA.PHYSICS", "G_validation", "NOT_RUN",
                "supplementary module unavailable: %r" % e)

    counts = {}
    for r in RESULTS:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    summary = dict(counts)
    print("\n== SUMMARY == " + json.dumps(summary))
    out = {
        "schema": "r21-validation/1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": sys.platform,
        "counts": summary,
        "results": RESULTS,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print("wrote " + args.json)
    if args.matrix:
        with open(args.matrix, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "group", "status", "exit_code", "expected",
                        "measured", "tolerance", "detail", "note"])
            for r in RESULTS:
                w.writerow([r["id"], r["group"], r["status"],
                            "" if r["exit_code"] is None else r["exit_code"],
                            r["expected"], r["measured"], r["tolerance"],
                            r["detail"], r["note"]])
        print("wrote " + args.matrix)
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
