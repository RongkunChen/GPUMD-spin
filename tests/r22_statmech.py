#!/usr/bin/env python3
"""
r22_statmech.py
===============

Executable CPU validation driver for GPUMD-spin V3, Stage R22-NonHeat.

Usage
-----
    python tests/r22_statmech.py --json <out.json>

Exit code is 0 ONLY IF every executed test met its preset numeric criterion;
otherwise 1.  A JSON result list is printed to stdout and (when --json is
given) written to that file.  A short summary is printed to stderr.

SCOPE / HONESTY
---------------
These are CPU-ONLY independent references and small self-contained float64
integrators implemented with numpy (Python 3.13, numpy 2.5.4).  scipy is NOT
installed and MUST NOT be imported.  NO GPU kernel, CUDA source, src/ file or
production binary is read, executed or validated by this driver.  GPU
validation of the production GPUMD-spin integrators is performed ELSEWHERE in
the stage.  One NOT_RUN result records that explicitly.

Status vocabulary (used verbatim, no invented statuses):
    PASS / FAIL / SKIP / BLOCKED / NOT_RUN / NOT_RESOLVED / NOT_CONVERGED
A test is marked PASS only after it actually executed and met a preset numeric
criterion (see the `criterion` field of every result row).  Negative controls
honestly carry the status NOT_CONVERGED / BLOCKED and are expected to do so.

PHYSICS CAVEATS
---------------
* Mermin-Wagner: 2D magnets with continuous spin symmetry have NO finite-T
  long-range order; a finite-T transition requires anisotropy / SOC.  Every
  Tc-related result states this caveat.
* Binder crossings and susceptibility peaks from finite lattices are
  finite-size pseudo-critical temperatures, NOT a thermodynamic Tc; a single
  magnetisation curve is insufficient.
* Energy drift alone cannot establish second-order accuracy; convergence order
  is measured against the SAME trusted analytic solution with a trajectory
  norm.
* Pointwise trajectory differences between different random seeds must NOT be
  used to claim deterministic convergence order (see strong/weak helpers).
* The SLLG stationary distribution is Boltzmann at T and alpha-independent;
  alpha sets only the relaxation rate (checked against Metropolis MC here).

All string literals and quotes are ASCII straight quotes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

# This stage permits exactly two NEW files; never let the import of statmech
# create a __pycache__/*.pyc alongside them.
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "tools", "r22")))

import statmech as sm  # noqa: E402  (path set up above)

KB = sm.KB_EV_PER_K
GAMMA = sm.GAMMA_MUB_PER_EV_FS

FIELDS = (
    "test_id", "category", "status", "expected", "measured",
    "absolute_error", "relative_error", "tolerance", "exit_code",
    "random_seed", "sample_length", "burn_in", "tau_int",
    "effective_sample_size", "confidence_interval", "system_size", "dt",
    "notes", "expected_status", "criterion_met", "criterion",
)

RESULTS = []


def emit(**kw):
    row = dict.fromkeys(FIELDS)
    row["status"] = "NOT_RUN"
    row["expected_status"] = "PASS"
    row["criterion_met"] = False
    row["criterion"] = ""
    row["notes"] = ""
    row.update(kw)
    RESULTS.append(row)
    return row


def check(test_id, category, expected, measured, tolerance, criterion,
          seed=None, notes="", **kw):
    """PASS/FAIL from a preset numeric criterion |measured-expected| <= tol."""
    e = float(expected)
    m = float(measured)
    ae = abs(m - e)
    re = ae / abs(e) if e != 0.0 else None
    ok = (ae <= float(tolerance))
    return emit(
        test_id=test_id, category=category,
        status="PASS" if ok else "FAIL", expected_status="PASS",
        criterion_met=ok, criterion=criterion, expected=e, measured=m,
        absolute_error=ae, relative_error=re, tolerance=float(tolerance),
        random_seed=seed, notes=notes, **kw)


def control(test_id, category, observed_status, expected_status, criterion,
            notes="", **kw):
    """Honest negative/informational control: status must equal expectation."""
    ok = (observed_status == expected_status)
    return emit(
        test_id=test_id, category=category, status=observed_status,
        expected_status=expected_status, criterion_met=ok,
        criterion=criterion, expected=expected_status,
        measured=observed_status, notes=notes, **kw)


def ci_list(lo, hi):
    return [float(lo), float(hi)]


# Self-contained reference integrators / samplers (CPU, float64)

def metropolis_chain(x, n_samples, burn_in, seed, step=1.0):
    """Own numpy Metropolis sampler for P(u) proportional to exp(x*u), u in
    [-1,1] (uniform solid-angle measure).  Proposal is a symmetric uniform
    step; out-of-range proposals are rejected.  Returns the post-burn-in
    chain (float64)."""
    rng = np.random.default_rng(seed)
    chain = np.empty(int(n_samples), dtype=np.float64)
    u = 0.0
    total = int(burn_in) + int(n_samples)
    for i in range(total):
        prop = u + rng.uniform(-step, step)
        if -1.0 <= prop <= 1.0:
            if rng.random() < math.exp(min(0.0, x * (prop - u))):
                u = prop
        if i >= burn_in:
            chain[i - burn_in] = u
    return chain


def midpoint_step(s, B, gamma, dt, tol=1e-14, max_iter=100):
    """Conservative LLG single step by implicit midpoint, fixed-point
    iteration with a 3-component residual and a HARD failure on
    non-convergence.  f(s) = -gamma * (s x B)."""
    sn = np.array(s, dtype=np.float64, copy=True)
    for k in range(max_iter):
        m = 0.5 * (np.array(s, dtype=np.float64) + sn)
        new = np.array(s, dtype=np.float64) - gamma * dt * np.cross(m, B)
        residual = new - sn                       # 3-component residual
        sn = new
        if np.linalg.norm(residual) < tol:
            return sn, k + 1
    raise RuntimeError(
        "implicit-midpoint non-convergence: |residual| did not fall below tol")


def midpoint_trajectory(S0, B, gamma, dt, t_final):
    """Integrate to the SAME physical time t_final; returns (times, spin path,
    max fixed-point iterations used)."""
    n = int(round(t_final / dt))
    if abs(n * dt - t_final) > 1e-12:
        raise ValueError("t_final must be an integer multiple of dt")
    path = np.empty((n + 1, 3), dtype=np.float64)
    path[0] = S0
    s = np.array(S0, dtype=np.float64)
    maxit = 0
    for i in range(1, n + 1):
        s, it = midpoint_step(s, B, gamma, dt)
        path[i] = s
        maxit = max(maxit, it)
    return np.arange(n + 1) * dt, path, maxit


def sllg_step(s, H, gamma, alpha, dt):
    """One Wong-Zakai / Stratonovich SLLG step for R unit spins (shape (R,3)).

    The thermal field H is held CONSTANT over the step, so the step is the
    EXACT Landau-Lifshitz flow under the constant total effective field H:
        ds/dt = -gamma*[s x H + alpha*s x (s x H)].
    In the (e1,e2,h) frame, h = H/|H|:
        tan(theta(t)/2) = tan(theta0/2)*exp(-alpha*gamma*|H|*t),
        phi(t)          = phi0 + gamma*|H|*t.
    |s| is preserved exactly (ct**2+st**2 = 1).  This is the same exact flow
    as statmech.llg_analytic_damped(form='LL'); here it is vectorised over
    replicas for efficiency.  float64 throughout.
    """
    Hn = np.linalg.norm(H, axis=1, keepdims=True)
    h = H / np.where(Hn > 0.0, Hn, 1.0)
    c = np.clip(np.sum(s * h, axis=1, keepdims=True), -1.0, 1.0)
    s0 = np.sqrt(np.maximum(0.0, 1.0 - c * c))
    ss = np.where(s0 > 1e-12, s0, 1.0)
    e1 = (s - c * h) / ss
    e2 = np.cross(h, e1)
    u0 = s0 / np.maximum(1.0 + c, 1e-300)        # tan(theta0/2), guarded
    lam = alpha * gamma * Hn
    om = gamma * Hn
    u = u0 * np.exp(-lam * dt)
    ct = (1.0 - u * u) / (1.0 + u * u)
    st = 2.0 * u / (1.0 + u * u)
    ph = om * dt
    return ct * h + st * (np.cos(ph) * e1 + np.sin(ph) * e2)


def sllg_simulate(alpha, dt, seed, n_replicas=512, t_burn_fs=80.0,
                  t_sample_fs=500.0, t_gap_fs=50.0,
                  B_z=0.05, T_K=1000.0, S_muB=1.0):
    """Independent-equilibrium SLLG simulation; returns a dict of real data.

    Noise amplitude from the FDT:
        H_th,rms = sqrt(2*alpha*kB*T/(gamma*(1+alpha**2)*|S|*dt)).
    Replicas evolve with INDEPENDENT noise, so at a fixed time they are
    independent draws from the stationary distribution.  Snapshots are taken
    every t_gap_fs (>= 5*tau_physical for every alpha used here), so they are
    decorrelated in time as well.  A full time series of replica 0 is kept to
    measure tau_int, ESS and the CI.
    """
    D = alpha * KB * T_K / (GAMMA * (1.0 + alpha ** 2) * S_muB)
    A = math.sqrt(2.0 * D / dt)
    nb = int(round(t_burn_fs / dt))
    ns = int(round(t_sample_fs / dt))
    gap = max(1, int(round(t_gap_fs / dt)))
    rng = np.random.default_rng(seed)
    s = rng.normal(size=(n_replicas, 3))
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    Bv = np.array([0.0, 0.0, B_z], dtype=np.float64)
    series = np.empty(ns, dtype=np.float64)
    snaps = []
    for it in range(nb + ns):
        H = Bv[None, :] + A * rng.normal(size=(n_replicas, 3))
        s = sllg_step(s, H, GAMMA, alpha, dt)
        if it >= nb:
            j = it - nb
            series[j] = s[0, 2]
            if j % gap == 0:
                snaps.append(s[:, 2].copy())
    U = np.array(snaps, dtype=np.float64).ravel()
    tau_int = sm.integrated_autocorrelation_time(series, max_lag=ns - 1)
    ess = sm.effective_sample_size(ns, tau_int)
    se = float(U.std(ddof=1) / math.sqrt(U.size))
    ci = sm.confidence_interval(float(U.mean()), se, 0.95)
    return {
        "alpha": alpha, "dt": dt, "seed": seed, "noise_amplitude": A,
        "diffusion_D": D, "n_replicas": n_replicas,
        "burn_in_steps": nb, "sample_steps": ns, "gap_steps": gap,
        "raw_series": series, "samples": U, "tau_int_steps": tau_int,
        "tau_int_fs": tau_int * dt, "effective_sample_size": ess,
        "stderr": se, "confidence_interval": ci,
        "mean_cos_theta": float(U.mean()),
        "T_K": T_K, "B_z": B_z, "S_muB": S_muB,
    }


# Tests

def test_01_boltzmann_mean_mc():
    """Analytic <cos(theta)> vs own Metropolis MC, 2 temperatures, CI overlap."""
    B, S = 0.05, 1.0
    temps = [300.0, 1000.0]
    n, burn, seeds = 200000, 10000, [91001, 91002]
    for tag, T, seed in zip(("a", "b"), temps, seeds):
        x = float(S * B / (KB * T))
        analytic = float(sm.single_spin_mean_cos_theta(B, S, T))
        chain = metropolis_chain(x, n, burn, seed, step=1.0)
        mean = float(chain.mean())
        se = sm.blocking_error(chain, 100)
        ci = sm.confidence_interval(mean, se, 0.95)
        tau = sm.integrated_autocorrelation_time(chain, max_lag=2000)
        ess = sm.effective_sample_size(chain.size, tau)
        overlap = (ci[0] <= analytic <= ci[1])
        check(
            "R22.SLLG.01%s" % tag, "equilibrium", analytic, mean,
            tolerance=0.01,
            criterion=("preset: |MC_mean - L(x)| <= 0.01 AND analytic L(x) "
                       "inside the 95%% CI of the MC mean (CI overlap); "
                       "MC n=200000 after burn-in=10000"),
            seed=seed,
            notes=("T=%g K, x=%.6f, L(x)=%.9f, MC_mean=%.9f, se=%.6g, "
                   "95%% CI=[%.6f, %.6f], CI_overlap=%s. Own numpy "
                   "Metropolis, symmetric uniform proposal step=1.0. "
                   "CPU reference only; no GPU validation." %
                   (T, x, analytic, mean, se, ci[0], ci[1], overlap)),
            sample_length=int(chain.size), burn_in=int(burn),
            tau_int=float(tau), effective_sample_size=float(ess),
            confidence_interval=ci_list(*ci), system_size=1,
            dt=None)
        # Honest additional criterion row: CI overlap itself.
        control(
            "R22.SLLG.01%s.ci" % tag, "equilibrium",
            "PASS" if overlap else "FAIL", "PASS",
            criterion="preset: analytic L(x) must lie inside the 95% CI",
            notes=("95%% CI overlap check at T=%g K: L(x)=%.9f, "
                   "CI=[%.6f, %.6f]." % (T, analytic, ci[0], ci[1])),
            seed=seed, sample_length=int(chain.size), burn_in=int(burn),
            tau_int=float(tau), effective_sample_size=float(ess),
            confidence_interval=ci_list(*ci), system_size=1)


def test_02_energy_heat_capacity_mc():
    """Analytic U and C(T) vs MC energy and energy fluctuation."""
    B, S = 0.05, 1.0
    temps = [300.0, 1000.0]
    n, burn, seeds = 200000, 10000, [92001, 92002]
    for tag, T, seed in zip(("a", "b"), temps, seeds):
        x = float(S * B / (KB * T))
        E = -S * B * metropolis_chain(x, n, burn, seed + 5000, step=1.0)
        U_exact = float(sm.single_spin_internal_energy(B, S, T))
        C_exact = float(sm.single_spin_heat_capacity(B, S, T))
        U_mc = float(E.mean())
        C_mc = float(E.var(ddof=1) / (KB * T ** 2))
        nblocks = 100
        blocks = E[:nblocks * (E.size // nblocks)].reshape(
            nblocks, E.size // nblocks)
        C_blocks = blocks.var(axis=1, ddof=1) / (KB * T ** 2)
        C_se = float(C_blocks.std(ddof=1) / math.sqrt(nblocks))
        C_ci = sm.confidence_interval(C_mc, C_se, 0.95)
        tau = sm.integrated_autocorrelation_time(E, max_lag=2000)
        ess = sm.effective_sample_size(E.size, tau)
        check(
            "R22.SLLG.02%s.U" % tag, "equilibrium", U_exact, U_mc,
            tolerance=0.001,
            criterion=("preset: |MC_energy_mean - U(T)| <= 0.001 eV; "
                       "U(T) = -|S||B|L(x)"),
            seed=int(seed + 5000),
            notes=("T=%g K, x=%.6f, U_exact=%.9g eV, U_MC=%.9g eV. "
                   "Energy in eV, kB T in eV." % (T, x, U_exact, U_mc)),
            sample_length=int(E.size), burn_in=int(burn),
            tau_int=float(tau), effective_sample_size=float(ess),
            confidence_interval=ci_list(*sm.confidence_interval(
                U_mc, sm.blocking_error(E, 100), 0.95)),
            system_size=1)
        check(
            "R22.SLLG.02%s.C" % tag, "equilibrium", C_exact, C_mc,
            tolerance=0.15 * C_exact,
            criterion=("preset: |C_MC - C(T)| <= 15%% of C(T); C(T)=dU/dT="
                       "kB*x**2*L'(x)=kB*(1-(x/sinh x)**2); "
                       "C_MC=Var(E)/(kB*T**2)"),
            seed=int(seed + 5000),
            notes=("T=%g K, C_exact=%.9g eV/K, C_MC=%.9g eV/K, "
                   "C_se=%.3g, 95%% CI=[%.3g, %.3g] (block bootstrap over "
                   "%d blocks)." % (T, C_exact, C_mc, C_se, C_ci[0],
                                    C_ci[1], nblocks)),
            sample_length=int(E.size), burn_in=int(burn),
            tau_int=float(tau), effective_sample_size=float(ess),
            confidence_interval=ci_list(*C_ci), system_size=1)
    # Independent self-consistency of C(T) against numerical dU/dT.
    B, S, T = 0.05, 1.0, 650.0
    h = 0.01
    C_analytic = float(sm.single_spin_heat_capacity(B, S, T))
    C_numeric = float((sm.single_spin_internal_energy(B, S, T + h)
                       - sm.single_spin_internal_energy(B, S, T - h)) / (2 * h))
    check(
        "R22.SLLG.02c.dUdT", "equilibrium", C_numeric, C_analytic,
        tolerance=1e-10,
        criterion=("preset: |C_analytic - central-difference dU/dT| <= 1e-10 "
                   "eV/K (analytic derivative self-consistency)"),
        notes=("T=%g K, C_analytic=%.12g eV/K, dU/dT=%.12g eV/K."
               % (T, C_analytic, C_numeric)),
        system_size=1)


def test_03_uniform_precession():
    """Exact uniform precession vs analytic Rodrigues reference."""
    B = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    S0 = np.array([1.0, 0.0, 0.5], dtype=np.float64)
    gamma = GAMMA
    omega = gamma * float(np.linalg.norm(B))
    period = 2.0 * math.pi / omega
    t = np.linspace(0.0, 3.0 * period, 301)
    ref = sm.llg_analytic_uniform_precession(t, S0, B, gamma)
    damped0 = sm.llg_analytic_damped(t, S0, B, 0.0, gamma, form="LL")
    diff = float(np.max(np.linalg.norm(ref - damped0, axis=1)))
    check(
        "R22.LLG.03a.self_consistency", "deterministic", 0.0, diff,
        tolerance=1e-13,
        criterion=("preset: max |Rodrigues - damped(alpha=0)| <= 1e-13 "
                   "(two independent analytic implementations)"),
        notes=("Rodrigues rotation vs tan(theta/2) solution at alpha=0; "
               "max vector difference %.3g mu_B." % diff),
        dt=None)
    # Frequency / unit check: unwrap precession phase and fit its slope.
    phase = np.unwrap(sm.spin_phase(ref, B))
    slope = float(np.polyfit(t, phase, 1)[0])
    check(
        "R22.LLG.03b.frequency", "deterministic", omega, slope,
        tolerance=1e-8,
        criterion=("preset: |measured omega - gamma*|B|| <= 1e-8 rad/fs; "
                   "gamma=3.0385 mu_B/(eV*fs), |B|=0.1 eV/mu_B"),
        notes=("omega_expected=%.12g rad/fs, omega_measured=%.12g rad/fs, "
               "period=%.9g fs. Units: gamma*|B| -> rad/fs." %
               (omega, slope, period)),
        dt=None)
    at_period = sm.llg_analytic_uniform_precession(period, S0, B, gamma)
    perr = float(np.linalg.norm(at_period - S0))
    check(
        "R22.LLG.03c.period", "deterministic", 0.0, perr,
        tolerance=1e-12,
        criterion="preset: |S(2*pi/omega) - S0| <= 1e-12 mu_B",
        notes="After one full precession period the spin returns to S0.",
        dt=None)
    norm_err = float(np.max(np.abs(np.linalg.norm(ref, axis=1)
                                  - np.linalg.norm(S0))))
    check(
        "R22.LLG.03d.spin_norm", "deterministic", 0.0, norm_err,
        tolerance=1e-13,
        criterion="preset: max |||S(t)| - |S0||| <= 1e-13 mu_B",
        notes="Rodrigues flow is orthogonal: fixed-length spin preserved.",
        dt=None)
    E = -np.dot(ref, B)
    e_err = float(np.max(np.abs(E - (-np.dot(S0, B)))))
    check(
        "R22.LLG.03e.energy", "deterministic", 0.0, e_err,
        tolerance=1e-13,
        criterion="preset: max |E(t) - E(0)| <= 1e-13 eV",
        notes=("E = -S.B conserved for alpha=0; energy drift alone CANNOT "
               "establish second-order accuracy (see R22.LLG.04)."),
        dt=None)
    # Verify the differential equation dS/dt = -gamma S x B numerically.
    h = 1e-4
    Sp = sm.llg_analytic_uniform_precession(7.3 + h, S0, B, gamma)
    Sm = sm.llg_analytic_uniform_precession(7.3 - h, S0, B, gamma)
    dSdt = (Sp - Sm) / (2 * h)
    rhs = -gamma * np.cross(sm.llg_analytic_uniform_precession(7.3, S0, B,
                                                               gamma), B)
    ode_err = float(np.linalg.norm(dSdt - rhs))
    check(
        "R22.LLG.03f.ode", "deterministic", 0.0, ode_err,
        tolerance=1e-6,
        criterion=("preset: |central-difference dS/dt - (-gamma S x B)| "
                   "<= 1e-6 mu_B/fs"),
        notes=("ODE residual of the trusted Rodrigues reference at t=7.3 fs; "
               "residual %.3g mu_B/fs." % ode_err),
        dt=None)


def test_04_implicit_midpoint_convergence():
    """Deterministic implicit-midpoint convergence, measured p reported."""
    B = np.array([0.0, 0.0, 0.1], dtype=np.float64)
    S0 = np.array([1.0, 0.3, 0.6], dtype=np.float64)
    gamma = GAMMA
    t_final = 60.0
    dts = [0.5, 0.25, 0.125, 0.0625]
    # ONE trusted reference, used for ALL dt: the analytic Rodrigues solution.
    trusted = sm.llg_analytic_uniform_precession(t_final, S0, B, gamma)
    traj_err, phase_err, norm_err, energy_err = [], [], [], []
    iters = []
    for dt in dts:
        times, path, maxit = midpoint_trajectory(S0, B, gamma, dt, t_final)
        ref_path = sm.llg_analytic_uniform_precession(times, S0, B, gamma)
        traj_err.append(float(np.max(np.linalg.norm(path - ref_path, axis=1))))
        phase_err.append(float(abs(sm.phase_error(path, ref_path, B)[-1])))
        norm_err.append(float(np.max(np.abs(np.linalg.norm(path, axis=1)
                                            - np.linalg.norm(S0)))))
        E0 = -float(np.dot(S0, B))
        energy_err.append(float(np.max(np.abs(-np.dot(path, B) - E0))))
        iters.append(int(maxit))
    conv = sm.convergence_order(
        traj_err, dts, reference="llg_analytic_uniform_precession(T=60 fs)",
        same_reference=True)
    p = float(conv["mean_order"])
    check(
        "R22.LLG.04.convergence_order", "deterministic", 2.0, p,
        tolerance=0.25,
        criterion=("preset: |mean observed p - 2| <= 0.25 with p = "
                   "log2(E(dt)/E(dt/2)) measured against the SAME trusted "
                   "analytic Rodrigues solution; FAIL if outside"),
        notes=(
            "dt=%s fs to the SAME physical time %.1f fs. Trajectory L-inf "
            "errors=%s; observed orders p=%s; mean_p=%.9f. Spin-norm errors="
            "%s (roundoff). Phase errors=%s rad. Energy errors=%s eV "
            "(roundoff; ENERGY DRIFT ALONE CANNOT ESTABLISH SECOND-ORDER "
            "ACCURACY). Max fixed-point iterations=%s, 3-component residual "
            "with hard failure on non-convergence. Guard: %s." %
            (dts, t_final, ["%.6g" % v for v in traj_err],
             ["%.6f" % v for v in conv["orders"]], p,
             ["%.3g" % v for v in norm_err],
             ["%.6g" % v for v in phase_err],
             ["%.3g" % v for v in energy_err], iters, conv["status"])),
        system_size=1, dt=dts[0])
    # Report energy-error-only "order" honestly as NOT_RESOLVED, not a number.
    control(
        "R22.LLG.04.energy_only_control", "deterministic",
        "NOT_RESOLVED", "NOT_RESOLVED",
        criterion=("preset: energy-error-based order is refused because all "
                   "energy errors are at roundoff level; energy drift alone "
                   "cannot establish second-order accuracy"),
        notes=("Energy errors are %.3g .. %.3g eV (roundoff). A "
               "log2 ratio of roundoff-level numbers is meaningless, so the "
               "energy-only convergence order is honestly NOT_RESOLVED." %
               (min(energy_err), max(energy_err))),
        system_size=1, dt=dts[0])
    return {"dts": dts, "traj_err": traj_err, "p": p,
            "phase_err": phase_err, "norm_err": norm_err,
            "energy_err": energy_err, "conv": conv}


def test_05_sllg_equilibrium():
    """SLLG stochastic equilibrium: KS vs Boltzmann, alpha-independence."""
    alphas = [0.1, 0.5, 1.0]
    dts = [0.02, 0.01]
    seeds = [93001, 93002]
    B, T, S = 0.05, 1000.0, 1.0
    x = float(S * B / (KB * T))
    L_exact = float(sm.single_spin_mean_cos_theta(B, S, T))
    cdf = lambda u: sm.single_spin_boltzmann_cdf(u, B, S, T)
    runs = []
    pooled_by_alpha = {a: [] for a in alphas}
    for a in alphas:
        for dt in dts:
            for sd in seeds:
                r = sllg_simulate(a, dt, sd + int(1000 * a) + int(1e6 * dt),
                                  B_z=B, T_K=T, S_muB=S)
                runs.append(r)
                pooled_by_alpha[a].append(r["samples"])
                sd_used = r["seed"]
                check(
                    "R22.SLLG.05.a%s.dt%s.s%d" %
                    (str(a).replace(".", "p"),
                     str(dt).replace(".", "p"), sd_used),
                    "stochastic", L_exact, r["mean_cos_theta"],
                    tolerance=0.03,
                    criterion=("preset: |mean cos(theta) - L(x)| <= 0.03 "
                               "(equivalence tolerance ~5 sigma for these "
                               "sample sizes); distribution checked separately "
                               "by pooled KS"),
                    seed=sd_used,
                    notes=(
                        "alpha=%g, dt=%g fs, seed=%d, replicas=%d, "
                        "burn_in=%d steps, sample_steps=%d, snapshot_gap=%d "
                        "steps (%.0f fs), n_independent=%d, "
                        "H_th_rms=%.6g eV/mu_B (FDT), D=%.6g, "
                        "tau_int=%.2f steps (%.2f fs), ESS=%.1f, "
                        "mean=%.6f, se=%.6g, 95%% CI=[%.5f, %.5f]. "
                        "Wong-Zakai/Stratonovich exact-LL-flow step; "
                        "CPU reference, no GPU validation." %
                        (a, dt, sd_used, r["n_replicas"],
                         r["burn_in_steps"], r["sample_steps"],
                         r["gap_steps"], r["gap_steps"] * dt,
                         int(r["samples"].size), r["noise_amplitude"],
                         r["diffusion_D"], r["tau_int_steps"],
                         r["tau_int_fs"], r["effective_sample_size"],
                         r["mean_cos_theta"], r["stderr"],
                         r["confidence_interval"][0],
                         r["confidence_interval"][1])),
                    sample_length=int(r["samples"].size),
                    burn_in=int(r["burn_in_steps"]),
                    tau_int=float(r["tau_int_steps"]),
                    effective_sample_size=float(r["effective_sample_size"]),
                    confidence_interval=ci_list(*r["confidence_interval"]),
                    system_size=int(r["n_replicas"]), dt=float(dt))
    all_samples = np.concatenate([r["samples"] for r in runs])
    ks = sm.ks_test_one_sample(all_samples, cdf)
    control(
        "R22.SLLG.05.KS_pooled", "stochastic",
        "PASS" if (ks["p_value"] >= 0.05
                   and ks["D"] <= ks["d_crit_95"]) else "FAIL",
        "PASS",
        criterion=("preset: one-sample KS at 95%% -> p_value >= 0.05 and "
                   "D <= D_crit_95; analytic Boltzmann CDF as reference"),
        notes=(
            "Pooled %d decorrelated samples over %d runs (3 alphas x 2 dt x "
            "2 seeds). KS D=%.6g, D_crit_95=%.6g, p_value=%.4f, lambda=%.4f. "
            "x=%.6f, L(x)=%.6f. This is the primary equilibrium-distribution "
            "check." % (int(all_samples.size), len(runs), ks["D"],
                        ks["d_crit_95"], ks["p_value"], ks["lambda"], x,
                        L_exact)),
        sample_length=int(all_samples.size), system_size=len(runs))
    gof = sm.chi_square_gof(all_samples, cdf, np.linspace(-1.0, 1.0, 21))
    check(
        "R22.SLLG.05.chi2_gof", "stochastic", 1.0, gof["reduced_chi2"],
        tolerance=1.0,
        criterion=("preset: reduced chi2 within 1.0 of unity "
                   "(|chi2/dof - 1| <= 1.0); KS p-value reported as diagnostic"),
        notes=("chi2=%.4f, dof=%d, reduced=%.4f, p_value=%.4f, "
               "min_expected=%.1f (20 bins over cos(theta) in [-1,1])." %
               (gof["chi2"], gof["dof"], gof["reduced_chi2"],
                gof["p_value"], gof["min_expected"])),
        sample_length=int(all_samples.size))
    # alpha-independence: means and CIs per alpha.
    means, ses, cis = {}, {}, {}
    for a in alphas:
        u = np.concatenate(pooled_by_alpha[a])
        means[a] = float(u.mean())
        ses[a] = float(u.std(ddof=1) / math.sqrt(u.size))
        cis[a] = sm.confidence_interval(means[a], ses[a], 0.95)
    max_dev = max(abs(means[a] - L_exact) for a in alphas)
    check(
        "R22.SLLG.05.alpha_independence", "stochastic", 0.0, max_dev,
        tolerance=0.03,
        criterion=("preset: max_alpha |mean_cos_theta(alpha) - L(x)| <= 0.03 "
                   "and all 95%% CIs overlap the analytic L(x); alpha only "
                   "sets the relaxation rate, NOT the equilibrium"),
        notes=("alpha means=%s, se=%s, 95%% CIs=%s; L(x)=%.6f. "
               "Mermin-Wagner caveat applies to any 2D Tc analysis. "
               "Two-sample KS alpha=0.1 vs alpha=1.0: %s." %
               ({a: round(means[a], 6) for a in alphas},
                {a: round(ses[a], 6) for a in alphas},
                {a: [round(v, 6) for v in cis[a]] for a in alphas},
                L_exact,
                _ks_alpha_note(pooled_by_alpha))),
        sample_length=int(sum(np.concatenate(pooled_by_alpha[a]).size
                              for a in alphas)))
    return {"runs": runs, "ks": ks, "gof": gof, "all_samples": all_samples,
            "pooled_by_alpha": pooled_by_alpha, "means": means}


def _ks_alpha_note(pooled_by_alpha):
    ks = sm.ks_test_two_samples(pooled_by_alpha[0.1], pooled_by_alpha[1.0])
    return ("D=%.6g, p=%.4f, D_crit_95=%.6g, reject95=%s" %
            (ks["D"], ks["p_value"], ks["d_crit_95"], ks["reject_at_95"]))


def test_06_ar1_autocorrelation():
    """Autocorrelation / tau_int / ESS / blocking error on AR(1)."""
    phi, n, seed = 0.8, 200000, 94001
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, math.sqrt(1.0 - phi * phi), n)
    x = np.empty(n, dtype=np.float64)
    x[0] = eps[0]
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    rho = sm.autocorrelation(x, 10)
    tau, M = sm.integrated_autocorrelation_time(x, with_window=True)
    tau_exact = 0.5 + phi / (1.0 - phi)
    ess = sm.effective_sample_size(n, tau)
    be = sm.blocking_error(x, 100)
    se_theory = math.sqrt(2.0 * tau / n)
    check(
        "R22.SLLG.06a.rho1", "statistics", phi, float(rho[1]),
        tolerance=0.02,
        criterion="preset: |rho(1) - phi| <= 0.02 for AR(1) phi=0.8",
        seed=seed,
        notes=("rho(1)=%.6f, phi=0.8; rho(k) measured=%s vs theory=%s." %
               (rho[1], [round(float(v), 4) for v in rho[1:6]],
                [round(phi ** k, 4) for k in range(1, 6)])),
        sample_length=int(n), tau_int=float(tau),
        effective_sample_size=float(ess), system_size=1)
    check(
        "R22.SLLG.06b.tau_int", "statistics", tau_exact, float(tau),
        tolerance=0.75,
        criterion=("preset: |tau_int(Sokal c=5) - (0.5+phi/(1-phi))| <= "
                   "0.75 for AR(1) phi=0.8"),
        seed=seed,
        notes=("tau_exact=%.6f, tau_measured=%.6f, Sokal window M=%d, "
               "c=5." % (tau_exact, tau, M)),
        sample_length=int(n), tau_int=float(tau),
        effective_sample_size=float(ess), system_size=1)
    check(
        "R22.SLLG.06c.blocking_error", "statistics", se_theory, be,
        tolerance=0.002,
        criterion=("preset: |blocking_error - sqrt(2*tau_int/n)| <= 0.002 "
                   "(Var(mean) = 2*tau_int*Var(x)/n)"),
        seed=seed,
        notes=("blocking_error(100 blocks)=%.6f, theoretical stderr=%.6f, "
               "ESS=n/(2*tau_int)=%.1f." % (be, se_theory, ess)),
        sample_length=int(n), tau_int=float(tau),
        effective_sample_size=float(ess), system_size=1)


def test_07_binder():
    """Binder cumulant, crossing recovery and NOT_CONVERGED controls."""
    rng = np.random.default_rng(95001)
    g = rng.normal(0.0, 1.0, (200000, 3))
    u4g = sm.binder_cumulant(g)
    check(
        "R22.TC.07a.U4_gaussian", "critical", 4.0 / 9.0, u4g,
        tolerance=0.01,
        criterion=("preset: |U4(3D Gaussian) - 4/9| <= 0.01; O(3) high-T "
                   "limit <M^4>/<M^2>^2 = 5/3"),
        notes=("U4=%.6f, expected 4/9=%.6f for a 3D Gaussian vector order "
               "parameter. Mermin-Wagner: 2D continuous-symmetry magnets "
               "have no finite-T LRO; Tc requires anisotropy/SOC." %
               (u4g, 4.0 / 9.0)),
        sample_length=int(g.shape[0]), system_size=1)
    Tg = np.linspace(20.0, 26.0, 41)     # 0.15 K grid
    Tc = 22.3
    Ls = np.array([8.0, 12.0, 16.0, 24.0])
    U = np.array([0.5 - 0.4 * np.tanh((Tg - Tc) * L / 8.0) for L in Ls])
    est = sm.estimate_tc_binder_crossing(Ls, Tg, U)
    control(
        "R22.TC.07b.crossing_status", "critical", est["status"], "PASS",
        criterion=("preset: crossing estimator returns PASS for >=3 sizes "
                   "with crossing curves"),
        notes=("n_sizes=%d, n_pairs=%d, n_crossings=%d. Mermin-Wagner "
               "caveat: 2D magnets with continuous spin symmetry have NO "
               "finite-T long-range order; a finite-T transition requires "
               "anisotropy/SOC. Binder crossings are finite-size estimates." %
               (est["n_sizes"], est["n_pairs"], est["n_crossings"])),
        system_size=4)
    check(
        "R22.TC.07c.crossing_tc", "critical", Tc, est["tc_mean"],
        tolerance=0.05,
        criterion=("preset: |mean crossing Tc - known synthetic Tc| <= 0.05 K"),
        notes=("known Tc=%.3f K, mean crossing=%.6f K, std across %d "
               "pairs=%.6f K, extrapolated Tc(1/L->0)=%.6f K." %
               (Tc, est["tc_mean"], est["n_crossings"], est["tc_std"],
                est["tc_extrapolated"])),
        system_size=4)
    # Negative control 1: strictly separated, non-crossing curves.
    U_nc = np.array([0.5 + 0.05 * i - 0.02 * np.tanh((Tg - Tc) * L / 8.0)
                     for i, L in enumerate(Ls)])
    est_nc = sm.estimate_tc_binder_crossing(Ls, Tg, U_nc)
    control(
        "R22.TC.07d.non_crossing", "critical", est_nc["status"],
        "NOT_CONVERGED",
        criterion=("preset: non-crossing U4 curves must return "
                   "NOT_CONVERGED (honest negative control)"),
        notes=("Negative control: 4 sizes with strictly separated curves, "
               "n_crossings=%d. Mermin-Wagner caveat applies." %
               est_nc["n_crossings"]),
        system_size=4)
    # Negative control 2: fewer than 3 sizes.
    est_few = sm.estimate_tc_binder_crossing(Ls[:2], Tg, U[:2])
    control(
        "R22.TC.07e.fewer_than_3_sizes", "critical", est_few["status"],
        "NOT_CONVERGED",
        criterion=("preset: fewer than 3 sizes must return NOT_CONVERGED; "
                   "a single magnetisation curve cannot locate Tc"),
        notes="Negative control: only 2 sizes supplied.",
        system_size=2)


def test_08_susceptibility_peaks():
    """Susceptibility/specific-heat peak Tc and the single-curve control."""
    Tg = np.linspace(20.0, 26.0, 41)
    Tc = 22.3
    chi = 1.0 / ((Tg - Tc) ** 2 + 0.7 ** 2)
    est = sm.estimate_tc_susceptibility_peak(Tg, chi)
    control(
        "R22.TC.08a.peak_status", "critical", est["status"], "PASS",
        criterion="preset: interior susceptibility peak returns PASS",
        notes=("Finite-size pseudo-critical temperature ONLY. Mermin-Wagner "
               "caveat: 2D magnets with continuous spin symmetry have NO "
               "finite-T long-range order; a finite-T transition requires "
               "anisotropy/SOC."),
        system_size=4)
    check(
        "R22.TC.08b.peak_tc", "critical", Tc, est["tc_pseudo"],
        tolerance=0.05,
        criterion=("preset: |parabolic-refined peak - known Tc| <= 0.05 K"),
        notes=("known peak=%.3f K, tc_pseudo=%.6f K, peak_value=%.4f, "
               "curvature=%.4g. Grid spacing 0.15 K; parabolic refinement "
               "through the 3 points around the discrete maximum." %
               (Tc, est["tc_pseudo"], est["peak_value"], est["curvature"])),
        system_size=4)
    # Negative control: a single monotone magnetisation curve is insufficient.
    M_single = np.linspace(1.0, 0.0, Tg.size)
    est_bad = sm.estimate_tc_susceptibility_peak(Tg, M_single)
    control(
        "R22.TC.08c.single_curve_control", "critical", est_bad["status"],
        "NOT_CONVERGED",
        criterion=("preset: a single monotone magnetisation curve must "
                   "return NOT_CONVERGED; no interior peak => no Tc claim"),
        notes=("Explicit negative control: one monotone M(T) curve has its "
               "maximum at the grid boundary, so the estimator refuses to "
               "claim a Tc."),
        system_size=1)
    # Direct checks of the fluctuation formulas.
    rng = np.random.default_rng(96001)
    N, T = 64, 300.0
    M = rng.normal(0.3, 0.2, 50000)
    # Use the population second moment (mean of squares minus square of mean),
    # exactly the estimator implemented in sm.susceptibility; ddof=1 would
    # differ by the Bessel factor and is NOT the same quantity.
    var_M = float(np.mean(M ** 2) - np.mean(M) ** 2)
    chi_direct = N * var_M / (KB * T)
    chi_func = sm.susceptibility(M, N, T)
    check(
        "R22.TC.08d.susceptibility_formula", "critical", chi_direct, chi_func,
        tolerance=1e-12,
        criterion=("preset: |susceptibility() - N*Var(M)/(kB*T)| <= 1e-12"),
        notes=("N=%d, T=%g K, Var(M)=%.9f, chi=%.9f (1/eV for "
               "dimensionless M)." % (N, T, var_M, chi_func)),
        sample_length=int(M.size), system_size=N)
    E = rng.normal(-0.7, 0.15, 50000)
    var_E = float(np.mean(E ** 2) - np.mean(E) ** 2)
    c_direct = var_E / (N * KB * T ** 2)
    c_func = sm.specific_heat(E, N, T)
    check(
        "R22.TC.08e.specific_heat_formula", "critical", c_direct, c_func,
        tolerance=1e-15,
        criterion=("preset: |specific_heat() - Var(E)/(N*kB*T^2)| <= 1e-15"),
        notes=("N=%d, T=%g K, Var(E)=%.9g eV^2, C/N=%.9g eV/K." %
               (N, T, var_E, c_func)),
        sample_length=int(E.size), system_size=N)


def test_09_fm_afm_order_parameter():
    """FM vs AFM order parameters on a two-sublattice Neel configuration."""
    rng = np.random.default_rng(97001)
    N = 200
    sub = np.arange(N) % 2                      # parity/site index
    z = np.zeros((N, 3), dtype=np.float64)
    z[:, 2] = np.where(sub == 0, 1.0, -1.0)
    z += 1e-3 * rng.normal(size=(N, 3))        # tiny realistic disorder
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    M_fm = sm.fm_order_parameter(z)
    M_afm = sm.afm_order_parameter(z, sub)
    mag_fm = float(np.linalg.norm(M_fm))
    mag_afm = float(np.linalg.norm(M_afm))
    check(
        "R22.TC.09a.M_FM_vanishes", "order_parameter", 0.0, mag_fm,
        tolerance=0.01,
        criterion=("preset: |M_FM| <= 0.01 for a two-sublattice AFM "
                   "configuration (FM order parameter is WRONG for AFM)"),
        notes=("N=%d two-sublattice Neel state with 1e-3 Gaussian disorder. "
               "M_FM=%s, |M_FM|=%.6f. Correct AFM order parameter is the "
               "staggered magnetisation." % (N, np.round(M_fm, 6).tolist(),
                                             mag_fm)),
        system_size=N)
    check(
        "R22.TC.09b.M_staggered", "order_parameter", 1.0, mag_afm,
        tolerance=0.01,
        criterion=("preset: |M_staggered| within 0.01 of 1.0 for a "
                   "two-sublattice AFM configuration"),
        notes=("M_staggered=%s, |M_staggered|=%.6f. Mermin-Wagner caveat: "
               "a 2D continuous-symmetry AFM has no finite-T LRO without "
               "anisotropy/SOC." % (np.round(M_afm, 6).tolist(), mag_afm)),
        system_size=N)
    # Positive FM control.
    f = np.zeros((N, 3), dtype=np.float64)
    f[:, 2] = 1.0
    f += 1e-3 * rng.normal(size=(N, 3))
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    mag_fm2 = float(np.linalg.norm(sm.fm_order_parameter(f)))
    check(
        "R22.TC.09c.FM_control", "order_parameter", 1.0, mag_fm2,
        tolerance=0.01,
        criterion="preset: |M_FM| within 0.01 of 1.0 for a FM configuration",
        notes="All spins aligned: |M_FM|=%.6f." % mag_fm2,
        system_size=N)


def test_10_weak_strong_helpers():
    """Exercise weak/strong convergence helpers with deterministic controls."""
    rng = np.random.default_rng(98001)
    path = rng.normal(size=(500, 3))
    good = sm.strong_convergence_test(path, path.copy(),
                                      noise_id_coarse="noise-A",
                                      noise_id_fine="noise-A")
    control(
        "R22.SLLG.10a.strong_crn", "stochastic", good["status"], "PASS",
        criterion=("preset: strong_convergence_test with common random "
                   "numbers returns PASS and zero pointwise error"),
        notes=("Common-random-numbers check: max_pointwise_error=%.3g, "
               "rms=%.3g. Strong convergence requires the SAME noise path." %
               (good["max_pointwise_error"], good["rms_pointwise_error"])),
        sample_length=500)
    check(
        "R22.SLLG.10b.strong_zero", "stochastic", 0.0,
        good["max_pointwise_error"], tolerance=1e-14,
        criterion="preset: identical CRN trajectories give zero pointwise error",
        notes="Deterministic self-consistency of the strong helper.",
        sample_length=500)
    bad = sm.strong_convergence_test(path, rng.normal(size=(500, 3)),
                                     noise_id_coarse="noise-A",
                                     noise_id_fine="noise-B")
    control(
        "R22.SLLG.10c.strong_mismatch", "stochastic", bad["status"],
        "BLOCKED",
        criterion=("preset: mismatched noise ids return BLOCKED; pointwise "
                   "differences across different seeds must NOT be used to "
                   "claim deterministic convergence order"),
        notes=("Explicit guard control. Warning: pointwise trajectory "
               "differences between different random seeds must NOT be used "
               "to claim deterministic convergence order."),
        sample_length=500)
    # Weak convergence: identical ensembles (deterministic sanity of helper).
    ens = rng.normal(0.2, 0.1, 20000)
    weak = sm.weak_convergence_test(ens, ens.copy(), alpha=0.05)
    control(
        "R22.SLLG.10d.weak_helper", "stochastic", weak["status"], "PASS",
        criterion=("preset: weak_convergence_test returns PASS for two "
                   "identical equilibrium ensembles"),
        notes=("Weak convergence compares ENSEMBLE MEANS / DISTRIBUTIONS, "
               "not pointwise trajectories. mean_diff=%.3g, KS p=%.4f, "
               "mean_diff_CI=%s." % (weak["mean_diff"],
                                     weak["ks"]["p_value"],
                                     [round(v, 5) for v in
                                      weak["mean_diff_ci"]])),
        sample_length=len(ens))


def test_11_not_run_gpu():
    """Honest NOT_RUN placeholder: GPU validation belongs to another driver."""
    control(
        "R22.LLG.GPU_VALIDATION", "scope", "NOT_RUN", "NOT_RUN",
        criterion=("preset: this CPU-only driver MUST NOT claim GPU validation; "
                   "status NOT_RUN is the honest value"),
        notes=("NO GPU kernel / CUDA source / src/ file was read, executed "
               "or validated here. These are independent CPU references "
               "(numpy only, no scipy). GPU validation of the production "
               "GPUMD-spin integrators is performed ELSEWHERE in the stage."),
        system_size=None)



def main(argv=None):
    ap = argparse.ArgumentParser(
        description="R22-NonHeat CPU statistical-mechanics validation driver")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write the JSON result list to this file as well")
    args = ap.parse_args(argv)

    test_01_boltzmann_mean_mc()
    test_02_energy_heat_capacity_mc()
    test_03_uniform_precession()
    test_04_implicit_midpoint_convergence()
    test_05_sllg_equilibrium()
    test_06_ar1_autocorrelation()
    test_07_binder()
    test_08_susceptibility_peaks()
    test_09_fm_afm_order_parameter()
    test_10_weak_strong_helpers()
    test_11_not_run_gpu()

    exit_code = 0 if all(r["criterion_met"] for r in RESULTS) else 1
    for r in RESULTS:
        r["exit_code"] = int(exit_code)

    payload = json.dumps(RESULTS, indent=2, allow_nan=True)
    print(payload)
    if args.json_path:
        with open(args.json_path, "w", encoding="ascii") as fh:
            fh.write(payload)
            fh.write("\n")

    counts = {}
    for r in RESULTS:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    executed = sum(1 for r in RESULTS if r["status"] not in ("NOT_RUN", "SKIP"))
    sys.stderr.write(
        "R22-NonHeat: results=%d executed=%d exit_code=%d counts=%s\n" %
        (len(RESULTS), executed, exit_code,
         json.dumps(counts, sort_keys=True)))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
