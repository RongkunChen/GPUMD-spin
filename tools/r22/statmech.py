"""
statmech.py
===========

CPU-ONLY independent statistical-mechanics / dynamics references for the
GPUMD-spin V3 validation stage "R22-NonHeat".

IMPORTANT SCOPE STATEMENT
-------------------------
Everything in this module is a *self-contained CPU reference* implemented with
numpy only (Python 3.13, numpy 2.5.4).  scipy is NOT installed and MUST NOT be
imported.  No GPU kernel, no CUDA file, no src/ file is read, executed or
validated here.  GPU validation of the production GPUMD-spin integrators is
performed ELSEWHERE in the stage; this module provides independent analytic,
statistical and small-integrator references against which GPU results can be
compared by other drivers.

PHYSICAL_MODEL (GPUMD-spin V3, single fixed-length classical spin)
------------------------------------------------------------------
The code integrates the Landau-Lifshitz-Gilbert (LLG) equation for classical
spins S_i of fixed length |S| (magnetic moments in units of mu_B).  Two
branches exist:

  * conservative (alpha = 0, T = 0): an implicit-midpoint iteration with a
    3-component residual and a hard failure on non-convergence;
  * stochastic (SLLG): a Wong-Zakai / Stratonovich interpretation with the
    thermal field held constant over a step.  The thermal-field component
    amplitude is

        H_th,rms = sqrt( 2*alpha*kB*T / ( gamma*(1+alpha**2)*|S|*dt ) )

    and the stationary distribution of the SLLG map is Boltzmann at T and
    does NOT depend on alpha (alpha only fixes the relaxation rate).  This
    alpha-independence was checked against Metropolis MC in earlier stages.

For the analytic references in this module we use the Landau-Lifshitz (LL)
explicit form

    dS/dt = -gamma * S x B  -  (gamma*alpha/|S|) * S x (S x B),

which for a unit spin s = S/|S| reduces to ds/dt = -gamma*[s x B +
alpha*s x (s x B)].  With the thermal field added to B in BOTH terms, the
Stratonovich Fokker-Planck equation has the Boltzmann stationary density
p(s) proportional to exp(beta*|S|*s.B) precisely for the FDT amplitude
quoted above (the (1+alpha**2) factor belongs to the noise strength, not to
the drift prefactor).  Section G documents the alternative Gilbert form.

EQUATIONS (summary, see individual docstrings)
---------------------------------------------
  x       = |S|*|B|/(kB*T)                     (dimensionless field)
  p(u)    = x*exp(x*u)/(2*sinh(x)), u = cos(theta) in [-1,1]
  F(u)    = (exp(x*u)-exp(-x))/(exp(x)-exp(-x))   (CDF)
  <u>     = L(x) = coth(x) - 1/x               (Langevin function)
  U(T)    = <E> = -|S|*|B|*L(x)
  C(T)    = dU/dT = kB*x**2*L'(x) = kB*(1-(x/sinh(x))**2)
  U4      = 1 - <M**4>/(3*<M**2>**2)          (3D vector order parameter)
  chi     = N*(<M**2>-<M>**2)/(kB*T)
  C/N     = (<E**2>-<E>**2)/(N*kB*T**2)
  tau_int = 1/2 + sum_{t>=1} rho(t)           (Sokal automatic windowing)
  ESS     = n/(2*tau_int)

UNITS
-----
  Energy      eV
  Time        fs
  Temperature K
  Field       eV/mu_B
  Spin        mu_B (|S|), often normalised to 1 for order parameters
  gamma       g*mu_B/hbar = 3.0385 mu_B/(eV*fs)
  kB          8.617333262e-5 eV/K
  gamma*|B|   rad/fs (precession angular frequency)

BOUNDARY_CONDITIONS
-------------------
  * u = cos(theta) is defined on the closed interval [-1,1]; the CDF is
    exactly 0 at u=-1 and 1 at u=+1.  Inputs are clipped to that interval.
  * T_K must be strictly positive.  B may be positive, negative or zero
    (the sign of x flips the favoured spin direction).
  * Spins are fixed-length: all analytic flows preserve |S| exactly.
  * Periodic boundary conditions / lattice geometry are NOT modelled here;
    N is only a counting factor in the fluctuation formulas.

LIMITATIONS / HONESTY
---------------------
  * Mermin-Wagner: a 2D magnet with continuous spin symmetry has NO
    finite-temperature long-range order.  A finite-T transition requires
    anisotropy / spin-orbit coupling (SOC).  ANY Tc analysis performed with
    the Binder-crossing or susceptibility-peak helpers below MUST state this
    caveat; the estimators assume a symmetry-broken (anisotropic) model.
  * Binder crossings and susceptibility peaks from finite lattices give
    finite-size pseudo-critical temperatures, NOT a thermodynamic Tc.  A
    single magnetisation curve is insufficient to locate Tc.
  * Energy drift (or a single conserved quantity) alone CANNOT establish
    second-order accuracy: a norm/energy-preserving scheme can still have a
    large phase error.  Convergence order must be measured against the SAME
    trusted analytic solution using a trajectory/phase norm.
  * Pointwise trajectory differences between runs with DIFFERENT random
    seeds must NOT be used to claim a deterministic convergence order; use
    common random numbers for strong convergence and ensemble observables
    for weak convergence.
  * KS / chi-square p-values use asymptotic null distributions; they are
    valid for large independent samples.  Correlated time series MUST be
    thinned / decorrelated first (see tau_int and ESS helpers).

STATUS VOCABULARY (R22-NonHeat)
-------------------------------
  PASS / FAIL / SKIP / BLOCKED / NOT_RUN / NOT_RESOLVED / NOT_CONVERGED.
  A test may be marked PASS only when it actually executed and met a preset
  numeric criterion.  Helpers below return this vocabulary verbatim.

All identifiers, strings and numeric literals use ASCII straight quotes only.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "KB_EV_PER_K",
    "GAMMA_MUB_PER_EV_FS",
    "VALID_STATUSES",
    "single_spin_boltzmann_pdf",
    "single_spin_boltzmann_cdf",
    "single_spin_mean_cos_theta",
    "single_spin_mean_energy",
    "single_spin_internal_energy",
    "single_spin_heat_capacity",
    "autocorrelation",
    "integrated_autocorrelation_time",
    "effective_sample_size",
    "blocking_error",
    "normal_quantile",
    "confidence_interval",
    "binder_cumulant",
    "fm_order_parameter",
    "afm_order_parameter",
    "magnetisation_magnitude",
    "susceptibility",
    "specific_heat",
    "estimate_tc_binder_crossing",
    "estimate_tc_susceptibility_peak",
    "convergence_order",
    "spin_phase",
    "phase_error",
    "weak_convergence_test",
    "strong_convergence_test",
    "ks_test_two_samples",
    "ks_test_one_sample",
    "chi_square_gof",
    "llg_analytic_uniform_precession",
    "llg_analytic_damped",
]

KB_EV_PER_K = 8.617333262e-5           # eV/K
GAMMA_MUB_PER_EV_FS = 3.0385           # mu_B/(eV*fs)

VALID_STATUSES = (
    "PASS", "FAIL", "SKIP", "BLOCKED", "NOT_RUN", "NOT_RESOLVED",
    "NOT_CONVERGED",
)


# Section A: exact single-spin Boltzmann statics

def _boltzmann_x(B_eV_per_muB, S_muB, T_K):
    """x = |S|*|B|/(kB*T); dimensionless, sign follows the field."""
    T = np.asarray(T_K, dtype=np.float64)
    if np.any(T <= 0.0):
        raise ValueError("T_K must be strictly positive (Kelvin).")
    B = np.asarray(B_eV_per_muB, dtype=np.float64)
    S = np.asarray(S_muB, dtype=np.float64)
    return S * B / (KB_EV_PER_K * T)


def _finish_scalar(out, x):
    """Return a python float when the input was scalar."""
    if np.asarray(x).ndim == 0:
        return float(np.asarray(out).reshape(-1)[0])
    return out


def _langevin(x):
    """L(x) = coth(x) - 1/x, numerically stable near x = 0.

    Series (|x| < 1e-2): x/3 - x**3/45 + 2*x**5/945 - x**7/4725 ...
    Direct : 1/tanh(x) - 1/x.
    """
    ax = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(ax)
    small = np.abs(ax) < 1e-2
    z = ax[small]
    out[small] = z / 3.0 - z**3 / 45.0 + 2.0 * z**5 / 945.0 - z**7 / 4725.0
    y = ax[~small]
    out[~small] = 1.0 / np.tanh(y) - 1.0 / y
    return _finish_scalar(out, x)


def _langevin_derivative(x):
    """L'(x) = 1/x**2 - 1/sinh(x)**2, stable near x = 0.

    Series (|x| < 1e-2): 1/3 - x**2/15 + 2*x**4/189 - x**6/675 ...
    """
    ax = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(ax)
    small = np.abs(ax) < 1e-2
    z = ax[small]
    out[small] = 1.0 / 3.0 - z**2 / 15.0 + 2.0 * z**4 / 189.0 - z**6 / 675.0
    y = ax[~small]
    e = np.exp(-2.0 * np.abs(y))              # overflow-free csch**2 form
    csch2 = 4.0 * e / (1.0 - e) ** 2
    out[~small] = 1.0 / y**2 - csch2
    return _finish_scalar(out, x)


def single_spin_boltzmann_pdf(cos_theta, B_eV_per_muB, S_muB, T_K):
    """Exact normalised P(u), u = cos(theta), for one classical spin.

    ASSUMPTIONS: fixed |S|, uniform solid-angle measure dOmega = 2*pi*du,
    energy E(u) = -|S|*|B|*u, Boltzmann weight exp(beta*|S|*|B|*u).

    EQUATION: p(u) = x*exp(x*u)/(2*sinh(x)), x = |S|*|B|/(kB*T),
    normalised over [-1,1] (integral = 1).  For x -> 0, p(u) -> 1/2.

    UNITS: u, x and p are dimensionless; B in eV/mu_B, S in mu_B, T in K.
    BOUNDARY_CONDITIONS: u clipped to [-1,1]; p(-1)*exp(x*...) etc. finite.
    LIMITATIONS: single spin only, no exchange, no anisotropy, no quantum
    corrections; valid for any sign of B (x may be negative).
    """
    x = float(np.asarray(_boltzmann_x(B_eV_per_muB, S_muB, T_K)))
    u = np.clip(np.asarray(cos_theta, dtype=np.float64), -1.0, 1.0)
    s = abs(x)
    if s < 1e-14:
        return np.full_like(u, 0.5) if u.ndim > 0 else 0.5
    q = 1.0 if x > 0.0 else -1.0
    # Stable for all |x|: exponent s*(q*u - 1) <= 0, den = 1-exp(-2s) > 0.
    den = -np.expm1(-2.0 * s)
    val = s * np.exp(s * (q * u - 1.0)) / den
    return _finish_scalar(val, cos_theta)


def single_spin_boltzmann_cdf(cos_theta_grid, B_eV_per_muB, S_muB, T_K):
    """Exact analytic CDF F(u) = P(U <= u) of the Boltzmann density.

    EQUATION: F(u) = (exp(x*u) - exp(-x)) / (exp(x) - exp(-x)),
    with x = |S|*|B|/(kB*T).  F(-1)=0, F(+1)=1, F(u)=(u+1)/2 for x=0.

    Implemented with sign-branched expm1-style expressions so that large
    positive or negative x never overflows.  BOUNDARY_CONDITIONS: input grid
    clipped to [-1,1]; monotone non-decreasing; exact endpoints.
    UNITS: dimensionless.  LIMITATIONS: single-spin, classical, fixed |S|.
    """
    x = float(np.asarray(_boltzmann_x(B_eV_per_muB, S_muB, T_K)))
    u = np.clip(np.asarray(cos_theta_grid, dtype=np.float64), -1.0, 1.0)
    s = abs(x)
    if s < 1e-14:
        val = 0.5 * (u + 1.0)
        return _finish_scalar(val, cos_theta_grid)
    if x > 0.0:
        num = np.exp(s * (u - 1.0)) - np.exp(-2.0 * s)
        den = -np.expm1(-2.0 * s)
    else:
        num = -np.expm1(-s * (u + 1.0))
        den = -np.expm1(-2.0 * s)
    val = num / den
    return _finish_scalar(val, cos_theta_grid)


def single_spin_mean_cos_theta(B_eV_per_muB, S_muB, T_K):
    """<cos(theta)> = Langevin function L(x) = coth(x) - 1/x.

    Stable at x=0 (L(0)=0) and odd in x (flips with the field direction).
    UNITS: dimensionless.  LIMITATIONS: classical single spin, no SOC.
    """
    return _langevin(_boltzmann_x(B_eV_per_muB, S_muB, T_K))


def single_spin_mean_energy(B_eV_per_muB, S_muB, T_K):
    """Mean energy <E> = -|S|*|B|*L(x) in eV.

    EQUATION: E(u) = -|S|*|B|*u and <u> = L(x), hence <E> = -|S|*|B|*L(x).
    UNITS: eV.  BOUNDARY_CONDITIONS: T>0; B may be any sign.
    """
    B = np.asarray(B_eV_per_muB, dtype=np.float64)
    S = np.asarray(S_muB, dtype=np.float64)
    return -S * B * _langevin(_boltzmann_x(B, S, T_K))


def single_spin_internal_energy(B_eV_per_muB, S_muB, T_K):
    """Internal energy U(T) = <E>(T) in eV, identical to mean_energy.

    Kept as a distinct name so that U(T) can be read as a function of
    temperature for the heat-capacity derivative below.  Vectorised in T_K.
    """
    return single_spin_mean_energy(B_eV_per_muB, S_muB, T_K)


def single_spin_heat_capacity(B_eV_per_muB, S_muB, T_K):
    """Heat capacity C(T) = dU/dT in eV/K (single spin, N=1).

    EQUATION: with x = a/T, a = |S|*|B|/kB,
        U = -|S|*|B|*L(x),
        C = dU/dT = kB*x**2*L'(x) = kB*(1 - (x/sinh(x))**2) >= 0,
    where L'(x) = 1/x**2 - 1/sinh(x)**2.  Limits: C -> 0 as T -> 0 and as
    T -> infinity (Schottky-like single-spin heat capacity).

    UNITS: eV/K.  LIMITATIONS: classical single spin, fixed |S|, no exchange.
    """
    x = _boltzmann_x(B_eV_per_muB, S_muB, T_K)
    return KB_EV_PER_K * np.asarray(x, dtype=np.float64) ** 2 * _langevin_derivative(x)


# Section B: time-series statistics (numpy only, no scipy)

def autocorrelation(x, max_lag):
    """FFT-based normalised autocorrelation rho(t), t = 0..max_lag.

    Uses the biased covariance estimator (divide by n, not n-t), which is
    positive semi-definite and standard for FFT autocorrelation:
        rho(t) = sum_{i} xc_i*xc_{i+t} / sum_i xc_i**2,  xc = x - mean(x).
    rho(0) = 1 exactly.  BOUNDARY_CONDITIONS: max_lag clipped to n-1.
    LIMITATIONS: assumes a stationary, mean-zero-after-centering series;
    a constant series returns zeros (rho(0)=0 sentinel).
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    if n < 2:
        return np.zeros(1, dtype=np.float64)
    xc = a - a.mean()
    v = float(np.dot(xc, xc))
    lim = int(min(int(max_lag), n - 1))
    if v <= 0.0:
        return np.zeros(lim + 1, dtype=np.float64)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(xc, nfft)
    acov = np.fft.irfft(f * np.conjugate(f), nfft)[:n]
    if acov[0] <= 0.0:
        return np.zeros(lim + 1, dtype=np.float64)
    return np.asarray(acov[:lim + 1] / acov[0], dtype=np.float64)


def integrated_autocorrelation_time(x, c=5.0, max_lag=None, with_window=False):
    """Sokal's self-consistent automatic windowing for tau_int.

    tau(M) = 1/2 + sum_{t=1..M} rho(t).  The window M is the smallest integer
    with M >= c*tau(M) (default c=5, Sokal's recommendation for exponential
    autocorrelation; use c=10 for a conservative bound).  The returned
    tau_int is tau(M) at that self-consistent window, floored at 0.5.

    BOUNDARY_CONDITIONS: if no self-consistent window exists, M falls back to
    max_lag.  LIMITATIONS: statistical noise in rho(t) at large t; do not use
    on non-stationary / too-short series.  Return (tau_int, M) if
    with_window=True.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    lim = n - 1 if max_lag is None else int(min(int(max_lag), n - 1))
    lim = max(1, lim)
    rho = autocorrelation(a, lim)
    tau_M = 0.5 + np.cumsum(rho[1:])        # tau_M[k-1] = tau(k)
    M_grid = np.arange(1, lim + 1, dtype=np.int64)
    ok = M_grid >= c * tau_M
    M = int(M_grid[int(np.argmax(ok))]) if np.any(ok) else lim
    tau = max(float(0.5 + np.sum(rho[1:M + 1])), 0.5)
    return (tau, M) if with_window else tau


def effective_sample_size(n, tau_int):
    """ESS = n/(2*tau_int): number of independent samples equivalent to n
    correlated ones (Var(mean) = 2*tau_int*Var(x)/n).  tau_int is floored at
    0.5 so ESS <= n.  LIMITATIONS: asymptotic; assumes stationarity.
    """
    tau = max(float(tau_int), 0.5)
    return float(n) / (2.0 * tau)


def blocking_error(x, nblocks):
    """Standard error of the mean from the block-averages method.

    Split the series into nblocks contiguous blocks of equal length (remainder
    dropped), take block means, and return
        stderr = std(block_means, ddof=1)/sqrt(nblocks).
    EQUATIONS/UNITS: same units as x.  BOUNDARY_CONDITIONS: nblocks clipped
    to [1, n]; returns 0.0 if fewer than 2 usable blocks.
    LIMITATIONS: block length must exceed tau_int or the error is
    underestimated; this is the responsibility of the caller.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    n = a.size
    nb = int(min(max(int(nblocks), 1), n))
    m = n // nb
    if m < 1 or nb < 2:
        return 0.0
    blocks = a[:nb * m].reshape(nb, m).mean(axis=1)
    return float(blocks.std(ddof=1) / math.sqrt(nb))


def normal_quantile(p):
    """Inverse standard-normal CDF (Acklam's rational approximation).

    Relative error < 1.15e-9 before one Halley refinement step using
    math.erfc; refined result is near machine precision.  Implemented here
    because scipy is NOT available.  Accepts scalars or numpy arrays.
    p must lie in (0,1).
    """
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    arr = np.asarray(p, dtype=np.float64)
    if np.any(arr <= 0.0) or np.any(arr >= 1.0):
        raise ValueError("normal_quantile: p must be in the open interval (0,1).")
    scalar = (arr.ndim == 0)
    arr = np.atleast_1d(arr)
    out = np.empty_like(arr)
    low = arr < 0.02425
    high = arr > 1.0 - 0.02425
    mid = ~(low | high)

    q = np.sqrt(-2.0 * np.log(arr[low]))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    out[low] = num / den

    q = arr[mid] - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
    out[mid] = num / den

    q = np.sqrt(-2.0 * np.log(np.maximum(1.0 - arr[high], 1e-300)))
    num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
    den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    out[high] = -num / den

    # One Halley refinement: e = Phi(x)-p; u = e*sqrt(2*pi)*exp(x**2/2).
    for i in range(out.size):
        xx = float(out[i])
        e = 0.5 * math.erfc(-xx / math.sqrt(2.0)) - float(arr[i])
        u = e * math.sqrt(2.0 * math.pi) * math.exp(xx * xx / 2.0)
        out[i] = xx - u / (1.0 + xx * u / 2.0)
    return float(out[0]) if scalar else out


def confidence_interval(mean, stderr, confidence=0.95):
    """Two-sided normal confidence interval (lo, hi) for a mean.

    EQUATION: z = Phi^{-1}(0.5 + confidence/2); CI = mean +/- z*stderr.
    Uses our own normal_quantile (no scipy).  LIMITATIONS: assumes the mean
    estimator is approximately Gaussian and stderr is trustworthy; for
    correlated data use blocking_error / ESS-adjusted stderr.
    """
    if not (0.0 < float(confidence) < 1.0):
        raise ValueError("confidence must be in (0,1).")
    z = float(normal_quantile(0.5 + 0.5 * float(confidence)))
    m = float(mean)
    s = float(stderr)
    return (m - z * s, m + z * s)


# Section C: order parameters, Binder cumulant, fluctuations

def magnetisation_magnitude(order_parameter_vector):
    """|M| of a (...,3) order-parameter vector, or |M| of scalar samples."""
    m = np.asarray(order_parameter_vector, dtype=np.float64)
    if m.ndim >= 1 and m.shape[-1] == 3 and m.ndim >= 2:
        return np.linalg.norm(m, axis=-1)
    return np.abs(m)


def binder_cumulant(magnetisation_samples):
    """Binder cumulant U4 = 1 - <M**4>/(3*<M**2>**2).

    DOCUMENTATION (order parameter convention):
      magnetisation_samples may be
        * shape (n,)   : scalar magnitudes m = |M| of a 3D vector M; or
        * shape (n, 3) : the 3D vector order-parameter samples M themselves.
      In both cases M**2 = M.M = |M|**2 and M**4 = (M.M)**2.
      For an O(3) order parameter at high T (Gaussian) <M**4>/<M**2>**2 =
      1 + 2/3 = 5/3, hence U4 -> 4/9.  In a fully ordered state
      (|M| fixed) U4 -> 2/3.  Crossings of U4(L,T) for different L locate
      Tc (see estimate_tc_binder_crossing and the Mermin-Wagner caveat).
    UNITS: dimensionless.  BOUNDARY_CONDITIONS: returns NaN if <M**2> = 0.
    LIMITATIONS: finite-size estimator; needs many independent samples.
    """
    m = np.asarray(magnetisation_samples, dtype=np.float64)
    if m.ndim == 1:
        m2 = m ** 2
    elif m.ndim == 2 and m.shape[1] == 3:
        m2 = np.sum(m ** 2, axis=1)
    else:
        raise ValueError("magnetisation_samples must be (n,) or (n,3).")
    avg_m2 = float(np.mean(m2))
    if avg_m2 <= 0.0:
        return float("nan")
    return 1.0 - float(np.mean(m2 ** 2)) / (3.0 * avg_m2 ** 2)


def fm_order_parameter(spin_traj):
    """Ferromagnetic order parameter: normalised vector magnetisation.

    DOCUMENTATION:
      spin_traj shape (..., N, 3): trajectory of N fixed-length spins
      (units mu_B).  Each spin is normalised to a UNIT vector before
      averaging, so M_FM = (1/N) sum_i s_i has magnitude in [0,1] and is
      dimensionless.  Returns a VECTOR of shape (..., 3); the scalar
      magnetisation is |M_FM| (see magnetisation_magnitude).

      For a FERROMAGNET the correct order parameter is M_FM.  For an
      ANTIFERROMAGNET M_FM vanishes by symmetry and one MUST use the
      staggered magnetisation (afm_order_parameter).

    UNITS: dimensionless (spins normalised).  LIMITATIONS: no lattice
    geometry / periodic boundaries modelled; sublattice labelling is external.
    """
    a = np.asarray(spin_traj, dtype=np.float64)
    if a.shape[-1] != 3:
        raise ValueError("spin_traj last axis must be 3 components.")
    nrm = np.linalg.norm(a, axis=-1, keepdims=True)
    s = a / np.where(nrm > 0.0, nrm, 1.0)
    return np.mean(s, axis=-2)


def afm_order_parameter(spin_traj, sublattice):
    """Antiferromagnetic (staggered) order parameter for two sublattices.

    DOCUMENTATION:
      sublattice: (N,) integer labels, either {0,1} (parity/site index) or
      arbitrary integer labels; two distinct values are required.  Sign
      sigma_i = +1 for the first label, -1 for the second.  The staggered
      magnetisation is M_stag = (1/N) sum_i sigma_i * s_i, a VECTOR of shape
      (..., 3), magnitude in [0,1].  For a Neel configuration
      (even sites +z, odd sites -z) |M_stag| -> 1 while |M_FM| -> 0.

    Correct order parameter: AFM => staggered M (this function), NOT M_FM.
    UNITS: dimensionless.  LIMITATIONS: two-sublattice only; a general
    ordering wavevector q would require sigma_i = exp(i q . r_i).
    """
    a = np.asarray(spin_traj, dtype=np.float64)
    sub = np.asarray(sublattice).ravel()
    if a.shape[-2] != sub.size:
        raise ValueError("sublattice length must equal the number of spins.")
    labels = np.unique(sub)
    if labels.size != 2:
        raise ValueError("afm_order_parameter requires exactly 2 sublattices.")
    sigma = np.where(sub == labels[0], 1.0, -1.0)
    nrm = np.linalg.norm(a, axis=-1, keepdims=True)
    s = a / np.where(nrm > 0.0, nrm, 1.0)
    w = sigma.reshape((1,) * (s.ndim - 2) + (sub.size, 1))
    return np.mean(w * s, axis=-2)


def susceptibility(M_samples, N, T_K):
    """Magnetic susceptibility chi = N*(<M**2>-<M>**2)/(kB*T).

    DOCUMENTATION:
      M_samples: samples of the scalar magnetisation magnitude (n,) or of the
      3D order-parameter vector (n,3); in the latter case |M| is used.  N is
      the number of spins, T_K in Kelvin, kB in eV/K.  The formula is the
      connected fluctuation of the ORDER PARAMETER, so for a zero-field O(3)
      magnet the scalar |M| (not a single component) must be used.

    UNITS: if M is dimensionless (spins normalised), chi has units 1/eV;
    if M is in mu_B, chi is in mu_B**2/eV.  BOUNDARY_CONDITIONS: T>0, N>0.
    LIMITATIONS: finite-size; assumes the samples are equilibrium and
    (approximately) independent.
    """
    m = np.asarray(M_samples, dtype=np.float64)
    if m.ndim == 2 and m.shape[1] == 3:
        m = np.linalg.norm(m, axis=1)
    else:
        m = m.ravel()
    T = float(T_K)
    if T <= 0.0 or N <= 0:
        raise ValueError("susceptibility: need T>0 and N>0.")
    var = float(np.mean(m ** 2) - np.mean(m) ** 2)
    return N * var / (KB_EV_PER_K * T)


def specific_heat(E_samples, N, T_K):
    """Specific heat per spin C/N = (<E**2>-<E>**2)/(N*kB*T**2).

    DOCUMENTATION: fluctuation-dissipation formula for the canonical
    ensemble.  E_samples in eV, N number of spins, T in K, kB in eV/K.
    Var(E)/(kB*T**2) is the TOTAL heat capacity in eV/K; dividing by N
    gives C per spin.  UNITS: eV/K per spin.
    BOUNDARY_CONDITIONS: T>0, N>0.
    LIMITATIONS: assumes equilibrium, independent samples; finite-size
    peaks give pseudo-critical temperatures, not a thermodynamic Tc.
    """
    e = np.asarray(E_samples, dtype=np.float64).ravel()
    T = float(T_K)
    if T <= 0.0 or N <= 0:
        raise ValueError("specific_heat: need T>0 and N>0.")
    var = float(np.mean(e ** 2) - np.mean(e) ** 2)
    return var / (N * KB_EV_PER_K * T ** 2)


# Section D: critical-temperature estimators (with honest statuses)

def estimate_tc_binder_crossing(sizes, temperatures, U4_curves):
    """Binder-cumulant crossing estimator of Tc.

    DOCUMENTATION / PHYSICAL_MODEL:
      U4_curves[i,:] is the Binder cumulant for linear size sizes[i] on the
      COMMON temperature grid temperatures.  For every pair (i,j) we form
      D(T) = U4_i(T) - U4_j(T), locate sign changes on the grid and
      linearly interpolate the crossing temperature.  The reported estimate
      is the mean +/- std over pairs, plus a linear extrapolation of the pair
      crossings versus 1/L_eff (L_eff = harmonic mean of the two sizes) to
      1/L -> 0.

    CAVEAT (Mermin-Wagner): 2D magnets with continuous spin symmetry have NO
    finite-T long-range order.  A finite-T transition requires anisotropy /
    SOC.  This estimator assumes a symmetry-broken (anisotropic) model; it
    must NOT be applied to an isotropic 2D Heisenberg model.

    STATUS RULES (honest):
      * fewer than 3 sizes              -> NOT_CONVERGED
      * zero / fewer than 2 crossings  -> NOT_CONVERGED (curves do not cross)
      * otherwise                      -> PASS, with tc_mean, tc_std,
                                          tc_extrapolated.
    It NEVER claims a precise Tc from a single magnetisation curve: >= 3
    independent system sizes are mandatory.  The result is a finite-size
    crossing temperature; thermodynamic Tc needs a proper scaling fit.

    UNITS: temperatures in K; sizes are dimensionless lattice lengths.
    """
    L = np.asarray(sizes, dtype=np.float64).ravel()
    T = np.asarray(temperatures, dtype=np.float64).ravel()
    U = np.asarray(U4_curves, dtype=np.float64)
    base = {
        "status": "NOT_CONVERGED", "tc_mean": None, "tc_std": None,
        "tc_extrapolated": None, "n_sizes": int(L.size),
        "n_pairs": 0, "n_crossings": 0, "crossings": [],
        "notes": (
            "Mermin-Wagner caveat: a 2D magnet with continuous spin symmetry "
            "has no finite-T long-range order; a finite-T transition requires "
            "anisotropy/SOC. Binder crossings give finite-size estimates only."
        ),
    }
    if L.size < 3:
        base["notes"] += " Fewer than 3 sizes supplied -> NOT_CONVERGED."
        return base
    if U.ndim != 2 or U.shape[0] != L.size or U.shape[1] != T.size:
        base["status"] = "NOT_RESOLVED"
        base["notes"] += " U4_curves shape mismatch with sizes/temperatures."
        return base
    pairs = [(i, j) for i in range(L.size) for j in range(i + 1, L.size)]
    base["n_pairs"] = len(pairs)
    crossings = []
    for (i, j) in pairs:
        d = U[i] - U[j]
        tc = None
        for k in range(T.size - 1):
            if d[k] == 0.0:
                tc = float(T[k])
                break
            if d[k] * d[k + 1] < 0.0:
                tc = float(T[k] - d[k] * (T[k + 1] - T[k]) / (d[k + 1] - d[k]))
                break
        if tc is not None:
            crossings.append({
                "pair": [int(i), int(j)],
                "sizes": [float(L[i]), float(L[j])],
                "tc": tc,
                "L_eff": float(2.0 * L[i] * L[j] / (L[i] + L[j])),
            })
    base["n_crossings"] = len(crossings)
    base["crossings"] = crossings
    if len(crossings) < 2:
        base["notes"] += " Curves do not cross (or only one pair crosses)."
        return base
    tc_arr = np.array([c["tc"] for c in crossings], dtype=np.float64)
    invL = np.array([1.0 / c["L_eff"] for c in crossings], dtype=np.float64)
    slope, intercept = np.polyfit(invL, tc_arr, 1)
    base.update({
        "status": "PASS",
        "tc_mean": float(np.mean(tc_arr)),
        "tc_std": float(np.std(tc_arr, ddof=1)) if tc_arr.size > 1 else 0.0,
        "tc_extrapolated": float(intercept),
        "extrapolation_slope": float(slope),
        "notes": base["notes"] + (
            " Finite-size crossing: mean/std over %d crossing pairs; "
            "extrapolated Tc from linear fit in 1/L_eff. "
            "A single magnetisation curve is insufficient." % len(crossings)
        ),
    })
    return base


def estimate_tc_susceptibility_peak(temperatures, chi_curve):
    """Finite-size pseudo-critical temperature from a susceptibility peak.

    DOCUMENTATION:
      Locates the discrete maximum of chi_curve, then refines it by a
      parabola through the three grid points (k-1, k, k+1):
        tc_pseudo = -b/(2*a) for chi(T) = a*T**2 + b*T + c.
      The result is EXPLICITLY a finite-size PSEUDO-critical temperature,
      NOT a thermodynamic Tc; extrapolating Tc requires several system sizes
      plus finite-size scaling.  CAVEAT (Mermin-Wagner): 2D magnets with
      continuous spin symmetry have no finite-T long-range order; a
      finite-T transition requires anisotropy/SOC.

    STATUS RULES:
      * maximum at a grid boundary (monotone curve, e.g. a SINGLE
        magnetisation curve) -> NOT_CONVERGED;
      * non-concave parabola or vertex outside the 3-point bracket
        -> NOT_CONVERGED;
      * otherwise -> PASS with tc_pseudo and peak_value.
      A single magnetisation curve is insufficient and returns
      NOT_CONVERGED; this function NEVER claims a precise Tc from one curve.

    UNITS: temperatures in K; chi_curve in the caller's units (e.g. 1/eV).
    """
    T = np.asarray(temperatures, dtype=np.float64).ravel()
    y = np.asarray(chi_curve, dtype=np.float64).ravel()
    out = {
        "status": "NOT_CONVERGED", "tc_pseudo": None, "peak_value": None,
        "curvature": None, "notes": (
            "Mermin-Wagner caveat: 2D magnets with continuous spin symmetry "
            "have no finite-T long-range order; a finite-T transition "
            "requires anisotropy/SOC. This is a finite-size pseudo-critical "
            "temperature from a susceptibility peak, NOT a thermodynamic Tc."
        ),
    }
    if T.size != y.size or T.size < 3:
        out["notes"] += " Grid/value mismatch or fewer than 3 points."
        return out
    k = int(np.argmax(y))
    if k == 0 or k == T.size - 1:
        out["notes"] += (
            " No interior peak (maximum at grid boundary): a single "
            "magnetisation curve is insufficient -> NOT_CONVERGED."
        )
        return out
    idx = np.array([k - 1, k, k + 1])
    a, b, c = np.polyfit(T[idx], y[idx], 2)
    if a >= 0.0:
        out["notes"] += " Parabola is not concave (no maximum)."
        return out
    tc = -b / (2.0 * a)
    if not (T[k - 1] <= tc <= T[k + 1]):
        out["notes"] += " Refined vertex outside the 3-point bracket."
        return out
    out.update({
        "status": "PASS", "tc_pseudo": float(tc),
        "peak_value": float(y[k]), "curvature": float(a),
        "notes": out["notes"] + (
            " Parabolic refinement around grid index %d; labelled as a "
            "finite-size pseudo-critical temperature." % k
        ),
    })
    return out


# Section E: convergence order and phase error

def convergence_order(errors, dts, reference=None, same_reference=True):
    """Observed convergence order p from a dt-refinement study.

    EQUATION: for consecutive pairs with dt_k/d_{k+1} = r (r=2 for halving),
        p_k = log2(E_k/E_{k+1}) / log2(r)   [= log2(E_k/E_{k+1}) for r=2].
    This is the requested p = log2(E(dt)/E(dt/2)) for halving sequences.

    EXPLICIT GUARD (trusted reference): `reference` MUST identify the SAME
    trusted analytic solution against which EVERY error was measured.  If
    reference is None, or same_reference is False, the status is
    NOT_RESOLVED: mixing references (e.g. comparing run k against run k+1,
    or against different analytic solutions) does NOT measure convergence
    order and is refused.  The caller must pass the identifier of the single
    trusted solution actually used.

    HONESTY NOTE: energy drift alone CANNOT establish second-order accuracy.
    A norm-preserving / energy-conserving scheme (e.g. implicit midpoint for
    LLG) can have zero energy error at every dt while its trajectory/phase
    error is O(dt**2).  Use a trajectory or phase norm for `errors`.

    UNITS: errors in any consistent norm; dts in fs.
    """
    e = np.asarray(errors, dtype=np.float64).ravel()
    d = np.asarray(dts, dtype=np.float64).ravel()
    res = {
        "status": "PASS", "orders": [], "mean_order": None,
        "dt_ratios": [], "notes": "",
    }
    if e.size != d.size or e.size < 2:
        res["status"] = "NOT_RESOLVED"
        res["notes"] = "errors and dts must have equal length >= 2."
        return res
    if np.any(e <= 0.0) or np.any(d <= 0.0):
        res["status"] = "NOT_RESOLVED"
        res["notes"] = "errors and dts must be strictly positive."
        return res
    ratios = d[:-1] / d[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.log2(e[:-1] / e[1:]) / np.log2(ratios)
    res["orders"] = [float(v) for v in p]
    res["mean_order"] = float(np.mean(p))
    res["dt_ratios"] = [float(v) for v in ratios]
    res["notes"] = (
        "p = log2(E(dt)/E(dt/2)) per consecutive pair (generalised by "
        "log2(dt_k/dt_{k+1})). Energy drift alone cannot establish "
        "second-order accuracy; use a trajectory/phase norm."
    )
    if not np.allclose(ratios, 2.0, rtol=1e-9):
        res["notes"] += " WARNING: dt ratios are not exactly 2."
    if reference is None or not same_reference:
        res["status"] = "NOT_RESOLVED"
        res["notes"] += (
            " GUARD TRIPPED: the SAME trusted solution was not certified as "
            "the reference for all errors; order is NOT_RESOLVED."
        )
    else:
        res["notes"] += (
            " All errors measured against the SAME trusted solution: %s." %
            str(reference)
        )
    return res


def _orthonormal_basis(axis):
    """Return (e1, e2, axis_unit) with e1,e2 perpendicular to axis."""
    ax = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(ax))
    if n <= 0.0:
        raise ValueError("axis must be non-zero.")
    a = ax / n
    tmp = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    return e1, e2, a


def spin_phase(spin_traj, axis):
    """Precession phase of a spin trajectory in the plane perpendicular to
    axis (the field direction).  Returns atan2(S.e2, S.e1) for shape
    (...,3) input; UNITS: radians, wrapped by arctan2 to (-pi, pi]."""
    a = np.asarray(spin_traj, dtype=np.float64)
    e1, e2, _ = _orthonormal_basis(axis)
    return np.arctan2(a @ e2, a @ e1)


def phase_error(spin_traj, reference, axis):
    """Wrapped phase error between a numerical and a trusted trajectory.

    EQUATION: delta_phi(t) = Arg(exp(i*(phi_num(t) - phi_ref(t)))), wrapped
    to [-pi, pi].  UNITS: radians.  LIMITATIONS: undefined if the transverse
    component vanishes (spin parallel to axis); requires the SAME physical
    time grid for both inputs.  A trajectory/phase norm (not energy drift)
    is the correct observable for measuring convergence order.
    """
    p_num = spin_phase(spin_traj, axis)
    p_ref = spin_phase(reference, axis)
    return np.angle(np.exp(1j * (np.asarray(p_num) - np.asarray(p_ref))))


# Section F: stochastic-convergence helpers and distribution tests

def _kolmogorov_q(lam):
    """Asymptotic two-sided Kolmogorov survival function Q(lam).

    Q(lam) = 2*sum_{k=1..inf} (-1)**(k-1)*exp(-2*k**2*lam**2).
    For lam < 0.2 the p-value is indistinguishable from 1.
    """
    if lam <= 0.0:
        return 1.0
    if lam < 0.2:
        return 1.0
    s = 0.0
    for k in range(1, 300):
        t = math.exp(-2.0 * k * k * lam * lam)
        s += t if (k % 2) else -t
        if abs(t) < 1e-17:
            break
    return min(1.0, max(0.0, 2.0 * s))


def _ks_statistic(x, y):
    """Two-sample KS D statistic for sorted-or-unsorted 1-D samples."""
    a = np.sort(np.asarray(x, dtype=np.float64).ravel())
    b = np.sort(np.asarray(y, dtype=np.float64).ravel())
    n1, n2 = a.size, b.size
    allv = np.concatenate([a, b])
    grid = np.unique(allv)
    fa = np.searchsorted(a, grid, side="right") / n1
    fb = np.searchsorted(b, grid, side="right") / n2
    return float(np.max(np.abs(fa - fb))), n1, n2


def ks_test_two_samples(x, y):
    """Two-sample Kolmogorov-Smirnov test (numpy only, no scipy).

    D = sup|F_n - F_m|.  Effective sample size n_eff = n*m/(n+m); the
    asymptotic p-value uses lam = (sqrt(n_eff)+0.12+0.11/sqrt(n_eff))*D and
    Q(lam) above.  Also returns the 95% critical value
    D_crit = sqrt(-0.5*ln(0.025))/sqrt(n_eff) ~ 1.3581/sqrt(n_eff).

    UNITS: dimensionless.  LIMITATIONS: asymptotic null; samples MUST be
    independent (thin correlated time series first, see tau_int/ESS).
    """
    D, n1, n2 = _ks_statistic(x, y)
    n_eff = n1 * n2 / (n1 + n2)
    lam = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * D
    p_value = _kolmogorov_q(lam)
    d_crit = math.sqrt(-0.5 * math.log(0.025)) / math.sqrt(n_eff)
    return {
        "D": D, "p_value": float(p_value), "n1": int(n1), "n2": int(n2),
        "n_eff": float(n_eff), "lambda": float(lam), "d_crit_95": float(d_crit),
        "reject_at_95": bool(p_value < 0.05),
    }


def ks_test_one_sample(x, cdf):
    """One-sample KS test of i.i.d. samples against a known analytic CDF.

    D = max(D+, D-) with D+ = max(i/n - F(x_i)), D- = max(F(x_i) - (i-1)/n)
    on sorted samples.  Returns D, asymptotic p_value, 95% critical value
    1.3581/sqrt(n) and a reject flag.  BOUNDARY_CONDITIONS: cdf must map the
    sample support to [0,1].  LIMITATIONS: same independence requirement as
    ks_test_two_samples.
    """
    xs = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = xs.size
    F = np.clip(np.asarray(cdf(xs), dtype=np.float64), 0.0, 1.0)
    i = np.arange(1, n + 1, dtype=np.float64)
    d_plus = float(np.max(i / n - F))
    d_minus = float(np.max(F - (i - 1.0) / n))
    D = max(d_plus, d_minus)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * D
    p_value = _kolmogorov_q(lam)
    d_crit = math.sqrt(-0.5 * math.log(0.025)) / math.sqrt(n)
    return {
        "D": D, "p_value": float(p_value), "n": int(n),
        "lambda": float(lam), "d_crit_95": float(d_crit),
        "reject_at_95": bool(p_value < 0.05),
    }


def _gamma_q(a, x):
    """Regularised upper incomplete gamma Q(a,x) (Numerical Recipes).

    Used for the chi-square p-value: p = Q(dof/2, chi2/2).  Implemented with
    math.lgamma only (no scipy).  Series for x < a+1, Lentz continued
    fraction otherwise.
    """
    if x <= 0.0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi_square_gof(samples, cdf, bin_edges, n_fitted_params=0):
    """Chi-square goodness-of-fit test against an analytic CDF (numpy only).

    EQUATIONS: O_k = counts in [e_k, e_{k+1}); p_k = F(e_{k+1})-F(e_k);
    E_k = n*p_k; chi2 = sum_k (O_k-E_k)**2/E_k over bins with E_k>0;
    dof = (#used bins) - 1 - n_fitted_params; p_value = Q(dof/2, chi2/2).

    BOUNDARY_CONDITIONS: bin_edges must span the sample support (cdf(-inf)=0,
    cdf(+inf)=1); probabilities renormalised to sum 1.  UNITS: chi2 and
    dof dimensionless.  LIMITATIONS: requires independent samples and
    E_k >= ~5 for the asymptotic null; correlated series must be thinned.
    """
    s = np.asarray(samples, dtype=np.float64).ravel()
    edges = np.asarray(bin_edges, dtype=np.float64).ravel()
    obs, _ = np.histogram(s, bins=edges)
    p = np.diff(np.clip(np.asarray(cdf(edges), dtype=np.float64), 0.0, 1.0))
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        return {"chi2": float("nan"), "dof": 0, "p_value": float("nan"),
                "reduced_chi2": float("nan"), "n": int(s.size),
                "note": "zero total probability in bins"}
    p /= total
    exp = s.size * p
    keep = exp > 0.0
    chi2 = float(np.sum((obs[keep] - exp[keep]) ** 2 / exp[keep]))
    dof = int(keep.sum()) - 1 - int(n_fitted_params)
    pv = _gamma_q(dof / 2.0, chi2 / 2.0) if dof > 0 else float("nan")
    return {
        "chi2": chi2, "dof": dof, "p_value": float(pv),
        "reduced_chi2": chi2 / dof if dof > 0 else float("nan"),
        "n": int(s.size), "n_bins_used": int(keep.sum()),
        "min_expected": float(np.min(exp[keep])) if np.any(keep) else 0.0,
        "reject_at_95": bool(pv < 0.05) if dof > 0 else True,
    }


def weak_convergence_test(obs_fine, obs_coarse, alpha=0.05):
    """WEAK convergence check for stochastic dynamics.

    DOCUMENTATION: weak convergence compares ENSEMBLE MEANS / OBSERVABLE
    DISTRIBUTIONS at two step sizes, NOT pointwise trajectories.  Here we
    compare two equilibrium ensembles of a scalar observable with a
    two-sample KS test and the CI of the mean difference (blocking errors).
    Different random seeds are allowed and expected.

    WARNING: pointwise trajectory differences between different random seeds
    must NOT be used to claim deterministic convergence order; they mix
    statistical noise with discretisation error.

    STATUS: PASS if the KS test does not reject at `alpha` AND the CI of the
    mean difference contains 0; FAIL otherwise.  UNITS: observable units.
    """
    a = np.asarray(obs_fine, dtype=np.float64).ravel()
    b = np.asarray(obs_coarse, dtype=np.float64).ravel()
    ks = ks_test_two_samples(a, b)
    diff = float(a.mean() - b.mean())
    se = math.sqrt(blocking_error(a, 20) ** 2 + blocking_error(b, 20) ** 2)
    lo, hi = confidence_interval(diff, se, 1.0 - alpha)
    ok = (ks["p_value"] >= alpha) and (lo <= 0.0 <= hi)
    return {
        "status": "PASS" if ok else "FAIL",
        "mean_fine": float(a.mean()), "mean_coarse": float(b.mean()),
        "mean_diff": diff, "mean_diff_ci": [float(lo), float(hi)],
        "ks": ks,
        "notes": (
            "Weak convergence: ensemble distributions/means compared; "
            "pointwise trajectory differences across different seeds must "
            "NOT be used to claim deterministic convergence order."
        ),
    }


def strong_convergence_test(coarse, fine, noise_id_coarse=None,
                            noise_id_fine=None):
    """STRONG convergence check: requires common random numbers (CRN).

    DOCUMENTATION: strong convergence compares POINTWISE trajectories at the
    SAME physical times for dt and dt/2 driven by the SAME noise path.
    `coarse` and `fine` are arrays of shape (n_times, 3) sampled on a common
    time grid (fine subsampled).  noise_id_* are identifiers of the driving
    noise path; they MUST be equal and non-None.

    STATUS: BLOCKED if the CRN guard fails (different / missing noise ids);
    the pointwise error is then NOT reported and must NOT be interpreted as
    a convergence order.  Otherwise PASS with max/rms pointwise errors.

    WARNING: pointwise trajectory differences between different random
    seeds must NOT be used to claim deterministic convergence order.
    """
    if (noise_id_coarse is None or noise_id_fine is None or
            noise_id_coarse != noise_id_fine):
        return {
            "status": "BLOCKED", "max_pointwise_error": None,
            "rms_pointwise_error": None,
            "notes": (
                "Common random numbers (same noise path) REQUIRED for a "
                "strong-convergence test; noise ids differ or are missing. "
                "Pointwise trajectory differences between different random "
                "seeds must NOT be used to claim deterministic convergence "
                "order. Status BLOCKED."
            ),
        }
    c = np.asarray(coarse, dtype=np.float64)
    f = np.asarray(fine, dtype=np.float64)
    if c.shape != f.shape:
        return {"status": "NOT_RESOLVED",
                "max_pointwise_error": None, "rms_pointwise_error": None,
                "notes": "coarse and fine must share the same time grid/shape."}
    d = np.linalg.norm(c - f, axis=-1)
    return {
        "status": "PASS", "max_pointwise_error": float(np.max(d)),
        "rms_pointwise_error": float(np.sqrt(np.mean(d ** 2))),
        "notes": "Common random numbers verified; pointwise errors reported.",
    }


# Section G: trusted analytic LLG references (float64)

def llg_analytic_uniform_precession(t, S0, B, gamma=GAMMA_MUB_PER_EV_FS):
    """Exact uniform precession of one spin about a constant field (Rodrigues).

    PHYSICAL_MODEL: conservative LLG, alpha = 0, T = 0,
        dS/dt = -gamma * S x B = gamma*|B| * (bhat x S).
    EQUATION (Rodrigues rotation about bhat by angle theta = gamma*|B|*t):
        S(t) = S0*cos(theta) + (bhat x S0)*sin(theta)
               + bhat*(bhat.S0)*(1-cos(theta)).
    The flow is an orthogonal rotation: |S| and S.bhat are exactly conserved.

    UNITS: t in fs, B in eV/mu_B, gamma in mu_B/(eV*fs) so gamma*|B| is
    rad/fs; S0 in mu_B; output in mu_B.  BOUNDARY_CONDITIONS: B = 0 returns
    S0 unchanged.  LIMITATIONS: constant field, single spin, no damping or
    noise.  This is a TRUSTED reference for dt-convergence tests.
    """
    S0 = np.asarray(S0, dtype=np.float64)
    Bv = np.asarray(B, dtype=np.float64)
    Bn = float(np.linalg.norm(Bv))
    t_arr = np.asarray(t, dtype=np.float64)
    scalar_t = (t_arr.ndim == 0)
    t_arr = np.atleast_1d(t_arr)
    if Bn == 0.0:
        res = np.tile(S0, (t_arr.size, 1))
        return res[0] if scalar_t else res
    bhat = Bv / Bn
    theta = float(gamma) * Bn * t_arr
    cth = np.cos(theta)
    sth = np.sin(theta)
    v_cross = np.cross(bhat, S0)
    v_par = float(np.dot(S0, bhat)) * bhat
    res = (cth[:, None] * S0[None, :] + sth[:, None] * v_cross[None, :]
           + (1.0 - cth)[:, None] * v_par[None, :])
    return res[0] if scalar_t else res


def llg_analytic_damped(t, S0, B, alpha, gamma=GAMMA_MUB_PER_EV_FS,
                        form="LL"):
    """Exact analytic Gilbert relaxation of tan(theta/2) for one spin.

    PHYSICAL_MODEL / EQUATIONS: field along bhat = B/|B|; theta is the angle
    between S and bhat, phi the azimuth.  For the Landau-Lifshitz (LL)
    explicit form used in this module
        ds/dt = -gamma*[s x B + alpha*s x (s x B)]:
        dtheta/dt = -lambda*sin(theta),  lambda = alpha*gamma*|B|,
        dphi/dt   = omega,               omega  = gamma*|B|,
        tan(theta(t)/2) = tan(theta0/2)*exp(-lambda*t).
    With form="Gilbert" the implicit-Gilbert prefactor 1/(1+alpha**2) is
    applied to both lambda and omega.  The solution is reconstructed as
        S(t) = |S0|*(cos(theta)*bhat
               + sin(theta)*(cos(phi)*e1 + sin(phi)*e2)),
    with e1 from the initial transverse component and e2 = bhat x e1.
    cos(theta)=(1-u**2)/(1+u**2), sin(theta)=2u/(1+u**2) with
    u = tan(theta/2), so |S| is preserved to roundoff.

    CONSISTENCY: with alpha=0 this reduces identically to
    llg_analytic_uniform_precession.  The LL form (default) is the one
    consistent with the SLLG FDT amplitude
    sqrt(2*alpha*kB*T/(gamma*(1+alpha**2)*|S|*dt)) used in this stage.

    UNITS: t fs, B eV/mu_B, gamma mu_B/(eV*fs), S0 mu_B; output mu_B.
    BOUNDARY_CONDITIONS: B=0 -> S0 unchanged; theta0=pi handled with a
    guarded half-angle; float64 throughout.  LIMITATIONS: constant field,
    no exchange/noise; trusted reference for dt-convergence tests only.
    """
    S0 = np.asarray(S0, dtype=np.float64)
    Bv = np.asarray(B, dtype=np.float64)
    Bn = float(np.linalg.norm(Bv))
    t_arr = np.asarray(t, dtype=np.float64)
    scalar_t = (t_arr.ndim == 0)
    t_arr = np.atleast_1d(t_arr)
    S_len = float(np.linalg.norm(S0))
    if Bn == 0.0 or S_len == 0.0:
        res = np.tile(S0, (t_arr.size, 1))
        return res[0] if scalar_t else res
    bhat = Bv / Bn
    s = S0 / S_len
    c0 = float(np.clip(np.dot(s, bhat), -1.0, 1.0))
    s0 = math.sqrt(max(0.0, 1.0 - c0 * c0))
    if s0 > 1e-12:
        e1 = (s - c0 * bhat) / s0
    else:
        e1, _, _ = _orthonormal_basis(bhat)
    e2 = np.cross(bhat, e1)
    u0 = s0 / max(1.0 + c0, 1e-300)          # tan(theta0/2), guarded
    if form == "Gilbert":
        pref = 1.0 / (1.0 + float(alpha) ** 2)
    elif form == "LL":
        pref = 1.0
    else:
        raise ValueError("form must be 'LL' or 'Gilbert'.")
    lam = float(alpha) * float(gamma) * Bn * pref
    om = float(gamma) * Bn * pref
    u = u0 * np.exp(-lam * t_arr)
    ct = (1.0 - u ** 2) / (1.0 + u ** 2)
    st = 2.0 * u / (1.0 + u ** 2)
    phi = om * t_arr
    res = S_len * (ct[:, None] * bhat[None, :]
                   + st[:, None] * (np.cos(phi)[:, None] * e1[None, :]
                                   + np.sin(phi)[:, None] * e2[None, :]))
    return res[0] if scalar_t else res
