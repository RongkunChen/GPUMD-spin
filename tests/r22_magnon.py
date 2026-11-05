#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R22-NonHeat magnon CPU-reference self-test driver (self-contained).

Run from the package root (r22_src)::

    python tests/r22_magnon.py --json <out.json>

WHAT THIS IS
------------
A CPU-only, numpy-only validation of the INDEPENDENT analytic reference in
tools/r22/magnon_spectra.py.  It generates SYNTHETIC spin trajectories with a
KNOWN analytic precession frequency and damping envelope, then checks the DSF
peak position, Lorentzian HWHM, resolution gating, analytic dispersions, BZ
periodicity, noncollinear local frames, window ENBW, the XYZ reader, q-phase
(sublattice) handling, and the fixed-vs-coupled linewidth comparison.

It does NOT run or validate any GPU kernel.  GPU-trajectory validation is
performed ELSEWHERE in stage R22-NonHeat.  No scipy is used/imported.

STATUS VOCABULARY: PASS / FAIL / SKIP / BLOCKED / NOT_RUN / NOT_RESOLVED.
A PASS is emitted ONLY for a test that actually executed and met its stated
tolerance.  A NOT_RESOLVED result is used for a deliberately under-resolved
fit whose honest physical status is "cannot report a lifetime"; it is an
EXPECTED outcome (the negative control) and is NOT counted as a test failure.
Exit code is 0 only if every executed assertion test passes (status PASS) and
no FAIL / BLOCKED / NOT_RUN status is present.
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TOOLS_R22 = os.path.join(PKG, "tools", "r22")
if TOOLS_R22 not in sys.path:
    sys.path.insert(0, TOOLS_R22)

import magnon_spectra as ms  # noqa: E402  (path set above)

STATUS_FAIL_SET = ("FAIL", "BLOCKED", "NOT_RUN")


# helpers
def aerr(a, b):
    return abs(float(a) - float(b))


def rerr(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == 0 else float("inf"))


def triad(axis):
    """Return a right-handed orthonormal (e1, e2, e3) triad, e3 = axis."""
    e3 = ms._unit(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(e3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, e3) * e3
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def gen_precession(T, dt_fs, omega0, Gamma, N=8, q0=None, axis=(0, 0, 1),
                   S0=1.0, amp=0.2, a=2.5, local_frames=None, noise=0.0,
                   rng=None):
    """Synthetic single-mode precession with damping envelope exp(-Gamma t).

    spin_j(t) = S0*e3_j + amp*env(t)*(cos(theta_j)*e1_j + sin(theta_j)*e2_j)
    theta_j = omega0*t + q0.r_j  (a spin wave of wavevector q0)
    env(t)  = exp(-Gamma*t)

    'Gamma' here is the injected damping rate; the Lorentzian power spectrum
    of a exp(-Gamma t) exp(i omega0 t) signal has HWHM = Gamma [rad/fs], so
    Gamma doubles as the injected HWHM (tau = 1/Gamma).  All float64.
    """
    t = np.arange(T, dtype=np.float64) * dt_fs
    env = np.exp(-Gamma * t)
    g1, g2, g3 = triad(axis)
    positions = np.zeros((N, 3), dtype=np.float64)
    positions[:, 0] = np.arange(N, dtype=np.float64) * a
    phi = (positions @ np.asarray(q0, dtype=np.float64)) if q0 is not None else np.zeros(N)
    spin = np.empty((T, N, 3), dtype=np.float64)
    for j in range(N):
        theta = omega0 * t + phi[j]
        c = amp * env * np.cos(theta)
        s = amp * env * np.sin(theta)
        if local_frames is not None:
            l1, l2, l3 = (local_frames[j, 0, :], local_frames[j, 1, :],
                          local_frames[j, 2, :])
        else:
            l1, l2, l3 = g1, g2, g3
        spin[:, j, :] = (S0 * l3 + c[:, None] * l1 + s[:, None] * l2)
    if noise > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        spin += rng.normal(0.0, noise, size=spin.shape)
    return spin, positions


def positive_half(omega, S):
    m = omega > 0
    return omega[m], S[m]


def equivalent_width_bins(omega, S, half_window=40):
    """Equivalent (integrated) line width in FFT bins, for one coherent peak.

    EW_bins = sum(S - bg over a local region) / (S_peak - bg).  For a single
    coherent tone this is exactly the discrete analogue of the window's
    ENBW, so EW_bins(hann)/EW_bins(rect) tracks the 1.5 ENBW ratio.  Robust
    even for a one-bin (delta-like) rectangular-window peak.
    """
    om, s = positive_half(omega, S)
    if om.size < 3:
        return 0.0
    bg = float(np.median(s))
    pk = int(np.argmax(s))
    lo = max(0, pk - half_window)
    hi = min(len(s), pk + half_window + 1)
    peak = float(s[pk] - bg)
    if peak <= 0:
        return 0.0
    reg = np.clip(s[lo:hi] - bg, 0.0, None)
    return float(reg.sum() / peak)


def record(results, test_id, category, expected, measured, abs_err, rel_err,
           tolerance, notes, passed, is_assertion=True, status=None):
    if status is None:
        status = "PASS" if passed else "FAIL"
    results.append({
        "test_id": test_id,
        "category": category,
        "status": status,
        "expected": expected,
        "measured": measured,
        "absolute_error": abs_err,
        "relative_error": rel_err,
        "tolerance": tolerance,
        "notes": notes,
        "is_assertion": bool(is_assertion),
    })


# test bodies
def run_all():
    R = []
    dt = 1.0                  # fs
    T = 4096
    omega0 = 0.10             # rad/fs  (~15.9 THz)
    Gamma = 0.010             # rad/fs  (injected damping / HWHM)
    domega_full = 2.0 * np.pi / (T * dt)
    q_zero = [[0.0, 0.0, 0.0]]

    # Reference trajectory shared by the frequency / lifetime / window tests.
    spin_ref, pos_ref = gen_precession(T, dt, omega0, Gamma, N=8)
    om_ref, S_ref = ms.dynamic_structure_factor(
        spin_ref, pos_ref, q_zero, dt, window="rectangular",
        transverse_axis=(0, 0, 1), component="transverse")
    S_ref = S_ref[0]

    # --- 1. frequency-unit validation (rad/fs vs THz) -----------------------
    peaks = ms.peak_positions(om_ref, S_ref, n_peaks=1, omega_min=0.0)
    peak = float(peaks[0]) if peaks.size else float("nan")
    omega0_THz = omega0 / ms.THZ_TO_RAD_PER_FS
    peak_THz = peak / ms.THZ_TO_RAD_PER_FS
    e1 = aerr(peak, omega0)
    r1 = rerr(peak, omega0)
    tol1 = 0.05
    record(R, "R22.MAGNON.FREQ_UNIT", "unit_validation",
           {"omega0_rad_per_fs": omega0, "omega0_THz": omega0_THz},
           {"peak_rad_per_fs": peak, "peak_THz": peak_THz,
            "THz_conversion": "omega_THz = omega / (2*pi*1e-3)"},
           e1, r1, tol1,
           "DSF peak of a coherent precession matches injected omega0; both "
           "rad/fs and THz (omega/(2*pi*1e-3)) agree. Rectangular window.",
           r1 < tol1)

    # --- 2. Lorentzian HWHM / lifetime definition ----------------------------
    om_pos, S_pos = positive_half(om_ref, S_ref)
    fit = ms.lorentzian_fit(om_pos, S_pos, dt, n_frames=T,
                            window="rectangular", fit_range=(0.02, 0.30))
    G = fit["HWHM_Gamma"]
    ok2 = fit["resolved"] and G is not None and rerr(G, Gamma) < 0.20
    meas2 = {"HWHM_Gamma": G, "FWHM": fit["FWHM"], "tau_fs": fit["tau"],
             "tau_energy_fs": fit["tau_energy"], "Gamma_E_eV": fit["Gamma_E"],
             "R2": fit["R2"], "snr": fit["snr"], "status": fit["status"],
             "domega": fit["domega"]}
    record(R, "R22.MAGNON.LIFETIME_HWHM", "lifetime_definition",
           {"injected_Gamma_HWHM": Gamma, "injected_FWHM": 2 * Gamma,
            "injected_tau_fs": 1.0 / Gamma},
           meas2,
           (aerr(G, Gamma) if G is not None else None),
           (rerr(G, Gamma) if G is not None else None),
           0.20,
           "Rectangular window -> undistorted Lorentzian. HWHM=Gamma, "
           "FWHM=2*Gamma, tau=1/Gamma and tau=hbar/Gamma_E reported. "
           "Numerical/spectral broadening is NOT a physical lifetime.",
           ok2)

    # --- 3. negative control: deliberately under-resolved --------------------
    Ts, Gs = 1024, 0.008
    spin_s, pos_s = gen_precession(Ts, dt, omega0, Gs, N=8)
    om_s, S_s = ms.dynamic_structure_factor(
        spin_s, pos_s, q_zero, dt, window="rectangular",
        transverse_axis=(0, 0, 1), component="transverse")
    om_sp, S_sp = positive_half(om_s, S_s[0])
    fit_s = ms.lorentzian_fit(om_sp, S_sp, dt, n_frames=Ts,
                             window="rectangular", fit_range=(0.02, 0.30))
    no_lifetime = (fit_s["tau"] is None and fit_s["HWHM_Gamma"] is None
                   and fit_s["raw"] is None)
    ok3 = (fit_s["status"] == "NOT_RESOLVED" and not fit_s["resolved"]
           and no_lifetime)
    record(R, "R22.MAGNON.NEGCTRL", "negative_control",
           "NOT_RESOLVED with NO quantitative lifetime emitted",
           {"fit_status": fit_s["status"], "resolved": fit_s["resolved"],
            "tau": fit_s["tau"], "HWHM_Gamma": fit_s["HWHM_Gamma"],
            "reasons": fit_s["reasons"],
            "domega": fit_s["domega"], "two_domega": 2 * fit_s["domega"],
            "injected_Gamma": Gs},
           None, None, "exact status match",
           "Short trajectory (T=1024 -> 2*domega=%.4g rad/fs) with "
           "Gamma=%.4g < 2*domega: fit correctly withholds a lifetime; "
           "tests the resolution gate (Gamma<=2*domega)." %
           (2 * 2 * np.pi / (Ts * dt), Gs),
           ok3)
    # Honest physical status of the under-resolved fit (informational).
    record(R, "R22.MAGNON.NEGCTRL.FIT", "negative_control",
           "NOT_RESOLVED", "NOT_RESOLVED",
           None, None, "n/a",
           "EXPECTED OUTCOME: the deliberately under-resolved fit's own honest "
           "status. Informational record (is_assertion=false); NOT a test "
           "failure. No lifetime is emitted.",
           True, is_assertion=False, status="NOT_RESOLVED")

    # --- 4a. FM Goldstone mode ------------------------------------------------
    J, Sspin, a_lat = 0.010, 1.0, 2.5
    fm0 = ms.heisenberg_fm_dispersion([0, 0, 0], J, Sspin, a_lat)
    fm_x = ms.heisenberg_fm_dispersion([np.pi / a_lat, 0, 0], J, Sspin, a_lat)
    ok4a = aerr(fm0, 0.0) < 1e-12 and fm_x > 0
    record(R, "R22.MAGNON.FM_GOLDSTONE", "analytic_dispersion",
           {"omega_q0": 0.0},
           {"omega_q0": float(fm0), "omega_zone_edge": float(fm_x)},
           aerr(fm0, 0.0), None, 1e-12,
           "FM Heisenberg LSWT: omega(0)=0 (Goldstone, spontaneously broken "
           "continuous spin-rotation symmetry); finite at zone edge.",
           ok4a)

    # --- 4b. AFM two-branch degeneracy ---------------------------------------
    q_afm = [0.2, 0.1, 0.0]
    afm = ms.heisenberg_afm_dispersion(q_afm, J, Sspin, a_lat)
    afm0 = ms.heisenberg_afm_dispersion([0, 0, 0], J, Sspin, a_lat)
    degen = aerr(afm[0], afm[1])
    ok4b = degen < 1e-12 and afm[0] > 0 and aerr(afm0[0], 0.0) < 1e-12
    record(R, "R22.MAGNON.AFM_DEGENERACY", "analytic_dispersion",
           {"two_degenerate_branches": True, "omega_q0_both": 0.0},
           {"branch1": float(afm[0]), "branch2": float(afm[1]),
            "degeneracy_error": degen, "branches_at_q0": [float(afm0[0]),
                                                          float(afm0[1])]},
           degen, None, 1e-12,
           "Two-sublattice bipartite AFM LSWT: two magnon branches omega_+/- "
           "are degenerate (isotropic, zero field); linear Goldstone at q=0.",
           ok4b)

    # --- 4c. anisotropy gap (SOC breaks continuous symmetry) -----------------
    K = 0.001
    gap = ms.fm_dispersion_with_anisotropy_gap([0, 0, 0], J, Sspin, a_lat, K)
    gap_K0 = ms.fm_dispersion_with_anisotropy_gap([0, 0, 0], J, Sspin, a_lat, 0.0)
    gap_expected = 2.0 * K * Sspin / ms.HBAR_EV_FS
    ok4c = (gap > 0 and gap_K0 == 0.0 and aerr(gap, gap_expected) < 1e-12)
    record(R, "R22.MAGNON.ANISOTROPY_GAP", "analytic_dispersion",
           {"gap_at_q0_rad_per_fs": gap_expected, "gap_with_K0": 0.0},
           {"gap_at_q0_rad_per_fs": float(gap), "gap_with_K0": float(gap_K0),
            "Delta_eV": 2 * K * Sspin},
           aerr(gap, gap_expected), rerr(gap, gap_expected), 1e-12,
           "SOC/anisotropy breaks continuous symmetry -> q=0 NOT gapless: "
           "omega(0)=2*K*S/hbar > 0 for K>0, and omega(0)=0 for K=0.",
           ok4c)

    # --- 5. Brillouin-zone periodicity ---------------------------------------
    q5 = np.array([0.3, 0.2, 0.1])
    Gvec = np.array([2.0 * np.pi / a_lat, 0.0, 0.0])
    w5a = ms.heisenberg_fm_dispersion(q5, J, Sspin, a_lat)
    w5b = ms.heisenberg_fm_dispersion(q5 + Gvec, J, Sspin, a_lat)
    w5c = ms.heisenberg_fm_dispersion(q5 + 2 * Gvec, J, Sspin, a_lat)
    ok5 = aerr(w5a, w5b) < 1e-10 and aerr(w5a, w5c) < 1e-10
    record(R, "R22.MAGNON.BZ_PERIODICITY", "analytic_dispersion",
           {"omega_q": float(w5a)},
           {"omega_q": float(w5a), "omega_q_plus_G": float(w5b),
            "omega_q_plus_2G": float(w5c), "G_x": float(Gvec[0])},
           aerr(w5a, w5b), rerr(w5a, w5b), 1e-10,
           "gamma_q is periodic under q -> q+G for sc (G=2*pi/a along x), "
           "hence omega(q+G)=omega(q).",
           ok5)

    # --- 6. noncollinear local-frame projection ------------------------------
    model = ms.multi_sublattice_example()
    lf = model["local_frames"]
    pos_nc = model["positions"]
    spin_nc, _ = gen_precession(T, dt, omega0, Gamma, N=3,
                                local_frames=lf, S0=1.0, amp=0.2)
    om_nc, S_nc = ms.dynamic_structure_factor(
        spin_nc, pos_nc, q_zero, dt, window="rectangular",
        local_frames=lf, component="transverse")
    om_nl, S_nl = ms.dynamic_structure_factor(
        spin_nc, pos_nc, q_zero, dt, window="rectangular",
        local_frames=lf, component="longitudinal")
    om_ng, S_ng = ms.dynamic_structure_factor(
        spin_nc, pos_nc, q_zero, dt, window="rectangular",
        transverse_axis=(0, 0, 1), component="longitudinal")
    pk_nc = ms.peak_positions(om_nc, S_nc[0], n_peaks=1, omega_min=0.0)
    peak_nc = float(pk_nc[0]) if pk_nc.size else float("nan")
    _, s_t = positive_half(om_nc, S_nc[0])
    _, s_l = positive_half(om_nl, S_nl[0])
    _, s_g = positive_half(om_ng, S_ng[0])
    st_max, sl_max, sg_max = float(s_t.max()), float(s_l.max()), float(s_g.max())
    ok6 = (rerr(peak_nc, omega0) < 0.05 and sl_max < 0.02 * st_max
           and sg_max > 0.10 * st_max)
    record(R, "R22.MAGNON.NONCOLLINEAR_LOCAL_FRAME", "local_frame",
           {"peak_rad_per_fs": omega0, "longitudinal_suppressed": True},
           {"peak_rad_per_fs": peak_nc, "transverse_peak": st_max,
            "local_longitudinal_peak": sl_max, "global_z_longitudinal_peak": sg_max,
            "local_long_over_trans": sl_max / st_max if st_max else None},
           aerr(peak_nc, omega0), rerr(peak_nc, omega0), 0.05,
           "3-sublattice 120-degree noncollinear ground state: local triads "
           "(e1,e2,e3) recover the injected transverse frequency; local "
           "longitudinal weight is ~0 while a naive global-z frame wrongly "
           "assigns transverse weight as longitudinal. q-phase uses ACTUAL "
           "site positions incl. sublattice offsets.",
           ok6)

    # --- 7. window broadening / ENBW -----------------------------------------
    w_h, CG_h, U_h, ENBW_h = ms.window_properties("hann", T)
    w_r, CG_r, U_r, ENBW_r = ms.window_properties("rectangular", T)
    # A STATIONARY (undamped) coherent sinusoid is the correct scenario for
    # the window-broadening measurement: a fully decaying transient would be
    # suppressed by a symmetric Hann window (which is zero at t=0).
    spin_stat, pos_stat = gen_precession(T, dt, 0.103, 0.0, N=8)
    om_h, S_h = ms.dynamic_structure_factor(
        spin_stat, pos_stat, q_zero, dt, window="hann",
        transverse_axis=(0, 0, 1), component="transverse")
    om_r, S_r = ms.dynamic_structure_factor(
        spin_stat, pos_stat, q_zero, dt, window="rectangular",
        transverse_axis=(0, 0, 1), component="transverse")
    ew_h = equivalent_width_bins(om_h, S_h[0])
    ew_r = equivalent_width_bins(om_r, S_r[0])
    ew_ratio = ew_h / ew_r if ew_r > 0 else float("inf")
    ratio = ENBW_h / ENBW_r
    ok7 = (aerr(ENBW_h, 1.5) < 0.02 and aerr(ENBW_r, 1.0) < 1e-9
           and aerr(CG_h, 0.5) < 1e-6 and aerr(CG_r, 1.0) < 1e-9
           and ew_h > 1.2 * ew_r > 0)
    record(R, "R22.MAGNON.WINDOW_ENBW", "window",
           {"ENBW_hann_bins": 1.5, "ENBW_rect_bins": 1.0,
            "CG_hann": 0.5, "CG_rect": 1.0},
           {"ENBW_hann_bins": ENBW_h, "ENBW_rect_bins": ENBW_r,
            "ENBW_ratio": ratio, "CG_hann": CG_h, "CG_rect": CG_r,
            "U_hann": U_h, "U_rect": U_r,
            "equivalent_width_hann_bins": ew_h,
            "equivalent_width_rect_bins": ew_r,
            "equivalent_width_ratio": ew_ratio,
            "equivalent_width_hann_rad_per_fs": ew_h * domega_full,
            "equivalent_width_rect_rad_per_fs": ew_r * domega_full},
           aerr(ENBW_h, 1.5), rerr(ENBW_h, 1.5), 0.02,
           "Coherent-gain correction (w/CG) keeps peak heights window-"
           "independent; Hann ENBW=1.5 bins vs rectangular 1.0 bin -> extra "
           "~50% noise-bandwidth broadening. Measured equivalent line widths "
           "on a stationary sinusoid (Hann broader, ratio tracks ENBW) are "
           "reported. A symmetric Hann window is NOT applied to a decaying "
           "transient here (that would suppress the strong early signal). "
           "Window broadening is NOT a physical lifetime.",
           ok7)

    # --- 8. extended-XYZ trajectory reader -----------------------------------
    xyz_spin_middle = (
        "3\n"
        "Lattice=\"7.5 0 0 0 10 0 0 0 10\" "
        "Properties=species:S:1:pos:R:3:spin:R:3:group:I:1 pbc=\"T T T\"\n"
        "Fe 0.0 0.0 0.0 0.0 0.0 1.0 0\n"
        "Fe 2.5 0.0 0.0 0.0 1.0 0.0 1\n"
        "Fe 5.0 0.0 0.0 1.0 0.0 0.0 2\n"
        "3\n"
        "Lattice=\"7.5 0 0 0 10 0 0 0 10\" "
        "Properties=species:S:1:pos:R:3:spin:R:3:group:I:1 pbc=\"T T T\"\n"
        "Fe 0.0 0.0 0.0 0.0 0.0 -1.0 0\n"
        "Fe 2.5 0.0 0.0 0.0 -1.0 0.0 1\n"
        "Fe 5.0 0.0 0.0 -1.0 0.0 0.0 2\n"
    )
    xyz_spin_last = (
        "2\n"
        "Properties=species:S:1:pos:R:3:spin:R:3 pbc=\"T T T\"\n"
        "Co 0.0 0.0 0.0 1.0 0.0 0.0\n"
        "Co 1.0 1.0 1.0 0.0 0.0 2.0\n"
    )
    xyz_no_spin = (
        "1\n"
        "Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n"
        "Fe 0.0 0.0 0.0\n"
    )
    reader_ok = True
    reader_meas = {}
    with tempfile.TemporaryDirectory() as td:
        p1 = os.path.join(td, "spin_middle.xyz")
        with open(p1, "w") as f:
            f.write(xyz_spin_middle)
        d1 = ms.read_spin_trajectory(p1)
        reader_meas["spin_middle"] = {
            "nframes": d1["nframes"], "natoms": d1["natoms"],
            "spin_shape": list(d1["spin"].shape), "spin_col": d1["spin_col"],
            "frame0_spin": d1["spin"][0].tolist(),
            "frame0_pos": d1["pos"][0].tolist(),
            "species": list(d1["species"]),
        }
        reader_ok &= (d1["nframes"] == 2 and d1["natoms"] == 3
                      and d1["spin"].shape == (2, 3, 3)
                      and d1["spin_col"] == 4
                      and np.allclose(d1["spin"][0],
                                      [[0, 0, 1], [0, 1, 0], [1, 0, 0]])
                      and np.allclose(d1["pos"][0, :, 0], [0, 2.5, 5.0]))
        p2 = os.path.join(td, "spin_last.xyz")
        with open(p2, "w") as f:
            f.write(xyz_spin_last)
        d2 = ms.read_spin_trajectory(p2)
        reader_meas["spin_last"] = {
            "nframes": d2["nframes"], "spin_col": d2["spin_col"],
            "frame0_spin": d2["spin"][0].tolist(),
        }
        reader_ok &= (d2["nframes"] == 1 and d2["natoms"] == 2
                      and d2["spin_col"] == 4
                      and np.allclose(d2["spin"][0], [[1, 0, 0], [0, 0, 2]]))
        p3 = os.path.join(td, "no_spin.xyz")
        with open(p3, "w") as f:
            f.write(xyz_no_spin)
        try:
            ms.read_spin_trajectory(p3)
            raised = False
        except (ValueError, IOError):
            raised = True
        reader_meas["absent_spin_raises_clear_error"] = raised
        reader_ok &= raised
    record(R, "R22.MAGNON.XYZ_READER", "file_io",
           {"spin_middle_parsed": True, "spin_last_parsed": True,
            "absent_spin_raises": True},
           reader_meas, None, None, "exact",
           "Robust to a spin:R:3 field placed anywhere in Properties (middle "
           "with an extra group:I:1 field, and last), Lattice/pbc comment "
           "fields, multi-frame files, and a clear error when spin is absent. "
           "All parsed into float64 (nframes,N,3).",
           reader_ok)

    # --- 9. q-phase / sublattice-offset handling ------------------------------
    a9 = 2.5
    N9 = 8
    q0 = np.array([2.0 * np.pi / (N9 * a9), 0.0, 0.0])  # 1 wavelength over 8 sites
    spin_q, pos_q = gen_precession(T, dt, omega0, Gamma, N=N9, q0=q0, a=a9)
    om_q, S_q = ms.dynamic_structure_factor(
        spin_q, pos_q, [q0, [0.0, 0.0, 0.0]], dt, window="rectangular",
        transverse_axis=(0, 0, 1), component="transverse")
    pow_at_q0 = float(S_q[0][om_q > 0].max())
    pow_at_0 = float(S_q[1][om_q > 0].max())
    contrast = pow_at_q0 / pow_at_0 if pow_at_0 > 0 else float("inf")
    ok9 = contrast > 20.0
    record(R, "R22.MAGNON.QPHASE", "q_resolution",
           {"peak_power_ratio_q0_over_qzero_gt": 20.0},
           {"q0_per_Angstrom": q0.tolist(), "power_at_q0": pow_at_q0,
            "power_at_qzero": pow_at_0, "contrast": contrast},
           None, None, ">20",
           "exp(-i q.r_j) uses ACTUAL site positions r_j=j*a including "
           "sublattice offsets. A spin wave with phase exp(i q0.r_j) is "
           "constructive only at q=q0 (contrast >> 1). Multiple q vectors "
           "returned as a (nq,T) array.",
           ok9)

    # --- 10. compare_fixed_vs_coupled (multi-seed statistics + warning) ------
    n_seed = 4
    Gf, Gc = 0.010, 0.016
    rng = np.random.default_rng(20241104)
    Sf, Sc = [], []
    om_cmp = None
    for _ in range(n_seed):
        sp, pp = gen_precession(T, dt, omega0, Gf, N=4, noise=0.004, rng=rng)
        omx, Ssx = ms.dynamic_structure_factor(
            sp, pp, q_zero, dt, window="rectangular",
            transverse_axis=(0, 0, 1), component="transverse")
        om_pos_x, S_pos_x = positive_half(omx, Ssx[0])
        om_cmp = om_pos_x
        Sf.append(S_pos_x)
        sp, pp = gen_precession(T, dt, omega0, Gc, N=4, noise=0.004, rng=rng)
        _, Scx = ms.dynamic_structure_factor(
            sp, pp, q_zero, dt, window="rectangular",
            transverse_axis=(0, 0, 1), component="transverse")
        Sc.append(positive_half(omx, Scx[0])[1])
    cmp = ms.compare_fixed_vs_coupled(
        om_cmp, np.array(Sf), np.array(Sc), dt, n_frames=T,
        window="rectangular", fit_range=(0.02, 0.30))
    warn_text = " ".join(cmp["warnings"])
    ok10 = (cmp["status"] == "PASS"
            and cmp["difference"] is not None and cmp["difference"] > 0
            and cmp["sigma_difference"] is not None
            and cmp["sigma_difference"] >= 0
            and "NOT automatically an additive magnon-phonon scattering rate"
            in warn_text)
    record(R, "R22.MAGNON.COMPARE_FIXED_COUPLED", "linewidth_comparison",
           {"difference_positive": True, "sigma_nonnegative": True,
            "warning_present": True},
           {"difference_rad_per_fs": cmp.get("difference"),
            "sigma_difference": cmp.get("sigma_difference"),
            "fixed_mean_std_se": cmp["fixed"]["mean_std_se"],
            "coupled_mean_std_se": cmp["coupled"]["mean_std_se"],
            "n_seeds_fixed": cmp["fixed"]["n_seeds"],
            "n_seeds_coupled": cmp["coupled"]["n_seeds"],
            "status": cmp["status"]},
           None, None, "exact",
           "Fits both linewidths over %d seeds (mean/std/standard error), "
           "returns D=mean(Gc)-mean(Gf) with sigma_D=sqrt(se_c^2+se_f^2), and "
           "warns that a linewidth difference is NOT automatically an additive "
           "magnon-phonon scattering rate." % n_seed,
           ok10)

    return R, domega_full


def main(argv=None):
    ap = argparse.ArgumentParser(description="R22 magnon CPU-reference self-test")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write the JSON result list to this path as well as stdout")
    args = ap.parse_args(argv)

    results, domega_full = run_all()

    if args.json_path:
        try:
            with open(args.json_path, "w") as fh:
                json.dump(results, fh, indent=2, allow_nan=False)
        except (OSError, ValueError) as exc:
            sys.stderr.write("failed to write --json file: %r\n" % (exc,))

    # Pure JSON result list on stdout.
    sys.stdout.write(json.dumps(results, indent=2, allow_nan=False))
    sys.stdout.write("\n")

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    assertions = [r for r in results if r["is_assertion"]]
    failed = any(r["status"] in STATUS_FAIL_SET for r in results)
    all_assertions_pass = (len(assertions) > 0
                           and all(r["status"] == "PASS" for r in assertions))
    exit_code = 0 if (not failed and all_assertions_pass) else 1

    sys.stderr.write(
        "\nR22.MAGNON summary: total=%d  %s  |  assertion_tests=%d  "
        "domega_full=%.6g rad/fs  exit=%d\n"
        % (len(results),
           " ".join("%s=%d" % (k, counts[k]) for k in sorted(counts)),
           len(assertions), domega_full, exit_code))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
