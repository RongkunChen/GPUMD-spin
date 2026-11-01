"""Stage PV1 -- analyse the heat-current experiments.

The three numbers that matter:

1. `cont`  : closure of  d/dt sum_i r_i e_i = J  (both sides from one trajectory)
2. `shift` : J is independent of the origin of the cell
3. `gk`    : correlation time, running Green-Kubo integral, block error

Unit conventions, spelled out once:

* velocities in model.xyz / traj.xyz are A/fs, kinetic energy per atom is
  ``0.5 * 103.6427 * m[u] * v^2`` eV  (1 u A^2/fs^2 = 103.6427 eV);
* positions are A, so  M = sum_i r_i e_i  is in eV A;
* ``heat_flux.out`` writes eV/(A^2 ps) and its header records the volume, so the
  extensive flux is ``J[col] * V`` in eV A/ps;
* dM/dt is therefore evaluated with dt in ps  (dt_fs / 1000).
"""

import json
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "evidence", "stage_PV1", "heat_audit")
CASES = os.path.join(OUT, "cases")
M_FE = 55.845
KE_FACTOR = 103.6427
KB_EV = 8.617333262e-5


def read_traj(path):
    frames = []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        n = int(lines[i])
        head = lines[i + 1]
        lat = np.array([float(x) for x in re.search(r'Lattice="([^"]+)"', head).group(1).split()])
        lat = lat.reshape(3, 3)
        props = re.search(r"Properties=(\S+)", head).group(1)
        toks = props.split(":")
        names = []
        k = 0
        while k < len(toks):
            names.append((toks[k], int(toks[k + 2])))
            k += 3
        cols = {}
        c = 0
        for name, ncol in names:
            cols[name] = (c, c + ncol)
            c += ncol
        rows = [ln.split() for ln in lines[i + 2 : i + 2 + n]]

        def get(name):
            a, b = cols[name]
            return np.array([[float(x) for x in r[a:b]] for r in rows])

        fr = dict(n=n, lat=lat, pos=get("pos"))
        for name in ("unwrapped_position", "vel", "spin", "torque"):
            if name in cols:
                fr[name] = get(name)
        if "energy_atom" in cols:
            fr["pot"] = get("energy_atom")[:, 0]
        frames.append(fr)
        i += 2 + n
    return frames


def read_heat(path):
    rows = []
    vol = None
    ncol = None
    for line in open(path):
        if line.startswith("#"):
            m = re.search(r"volume\s*=\s*([0-9.eE+-]+)", line)
            if m:
                vol = float(m.group(1))
            if "columns" in line:
                ncol = 3 + (1 if "|J|" in line else 0)
            continue
        if not line.strip():
            continue
        rows.append([float(x) for x in line.split()])
    return np.array(rows), vol, ncol


EV_A_PER_PS_TO_W = 1.602176634e-14   # 1 eV A/ps = 1.602e-14 W
KB_J = 1.380649e-23


def _corr(dJ, dt_ps):
    """Autocorrelation of the fluctuations, averaged over the three directions."""
    n = len(dJ)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    C = np.zeros(nfft // 2)
    ncomp = dJ.shape[1] if dJ.ndim > 1 else 1
    for k in range(ncomp):
        f = np.fft.rfft(dJ[:, k] - dJ[:, k].mean(), nfft)
        ac = np.fft.irfft(f * np.conj(f), nfft)[: nfft // 2]
        C += ac
    denom = np.maximum(n - np.arange(len(C)), 1)
    C = C / denom / float(ncomp)
    return C


def _integral_at(C, dt_ps, t_ps):
    """Integral of C from 0 to t (trapezoid over whole samples)."""
    m = max(2, int(round(t_ps / dt_ps)))
    m = min(m, len(C))
    return float(np.trapezoid(C[:m], dx=dt_ps))


def kappa_from_trace(J, dt_ps, vol_A3, T, min_segments=8):
    """Green-Kubo integral of a heat-current trace, in W/(m K).

    Two different error estimates are produced, because they answer different
    questions:

    * ``kappa_err`` -- standard error over ``min_segments`` independent
      sub-trajectories, each at least 10 decay times long.  This is the honest
      statistical error of the reported number.
    * ``kappa_err_block`` -- the 8-block average over the plateau window, which
      is what a naive implementation would report.  It is kept so that the
      audit can show how much it underestimates (it does).

    A single code path is used for the MD data and for the synthetic self-test,
    so a wrong prefactor cannot hide in one of them.
    """
    n = len(J)
    C = _corr(J, dt_ps)
    c0 = C[0]
    below = np.where(C < 0.1 * c0)[0]
    n_decay = int(below[0]) if len(below) else max(2, len(C) // 4)
    tau = float(np.trapezoid(C[: max(2, n_decay)], dx=dt_ps) / c0)
    t_plateau = 5.0 * tau
    kappa_raw = _integral_at(C, dt_ps, t_plateau)
    integ = np.cumsum(C) * dt_ps

    V_m3 = vol_A3 * 1e-30
    pref = 1.0 / (V_m3 * KB_J * T * T)
    to_W = EV_A_PER_PS_TO_W ** 2 * 1e-12

    total_ps = n * dt_ps
    nseg = int(max(1, min(min_segments, total_ps / max(10 * tau, 1e-12))))
    kappas = []
    if nseg >= 4:
        seg = n // nseg
        for i in range(nseg):
            sl = J[i * seg : (i + 1) * seg]
            if len(sl) < 10:
                continue
            Ci = _corr(sl, dt_ps)
            kappas.append(_integral_at(Ci, dt_ps, t_plateau))
        kappas = np.array(kappas)
        err = float(kappas.std(ddof=1) / np.sqrt(len(kappas)))
    else:
        kappas = np.array([])
        err = float("nan")

    # naive block error over the plateau window, for comparison only
    lo = int(min(len(integ) - 2, 5 * tau / dt_ps))
    hi = int(min(len(integ) - 1, 12 * tau / dt_ps))
    blocks = np.array_split(integ[lo:hi], 8)
    bmeans = np.array([b.mean() for b in blocks])
    err_block = float(bmeans.std(ddof=1) / np.sqrt(8))

    return dict(C=C, tau0=tau, lo=lo, hi=hi, integral=integ,
                kappa=kappa_raw * to_W * pref,
                kappa_err=err * to_W * pref,
                kappa_err_block=err_block * to_W * pref,
                plateau_mean=kappa_raw * to_W,
                plateau_err=err * to_W,
                n_segments=int(len(kappas)),
                segment_kappa_rel_spread=float(kappas.std(ddof=1) / abs(kappas.mean()))
                if len(kappas) > 1 and kappas.mean() else float("nan"))


def estimator_selftest(n=40000, dt_ps=0.0005, tau_ps=0.05, vol_A3=1415.4, T=250.0,
                       trials=60):
    """Feed an Ornstein-Uhlenbeck process with a known integral through the estimator.

    For  <J(0)J(t)> = sigma^2 exp(-t/tau)  the analytic result is
    kappa = sigma^2 tau / (V kB T^2).  The test reports the ratio of the
    estimated to the analytic value and how often the block error bar covers
    the analytic value -- an estimator can be wrong in the prefactor, in the
    integration, or in its own error estimate, and these three failures look
    different here.
    """
    rng = np.random.default_rng(20261031)
    phi = np.exp(-dt_ps / tau_ps)
    sigma = 1.0
    to_W = EV_A_PER_PS_TO_W ** 2 * 1e-12
    pref = 1.0 / (vol_A3 * 1e-30 * KB_J * T * T)
    kappa_true = sigma ** 2 * tau_ps * to_W * pref
    ratios, errs, cover = [], [], 0
    for _ in range(trials):
        eps = rng.normal(size=(n, 3))
        J = np.empty((n, 3))
        J[0] = rng.normal(scale=sigma, size=3)
        s = np.sqrt(1.0 - phi ** 2)
        for i in range(1, n):
            J[i] = phi * J[i - 1] + sigma * s * eps[i]
        r = kappa_from_trace(J, dt_ps, vol_A3, T)
        ratios.append(r["kappa"] / kappa_true)
        errs.append(r["kappa_err"] / kappa_true)
        if abs(r["kappa"] - kappa_true) <= r["kappa_err"]:
            cover += 1
    ratios = np.array(ratios)
    errs = np.array(errs)
    return dict(n_samples=n, dt_ps=dt_ps, tau_ps=tau_ps, trials=trials,
                kappa_true_W_per_mK=float(kappa_true),
                ratio_mean=float(ratios.mean()), ratio_std=float(ratios.std()),
                ratio_min=float(ratios.min()), ratio_max=float(ratios.max()),
                reported_rel_error_mean=float(errs.mean()),
                actual_rel_scatter=float(ratios.std()),
                coverage_1sigma=float(cover / trials),
                verdict=("PASS" if (abs(ratios.mean() - 1) < 0.10 and
                                    cover / trials >= 0.5) else "CHECK"),
                error_bar_underestimate=float(ratios.std() / errs.mean()))


def continuity():
    d = os.path.join(CASES, "cont_N128")
    tr = read_traj(os.path.join(d, "traj.xyz"))
    jf, vol, ncol = read_heat(os.path.join(d, "heat_flux.out"))
    dt_fs = 0.05
    dt_ps = dt_fs / 1000.0
    n = tr[0]["n"]

    # energy per atom and the extensive energy moment
    E = []
    conv = []
    for fr in tr:
        e = fr["pot"] + 0.5 * KE_FACTOR * M_FE * np.sum(fr["vel"] ** 2, axis=1)
        E.append(e)
        conv.append(np.sum(e[:, None] * (fr["vel"] * 1000.0), axis=0))  # eV A / ps
    E = np.array(E)
    conv = np.array(conv)
    R = np.stack([fr["unwrapped_position"] for fr in tr])  # (nf, n, 3)
    M = np.einsum("fi,fi a->f a", E, R)  # eV A

    # central difference over whole frames
    J_fd = (M[2:] - M[:-2]) / (2.0 * dt_ps)  # eV A / ps
    J_cond = J_fd - conv[1:-1]               # remove sum_i v_i e_i
    # heat_flux rows correspond to step 1..400 -> frames 0..399
    jf = jf[: len(tr)]
    J_code = jf[:, 1:4] * vol
    # align: J_fd[t] belongs to frame t+1
    idx = np.arange(1, len(tr) - 1)
    Jc = J_code[idx]

    # third variant: the code carries only the *spin* part of the site energy
    # in its convective term (espin = -1/2 S.B); the lattice part e_i^lat v_i is
    # not there.  J_code + sum e_i^lat v_i should then close the identity if the
    # only missing piece is that convective term.
    espin = -0.5 * np.sum(np.stack([f["spin"] for f in tr]) *
                          np.stack([f["torque"] for f in tr]), axis=2)
    e_lat = np.stack([f["pot"] for f in tr]) - espin
    conv_lat = np.einsum("fi,fia->fa", e_lat, np.stack([f["vel"] for f in tr]) * 1000.0)
    J_plus = Jc + conv_lat[idx]

    def score(A, B, label):
        num = np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1)))
        den = np.sqrt(np.mean(np.sum(B ** 2, axis=1)))
        cc = np.corrcoef(A.ravel(), B.ravel())[0, 1]
        per = []
        for k in range(3):
            per.append(float(np.sqrt(np.mean((A[:, k] - B[:, k]) ** 2))
                             / (np.sqrt(np.mean(B[:, k] ** 2)) + 1e-30)))
        return dict(label=label, rel_rms=float(num / max(den, 1e-30)), corr=float(cc),
                    rel_rms_per_component=per,
                    rms_A=float(np.sqrt(np.mean(np.sum(A ** 2, axis=1)))),
                    rms_B=float(np.sqrt(np.mean(np.sum(B ** 2, axis=1)))))

    out = dict(
        case="cont_N128",
        n_atoms=n,
        frames=len(tr),
        dt_fs=dt_fs,
        volume=vol,
        heat_flux_columns=ncol,
        T_K=float(np.mean(2.0 * np.sum(E - np.stack([f["pot"] for f in tr]), axis=1) / (3 * n * KB_EV))),
        moment_direct=score(J_fd, Jc, "d/dt sum r e  vs  J_code"),
        moment_conduction_only=score(J_cond, Jc, "d/dt sum r e - sum v e  vs  J_code"),
        moment_with_lattice_convection=score(J_fd, J_plus,
                                             "d/dt sum r e  vs  J_code + sum e_lat v"),
        convective_lattice_rms=float(np.sqrt(np.mean(np.sum(conv_lat[idx] ** 2, axis=1)))),
        convective_rms=float(np.sqrt(np.mean(np.sum(conv[1:-1] ** 2, axis=1)))),
        j_code_rms=float(np.sqrt(np.mean(np.sum(Jc ** 2, axis=1)))),
    )
    return out


def shift():
    a = os.path.join(CASES, "cont_N128")
    b = os.path.join(CASES, "shift_N128")
    ja, vola, _ = read_heat(os.path.join(a, "heat_flux.out"))
    jb, volb, _ = read_heat(os.path.join(b, "heat_flux.out"))
    m = min(len(ja), len(jb))
    A = ja[:m, 1:4]
    B = jb[:m, 1:4]
    scale = np.sqrt(np.mean(np.sum(A ** 2, axis=1)))
    dev = np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))) / scale
    # how long the two stay pointwise equal
    close = np.abs(A - B).max(axis=1)
    tol = 1e-9 * scale
    n_equal = int(np.argmax(close > tol)) if np.any(close > tol) else m
    return dict(case="shift_N128", compared_steps=int(m), volume_a=vola, volume_b=volb,
                rel_rms_deviation=float(dev), first_step_deviation=float(close[0]),
                steps_before_divergence=n_equal,
                j_rms=float(scale),
                note="the same physical state with every coordinate shifted by (0.317,0.191,0.233) A")


def gk():
    d = os.path.join(CASES, "gk_N1024")
    jf, vol, _ = read_heat(os.path.join(d, "heat_flux.out"))
    t = jf[:, 0]
    J = jf[:, 1:4] * vol
    dt = float(np.median(np.diff(t)))
    r = kappa_from_trace(J, dt, vol, 250.0)
    dJ = J - J.mean(axis=0)
    n = len(J)
    out = dict(
        case="gk_N1024", n_samples=int(n), sample_interval_ps=dt, total_ps=float(t[-1]),
        n_atoms=1024, volume=vol, T_K=250.0,
        J_mean=[float(x) for x in J.mean(axis=0)],
        J_std=[float(x) for x in J.std(axis=0)],
        tau_zero_crossing_ps=r["tau0"],
        independent_samples=float(t[-1] / (2 * r["tau0"])) if r["tau0"] > 0 else 0.0,
        plateau_window_ps=[float(r["lo"] * dt), float(r["hi"] * dt)],
        plateau_mean=r["plateau_mean"], plateau_block_error=r["plateau_err"],
        kappa_W_per_mK=r["kappa"], kappa_block_error=r["kappa_err"],
        relative_error=float(r["kappa_err"] / r["kappa"]) if r["kappa"] else None,
        off_diagonal=[float(np.mean(dJ[:, i] * dJ[:, j])) for i, j in ((0, 1), (0, 2), (1, 2))],
        diagonal=[float(np.mean(dJ[:, k] ** 2)) for k in range(3)],
        isotropy_ratio=float(np.mean(dJ[:, 0] ** 2) / np.mean(dJ[:, 1] ** 2)),
        note="kappa is a convergence diagnostic only; S09 stays blocked, see the audit report",
    )
    # correlation function shape: normalised C(t) at a few lags
    C = r["C"]
    out["corr_normalised"] = [float(C[i] / C[0]) for i in
                              (0, 10, 50, 100, 200, 500, 1000)
                              if i < len(C)]
    out["corr_lags_ps"] = [float(i * dt) for i in (0, 10, 50, 100, 200, 500, 1000)
                           if i < len(C)]
    return out


def hac_cross_check():
    """Compare the shipped `compute_hac` (z column) with `kappa_from_trace` (zz).

    `hac.out` columns are: time(ps), hac[jx_in], hac[jx_out], hac[jy_in],
    hac[jy_out], hac[jz], rtc[...] (same order).  rtc is the running integral in
    W/(m K), already carrying GPUMD's KAPPA_UNIT_CONVERSION.
    """
    d = os.path.join(CASES, "hac_N1024")
    jf, vol, _ = read_heat(os.path.join(d, "heat_flux.out"))
    t = jf[:, 0]
    Jz = jf[:, 2:3] * vol
    dt = float(np.median(np.diff(t)))
    r = kappa_from_trace(Jz, dt, vol, 250.0)
    ours = r["kappa"]
    rows = []
    for line in open(os.path.join(d, "hac.out")):
        if line.startswith("#") or not line.strip():
            continue
        v = [float(x) for x in line.split()]
        rows.append(v)
    rows = np.array(rows)
    # columns: 0 time, 1..5 hac, 6..10 rtc
    t_hac = rows[:, 0]
    rtc_z = rows[:, 9]
    plateau = (t_hac > 5 * r["tau0"]) & (t_hac < 12 * r["tau0"])
    if plateau.sum() < 3:
        plateau = (t_hac > 0.5 * t_hac.max()) & (t_hac < 0.9 * t_hac.max())
    gpumd = float(rtc_z[plateau].mean())
    return dict(case="hac_N1024", n_samples=int(len(t)), total_ps=float(t[-1]),
                tau_ps=r["tau0"], plateau_window_ps=[float(t_hac[plateau][0]),
                                                    float(t_hac[plateau][-1])],
                kappa_zz_ours_W_per_mK=ours,
                kappa_zz_gpumd_compute_hac=gpumd,
                ratio_gpumd_over_ours=float(gpumd / ours) if ours else None,
                note="same trajectory, same 10-step sampling, z direction only")


if __name__ == "__main__":
    res = {}
    for name, fn in (("continuity", continuity), ("translation", shift),
                     ("green_kubo", gk), ("estimator_selftest", estimator_selftest),
                     ("hac_cross_check", hac_cross_check)):
        try:
            res[name] = fn()
            print(f"--- {name}")
            print(json.dumps(res[name], indent=1)[:1200])
        except Exception as exc:
            res[name] = dict(error=str(exc))
            print(f"--- {name}: FAILED {exc}")
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "heat_audit_results.json"), "w"), indent=1)
