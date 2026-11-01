"""Full magnetic-property battery for one model, cached per checkpoint.

For every 100k checkpoint of the 5-descriptor training run this computes:

  * DEV metrics (energy / force / torque MAE on the held-out split)
  * equilibrium lattice constant a0 and bulk modulus (EOS)
  * equilibrium moment m_eq (argmin E over |S|)
  * magnetic ground-state ordering (FM, NC-90, AFM-A, DLM, NM, AFM-G)
  * exchange constants J1..J6 and the spin-wave stiffness D (frozen spirals)
  * Curie temperature from M(T) with the SLLG-Langevin sampler
  * FM phonon maximum frequency and the lowest branch at N

Everything uses the production binaries; nothing is fitted to make a number look
right.  Results are cached in `series/<tag>.json` so a re-run only computes what
is missing, and `make_series_figures()` turns the cache into the comparison
figures.

usage
  python model_series.py --add tag=path/to/nep.txt [...]   # register + run
  python model_series.py --scan DIR --pattern 'gen_*k.nep.txt'
  python model_series.py --figures
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                "validation", "BCC_Fe_full_physics", "scripts"))
sys.path.insert(0, _HERE)
import feutils as U
import phonon_fd as P
import relax_and_phonons as R

WORK = os.path.join(U.ROOT, "work_20261101_retrain")
SER = os.path.join(WORK, "C_5desc_500k", "series")
DEV = os.path.join(WORK, "data", "dev.xyz")
NEP_IN = os.path.join(WORK, "C_5desc_500k", "nep.in")
NEP_EXE = os.path.join(WORK, "bin", "nep.exe")
GPUMD = U.EXE
os.makedirs(SER, exist_ok=True)

TC_TEMPS = (600, 800, 900, 1000, 1100, 1300)
A0_REF = 2.866
M_REF = 2.2
D_REF = (280, 320)


# ------------------------------------------------------------------ DEV metrics
def dev_metrics(nep_path, tag):
    d = os.path.join(SER, "eval", tag)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "train.xyz")
    with open(p, "wb") as fh, open(DEV, "rb") as src:
        fh.write(src.read())
    with open(os.path.join(d, "nep.txt"), "wb") as fh, open(nep_path, "rb") as src:
        fh.write(src.read())
    txt = open(NEP_IN).read()
    if "prediction" not in txt:
        txt += "prediction 1\n"
    open(os.path.join(d, "nep.in"), "w").write(txt)
    for junk in ("energy_train.out", "force_train.out", "spin_train.out", "stdout.txt"):
        q = os.path.join(d, junk)
        if os.path.exists(q):
            open(q, "w").close()
    with open(os.path.join(d, "stdout.txt"), "w") as fh:
        subprocess.run([NEP_EXE], cwd=d, stdout=fh, stderr=subprocess.STDOUT)
    e = np.loadtxt(os.path.join(d, "energy_train.out"))
    f = np.loadtxt(os.path.join(d, "force_train.out"))
    s = np.loadtxt(os.path.join(d, "spin_train.out"))
    fp, fr = f[:, :3], f[:, 3:]
    sp, sr = s[:, :3], s[:, 3:]
    return dict(n_frames=len(e),
                E_MAE_meV=float(np.abs(e[:, 0] - e[:, 1]).mean() * 1000),
                E_RMSE_meV=float(np.sqrt(((e[:, 0] - e[:, 1]) ** 2).mean()) * 1000),
                F_MAE=float(np.abs(fp - fr).mean()),
                F_RMSE=float(np.sqrt(((fp - fr) ** 2).mean())),
                M_MAE=float(np.abs(sp - sr).mean()))


# ------------------------------------------------------------------------ EOS
def eos(nep, tag):
    """a0 and B0 from a local quadratic around the minimum (54-atom cell)."""
    def E(a):
        c = U.bcc_cell(3, 3, 3, a)
        d = os.path.join(SER, "eos", f"{tag}_{a:.3f}")
        return U.static_energy(d, c, U.fm_spins(c["n"], 2.4), nep=nep) / c["n"]

    grid = np.arange(2.76, 2.921, 0.02)
    Ev = np.array([E(a) for a in grid])
    i = int(np.argmin(Ev))
    lo, hi = max(0, i - 1), min(len(grid) - 1, i + 1)
    c = np.polyfit(grid[lo:hi + 1], Ev[lo:hi + 1], 2)
    a0 = float(-c[1] / (2 * c[0]))
    fine = np.linspace(a0 - 0.008, a0 + 0.008, 5)
    Ef = np.array([E(a) for a in fine])
    c2 = np.polyfit(fine, Ef, 2)
    a0 = float(-c2[1] / (2 * c2[0])) if c2[0] > 0 else a0
    # B = V d2E/dV2 with V = a^3/2 per atom -> B = 2 c2 / (9 a).
    # (The first version used V*2*c2, which is dimensionally wrong and gave
    # 41531 GPa instead of ~287 GPa.)
    # polyfit returns a2 with E = a2 x^2 + ..., so the curvature is 2*a2.
    # B = 2*curvature/(9a) with a in A -> eV/A^3; 1 eV/A^3 = 160.2177 GPa.
    B0 = float(2 * (2 * c2[0]) / (9 * a0) * 160.2176634)
    return dict(a0_A=a0, B0_GPa=B0, E_min_eV_per_atom=float(np.polyval(c2, a0)),
                curvature=float(2 * c2[0]), grid=[float(x) for x in fine],
                E=[float(x) for x in Ef])


def moment_curve(nep, a0, tag):
    def E(m):
        c = U.bcc_cell(3, 3, 3, a0)
        d = os.path.join(SER, "moment", f"{tag}_{m:.2f}")
        return U.static_energy(d, c, U.fm_spins(c["n"], m), nep=nep) / c["n"]

    ms = np.arange(0.0, 3.001, 0.25)
    Ev = np.array([E(m) for m in ms])
    i = int(np.argmin(Ev))
    m0 = float(ms[i])
    if 0 < i < len(ms) - 1:
        c = np.polyfit(ms[i - 1:i + 2], Ev[i - 1:i + 2], 2)
        m0 = float(-c[1] / (2 * c[0]))
    return dict(m_eq_uB=m0, grid=[float(x) for x in ms], E=[float(x) for x in Ev],
                E0_eV_per_atom=float(Ev[i]))


def magnetic_states(nep, a0, tag):
    """dE per atom of each order relative to FM, at the model's own a0."""
    m = 2.4
    out = {}
    e_fm = None
    for kind, n in (("FM", 7), ("NC-90", 5), ("AFM-A", 5), ("DLM", 7), ("AFM-G", 5)):
        st = R.replicate(R.state_cell(kind, a0), n)
        if kind == "DLM":
            acc = []
            for sd in (1, 2, 3):
                rng = np.random.default_rng(sd)
                v = rng.normal(size=(st["n"], 3))
                v /= np.linalg.norm(v, axis=1)[:, None]
                acc.append(U.static_energy(os.path.join(SER, "states", f"{tag}_{kind}{sd}"),
                                           st, v * m, nep=nep) / st["n"])
            e = float(np.mean(acc))
        else:
            e = U.static_energy(os.path.join(SER, "states", f"{tag}_{kind}"), st,
                                st["spins"] * m, nep=nep) / st["n"]
        if kind == "FM":
            e_fm = e
        out[kind] = e
    st = R.replicate(R.state_cell("FM", a0), 7)
    out["NM"] = U.static_energy(os.path.join(SER, "states", f"{tag}_NM"), st,
                                st["spins"] * 0.0, nep=nep) / st["n"]
    return {k: 1000.0 * (v - e_fm) for k, v in out.items()}


def exchange_and_D(nep, a0, tag, n=8):
    """D and the magnon energy at H from frozen spirals.

    Two traps, both hit before they were fixed:
      * the wavevector must be commensurate with the supercell.  An n x n x n
        *primitive* supercell is only (n/2) x (n/2) x (n/2) conventional cubes
        wide, so along a cubic axis the step is 2*pi/((n/2)*a), not 2*pi/(n*a).
        Using the wrong step gave D = 1347 meV.A^2 (4x too large) and a
        non-monotonic omega(q).
      * the quadratic regime only extends to ~25% of the zone, so D is fitted
        from the first three commensurate points.
    Here the spiral is put on an n x n x n *conventional* cell (2*n^3 atoms),
    which makes every multiple of 2*pi/(n*a) commensurate.
    """
    cell = U.bcc_cell(n, n, n, a0)
    fm = U.fm_spins(cell["n"], 2.4)
    e_fm = U.static_energy(os.path.join(SER, "spiral", f"{tag}_FM"), cell, fm, nep=nep)
    S = 1.2                     # m/2 at m = 2.4, g = 2
    rows = []
    for m in range(0, n + 1):
        q = m * 2 * np.pi / (n * a0)
        ang = cell["pos"][:, 2] * q
        sp = np.zeros((cell["n"], 3))
        sp[:, 0] = 2.4 * np.sin(ang)
        sp[:, 2] = 2.4 * np.cos(ang)
        e = U.static_energy(os.path.join(SER, "spiral", f"{tag}_q{m}"), cell, sp, nep=nep)
        # hbar*omega = 2 [E(q)-E(FM)]/(N S), NOT 2S[...]: see series_physics.lswt_omega
        rows.append(dict(m=m, q=float(q),
                         omega_meV=2 * (e - e_fm) / cell["n"] * 1000 / S))
    qq = np.array([r["q"] for r in rows])
    ww = np.array([r["omega_meV"] for r in rows])
    # D is a second derivative, so a straight omega-vs-q^2 slope over 12-37% of
    # the zone is biased low by the quartic term.  Extrapolate omega/q^2 -> D as
    # q^2 -> 0 instead, and keep the naive slope for comparison.
    sel = (np.arange(len(rows)) >= 1) & (np.arange(len(rows)) <= 3)
    D_naive = float(np.polyfit(qq[sel] ** 2, ww[sel], 1)[0]) if sel.sum() >= 3 else float("nan")
    if sel.sum() >= 3:
        y = ww[sel] / qq[sel] ** 2
        c = np.polyfit(qq[sel] ** 2, y, 1)
        D = float(c[1])
    else:
        D = float("nan")
    return dict(D_meV_A2=D, D_naive_slope_meV_A2=D_naive,
                omega_max_meV=float(ww.max()),
                omega_at_H_meV=float(ww[-1]), points=rows, n_atoms=cell["n"], S=S,
                commensurate=True, q_step_per_A=float(2 * np.pi / (n * a0)),
                fit_points=[1, 2, 3],
                D_method="omega/q^2 extrapolated to q^2 -> 0 over the three smallest "
                         "commensurate points; the naive omega-vs-q^2 slope is "
                         "D_naive_slope_meV_A2")


# ------------------------------------------------------------------------- Tc
def curie(nep, a0, tag, ps=1.0, dt=0.05, seeds=1):
    """|M|(T) with the SLLG-Langevin sampler; half-magnetisation crossing."""
    rows = []
    steps = int(ps * 1000 / dt)
    for T in TC_TEMPS:
        d = os.path.join(SER, "tc", f"{tag}_{T}K")
        os.makedirs(d, exist_ok=True)
        cell = U.bcc_cell(6, 6, 6, a0)
        sp = U.fm_spins(cell["n"], 2.2)
        U.write_xyz(os.path.join(d, "model.xyz"), cell, sp)
        with open(os.path.join(d, "nep.txt"), "wb") as fh, open(nep, "rb") as src:
            fh.write(src.read())
        with open(os.path.join(d, "run.in"), "w") as fh:
            fh.write(f"potential nep.txt\ntime_step {dt} 100\n"
                     f"ensemble nvt_lan {T} {T} 100\n"
                     f"spin_llg gamma=3.0385 alpha=0.3 T={T} seed=7\n"
                     f"dump_magnetization 200 magnetization.out warmup 10\n"
                     f"seed 7\nrun {steps}\n")
        for junk in ("magnetization.out", "thermo.out", "stdout.txt"):
            q = os.path.join(d, junk)
            if os.path.exists(q):
                open(q, "w").close()
        subprocess.run([GPUMD], cwd=d, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        f = os.path.join(d, "magnetization.out")
        if not os.path.getsize(f):
            rows.append(dict(T=T, M=float("nan"), chi=float("nan")))
            continue
        a = np.loadtxt(f, comments="#")
        half = a[len(a) // 2:]
        # the M columns of dump_magnetization are the CELL TOTAL, not per atom
        # (432 atoms x ~1.6 mu_B = ~700); the Tc crossing is scale invariant but
        # the plotted curve is not, so normalise here.
        rows.append(dict(T=T, M=float(half[:, 5].mean()) / cell["n"],
                         M_std=float(half[:, 5].std()) / cell["n"],
                         chi=float(half[:, 8].mean())))
    ms = {r["T"]: r["M"] for r in rows}
    tt = np.array(sorted(ms))
    mm = np.array([ms[t] for t in tt])
    half = 0.5 * mm[0]
    Tc = float("nan")
    for i in range(len(tt) - 1):
        if mm[i] >= half > mm[i + 1]:
            f = (mm[i] - half) / (mm[i] - mm[i + 1])
            Tc = float(tt[i] + f * (tt[i + 1] - tt[i]))
            break
    return dict(points=rows, Tc_half_M_K=Tc, n_atoms=432,
                protocol=f"{ps} ps per T, L=6 (432 atoms), dt={dt} fs, alpha=0.3")


# --------------------------------------------------------------- FM phonons
def fm_phonons(nep, a0, tag, n=7, delta=0.02):
    st = R.replicate(R.state_cell("FM", a0), n)
    st["spins"] = st["spins"] * 2.4
    phi, meta = R.force_constants(st, delta, nep=nep, tag=f"ser_{tag}", symmetrise=True)
    qN = np.array([0.5, 0.5, 0.0]) * 2 * np.pi / a0
    wG = R.frequencies(phi, st, np.zeros(3))[0]
    wN = R.frequencies(phi, st, qN)[0]
    pts, _ = R.qpath(a0)
    W = np.array([R.frequencies(phi, st, q)[0] for q in pts])
    return dict(omega_max_THz=float(W.max()), omega_N_THz=[float(x) for x in wN],
                gamma_THz=[float(x) for x in wG],
                imaginary=int((W < -1e-6).sum()),
                asr_before=meta["asr_before_eV_A2"],
                dispersion=W.tolist())


# ------------------------------------------------------------------ one model
def _checks(res):
    """Pass/fail flags against experiment, recomputed after every extension."""
    a0 = res["eos"]["a0_A"]
    ex = res.get("exchange") or {}
    mp = res.get("magnon_path")
    md = res.get("magnon_direct") or {}
    # consistency of the two independent magnon routes: LSWT from the fitted J_n
    # and the direct spiral energies.  Both are hbar*omega in meV.  Along Gamma-H
    # the path coordinate is linear in |q|, so the LSWT grid maps onto the spiral
    # q points directly (m = 1..8 of the 8^3 conventional cell).
    ratio = ratio_half = c_full = None
    if mp and md:
        nseg = mp["npts"]
        # along Gamma-H the path coordinate IS |q| in 1/A (straight segment from
        # the zone centre), so no rescaling is needed -- the first version divided
        # by 2*pi/a and compared LSWT at q<1 with spirals at q~2, which gave 2.64.
        qg = np.array(mp["x"][:nseg + 1])
        wg = np.array(mp["omega_meV"][:nseg + 1])
        pts = md["001"]["points"][1:]
        qn = np.array([p["q_norm"] for p in pts])
        wd = np.array([p["omega_meV"] for p in pts])
        # the J_n are fitted to spirals with mq = 0..4 of the 8-cell grid, i.e. up
        # to half of Gamma-H; comparing beyond that tests an extrapolation of the
        # fit, not the fit.  Both numbers are stored: the half-zone ratio is the
        # self-check, the full-zone one is a diagnostic of how Heisenberg-like the
        # model is out to the zone boundary.
        ok = np.abs(wd) > 20
        wl = np.interp(qn[ok], qg, wg)
        half = ok & (qn <= 0.5 * qg[-1])
        ratio_half = float(np.mean(wl[qn[ok] <= 0.5 * qg[-1]] /
                                    wd[ok][qn[ok] <= 0.5 * qg[-1]])) \
            if half.sum() >= 3 else None
        ratio = ratio_half
        c_full = float(np.mean(wl / wd[ok])) if ok.sum() >= 3 else None
    c = dict(
        a0_error_percent=100 * (a0 / A0_REF - 1),
        moment_error_percent=100 * (res["moment"]["m_eq_uB"] / M_REF - 1),
        D_in_range=bool(D_REF[0] <= res["magnon"]["D_meV_A2"] <= D_REF[1]),
        FM_lowest=bool(min(res["states"], key=res["states"].get) == "FM"),
        Tc_error_percent=100 * (res["curie"]["Tc_half_M_K"] / 1043 - 1)
        if res["curie"]["Tc_half_M_K"] == res["curie"]["Tc_half_M_K"] else None)
    c["J1_meV"] = float(ex["J_meV"][0]) if ex else None
    # standard Heisenberg convention (H = -1/2 sum J S_i.S_j, |S| in hbar)
    c["J1_S_meV"] = float(ex["J_S_meV"][0]) if ex else None
    c["J0_S_meV"] = float(ex["J0_meV"]) if ex else None
    ex2 = res.get("exchange_at_m2p2") or {}
    c["J1_S_at_2p2_meV"] = float(ex2["J_S_meV"][0]) if ex2 else None
    c["J0_S_at_2p2_meV"] = float(ex2["J0_meV"]) if ex2 else None
    c["D_at_2p2_lswt_meV_A2"] = (float(ex2["D_lswt_closed_form_meV_A2"])
                                 if ex2 else None)
    c["J_fit_rms_meV_per_atom"] = float(ex["fit_rms_meV_per_atom"] * 1000) if ex else None
    # D from the LSWT curve, closed form for a cubic crystal:
    # hbar*omega = D q^2 with D = (S/6) sum_n J_n z_n r_n^2.  A finite-q fit of
    # omega/q^2 would be biased high by the quartic term.
    c["D_lswt_closed_form_meV_A2"] = float(ex["D_lswt_closed_form_meV_A2"]) if ex else None
    c["D_direct_meV_A2"] = float(md["D_meV_A2"]) if md else None
    c["lswt_over_direct"] = ratio
    c["lswt_over_direct_half_zone"] = ratio_half
    c["lswt_over_direct_full_zone"] = c_full
    if "phonons" in res:
        ph = res["phonons"]
        c["omega_max_by_state_THz"] = {k: v["omega_max_THz"] for k, v in ph.items()}
        c["phonon_unstable_states"] = [k for k, v in ph.items()
                                      if v.get("imaginary_below_0p1_THz", v["imaginary"]) > 0]
        c["a0_by_state_A"] = {k: v["a0_A"] for k, v in ph.items()}
    if "coupling" in res:
        import series_physics as _S
        v = _S.coupling_verdict(res["coupling"])
        c["vertex_noise_floor_eV_per_A_per_muB"] = v["floor_eV_per_A_per_muB"]
        c["transverse_vertex_resolved"] = v["transverse_resolved"]
        c["longitudinal_vertex_resolved"] = v["longitudinal_resolved"]
    res["checks"] = c
    return c


def run_one(tag, nep, physics=True):
    out = os.path.join(SER, f"{tag}.json")
    if os.path.exists(out):
        print(f"{tag}: base cache found, filling in what is missing", flush=True)
        res = json.load(open(out))
    else:
        print(f"{tag}: running", flush=True)
        res = dict(tag=tag, nep=nep,
                   nep_sha256=__import__("hashlib").sha256(open(nep, "rb").read()).hexdigest(),
                   dev=dev_metrics(nep, tag))
        res["eos"] = eos(nep, tag)
        a0 = res["eos"]["a0_A"]
        res["moment"] = moment_curve(nep, a0, tag)
        res["states"] = magnetic_states(nep, a0, tag)
        res["magnon"] = exchange_and_D(nep, a0, tag)
        res["phonon"] = fm_phonons(nep, a0, tag)
        res["curie"] = curie(nep, a0, tag)
        # save before the derived flags: a crash in _checks must not throw away
        # the ~8 minutes of force evaluations that came before it
        U.save_json(out, res)
        _checks(res)
        U.save_json(out, res)
    if physics:
        import series_physics as SPS
        SPS.run_physics(res, nep, tag)
    _checks(res)
    U.save_json(out, res)
    a0 = res["eos"]["a0_A"]
    c = res["checks"]
    print(f"{tag}: done  a0={a0:.4f}  m={res['moment']['m_eq_uB']:.3f}  "
          f"J1={c['J1_meV']:.2f}  D_lswt={c['D_lswt_closed_form_meV_A2']:.1f}  "
          f"D_dir={c['D_direct_meV_A2']:.1f}  Tc={res['curie']['Tc_half_M_K']:.0f}  "
          f"E_MAE={res['dev']['E_MAE_meV']:.2f}", flush=True)
    return res


def gen_of(tag):
    m = re.search(r"(\d+)k", tag)
    return int(m.group(1)) * 1000 if m else 10 ** 9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[], help="tag=path")
    ap.add_argument("--scan", default=None)
    ap.add_argument("--pattern", default="gen_*k.nep.txt")
    ap.add_argument("--figures", action="store_true")
    A = ap.parse_args()
    jobs = []
    for spec in A.add:
        tag, path = spec.split("=", 1)
        jobs.append((tag, path))
    if A.scan:
        for p in sorted(glob.glob(os.path.join(A.scan, A.pattern))):
            tag = os.path.basename(p).replace(".nep.txt", "")
            jobs.append((tag, p))
    for tag, path in jobs:
        run_one(tag, path)
    if A.figures:
        import make_series_figures
        make_series_figures.main()
