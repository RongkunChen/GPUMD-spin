"""Comparison figures for the 100k-checkpoint series of the 5-descriptor run.

Reads the per-checkpoint JSON cache written by model_series.py and produces:
  series_dev.png          DEV errors vs generation
  series_static.png       a0, m_eq, D, Tc vs generation
  series_magnetization.png M(T) for every checkpoint
  series_magnon.png       magnon dispersion for every checkpoint
  series_states.png       magnetic-state energies relative to FM
  series_phonon.png       FM phonon maximum / N-point frequency vs generation
"""
import glob
import json
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                "validation", "BCC_Fe_full_physics", "scripts"))
sys.path.insert(0, _HERE)
import feutils as U
import model_series as MS
import series_physics as SPS

MS_STATES = SPS.STATES

SER = MS.SER
FIG = os.path.join(U.VAL, "figures")
os.makedirs(FIG, exist_ok=True)


def _savefig(fig, name, dpi=160):
    """Save via a temp file + atomic replace.

    A plain savefig occasionally fails here with OSError 22 (the file is being
    held open by a preview/AV scanner); writing to a temp name first makes the
    step idempotent and retryable.
    """
    final = os.path.join(FIG, name)
    tmp = final + ".tmp.png"
    for attempt in range(5):
        try:
            fig.savefig(tmp, dpi=dpi)
            os.replace(tmp, final)
            return
        except OSError:
            time.sleep(0.5)
    fig.savefig(final, dpi=dpi)


CUT = {"a0": 2.866, "m": 2.2, "D": (280, 320), "Tc": 1043}


def _fix_units(d):
    """Older caches stored |M| as the cell total; normalise in place (idempotent)."""
    c = d.get("curie", {})
    if c.get("n_atoms") is None:
        n = 432.0
        for p in c.get("points", []):
            p["M"] = p["M"] / n
            if "M_std" in p:
                p["M_std"] = p["M_std"] / n
        c["n_atoms"] = int(n)
        json.dump(d, open(os.path.join(SER, d["tag"] + ".json"), "w"), indent=1,
                  default=float)
    return d


def load():
    rows = []
    for p in glob.glob(os.path.join(SER, "*.json")):
        d = _fix_units(json.load(open(p)))
        d["gen"] = MS.gen_of(d["tag"])
        rows.append(d)
    rows.sort(key=lambda r: r["gen"])
    return rows


def _x(rows):
    return np.array([r["gen"] / 1000.0 for r in rows])


def fig_dev(rows):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    g = _x(rows)
    for a, key, lab in zip(ax, ("E_MAE_meV", "F_MAE", "M_MAE"),
                           ("energy MAE (meV/atom)", "force MAE (eV/Å)",
                            "torque MAE (eV/μB)")):
        a.plot(g, [r["dev"][key] for r in rows], "o-", ms=4, color="C0")
        a.set_xlabel("generation (×10³)")
        a.set_ylabel(lab)
        a.grid(alpha=0.3)
    ax[0].set_title("DEV errors vs training generation")
    fig.tight_layout()
    _savefig(fig, "series_dev.png", 160)
    plt.close(fig)


def fig_static(rows):
    fig, ax = plt.subplots(2, 2, figsize=(8.4, 5.8))
    g = _x(rows)
    ax[0, 0].plot(g, [r["eos"]["a0_A"] for r in rows], "o-", color="C0")
    ax[0, 0].axhline(CUT["a0"], color="0.4", ls="--", lw=0.9, label="experiment 2.866")
    ax[0, 0].set_ylabel("$a_0$ (Å)")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(g, [r["moment"]["m_eq_uB"] for r in rows], "o-", color="C3")
    ax[0, 1].axhline(CUT["m"], color="0.4", ls="--", lw=0.9, label="experiment 2.2")
    ax[0, 1].set_ylabel("$m_{eq}$ ($\\mu_B$/Fe)")
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(g, [r["magnon"]["D_meV_A2"] for r in rows], "o-", color="C2")
    ax[1, 0].axhspan(*CUT["D"], color="0.75", alpha=0.6, label="280-320 meV·Å²")
    ax[1, 0].set_ylabel("$D$ (meV·Å²)")
    ax[1, 0].set_xlabel("generation (×10³)")
    ax[1, 0].legend(fontsize=8)

    tc = [r["curie"]["Tc_half_M_K"] for r in rows]
    ax[1, 1].plot(g, tc, "o-", color="C4")
    ax[1, 1].axhline(CUT["Tc"], color="0.4", ls="--", lw=0.9, label="experiment 1043 K")
    ax[1, 1].set_ylabel("$T_c$ (K)")
    ax[1, 1].set_xlabel("generation (×10³)")
    ax[1, 1].legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("magnetic properties vs training generation", y=1.0)
    fig.tight_layout()
    _savefig(fig, "series_static.png", 160)
    plt.close(fig)


def fig_magnetization(rows):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for i, r in enumerate(rows):
        T = [p["T"] for p in r["curie"]["points"]]
        M = [p["M"] for p in r["curie"]["points"]]
        ax.plot(T, M, "o-", ms=3.5, color="C%d" % (i % 10),
                label=f"{r['tag']}  (Tc={r['curie']['Tc_half_M_K']:.0f} K)")
    ax.axvline(1043, color="0.4", ls="--", lw=0.9)
    ax.set_xlabel("T (K)")
    ax.set_ylabel("$|M|$ ($\\mu_B$/atom)")
    ax.set_title("magnetisation vs temperature, one curve per checkpoint")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "series_magnetization.png", 160)
    plt.close(fig)


def fig_magnon(rows):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for i, r in enumerate(rows):
        q = [p["q"] for p in r["magnon"]["points"]]
        w = [p["omega_meV"] for p in r["magnon"]["points"]]
        ax.plot(q, w, "o-", ms=3, color="C%d" % (i % 10), label=r["tag"])
    ax.set_xlabel("q along [001] (Å$^{-1}$)")
    ax.set_ylabel("$\\hbar\\omega$ (meV)")
    ax.set_title("magnon dispersion (frozen spiral) per checkpoint")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "series_magnon.png", 160)
    plt.close(fig)


def fig_states(rows):
    keys = ["NC-90", "AFM-A", "DLM", "NM", "AFM-G"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(keys))
    w = 0.8 / max(len(rows), 1)
    for i, r in enumerate(rows):
        ax.bar(x + i * w, [r["states"][k] for k in keys], w,
               label=r["tag"], color="C%d" % (i % 10))
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(keys)
    ax.set_yscale("log")
    ax.set_ylabel("$\\Delta E$ vs FM (meV/atom, log)")
    ax.set_title("magnetic-state energies relative to FM")
    ax.legend(fontsize=7)
    fig.tight_layout()
    _savefig(fig, "series_states.png", 160)
    plt.close(fig)


def fig_phonon(rows):
    """Legacy single-state view; prefers the state-resolved FM entry when present.

    `phonon` (first version) fixed |S| = 2.4 and used the EOS a0; `phonons.FM`
    uses the equilibrium moment and the relaxed a0, which moves omega_max by
    about 1%.  Plot whichever exists, and say which one it is.
    """
    which = "phonons" if all("phonons" in r for r in rows) else "phonon"
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.6))
    g = _x(rows)
    if which == "phonons":
        om = [r["phonons"]["FM"]["omega_max_THz"] for r in rows]
        dis = [r["phonons"]["FM"]["omega"] for r in rows]
        tit = "FM phonon maximum ($m_{eq}$, relaxed $a_0$)"
    else:
        om = [r["phonon"]["omega_max_THz"] for r in rows]
        dis = [r["phonon"]["dispersion"] for r in rows]
        tit = "FM phonon maximum (|S| = 2.4, EOS $a_0$)"
    ax[0].plot(g, om, "o-", color="C0")
    ax[0].set_xlabel("generation (×10³)")
    ax[0].set_ylabel("$\\omega_{max}$ (THz)")
    ax[0].set_title(tit)
    ax[0].grid(alpha=0.3)
    for i, r in enumerate(rows):
        W = np.array(dis[i])
        ax[1].plot(np.arange(W.shape[0]), W[:, 0], "-", lw=1.0, color="C%d" % (i % 10),
                   label=r["tag"])
    ax[1].set_xlabel("path index  ($\\Gamma$-H-N-$\\Gamma$-P-H)")
    ax[1].set_ylabel("lowest branch (THz)")
    ax[1].set_title("lowest phonon branch per checkpoint")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "series_phonon.png", 160)
    plt.close(fig)


def _marks(ax, d, ymax=None):
    """Vertical guide lines at the high-symmetry points; labels come from the ticks."""
    for m in d["high_symmetry"]:
        ax.axvline(m["x"], color="0.85", lw=0.8, zorder=0)


def _path_xticks(ax, d):
    ax.set_xticks([m["x"] for m in d["high_symmetry"]])
    ax.set_xticklabels([m["label"] for m in d["high_symmetry"]])


def fig_exchange(rows):
    """J1..J6, the fit residual, and the two independent stiffness estimates."""
    figs, ax = plt.subplots(2, 2, figsize=(9.4, 6.2))
    g = _x(rows)
    sh = rows[-1]["exchange"]["shells"]
    nsh = len(sh)
    for n in range(nsh):
        ax[0, 0].plot(g, [r["exchange"]["J_meV"][n] for r in rows], "o-", ms=3.5,
                      color="C%d" % (n % 10),
                      label=f"$J_{n+1}$  r={sh[n]['r_A']:.2f} Å  z={sh[n]['z']}")
    ax[0, 0].axhline(0, color="0.5", lw=0.7)
    ax[0, 0].set_ylabel("$J_n$ (meV)")
    ax[0, 0].set_title("exchange constants vs generation")
    ax[0, 0].legend(fontsize=6.5, ncol=2)

    x = np.arange(nsh)
    w = 0.8 / max(len(rows), 1)
    for i, r in enumerate(rows):
        ax[0, 1].bar(x + i * w, r["exchange"]["J_meV"], w, color="C%d" % (i % 10),
                     label=r["tag"])
    ax[0, 1].set_xticks(x + 0.4 - w / 2)
    ax[0, 1].set_xticklabels([f"$J_{n+1}$" for n in range(nsh)])
    ax[0, 1].axhline(0, color="0.5", lw=0.7)
    ax[0, 1].set_ylabel("$J_n$ (meV)")
    ax[0, 1].set_title("fitted exchange constants")
    ax[0, 1].legend(fontsize=6.5)

    ax[1, 0].plot(g, [1000 * r["exchange"]["fit_rms_meV_per_atom"] for r in rows],
                  "o-", color="C3")
    ax[1, 0].set_ylabel("fit residual (meV/atom)")
    ax[1, 0].set_xlabel("generation (×10³)")
    ax[1, 0].set_title("Heisenberg-fit residual of $E(q)-E_{FM}$")

    ax[1, 1].plot(g, [r["checks"]["D_direct_meV_A2"] for r in rows], "o-",
                  color="C2", label="direct frozen spirals")
    ax[1, 1].plot(g, [r["checks"]["D_lswt_closed_form_meV_A2"] for r in rows],
                  "s--", color="C0", label="LSWT from fitted $J_n$")
    ax[1, 1].axhspan(*CUT["D"], color="0.78", alpha=0.6, label="280-320 meV·Å²")
    ax[1, 1].set_ylabel("$D$ (meV·Å²)")
    ax[1, 1].set_xlabel("generation (×10³)")
    ax[1, 1].set_title("spin-wave stiffness, two independent routes")
    ax[1, 1].legend(fontsize=7)
    for a in ax.ravel():
        a.grid(alpha=0.25)
    figs.tight_layout()
    figs.savefig(os.path.join(FIG, "series_exchange.png"), dpi=160)
    plt.close(figs)


def fig_magnon_path(rows):
    """Full magnon dispersion along the path, plus the direct spiral points."""
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9))
    d0 = rows[-1]["magnon_path"]
    for i, r in enumerate(rows):
        mp = r["magnon_path"]
        ax[0].plot(mp["x"], mp["omega_meV"], "-", lw=1.2, color="C%d" % (i % 10),
                   label=f"{r['tag']}  D={r['checks']['D_lswt_closed_form_meV_A2']:.0f}")
    _marks(ax[0], d0, ax[0].get_ylim()[1])
    _path_xticks(ax[0], d0)
    ax[0].set_xlabel("q path")
    ax[0].set_ylabel("$\\hbar\\omega$ (meV)")
    ax[0].set_title("magnon dispersion, LSWT of the fitted $J_n$")
    ax[0].legend(fontsize=6.5)
    ax[0].grid(alpha=0.25)

    for i, r in enumerate(rows):
        md = r["magnon_direct"]
        for name, ls in (("001", "-"), ("111", "--")):
            p = md[name]["points"]
            ax[1].plot([x["q_norm"] for x in p], [x["omega_meV"] for x in p], ls + "o",
                       ms=3, lw=1.0, color="C%d" % (i % 10),
                       label=f"{r['tag']} [{name}]" if i == 0 or True else None)
    ax[1].set_xlabel("|q| (Å$^{-1}$)")
    ax[1].set_ylabel("$\\hbar\\omega$ (meV)")
    ax[1].set_title("direct frozen spirals along [001] and [111]")
    ax[1].legend(fontsize=6, ncol=2)
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, "series_magnon_path.png", 160)
    plt.close(fig)


def fig_phonon_states(rows):
    """omega_max per magnetic state, its shift relative to FM, and the a0 it sits at."""
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.6))
    g = _x(rows)
    kinds = [k for k, _ in MS_STATES if any(k in r.get("phonons", {}) for r in rows)]
    for i, k in enumerate(kinds):
        y = [r["phonons"][k]["omega_max_THz"] if k in r.get("phonons", {}) else np.nan
             for r in rows]
        ax[0].plot(g, y, "o-", ms=3.5, color="C%d" % (i % 10), label=k)
        y0 = [r["phonons"]["FM"]["omega_max_THz"] if "FM" in r.get("phonons", {}) else np.nan
              for r in rows]
        ax[1].plot(g, np.array(y) - np.array(y0), "o-", ms=3.5, color="C%d" % (i % 10),
                   label=k)
        ya = [r["phonons"][k]["a0_A"] if k in r.get("phonons", {}) else np.nan for r in rows]
        ax[2].plot(g, ya, "o-", ms=3.5, color="C%d" % (i % 10), label=k)
    ax[0].set_ylabel("$\\omega_{max}$ (THz)")
    ax[1].set_ylabel("$\\omega_{max}-\\omega_{max}^{FM}$ (THz)")
    ax[2].set_ylabel("$a_0$ of that state (Å)")
    ax[2].axhline(CUT["a0"], color="0.4", ls="--", lw=0.9)
    for a, t in zip(ax, ("phonon maximum per magnetic state", "shift relative to FM",
                         "relaxed lattice constant per state")):
        a.set_xlabel("generation (×10³)")
        a.set_title(t, fontsize=9)
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=6.5)
    fig.tight_layout()
    _savefig(fig, "series_phonon_states.png", 160)
    plt.close(fig)


def fig_phonon_dispersion(rows):
    """One panel per magnetic state, every checkpoint overlaid."""
    kinds = [k for k, _ in MS_STATES if any(k in r.get("phonons", {}) for r in rows)]
    n = len(kinds)
    fig, axs = plt.subplots(1, n, figsize=(2.5 * n + 0.6, 3.6), sharey=True,
                            squeeze=False)
    for j, k in enumerate(kinds):
        a = axs[0, j]
        for i, r in enumerate(rows):
            ph = r.get("phonons", {}).get(k)
            if ph is None:
                continue
            W = np.array(ph["omega"])
            for b in range(W.shape[1]):
                a.plot(ph["x"], W[:, b], "-", lw=0.8, color="C%d" % (i % 10),
                       label=(r["tag"] if (j == 0 and b == 0) else None))
        a.axhline(0, color="0.6", lw=0.7)
        a.set_title(k, fontsize=9)
        _path_xticks(a, rows[-1]["magnon_path"])
        a.grid(alpha=0.2)
        if j == 0:
            a.set_ylabel("$\\omega$ (THz)")
            a.legend(fontsize=6)
    fig.suptitle("phonon dispersion of every magnetic state, one curve per checkpoint",
                 fontsize=10)
    fig.tight_layout()
    _savefig(fig, "series_phonon_dispersion.png", 160)
    plt.close(fig)


def fig_phonon_compare_states(rows):
    """All magnetic states of the same checkpoint in one panel, and the shift."""
    r = rows[-1]
    ph = r.get("phonons") or {}
    if not ph:
        return
    kinds = [k for k, _ in MS_STATES if k in ph]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for i, k in enumerate(kinds):
        W = np.array(ph[k]["omega"])
        for b in range(W.shape[1]):
            ax[0].plot(ph[k]["x"], W[:, b], "-", lw=0.8, color="C%d" % (i % 10),
                       label=k if b == 0 else None)   # label once per state
    # the magnetic cells fold the BZ, so AFM-G has 6 and AFM-A 12 branches; the
    # difference is only meaningful branch-by-branch for the 3 that all cells share
    nb = min(np.array(ph[k]["omega"]).shape[1] for k in kinds)
    Wfm = np.array(ph["FM"]["omega"])[:, :nb]
    for i, k in enumerate(kinds):
        W = np.array(ph[k]["omega"])[:, :nb]
        for b in range(nb):
            ax[1].plot(ph[k]["x"], W[:, b] - Wfm[:, b], "-", lw=0.9,
                       color="C%d" % (i % 10), label=k if b == 0 else None)
    ax[0].axhline(0, color="0.6", lw=0.7)
    ax[0].set_title(f"phonons of all magnetic states ({r['tag']})", fontsize=9)
    ax[1].axhline(0, color="0.6", lw=0.7)
    ax[1].set_title("$\\omega_{state}-\\omega_{FM}$: dispersion correction by magnetism",
                    fontsize=9)
    for a in ax:
        _path_xticks(a, r["magnon_path"])
        a.set_ylabel("$\\omega$ (THz)")
        a.grid(alpha=0.22)
        a.legend(fontsize=6.5)
    fig.tight_layout()
    _savefig(fig, "series_phonon_compare_states.png", 160)
    plt.close(fig)


def fig_coupling(rows):
    """The magnon-phonon vertex, the crossings, and their evolution with training."""
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 3.5))
    g = _x(rows)
    # Plot |C| relative to that checkpoint's own noise floor.  Absolute |C| is not
    # interpretable across checkpoints because the finite-difference noise itself
    # changes; the question is only "is the vertex above the floor?", and the
    # elastic control is that floor by construction (ratio 1).
    for i, (key, lab) in enumerate((("transverse", "transverse (magnon channel)"),
                                    ("longitudinal", "longitudinal (exchange striction)"))):
        y = []
        for r in rows:
            cp = r.get("coupling")
            if not cp:
                y.append(np.nan)
                continue
            fl = SPS.coupling_verdict(cp)["floor_eV_per_A_per_muB"]
            y.append(abs(cp[key]["C_eV_per_A_per_muB"]) / fl)
        ax[0].semilogy(g, y, "o-", ms=3.5, color="C%d" % i, label=lab)
    ax[0].axhline(1.0, color="C2", ls="--", lw=1.0, label="noise floor (elastic control)")
    ax[0].axhline(2.0, color="0.5", ls=":", lw=1.0, label="2x floor = resolved")
    ax[0].set_ylabel("$|C|$ / noise floor")
    ax[0].set_title("magnon-phonon vertex vs its own noise floor", fontsize=9)
    ax[0].legend(fontsize=5.5)

    for i, r in enumerate(rows):
        cr = r.get("crossings") or {}
        if not cr.get("crossings"):
            continue
        ax[1].plot([c["fraction_along_path"] for c in cr["crossings"]],
                   [c["omega_meV"] for c in cr["crossings"]], "o", ms=4,
                   color="C%d" % (i % 10), label=f"{r['tag']} ({cr['n']})")
    ax[1].set_xlabel("fraction along the q path (exact, not avoided)")
    ax[1].set_ylabel("$\\hbar\\omega$ at the crossing (meV)")
    ax[1].set_title("magnon-phonon crossings", fontsize=9)
    ax[1].legend(fontsize=6.5)

    for i, k in enumerate([kk for kk, _ in MS_STATES if kk != "FM"]):
        y = []
        for r in rows:
            ph = r.get("phonons") or {}
            if "FM" not in ph or k not in ph:
                y.append(np.nan)
                continue
            W = np.array(ph[k]["omega"])
            Wf = np.array(ph["FM"]["omega"])
            nb = min(W.shape[1], Wf.shape[1])      # folded cells have more branches
            y.append(float(np.abs(W[:, :nb] - Wf[:, :nb]).max()))
        ax[2].plot(g, y, "o-", ms=3.5, color="C%d" % (i % 10), label=f"FM vs {k}")
    ax[2].set_ylabel("max $|\\Delta\\omega|$ (THz)")
    ax[2].set_title("max phonon shift from magnetic order", fontsize=9)
    ax[2].legend(fontsize=6.5)
    for a in ax:
        a.set_xlabel("generation (×10³)") if a is not ax[1] else None
        a.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, "series_coupling.png", 160)
    plt.close(fig)


def fig_moment_coupling(rows):
    """D at fixed |S| (why D is stiff) and the magnetostrictive phonon shift."""
    r = rows[-1]
    sv = r.get("stiffness_vs_moment")
    ms = r.get("magnetostriction")
    if not sv and not ms:
        return
    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.6))
    if sv:
        ax[0].plot([p["m_uB"] for p in sv["points"]],
                   [p["D_meV_A2"] for p in sv["points"]], "o-", color="C0")
        ax[0].axhline(320, color="0.5", ls="--", lw=0.9)
        ax[0].axhline(280, color="0.5", ls="--", lw=0.9)
        ax[0].axvline(2.2, color="C3", ls=":", lw=1.0, label="experiment 2.2 μ$_B$")
        ax[0].set_xlabel("$|S|$ (μ$_B$/Fe)")
        ax[0].set_ylabel("$D$ (meV·Å²)")
        ax[0].set_title(f"D(|S|):  dlnD/dlnm = {sv['exponent_dlnD_dlnm']:.2f} "
                        f"(0 = Heisenberg)", fontsize=8)
        ax[0].legend(fontsize=7)
    g = _x(rows)
    ax[1].plot(g, [r["checks"]["D_direct_meV_A2"] for r in rows], "o-", color="C2",
               label="at the model's own $m_{eq}$")
    ax[1].plot(g, [r["stiffness_vs_moment"]["D_at_2p2_uB_meV_A2"]
                   if "stiffness_vs_moment" in r else np.nan for r in rows],
               "s--", color="C1", label="renormalised to m = 2.2 μ$_B$")
    ax[1].axhspan(*CUT["D"], color="0.78", alpha=0.6)
    ax[1].set_xlabel("generation (×10³)")
    ax[1].set_ylabel("$D$ (meV·Å²)")
    ax[1].set_title("D at the model moment vs at 2.2 $\\mu_B$ (must coincide)", fontsize=8)
    ax[1].legend(fontsize=7)
    if ms:
        for i, rr in enumerate(rows):
            m2 = rr.get("magnetostriction")
            if not m2:
                continue
            W = np.array([[np.nan if v is None else v for v in row]
                          for row in m2["dlnomega_dlnS"]])
            for b in range(W.shape[1]):
                ax[2].plot(m2["x"], W[:, b], "-", lw=1.0, color="C%d" % (i % 10),
                           label=(f"{rr['tag']} ({m2['max_abs']:.2f})" if b == 0 else None))
        ax[2].axhline(0, color="0.6", lw=0.7)
        _path_xticks(ax[2], r["magnon_path"])
        ax[2].set_xlabel("q path")
        ax[2].set_ylabel("$\\partial\\ln\\omega/\\partial\\ln|S|$")
        ax[2].set_title("magnetostrictive spin-phonon coupling", fontsize=9)
        ax[2].legend(fontsize=6)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    _savefig(fig, "series_moment_coupling.png", 160)
    plt.close(fig)


def fig_table(rows):
    """One-page summary table as a figure, so the numbers travel with the plots."""
    cols = ["gen", "E_MAE", "F_MAE", "a0", "m_eq", "J1", "J rms", "D_direct",
            "D_LSWT", "Tc", "ω_max", "unstable", "FM low"]
    cells = []
    for r in rows:
        c = r["checks"]
        cells.append([f"{r['gen']//1000}k", f"{r['dev']['E_MAE_meV']:.2f}",
                      f"{r['dev']['F_MAE']:.3f}",
                      f"{r['eos']['a0_A']:.4f}", f"{r['moment']['m_eq_uB']:.3f}",
                      "-" if c["J1_meV"] is None else f"{c['J1_meV']:.1f}",
                      "-" if c["J_fit_rms_meV_per_atom"] is None
                      else f"{c['J_fit_rms_meV_per_atom']:.2f}",
                      "-" if c["D_direct_meV_A2"] is None else f"{c['D_direct_meV_A2']:.0f}",
                      "-" if c["D_lswt_closed_form_meV_A2"] is None
                      else f"{c['D_lswt_closed_form_meV_A2']:.0f}",
                      f"{r['curie']['Tc_half_M_K']:.0f}",
                      # state-resolved FM entry when present, so this column
                      # matches series_phonon.png instead of the legacy |S|=2.4 run
                      f"{(r.get('phonons', {}).get('FM') or r['phonon'])['omega_max_THz']:.2f}",
                      ",".join(x.replace("AFM-", "") for x in
                               c.get("phonon_unstable_states", [])) or "-",
                      "yes" if c["FM_lowest"] else "NO"])
    fig, ax = plt.subplots(figsize=(11.0, 0.6 + 0.32 * len(rows)))
    ax.axis("off")
    t = ax.table(cellText=cells, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    ax.set_title("5-descriptor run: physics per 100k checkpoint "
                 "(D in meV·Å², ω_max in THz, J in meV)", fontsize=9)
    fig.tight_layout()
    _savefig(fig, "series_table.png", 170)
    plt.close(fig)


def main():
    rows = load()
    if not rows:
        print("no cached checkpoints yet")
        return
    fig_dev(rows)
    fig_static(rows)
    fig_magnetization(rows)
    fig_magnon(rows)
    fig_states(rows)
    fig_phonon(rows)
    if "exchange" in rows[-1]:
        fig_exchange(rows)
        fig_magnon_path(rows)
    if "phonons" in rows[-1]:
        fig_phonon_states(rows)
        fig_phonon_dispersion(rows)
        fig_phonon_compare_states(rows)
    if "coupling" in rows[-1]:
        fig_coupling(rows)
    fig_moment_coupling(rows)
    fig_table(rows)
    print(f"{len(rows)} checkpoints, figures -> {FIG}")


if __name__ == "__main__":
    main()
