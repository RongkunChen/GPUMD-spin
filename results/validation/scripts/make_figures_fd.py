"""Figures for the external phonon / magnon analysis.

Reads only the .dat/.csv files that phonon_fd.py and stage_C.py wrote; nothing is
recomputed here.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import feutils as U

OUT = os.path.join(U.VAL, "raw_data", "E_fd")
FIG = os.path.join(U.VAL, "figures")
os.makedirs(FIG, exist_ok=True)

HSP = {"G": [0.0, 0.0, 0.0], "H": [1.0, 0.0, 0.0], "N": [0.5, 0.5, 0.0],
       "P": [0.5, 0.5, 0.5]}
PATH = [("G", "H"), ("H", "N"), ("N", "G"), ("G", "P"), ("P", "H")]


def path_axis():
    """Cumulative |q| distance of every high-symmetry point along the path."""
    d, ticks = 0.0, [0.0]
    for pa, pb in PATH:
        d += np.linalg.norm(np.array(HSP[pb]) - np.array(HSP[pa]))
        ticks.append(d)
    return ticks


def load_dispersion(path):
    seg, x, W = [], [], []
    ticks = path_axis()
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        s = p[0]
        q = np.array([float(p[2]), float(p[3]), float(p[4])])
        i = [k for k, (pa, pb) in enumerate(PATH) if f"{pa}-{pb}" == s][0]
        x.append(ticks[i] + np.linalg.norm(q - np.array(HSP[PATH[i][0]])))
        W.append([float(v) for v in p[5:]])
        seg.append(s)
    return np.array(x), np.array(W), np.array(seg), ticks


def fig_phonon():
    x, W, seg, ticks = load_dispersion(os.path.join(OUT, "phonon_dispersion.dat"))
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for b in range(W.shape[1]):
        ax.plot(x, W[:, b], "-", lw=1.2, color="C%d" % b)
    for t in ticks:
        ax.axvline(t, color="0.7", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$\\Gamma$", "H", "N", "$\\Gamma$", "P", "H"])
    ax.set_ylabel("phonon frequency (THz)")
    ax.set_title("bcc Fe phonons, finite-displacement IFC2\n"
                 "(spin-aware forces, 7$\\times$7$\\times$7 primitive, ASR corrected)")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, max(12.0, W.max() * 1.08))
    ax.axhline(0, color="0.4", lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "phonon_dispersion_fd.png"), dpi=160)
    plt.close(fig)

    # DOS
    d = np.loadtxt(os.path.join(OUT, "phonon_dos.dat"))
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    ax.plot(d[:, 0], d[:, 1], "-", color="C3")
    ax.fill_between(d[:, 0], 0, d[:, 1], color="C3", alpha=0.25)
    ax.set_xlabel("$\\omega$ (THz)")
    ax.set_ylabel("g($\\omega$) (states/THz/cell)")
    ax.set_title("phonon DOS, 16$^3$ q grid")
    ax.set_xlim(0, d[:, 0].max())
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "phonon_dos_fd.png"), dpi=160)
    plt.close(fig)


def fig_shift():
    p = os.path.join(OUT, "spin_phonon_frequency_shift.csv")
    if not os.path.exists(p):
        return
    rows = [l.split() for l in open(p) if not l.startswith("#")]
    segs = [r[0] for r in rows]
    labels = []
    for s in segs:
        if not labels or labels[-1] != s:
            labels.append(s)
    x, per_seg = [], {}
    ticks = path_axis()
    for r in rows:
        s = r[0]
        i = [k for k, (pa, pb) in enumerate(PATH) if f"{pa}-{pb}" == s][0]
        t = float(r[1])
        xx = ticks[i] + t * np.linalg.norm(np.array(HSP[PATH[i][1]]) - np.array(HSP[PATH[i][0]]))
        per_seg.setdefault((s, r[2]), []).append((xx, float(r[5])))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for (s, b), v in per_seg.items():
        v = np.array(v)
        ax.plot(v[:, 0], v[:, 1], "-", lw=1.0, color="C%d" % (int(b) - 1), alpha=0.85)
    for t in ticks:
        ax.axvline(t, color="0.7", lw=0.6)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$\\Gamma$", "H", "N", "$\\Gamma$", "P", "H"])
    ax.set_ylabel("$\\omega_{FM}-\\omega_{DLM}$ (THz)")
    ax.set_title("spin-disorder phonon softening (4 random spin states, 6$\\times$6$\\times$6)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "spin_phonon_shift_fd.png"), dpi=160)
    plt.close(fig)


def fig_magnon():
    p = os.path.join(U.VAL, "raw_data", "C", "magnon_dispersion.dat")
    if not os.path.exists(p):
        return
    x, y = [], []
    ticks = path_axis()
    cur = ticks[0]
    for line in open(p):
        if line.startswith("#"):
            continue
        s, t, kx, ky, kz, w = line.split()
        i = [k for k, (pa, pb) in enumerate(PATH) if f"{pa}-{pb}" == s][0]
        q = np.array([float(kx), float(ky), float(kz)])
        x.append(ticks[i] + np.linalg.norm(q - np.array(HSP[PATH[i][0]])))
        y.append(float(w))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(x, y, "-", color="C0", lw=1.4)
    for t in ticks:
        ax.axvline(t, color="0.7", lw=0.6)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["$\\Gamma$", "H", "N", "$\\Gamma$", "P", "H"])
    ax.set_ylabel("$\\hbar\\omega$ (meV)")
    ax.set_title("magnon dispersion from linear spin-wave theory\n"
                 "on the exchange constants fitted to the model's spiral energies")
    ax.set_xlim(min(x), max(x))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "magnon_dispersion_lswt.png"), dpi=160)
    plt.close(fig)


def fig_topo():
    p = os.path.join(OUT, "phonon_degeneracies.csv")
    if not os.path.exists(p):
        return
    d = np.loadtxt(p, delimiter=",", skiprows=1)
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for b, c in ((0, "C0"), (1, "C1")):
        m = d[:, 3] == b
        ax.semilogy(np.abs(d[m, 4]) + 1e-6, ".", ms=1.2, color=c,
                    label=f"bands {b+1}-{b+2}")
    ax.set_xlabel("grid point index")
    ax.set_ylabel("|gap| (THz, log)")
    ax.set_title("gap distribution over a 31$^3$ q grid")
    ax.legend(markerscale=8, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "phonon_gaps_fd.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig_phonon()
    fig_shift()
    fig_magnon()
    fig_topo()
    print("figures:", sorted(f for f in os.listdir(FIG) if f.endswith(".png")))
