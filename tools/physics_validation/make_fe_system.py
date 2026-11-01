"""Build Fe spin-lattice test cells for stage PV1.

Two sources of structure are supported and they are NOT equivalent:

  * ``dataset``  -- a frame copied verbatim (positions and per-atom |S|) from
                    ``Fe/test.xyz`` and replicated.  In-distribution for the
                    selected model, but the cells are small (16/32 atoms) and
                    heavily distorted, so they are only useful for numerical
                    stability, not for materials physics.
  * ``bcc``      -- perfect bcc Fe built from a lattice constant.  Out of the
                    training distribution; used only as a control and its
                    results are labelled OUT_OF_DISTRIBUTION.

Spin states: ``dataset`` keeps the direction and magnitude found in the file,
``fm`` sets every spin along +z with |S| = spin_length, ``random`` draws
isotropic directions at fixed |S|.

The output is a GPUMD ``model.xyz`` with ``Properties=species:S:1:pos:R:3:vel:R:3:spin:R:3``.
Velocities are left at zero unless ``--temperature`` is given, in which case a
Maxwell-Boltzmann sample with the correct Fe mass is written.
"""

import argparse
import json
import os
import re

import numpy as np

FE_MASS = 55.845  # u
KB_EV = 8.617333262e-5  # eV/K


def read_xyz_frames(path):
    with open(path) as f:
        lines = f.readlines()
    frames = []
    i = 0
    while i < len(lines):
        n = int(lines[i])
        header = lines[i + 1]
        lattice = np.array(
            [float(x) for x in re.search(r'Lattice="([^"]+)"', header).group(1).split()]
        ).reshape(3, 3)
        props = re.search(r"Properties=(\S+)", header).group(1)
        # name:type:ncol triples
        toks = props.split(":")
        names = []
        k = 0
        while k < len(toks):
            names.append((toks[k], toks[k + 1], int(toks[k + 2])))
            k += 3
        rows = [ln.split() for ln in lines[i + 2 : i + 2 + n]]
        cols = {}
        c = 0
        for name, typ, ncol in names:
            cols[name] = (c, c + ncol)
            c += ncol
        pos = np.array([[float(x) for x in r[cols["pos"][0] : cols["pos"][1]]] for r in rows])
        spin = None
        if "spin" in cols:
            spin = np.array(
                [[float(x) for x in r[cols["spin"][0] : cols["spin"][1]]] for r in rows]
            )
        elif "initial_magmoms" in cols and "spin_length" in cols:
            m = np.array(
                [[float(x) for x in r[cols["initial_magmoms"][0] : cols["initial_magmoms"][1]]]
                 for r in rows]
            )
            sl = np.array(
                [float(r[cols["spin_length"][0]]) for r in rows]
            )
            m = m / np.linalg.norm(m, axis=1)[:, None]
            spin = m * sl[:, None]
        vel = None
        if "vel" in cols:
            vel = np.array([[float(x) for x in r[cols["vel"][0] : cols["vel"][1]]] for r in rows])
        frames.append(dict(n=n, lattice=lattice, pos=pos, spin=spin, vel=vel, names=names))
        i += 2 + n
    return frames


def wrap(pos, lattice):
    inv = np.linalg.inv(lattice)
    s = pos @ inv
    s -= np.floor(s)
    return s @ lattice


def replicate(frame, nx, ny, nz):
    L = frame["lattice"]
    pos = frame["pos"]
    spin = frame["spin"]
    P, S = [], []
    for a in range(nx):
        for b in range(ny):
            for c in range(nz):
                shift = a * L[0] + b * L[1] + c * L[2]
                P.append(pos + shift)
                S.append(spin)
    P = np.vstack(P)
    S = np.vstack(S)
    new_L = np.array([nx * L[0], ny * L[1], nz * L[2]])
    return dict(n=len(P), lattice=new_L, pos=wrap(P, new_L), spin=S, vel=None)


def bcc(nx, ny, nz, a):
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    pts = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for b in basis:
                    pts.append((np.array([i, j, k]) + b) * a)
    P = np.array(pts)
    L = np.array([[nx * a, 0, 0], [0, ny * a, 0], [0, 0, nz * a]])
    return dict(n=len(P), lattice=L, pos=P, spin=None, vel=None)


def set_spins(frame, mode, spin_length, seed):
    rng = np.random.default_rng(seed)
    n = frame["n"]
    if mode == "dataset":
        s = frame["spin"].copy()
        mags = np.linalg.norm(s, axis=1)
        return s, mags
    if mode == "fm":
        s = np.zeros((n, 3))
        s[:, 2] = spin_length
        return s, np.full(n, spin_length)
    if mode == "random":
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        s = v * spin_length
        return s, np.full(n, spin_length)
    raise ValueError(mode)


def maxwell_velocities(n, temperature, seed):
    """Maxwell-Boltzmann velocities in the GPUMD unit system.

    GPUMD stores v in A/fs and evaluates the kinetic energy as
    ``0.5 * 103.6427 * m[u] * v^2`` (1 u*A^2/fs^2 = 103.6427 eV).  Solving
    ``0.5 * 103.6427 * m * sigma^2 = 0.5 * kB * T`` per degree of freedom gives
    the sigma below; at 300 K for Fe it is 2.11e-3 A/fs.
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(KB_EV * temperature / (FE_MASS * 103.6427))
    v = rng.normal(scale=sigma, size=(n, 3))
    v -= v.mean(axis=0)  # zero total momentum
    return v


def write_xyz(path, frame, spins, comment_extra=""):
    L = frame["lattice"]
    n = frame["n"]
    vel = frame["vel"]
    with open(path, "w") as f:
        f.write(f"{n}\n")
        lat = " ".join(f"{x:.10f}" for x in L.reshape(-1))
        f.write(
            f'Lattice="{lat}" Properties=species:S:1:pos:R:3'
            + (":vel:R:3" if vel is not None else "")
            + ":spin:R:3"
            + (f" {comment_extra}" if comment_extra else "")
            + "\n"
        )
        for i in range(n):
            row = ["Fe"] + [f"{x:.10f}" for x in frame["pos"][i]]
            if vel is not None:
                row += [f"{x:.10e}" for x in vel[i]]
            row += [f"{x:.10f}" for x in spins[i]]
            f.write(" ".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["dataset", "bcc"], default="dataset")
    ap.add_argument("--xyz", default="Fe/test.xyz")
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--rep", default="2,2,2", help="nx,ny,nz")
    ap.add_argument("--bcc-rep", default="4,4,4")
    ap.add_argument("--a", type=float, default=2.87)
    ap.add_argument("--spin-mode", choices=["dataset", "fm", "random"], default="fm")
    ap.add_argument("--spin-length", type=float, default=2.2)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.source == "dataset":
        frames = read_xyz_frames(args.xyz)
        nx, ny, nz = [int(x) for x in args.rep.split(",")]
        fr = replicate(frames[args.frame], nx, ny, nz)
        origin = f"{args.xyz} frame {args.frame} replicated {nx}x{ny}x{nz}"
    else:
        nx, ny, nz = [int(x) for x in args.bcc_rep.split(",")]
        fr = bcc(nx, ny, nz, args.a)
        origin = f"perfect bcc a={args.a} A, {nx}x{ny}x{nz}"

    spins, mags = set_spins(fr, args.spin_mode, args.spin_length, args.seed)
    if args.temperature > 0:
        fr["vel"] = maxwell_velocities(fr["n"], args.temperature, args.seed)

    mean_bond = np.linalg.det(fr["lattice"]) / fr["n"]
    write_xyz(args.out, fr, spins, comment_extra=f'source="{origin}"')
    info = dict(
        n=fr["n"],
        origin=origin,
        volume=float(abs(np.linalg.det(fr["lattice"]))),
        volume_per_atom=float(abs(mean_bond)),
        spin_mode=args.spin_mode,
        spin_length_mean=float(mags.mean()),
        spin_length_min=float(mags.min()),
        spin_length_max=float(mags.max()),
        temperature=args.temperature,
        seed=args.seed,
    )
    print(json.dumps(info, indent=1))
    with open(os.path.splitext(args.out)[0] + ".meta.json", "w") as f:
        json.dump(info, f, indent=1)


if __name__ == "__main__":
    main()
