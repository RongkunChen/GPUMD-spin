"""Build B20 FeGe supercells with arbitrary spin textures.

The 8-site basis (4 Fe + 4 Ge) and the per-species moment are read straight from
`48/b20/FM001.xyz`, so the structures here are the same crystal the trained
potential saw, just larger.

Used by the skyrmion / helix comparison with the literature.
"""

import numpy as np

A_DEFAULT = 4.8
# |S| in mu_B of the trained model
MOMENT = {"Fe": 2.0, "Ge": 0.11}


def read_basis(path):
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    n = int(lines[0])
    species, pos = [], []
    for l in lines[2: 2 + n]:
        t = l.split()
        species.append(t[0])
        pos.append([float(x) for x in t[1:4]])
    pos = np.array(pos)
    a = A_DEFAULT
    frac = np.mod(pos, a) / a
    # unique basis sites
    key = np.round(frac, 4)
    seen, basis = {}, []
    for k, sp in zip(key, species):
        kk = tuple(k)
        if kk not in seen:
            seen[kk] = sp
            basis.append((sp, k))
    basis.sort(key=lambda x: (x[1][2], x[1][1], x[1][0]))
    return a, basis


def build(nx, ny, nz, spin_func, a=A_DEFAULT, basis=None):
    """spin_func(r_cart, species) -> 3-vector in mu_B (need not be normalised)."""
    lines_species, pos, spin = [], [], []
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                for sp, f in basis:
                    r = np.array([(ix + f[0]) * a, (iy + f[1]) * a, (iz + f[2]) * a])
                    pos.append(r)
                    lines_species.append(sp)
                    v = np.asarray(spin_func(r, sp), float)
                    n = np.linalg.norm(v)
                    m = MOMENT[sp]
                    spin.append(v / n * m if n > 0 else np.zeros(3))
    pos = np.array(pos)
    spin = np.array(spin)
    L = np.array([nx * a, ny * a, nz * a])
    return lines_species, pos, spin, L


def write_xyz(path, species, pos, spin, L):
    n = len(species)
    hdr = ('Lattice="%.8f 0 0 0 %.8f 0 0 0 %.8f" '
           'Properties=species:S:1:pos:R:3:spin:R:3 pbc="T T T"' % (L[0], L[1], L[2]))
    with open(path, "w") as f:
        f.write("%d\n%s\n" % (n, hdr))
        for sp, p, s in zip(species, pos, spin):
            f.write("%s %.10f %.10f %.10f %.9f %.9f %.9f\n" % (sp, *p, *s))


def fm(spin_axis=(0, 0, 1)):
    ax = np.asarray(spin_axis, float)
    ax /= np.linalg.norm(ax)
    return lambda r, sp: ax


def helix(q_vec, plane=(1, 0, 0), phase=0.0):
    """S = cos(q.r+phi) e1 + sin(q.r+phi) e2, with e1 x e2 along the propagation.

    NOTE: this function is degenerate in the sign of q.  `e2 = q_hat x e1` flips
    with q, and so does `sin(q.r)`, so `helix(-q)` returns the *same* texture as
    `helix(q)`: a scan over +-q can never detect a chiral energy splitting here.
    Use `dmi_scan.spiral(direction, q, polarisation)` (fixed direction, signed q)
    for any chirality work; that is what the FeGe comparison uses.
    """
    q = np.asarray(q_vec, float)
    e1 = np.asarray(plane, float)
    e1 = e1 - q * (e1 @ q) / max(q @ q, 1e-30)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(q / max(np.linalg.norm(q), 1e-30), e1)
    e2 /= np.linalg.norm(e2)

    def f(r, sp):
        ph = q @ r + phase
        return np.cos(ph) * e1 + np.sin(ph) * e2

    return f


def skyrmion(center, radius, helicity=0.0, q_vec=(0, 0, 1), core_sign=1, box=None):
    """Neel/Bloch skyrmion ansatz: theta(r) from pi at the core to 0 outside.

    `box` enables the minimum-image convention, which is what makes the texture
    periodic: without it the atoms near the cell boundary sit at a distance of
    up to L*sqrt(2)/2 from the core and the profile is stretched.
    """
    c = np.asarray(center, float)
    e3 = np.asarray(q_vec, float)
    e3 /= np.linalg.norm(e3)

    def f(r, sp):
        d = r - c
        if box is not None:
            for k in range(3):
                d[k] -= box[k] * np.round(d[k] / box[k])
        # only the component perpendicular to e3 matters
        d = d - e3 * (d @ e3)
        rho = np.linalg.norm(d)
        phi = np.arctan2(d[1], d[0]) if abs(e3[2]) > 0.5 else np.arctan2(d[2], d[0])
        theta = 2.0 * np.arctan(radius / max(rho, 1e-6)) * core_sign
        # rotate e3 by theta towards the radial direction
        er = np.array([np.cos(phi), np.sin(phi), 0.0])
        if abs(e3[2]) < 0.5:
            er = np.array([np.cos(phi), 0.0, np.sin(phi)])
        ephi = np.cross(e3, er)
        return (np.cos(theta) * e3 +
                np.sin(theta) * (np.cos(helicity) * er + np.sin(helicity) * ephi))

    return f


def planar_grid(pos, spin, L, sublayer, a=A_DEFAULT, n_basis=8):
    """Pick one z-sublayer (one basis site) and index it on the (nx, ny) grid.

    A z-sublayer is a single square lattice of one basis atom, so the topology
    can be evaluated on a proper quad mesh.  The sublattice is selected by the
    atom index (the file order is cell-major, basis-minor), which survives a
    relaxation; selecting by z would not, because two basis sites can be only
    0.11 A apart.
    """
    n = len(pos)
    sel = np.arange(n) % n_basis == sublayer
    p = pos[sel][:, :2]
    s = spin[sel]
    s = s / np.linalg.norm(s, axis=1)[:, None]
    nx = int(round(L[0] / a))
    ny = int(round(L[1] / a))
    ix = np.mod(np.round(p[:, 0] / a).astype(int), nx)
    iy = np.mod(np.round(p[:, 1] / a).astype(int), ny)
    grid = np.full((nx, ny, 3), np.nan)
    grid[ix, iy] = s
    if np.isnan(grid).any():
        raise RuntimeError("sublayer is not a complete square lattice")
    return grid, p, s, (nx, ny)


def topological_charge(pos, spin, L, a=A_DEFAULT, sublayer=0):
    """Q = (1/4 pi) sum of the solid angles over the quad mesh of one sublattice."""
    grid, _, _, (nx, ny) = planar_grid(pos, spin, L, sublayer, a)
    tot = 0.0
    for i in range(nx):
        for j in range(ny):
            m00 = grid[i, j]
            m10 = grid[(i + 1) % nx, j]
            m01 = grid[i, (j + 1) % ny]
            m11 = grid[(i + 1) % nx, (j + 1) % ny]
            tot += _solid_angle(m00, m10, m11)
            tot += _solid_angle(m00, m11, m01)
    return tot / (4 * np.pi)


def radial_profile(pos, spin, L, center, dr=1.0, rmax=None, a=A_DEFAULT, sublayer=0):
    """theta(r) = angle between the spin and +z, binned by in-plane distance."""
    grid, p, s, (nx, ny) = planar_grid(pos, spin, L, sublayer, a)
    c = np.asarray(center, float)[:2]
    d = p - c
    d[:, 0] -= L[0] * np.round(d[:, 0] / L[0])
    d[:, 1] -= L[1] * np.round(d[:, 1] / L[1])
    r = np.linalg.norm(d, axis=1)
    if rmax is None:
        rmax = 0.5 * min(L[0], L[1])
    nb = int(rmax / dr)
    rs, ts = [], []
    for k in range(nb):
        m = (r >= k * dr) & (r < (k + 1) * dr)
        if m.sum() == 0:
            continue
        cz = np.clip(s[m][:, 2], -1, 1)
        rs.append((k + 0.5) * dr)
        ts.append(np.degrees(np.arccos(cz.mean())))
    return np.array(rs), np.array(ts)


def skyrmion_radius(pos, spin, L, center, dr=1.0, sublayer=0):
    """Radius where <theta(r)> crosses 90 degrees."""
    r, t = radial_profile(pos, spin, L, center, dr=dr, sublayer=sublayer)
    if len(r) < 2:
        return float("nan")
    for k in range(len(r) - 1):
        if t[k] >= 90.0 > t[k + 1]:
            f = (t[k] - 90.0) / (t[k] - t[k + 1])
            return float(r[k] + f * (r[k + 1] - r[k]))
    return float("nan")


def _solid_angle(s1, s2, s3):
    num = s1 @ np.cross(s2, s3)
    den = 1.0 + s1 @ s2 + s2 @ s3 + s3 @ s1
    return 2.0 * np.arctan2(num, den)
