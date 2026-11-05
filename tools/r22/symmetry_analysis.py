#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R22-NonHeat independent symmetry / Hamiltonian reference (numpy-only).

SCOPE / HONESTY STATEMENT
-------------------------
This module provides INDEPENDENT ANALYTIC REFERENCES derived directly from
spin Hamiltonians and from the algebra of spin-dependent descriptors.  It is
CPU-only, numpy-only (no scipy), and does NOT read, call, or validate ANY
GPUMD CUDA kernel.  It never imports a kernel, never parses a nep.txt, and
never claims that a GPUMD result is validated.  Every reference here is a
closed-form expression or a finite-difference measurement performed on that
closed form.

The module exists so that physics statements can be checked against an
independent source of truth.  It deliberately separates:
  * analytic Hamiltonians (Heisenberg, MAE, DMI, scalar chirality, a coupled
    U(r,S) test model, a continuum/lattice 180-degree wall);
  * symmetry algebra (time reversal, space inversion, spin rotation, joint
    lattice-spin rotation, chirality reversal, atom permutation);
  * a feasibility checklist for Berry curvature that returns an honest
    BLOCKED / PARTIAL status and NEVER a quantitative Chern number.

STATUS VOCABULARY used by the companion driver (not by this module's
internals): PASS / FAIL / SKIP / BLOCKED / NOT_RUN / PARTIAL / NOT_RESOLVED.
This module returns plain data; the driver assigns statuses.

CONVENTIONS
-----------
  * Spins are classical 3-vectors; when constrained they are UNIT vectors.
    Components are magnetic moments (mu_B by convention).
  * Positions / lengths are in Angstrom, energies in eV.
  * 'pairs' is an (M,2) integer array of UNORDERED atom indices (i<j); each
    unordered bond appears once.  Ordered-pair sums are handled explicitly
    where needed and are labelled.
  * A 'field' is B_i = -dU/dS_i.  A 'force' is F_i = -dU/dr_i.  The
    'spin gradient' is G_i = +dU/dS_i.  These sign conventions are stated
    again in every function docstring.
  * All cross products are the standard right-handed product.
"""

import numpy as np

# small shared helpers (private)
def _unit(v):
    """Return the unit vector of v; zero vectors map to zero."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0.0:
        return np.zeros_like(v)
    return v / n


def _rotmat(axis, angle):
    """Right-handed rotation matrix (Rodrigues) about a unit axis by angle."""
    k = _unit(np.asarray(axis, dtype=np.float64))
    kx = np.array([[0.0, -k[2], k[1]],
                   [k[2], 0.0, -k[0]],
                   [-k[1], k[0], 0.0]], dtype=np.float64)
    i3 = np.eye(3)
    c = np.cos(angle)
    s = np.sin(angle)
    return c * i3 + (1.0 - c) * np.outer(k, k) + s * kx


def _rand_rot(rng):
    """A uniformly distributed proper rotation (det=+1)."""
    m = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(m)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0.0:
        q[:, 0] *= -1.0
    return q


def _as_spins(spins):
    return np.asarray(spins, dtype=np.float64).reshape(-1, 3)


def _as_pairs(pairs):
    p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return p


# 1. Isotropic Heisenberg exchange
def heisenberg_energy(spins, pairs, J):
    """Isotropic Heisenberg exchange energy.

    PHYSICAL_MODEL
        H = - sum_{<ij>} J_ij (S_i . S_j), summed once per UNORDERED bond.
        This is the "state sign convention": a POSITIVE J_ij favours PARALLEL
        spins (ferromagnetic), because the energy -J (S_i.S_j) is then most
        negative.  A negative J_ij is antiferromagnetic.

    ASSUMPTIONS
        Classical spins, no SOC, no spin-orbit coupling, J_ij independent of
        position unless the caller supplies a position-dependent J elsewhere.
        Spins need not be unit vectors here.

    EQUATIONS
        H = - sum_{m=0}^{M-1} J_m (S_{p_m} . S_{q_m}).

    UNITS
        J in eV (per |S|^2), spins in mu_B, H in eV.  With dimensionless
        O(1) parameters the numbers are in the corresponding reduced units.

    BOUNDARY_CONDITIONS
        None: an explicit bond list defines the topology.  For a periodic
        uniform system the caller supplies the periodic image bonds.

    LIMITATIONS
        Pure isotropic exchange: cannot express anisotropy, DMI, chirality,
        or any spin-lattice-induced change of J.  It is T-even and P-even and
        invariant under an INDEPENDENT global spin rotation (no-SOC symmetry).
    """
    S = _as_spins(spins)
    P = _as_pairs(pairs)
    Jm = np.asarray(J, dtype=np.float64)
    if Jm.ndim == 0:
        Jm = np.full(P.shape[0], float(Jm))
    dots = np.einsum('ij,ij->i', S[P[:, 0]], S[P[:, 1]])
    return float(-np.sum(Jm * dots))


def heisenberg_field(spins, pairs, J):
    """Effective field B_i = -dH/dS_i for the Heisenberg Hamiltonian.

    PHYSICAL_MODEL
        Same H = - sum_{<ij>} J_ij S_i.S_j as heisenberg_energy.

    ASSUMPTIONS
        J_ij does not depend on S (linear-in-S derivative only).

    EQUATIONS
        dH/dS_i = - sum_{j: (i,j) in bonds} J_ij S_j,
        B_i = -dH/dS_i = + sum_{j} J_ij S_j.

    UNITS
        B in eV/mu_B if J in eV/mu_B^2 and S in mu_B.

    BOUNDARY_CONDITIONS
        Bond list as given; no self term.

    LIMITATIONS
        No tangential projection is applied: this is the FULL unconstrained
        derivative.  For fixed-|S| dynamics only the tangential part
        S_i x (B_i x S_i) enters the equation of motion.
    """
    S = _as_spins(spins)
    P = _as_pairs(pairs)
    N = S.shape[0]
    Jm = np.asarray(J, dtype=np.float64)
    if Jm.ndim == 0:
        Jm = np.full(P.shape[0], float(Jm))
    B = np.zeros((N, 3), dtype=np.float64)
    for m in range(P.shape[0]):
        p, q = P[m, 0], P[m, 1]
        B[p] += Jm[m] * S[q]
        B[q] += Jm[m] * S[p]
    return B


def _heisenberg_pair_force(S, positions, pairs, dJ_dr):
    """Internal: forces from a position-dependent J(d) on each bond."""
    P = _as_pairs(pairs)
    N = S.shape[0]
    F = np.zeros((N, 3), dtype=np.float64)
    if dJ_dr is None:
        return F
    g = np.asarray(dJ_dr, dtype=np.float64)
    if g.ndim == 0:
        g = np.full(P.shape[0], float(g))
    for m in range(P.shape[0]):
        p, q = P[m, 0], P[m, 1]
        u = positions[q] - positions[p]
        d = np.linalg.norm(u)
        if d == 0.0:
            continue
        rhat = u / d
        dot = float(np.dot(S[p], S[q]))
        # H = -J(d) (S_p.S_q); F_p = -dH/dr_p = -J'(d)(S_p.S_q) rhat; F_q=-F_p
        F[p] += -g[m] * dot * rhat
        F[q] += +g[m] * dot * rhat
    return F


def heisenberg_force(spins, positions, pairs, J, dJ_dr=None):
    """Force F_i = -dH/dr_i for a position-dependent exchange J(r_ij).

    PHYSICAL_MODEL
        H = - sum_{<ij>} J(r_ij) S_i.S_j.  For a CONSTANT J (dJ_dr None or 0)
        the exchange carries no position dependence and every force is
        EXACTLY zero.

    ASSUMPTIONS
        J is a smooth function of the bond distance; dJ_dr = dJ/dd evaluated
        at each bond.  Spins are held fixed while differentiating positions.

    EQUATIONS
        With rhat = (r_q - r_p)/d and F = -dH/dr,
        F_p = -J'(d) (S_p.S_q) rhat,  F_q = +J'(d) (S_p.S_q) rhat,
        so F_p = -F_q (Newton's third law) and sum_i F_i = 0 exactly.

    UNITS
        F in eV/Angstrom if J' in eV/Angstrom.

    BOUNDARY_CONDITIONS
        Explicit bond list.  For a periodic system supply minimum-image bonds.

    LIMITATIONS
        The virial returned by heisenberg_force_virial is the standard
        sum_i r_i (x) F_i and assumes zero net force (true here).
    """
    S = _as_spins(spins)
    R = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    return _heisenberg_pair_force(S, R, pairs, dJ_dr)


def heisenberg_virial(spins, positions, pairs, J, dJ_dr=None):
    """Virial tensor W = sum_i r_i (x) F_i (3x3) for the exchange model.

    PHYSICAL_MODEL / ASSUMPTIONS / EQUATIONS / UNITS / BOUNDARY_CONDITIONS
        Same as heisenberg_force.  W_ab = sum_i r_{i,a} F_{i,b}.  Also
        returns the 9-component GPUMD layout
        [xx, yy, zz, xy, xz, yz, yx, zx, zy].

    LIMITATIONS
        Uses raw Cartesian positions; for a triclinic cell the slab/virial
        geometry needs the proper cell Jacobian (outside this reference).
    """
    F = heisenberg_force(spins, positions, pairs, J, dJ_dr=dJ_dr)
    R = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    W = R.T @ F  # W[a,b] = sum_i R[i,a] F[i,b]
    W9 = np.array([W[0, 0], W[1, 1], W[2, 2],
                   W[0, 1], W[0, 2], W[1, 2],
                   W[1, 0], W[2, 0], W[2, 1]], dtype=np.float64)
    return W, W9


def heisenberg_force_virial(spins, positions, pairs, J, dJ_dr=None):
    """Combined helper: forces, virial tensor, 9-component virial, net force.

    PHYSICAL_MODEL / ASSUMPTIONS / EQUATIONS / UNITS
        See heisenberg_force and heisenberg_virial.  For a CONSTANT J the net
        force and the virial are exactly zero.

    BOUNDARY_CONDITIONS
        Explicit bond list; net force is zero by Newton's third law for any
        dJ_dr, so a periodic uniform system has zero total force.

    LIMITATIONS
        Reference only; never a GPU-kernel check.
    """
    F = heisenberg_force(spins, positions, pairs, J, dJ_dr=dJ_dr)
    W, W9 = heisenberg_virial(spins, positions, pairs, J, dJ_dr=dJ_dr)
    return {"force": F, "virial": W, "virial_9": W9,
            "total_force": F.sum(axis=0)}


# 2. Uniaxial magnetocrystalline anisotropy (MAE)
def mae_energy(spins, easy_axis, K):
    """Uniaxial magnetocrystalline anisotropy energy.

    PHYSICAL_MODEL
        E = -K sum_i (S_i . e)^2, where e is the UNIT easy axis and K is the
        anisotropy constant.  CONVENTION: K > 0 makes e the EASY axis (energy
        is minimised by S_i parallel or antiparallel to e); K < 0 makes the
        plane perpendicular to e the EASY PLANE (e is then a HARD axis).

    ASSUMPTIONS
        Single-ion, onsite, rank-2 (quadrupolar) anisotropy.  No exchange,
        no dipole, no SOC-derived higher-rank terms.  The axis e is a FIXED
        lattice-locked direction (this is an SOC-like term).

    EQUATIONS
        E = -K sum_i (S_i . e)^2;  B_i = -dE/dS_i = +2K (S_i.e) e.

    UNITS
        K in eV/mu_B^2, S in mu_B, E in eV.

    BOUNDARY_CONDITIONS
        None (onsite sum over atoms).

    LIMITATIONS
        This is the l=2 uniaxial (quadrupolar) form only.  A cubic rank-4
        term K1 (Sx^4+Sy^4+Sz^4), the leading magnetocrystalline anisotropy
        of bcc Fe, is NOT representable by (S.e)^2 and is absent here.
        Because e is lattice-locked, E is invariant under a JOINT rotation of
        the lattice and the spins, but NOT under an independent global spin
        rotation at FIXED lattice (that is the SOC case).
    """
    S = _as_spins(spins)
    e = _unit(easy_axis)
    Kf = float(K)
    proj = S @ e
    return float(-Kf * np.sum(proj ** 2))


def mae_field(spins, easy_axis, K):
    """Analytic MAE field B_i = -dE/dS_i = 2K (S_i.e) e.

    PHYSICAL_MODEL / ASSUMPTIONS / EQUATIONS / UNITS
        Same as mae_energy.  B_i = +2 K (S_i . e) e is a FULL (unconstrained)
        derivative; its tangential part drives fixed-|S| dynamics.

    BOUNDARY_CONDITIONS
        Onsite, no neighbours.

    LIMITATIONS
        Longitudinal component is retained; project with P_i = I - S_i S_i^T
        for constrained dynamics.
    """
    S = _as_spins(spins)
    e = _unit(easy_axis)
    Kf = float(K)
    proj = (S @ e)[:, None]
    return 2.0 * Kf * proj * e[None, :]


def mae_scan_energy(angles, easy_axis, K, spin_magnitude=1.0):
    """MAE energy E(theta) for a single spin tilted by theta from the axis.

    PHYSICAL_MODEL
        A single unit spin S(theta) = |S| (cos(theta) e + sin(theta) e_perp),
        where e is the easy axis and e_perp is a fixed perpendicular direction.
        Then E(theta) = -K |S|^2 cos^2(theta).

    ASSUMPTIONS
        One spin, fixed magnitude, uniaxial anisotropy of mae_energy.

    EQUATIONS
        E(theta) = -K |S|^2 cos^2(theta);
        E(0) = -K|S|^2 (extremum along e), E(pi/2) = 0 (extremum in plane).
        For K>0, E(0)<E(pi/2) so e is EASY.  For K<0 the inequality reverses
        and e is HARD (easy plane).

    UNITS
        Same as mae_energy; angles in radians.

    BOUNDARY_CONDITIONS
        theta in [0, 2pi); continuous periodic function of theta.

    LIMITATIONS
        Single-spin scan; no exchange, no dipole.  Easy/hard identification
        assumes the SIN^2 (quadrupolar) form; a rank-4 term would add a
        cos(4 theta) component not present here.
    """
    a = np.asarray(angles, dtype=np.float64)
    e = _unit(easy_axis)
    # a fixed perpendicular direction used only for the parameterisation
    helper = np.array([1.0, 0.0, 0.0]) if abs(e[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    eperp = _unit(helper - np.dot(helper, e) * e)
    S = spin_magnitude * (np.cos(a)[:, None] * e[None, :]
                          + np.sin(a)[:, None] * eperp[None, :])
    proj = S @ e
    return np.asarray(-float(K) * proj ** 2, dtype=np.float64)


# 3. Dzyaloshinskii-Moriya interaction (DMI)
def dmi_energy(spins, pairs, D_vectors):
    """DMI energy for UNORDERED bonds: H_DMI = sum_{i<j} D_ij . (S_i x S_j).

    PHYSICAL_MODEL
        Antisymmetric exchange.  The Moriya vector D_ij is a structural
        (spin-orbit derived) axial vector.  The energy is a sum over UNORDERED
        bonds, each counted once.

    ASSUMPTIONS
        D_vectors[m] is the D for the m-th unordered pair (i<j).  Spins are
        classical 3-vectors.

    EQUATIONS
        H_DMI = sum_m D_m . (S_{p_m} x S_{q_m}).

    UNITS
        D in eV/(mu_B^2), S in mu_B, H in eV.

    BOUNDARY_CONDITIONS
        Explicit unordered bond list.  For a consistent ORDERED-pair sum see
        dmi_energy_ordered and dmi_antisymmetry_check.

    LIMITATIONS
        TIME REVERSAL: S -> -S leaves S_i x S_j INVARIANT (two minus signs
        cancel), so the DMI energy term is TIME-REVERSAL EVEN when D is held
        fixed as a Hamiltonian parameter.  The D vector ITSELF is odd under
        time reversal (it is proportional to an orbital current / angular
        momentum); a full microscopic TR transformation sends D -> -D while
        S -> -S.  SPACE INVERSION: D is an AXIAL vector (P-even) while
        S_i x S_j is P-even, so an isolated D.DMI term with a FIXED D is
        P-even; however a NONZERO D is forbidden by symmetry on any bond that
        possesses an inversion centre (Moriya's rule).  For a bond-parallel
        D_ij = D(r) rhat_ij (as in the SDM descriptor), the term is P-ODD
        because rhat is polar.  A general (interfacial / Cnv) D_ij with
        out-of-plane or bond-perpendicular components is NOT restricted to
        the bond-parallel form.
    """
    S = _as_spins(spins)
    P = _as_pairs(pairs)
    D = np.asarray(D_vectors, dtype=np.float64).reshape(-1, 3)
    c = np.cross(S[P[:, 0]], S[P[:, 1]])
    return float(np.sum(D * c))


def dmi_field(spins, pairs, D_vectors):
    """DMI field B_i = -dH_DMI/dS_i.

    PHYSICAL_MODEL
        Same H_DMI = sum_{i<j} D_ij . (S_i x S_j) as dmi_energy.

    ASSUMPTIONS
        D_ij is fixed (no position dependence) when differentiating S.

    EQUATIONS
        Using D.(S_p x S_q) = (D x S_p).S_q = (S_q x D).S_p,
        dH/dS_p = S_q x D_ij,  dH/dS_q = D_ij x S_p,
        B_p = -(S_q x D_ij) = D_ij x S_q,  B_q = -(D_ij x S_p) = S_p x D_ij.

    UNITS
        B in eV/mu_B.

    BOUNDARY_CONDITIONS
        Unordered bond list; each bond contributes to both of its atoms.

    LIMITATIONS
        Full unconstrained derivative; constrain to the tangent plane for
        fixed-|S| dynamics.
    """
    S = _as_spins(spins)
    P = _as_pairs(pairs)
    D = np.asarray(D_vectors, dtype=np.float64).reshape(-1, 3)
    N = S.shape[0]
    dHdS = np.zeros((N, 3), dtype=np.float64)
    for m in range(P.shape[0]):
        p, q = P[m, 0], P[m, 1]
        dHdS[p] += np.cross(S[q], D[m])   # S_q x D
        dHdS[q] += np.cross(D[m], S[p])   # D x S_p
    return -dHdS


def dmi_energy_ordered(spins, ordered_pairs, D_vectors, factor=0.5):
    """DMI energy from an ORDERED-pair sum (both (i,j) and (j,i) present).

    PHYSICAL_MODEL / ASSUMPTIONS
        Sum over ordered pairs with an explicit 1/2 (or other) prefactor.

    EQUATIONS
        (1/2) sum_{(i,j) ordered} D_ij . (S_i x S_j)
            = sum_{i<j} (D_ij - D_ji)/2 . (S_i x S_j).
        This equals the unordered sum sum_{i<j} D_ij . (S_i x S_j) IF AND
        ONLY IF D_ji = -D_ij (D_ij - D_ji = 2 D_ij).  Antisymmetry of the
        ordered-pair D tensor is therefore REQUIRED for a consistent sum.

    UNITS / BOUNDARY_CONDITIONS
        As dmi_energy; ordered bond list.

    LIMITATIONS
        Demonstration of the antisymmetry requirement only.
    """
    S = _as_spins(spins)
    P = _as_pairs(ordered_pairs)
    D = np.asarray(D_vectors, dtype=np.float64).reshape(-1, 3)
    c = np.cross(S[P[:, 0]], S[P[:, 1]])
    return float(factor * np.sum(D * c))


def dmi_antisymmetry_check(ordered_pairs, D_vectors, tol=1e-10):
    """Check D_ij = -D_ji for every unordered bond in an ordered-pair list.

    PHYSICAL_MODEL / ASSUMPTIONS / EQUATIONS
        For an ordered list, group (p,q) with (q,p); require D_(p,q) =
        -D_(q,p).  Returns the maximum violation max ||D_ij + D_ji||.

    UNITS / BOUNDARY_CONDITIONS
        Euclidean norm of D; explicit ordered pair list.

    LIMITATIONS
        Purely algebraic check on the supplied D vectors; does not know the
        underlying lattice symmetry (Moriya rules are separate).
    """
    P = _as_pairs(ordered_pairs)
    D = np.asarray(D_vectors, dtype=np.float64).reshape(-1, 3)
    lookup = {}
    for m in range(P.shape[0]):
        lookup[(int(P[m, 0]), int(P[m, 1]))] = D[m]
    worst = 0.0
    seen = set()
    for (p, q), Dpq in lookup.items():
        if (p, q) in seen or (q, p) in seen:
            continue
        seen.add((p, q))
        seen.add((q, p))
        Dqp = lookup.get((q, p))
        if Dqp is None:
            worst = float("inf")
            continue
        worst = max(worst, float(np.linalg.norm(Dpq + Dqp)))
    return {"all_antisymmetric": bool(worst <= tol),
            "max_violation": worst,
            "explanation": ("D_ij = -D_ji is required: the ordered-pair sum "
                            "(1/2) sum D_ij.(S_i x S_j) reproduces the "
                            "unordered bond sum only under this condition "
                            "(D_ij - D_ji = 2 D_ij).")}


def dmi_energy_bond_parallel(spins, positions, pairs, D0):
    """Bond-parallel DMI: H = sum_{i<j} D0 rhat_ij . (S_i x S_j).

    PHYSICAL_MODEL
        D_ij = D0 * rhat_ij (the ONLY DMI component the SDM descriptor can
        express).  rhat is POLAR so the term is P-ODD (pseudoscalar-like).

    ASSUMPTIONS / EQUATIONS / UNITS
        rhat = (r_j - r_i)/|r_j - r_i|.  Position dependence gives a FORCE.

    BOUNDARY_CONDITIONS
        Explicit unordered bond list.

    LIMITATIONS
        Cannot express out-of-plane or bond-perpendicular D components.
    """
    S = _as_spins(spins)
    R = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    P = _as_pairs(pairs)
    u = R[P[:, 1]] - R[P[:, 0]]
    d = np.linalg.norm(u, axis=1)
    d = np.where(d == 0.0, 1.0, d)
    rhat = u / d[:, None]
    c = np.cross(S[P[:, 0]], S[P[:, 1]])
    return float(D0 * np.sum(rhat * c))


def dmi_force_bond_parallel(spins, positions, pairs, D0):
    """Analytic force F = -dH/dr for the bond-parallel DMI.

    PHYSICAL_MODEL / ASSUMPTIONS
        H = D0 (u.c)/d with u = r_j-r_i, d=|u|, c = S_i x S_j fixed.

    EQUATIONS
        Let P_perp = I - rhat rhat^T.  Then
        F_i = +(D0/d) P_perp c,  F_j = -(D0/d) P_perp c (Newton's 3rd law).

    UNITS / BOUNDARY_CONDITIONS
        F in eV/Angstrom; explicit unordered bond list.

    LIMITATIONS
        Reference for the r-dependent DMI force only.
    """
    S = _as_spins(spins)
    R = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    P = _as_pairs(pairs)
    N = S.shape[0]
    F = np.zeros((N, 3), dtype=np.float64)
    for m in range(P.shape[0]):
        p, q = P[m, 0], P[m, 1]
        u = R[q] - R[p]
        d = np.linalg.norm(u)
        if d == 0.0:
            continue
        rhat = u / d
        c = np.cross(S[p], S[q])
        perp = c - np.dot(rhat, c) * rhat
        F[p] += (D0 / d) * perp
        F[q] -= (D0 / d) * perp
    return F


def scalar_spin_chirality(S_i, S_j, S_k):
    """Scalar spin chirality chi = S_i . (S_j x S_k).

    PHYSICAL_MODEL
        The oriented volume (triple product) of three spins, equal to
        det[S_i, S_j, S_k].  It is the lattice-independent chirality order
        parameter used for noncoplanar spin textures.

    ASSUMPTIONS
        Classical 3-vectors.  The ATOM-ORDER / SIGN CONVENTION is that the
        arguments are the ORDERED triple (i, j, k): chi changes sign under a
        TRANSPOSITION of any two arguments (e.g. swapping j and k gives
        -chi) and is invariant under CYCLIC permutations.

    EQUATIONS
        chi = S_i . (S_j x S_k) = det[[S_i],[S_j],[S_k]].
        TIME REVERSAL (S -> -S for all three): chi -> (-1)^3 chi = -chi,
        i.e. ODD.  SPACE INVERSION / chirality reversal: chi is a
        PSEUDOSCALAR; for a texture whose site ordering defines an oriented
        triangle, inversion (or a mirror) reverses the triangle orientation
        and flips the sign of the chirality.  A NONZERO net chirality therefore
        requires an inversion-breaking (chiral) spin texture.
        DEGENERATE COPLANAR LIMIT: if all three spins lie in one plane then
        S_j x S_k is perpendicular to that plane while S_i is in it, hence
        chi = 0.  NONCOPLANAR EXAMPLE: S_i=x, S_j=y, S_k=z gives chi = +1;
        reversing one spin (or swapping two) gives -1.

    UNITS
        mu_B^3 (or dimensionless for unit spins).

    BOUNDARY_CONDITIONS
        Ordered triple of site labels; no boundary conditions.

    LIMITATIONS
        A single-triangle scalar; a lattice chirality observable is a SUM of
        such triples with a consistent orientation convention.  Time-reversal
        oddness here is distinct from the DMI energy term (which is T-even).
    """
    Si = np.asarray(S_i, dtype=np.float64).reshape(3)
    Sj = np.asarray(S_j, dtype=np.float64).reshape(3)
    Sk = np.asarray(S_k, dtype=np.float64).reshape(3)
    return float(np.dot(Si, np.cross(Sj, Sk)))


# 5. Mixed derivative d2U/(dr dS) and the coupled test Hamiltonian
def _coupled_eval(positions, spins, pairs, prm):
    """Evaluate U, F=-dU/dr, G=dU/dS for the coupled test model."""
    R = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    S = np.asarray(spins, dtype=np.float64).reshape(-1, 3)
    P = _as_pairs(pairs)
    N = S.shape[0]
    J0 = float(prm["J0"])
    g = float(prm["g"])          # dJ/dd (constant for the linear J(d))
    d0 = float(prm["d0"])
    K0 = float(prm["K0"])
    k1 = float(prm["k1"])
    w = np.asarray(prm["w"], dtype=np.float64).reshape(3)
    e = np.asarray(prm["e"], dtype=np.float64).reshape(3)
    e = _unit(e)

    p = P[:, 0]
    q = P[:, 1]
    u = R[q] - R[p]
    d = np.linalg.norm(u, axis=1)
    d = np.where(d == 0.0, 1.0, d)
    rhat = u / d[:, None]
    Sp = S[p]
    Sq = S[q]
    dot = np.einsum('ij,ij->i', Sp, Sq)
    J = J0 + g * (d - d0)
    Ki = K0 + k1 * (R @ w)
    Se = S @ e

    U = float(np.sum(J * dot) + np.sum(Ki * Se ** 2))

    # gradS = dU/dS
    G = np.zeros((N, 3), dtype=np.float64)
    np.add.at(G, p, J[:, None] * Sq)
    np.add.at(G, q, J[:, None] * Sp)
    G += (2.0 * Ki * Se)[:, None] * e[None, :]

    # force F = -dU/dr
    F = np.zeros((N, 3), dtype=np.float64)
    coeff = (g * dot)[:, None] * rhat          # (M,3)
    np.add.at(F, p, coeff)                     # F_p += g*dot*rhat
    np.add.at(F, q, -coeff)                    # F_q -= g*dot*rhat
    F -= (k1 * Se ** 2)[:, None] * w[None, :]  # F_i -= k1 (S_i.e)^2 w

    return {"energy": U, "force": F, "gradS": G}


def _coupled_mixed_C(positions, spins, pairs, prm):
    """Closed-form C[i,a,j,b] = d2U/(dr_{i,a} dS_{j,b})."""
    R = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    S = np.asarray(spins, dtype=np.float64).reshape(-1, 3)
    P = _as_pairs(pairs)
    N = S.shape[0]
    g = float(prm["g"])
    k1 = float(prm["k1"])
    w = np.asarray(prm["w"], dtype=np.float64).reshape(3)
    e = _unit(np.asarray(prm["e"], dtype=np.float64).reshape(3))
    C = np.zeros((N, 3, N, 3), dtype=np.float64)
    for m in range(P.shape[0]):
        p, q = int(P[m, 0]), int(P[m, 1])
        u = R[q] - R[p]
        d = np.linalg.norm(u)
        if d == 0.0:
            continue
        rhat = u / d
        cp = -g * rhat           # dJ/dd * dd/dr_p  (3-vector over a)
        cq = +g * rhat           # dJ/dd * dd/dr_q
        # C[i,a,j,:] += c_i[a] * S_other(j) ; other(p)=q, other(q)=p
        C[p, :, p, :] += np.outer(cp, S[q])
        C[p, :, q, :] += np.outer(cp, S[p])
        C[q, :, p, :] += np.outer(cq, S[q])
        C[q, :, q, :] += np.outer(cq, S[p])
    # anisotropy: C[i,a,i,:] += 2 k1 w_a (S_i.e) e_b
    Se = S @ e
    for i in range(N):
        C[i, :, i, :] += 2.0 * k1 * float(Se[i]) * np.outer(w, e)
    return C


def coupled_hamiltonian(positions, spins, pairs, params):
    """Analytic coupled test Hamiltonian U(r,S) with a CLOSED-FORM mixed C.

    PHYSICAL_MODEL
        A smooth synthetic spin-lattice model (NOT a physical potential):
            U = sum_{<ij>} J(r_ij) (S_i.S_j) + sum_i K_i(r) (S_i.e)^2,
            J(d) = J0 + g (d - d0)   (so dJ/dd = g is constant),
            K_i  = K0 + k1 (w . r_i),
        where e is a fixed easy-axis unit vector and w a fixed direction.
        The linear-in-distance J is chosen so that the mixed derivative has
        an exact closed form.  NOTE the PLUS sign on the exchange here; it is
        a generic scalar test function, distinct from the Heisenberg
        convention H = -sum J S_i.S_j used elsewhere in this module.

    ASSUMPTIONS
        Smooth U, no constraint on |S| in the analytic derivatives.  The
        companion mixed_derivative_reference probes TANGENTIAL directions.

    EQUATIONS
        F_i = -dU/dr_i ; G_i = dU/dS_i ;
        C[i,a,j,b] = d2U/(dr_{i,a} dS_{j,b}) = dG_{j,b}/dr_{i,a}
                   = -dF_{i,a}/dS_{j,b}   (Schwarz / Maxwell reciprocity).
        Closed form: for each bond (p,q) with rhat=(r_q-r_p)/d,
            C[p,:,p,:] += outer(-g rhat, S_q),
            C[p,:,q,:] += outer(-g rhat, S_p),
            C[q,:,p,:] += outer(+g rhat, S_q),
            C[q,:,q,:] += outer(+g rhat, S_p),
        plus the onsite anisotropy C[i,:,i,:] += 2 k1 (S_i.e) outer(w,e).

    UNITS
        Reduced units: r in Angstrom, S in mu_B, U in eV; C in
        eV / (Angstrom * mu_B).  Signs follow F=-dU/dr and G=dU/dS.

    BOUNDARY_CONDITIONS
        Explicit unordered bond list; open (non-periodic) reference.

    LIMITATIONS
        Synthetic linear J(d); a realistic J(d) adds higher derivatives that
        do not change C (first order in r) but would change higher checks.
        This is an INDEPENDENT reference, not a GPUMD kernel.
    """
    prm = dict(params)
    R = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    S = np.asarray(spins, dtype=np.float64).reshape(-1, 3)
    P = _as_pairs(pairs)
    e = _unit(np.asarray(prm["e"], dtype=np.float64).reshape(3))
    prm["e"] = e
    out = _coupled_eval(R, S, P, prm)
    out["mixed_C"] = _coupled_mixed_C(R, S, P, prm)

    def func(r, s):
        return _coupled_eval(r, s, P, prm)

    out["func"] = func
    return out


def _tangent_basis(spins):
    """Two orthonormal tangent vectors per unit spin: (N,2,3)."""
    S = _as_spins(spins)
    N = S.shape[0]
    T = np.zeros((N, 2, 3), dtype=np.float64)
    for i in range(N):
        s = _unit(S[i])
        helper = np.array([1.0, 0.0, 0.0]) if abs(s[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        t1 = helper - np.dot(helper, s) * s
        t1 = _unit(t1)
        t2 = np.cross(s, t1)
        T[i, 0] = t1
        T[i, 1] = t2
    return T


def _tangential_perturb(S, t, eps, mode):
    """Constraint-preserving spin perturbation along tangent direction t.

    mode 'normalize' : S -> normalize(S + eps*t)  (|S| exactly preserved).
    mode 'generator' : deltaS = eps*(t x S) with t a rotation axis; here the
                       caller passes the desired tangent direction and we use
                       axis = S x t, so deltaS = eps*t (|S| preserved to
                       O(eps^2); the O(eps^2) radial term is even in eps and
                       cancels in a central difference).
    """
    if mode == "normalize":
        v = S + eps * t
        return v / np.linalg.norm(v)
    if mode == "generator":
        axis = np.cross(S, t)          # S x t ; axis x S = t for |S|=1, t.S=0
        return S + eps * np.cross(axis, S)
    raise ValueError("unknown tangent_mode: %r" % (mode,))


def mixed_derivative_reference(func_r_S, positions, spins, h=1e-4,
                               tangent_mode="normalize"):
    """C[i,a,j,b]=d2U/(dr_{i,a} dS_{j,b}) by central finite differences.

    PHYSICAL_MODEL
        Given a scalar energy model exposed as func_r_S(r,S) returning a dict
        with 'force' = -dU/dr and 'gradS' = dU/dS, this computes the mixed
        derivative two independent ways and checks reciprocity.

    ASSUMPTIONS
        Smooth U.  Spins are UNIT vectors; the spin derivative is probed with
        TANGENTIAL (constraint-preserving) perturbations so |S| stays fixed.
        Two tangential directions per spin span the tangent plane.

    EQUATIONS
        Route A (gradient-vs-displacement, full unconstrained):
            C_grad_full[i,a,j,b] = (G_j(r + h e_{i,a}) - G_j(r - h e_{i,a}))_b / (2h).
        Route B (force-vs-spin, tangential):
            C_force_tang[i,a,j,m] = -(F_i(S + h t_{j,m}) - F_i(S - h t_{j,m}))_a / (2h).
        Project A onto the same tangent basis:
            C_grad_tang[i,a,j,m] = sum_b C_grad_full[i,a,j,b] t_{j,m,b}.
        Schwarz / Maxwell reciprocity requires
            C_grad_tang == C_force_tang == (closed-form C) . t.
        max_asymmetry = max |C_grad_tang - C_force_tang|.

    UNITS
        C in eV/(Angstrom*mu_B) (or reduced units).  SIGN: with F=-dU/dr and
        G=dU/dS, C = -dF/dS = dG/dr = d2U/(dr dS).

    BOUNDARY_CONDITIONS
        Open reference; positions perturbed component-wise, one atom at a
        time.  Tangential spin perturbations preserve unit length.

    LIMITATIONS
        O(h^2) truncation error; too small h hits float64 round-off.  Only
        the TANGENTIAL (2D) spin derivative is sampled, which is the physical
        one for fixed-|S| spins.  Does not call any GPUMD kernel.
    """
    R = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    S = _as_spins(spins)
    N = S.shape[0]

    # Route A: full C from position FD of the analytic spin gradient.
    C_grad_full = np.zeros((N, 3, N, 3), dtype=np.float64)
    for i in range(N):
        for a in range(3):
            rp = R.copy()
            rm = R.copy()
            rp[i, a] += h
            rm[i, a] -= h
            Gp = func_r_S(rp, S)["gradS"]
            Gm = func_r_S(rm, S)["gradS"]
            C_grad_full[i, a, :, :] = (Gp - Gm) / (2.0 * h)

    T = _tangent_basis(S)
    C_grad_tang = np.einsum("iajb,jmb->iajm", C_grad_full, T)

    # Route B: tangential spin FD of the analytic force.
    C_force_tang = np.zeros((N, 3, N, 2), dtype=np.float64)
    for j in range(N):
        for m in range(2):
            t = T[j, m]
            Sp = S.copy()
            Sm = S.copy()
            Sp[j] = _tangential_perturb(S[j], t, +h, tangent_mode)
            Sm[j] = _tangential_perturb(S[j], t, -h, tangent_mode)
            Fp = func_r_S(R, Sp)["force"]
            Fm = func_r_S(R, Sm)["force"]
            dF = (Fp - Fm) / (2.0 * h)      # dF_{i,a}/dS_j . t_{j,m}
            C_force_tang[:, :, j, m] = -dF  # C.t = -dF/dS . t

    asym = float(np.max(np.abs(C_grad_tang - C_force_tang)))
    return {
        "C_grad_full": C_grad_full,
        "C_grad_tang": C_grad_tang,
        "C_force_tang": C_force_tang,
        "tangent_basis": T,
        "max_asymmetry": asym,
        "h": float(h),
        "tangent_mode": tangent_mode,
        "units": "eV/(Angstrom*mu_B) (reduced units)",
        "sign_convention": "C = -dF/dS = dG/dr = d2U/(dr dS)",
    }


# 6. 180-degree domain wall: continuum profile + discrete lattice relaxation
def domain_wall_width(A, K):
    """Continuum Bloch wall width parameter delta = sqrt(A/K).

    PHYSICAL_MODEL / ASSUMPTIONS
        Micromagnetic CONTINUUM uniaxial ferromagnet with exchange stiffness
        A and anisotropy density K (energy density A (dtheta/dz)^2 + K sin^2theta).

    EQUATIONS
        delta = sqrt(A / K).  PRECISE COEFFICIENT: this is the width PARAMETER
        multiplying z in theta(z)=2 arctan(exp(z/delta)); the full wall width
        is commonly quoted as pi*delta (Lilley) or 2*delta depending on the
        definition.  It is a CONTINUUM relation, NOT an exact lattice formula.

    UNITS
        A in energy/length, K in energy/volume; delta in length.

    BOUNDARY_CONDITIONS
        Infinite chain, theta(-inf)=0, theta(+inf)=pi.

    LIMITATIONS
        Valid only when delta >> lattice spacing a; a discrete chain shows
        corrections of order (a/delta)^2 and the relation FAILS for delta ~ a.
    """
    return float(np.sqrt(float(A) / float(K)))


def domain_wall_energy(A, K):
    """Continuum 180-degree wall energy sigma = 4 sqrt(A K).

    PHYSICAL_MODEL / ASSUMPTIONS / EQUATIONS
        sigma = integral of the wall energy density = 4 sqrt(A K), per unit
        cross-sectional area (here total energy of a 1D chain).  CONTINUUM
        result, exact only as delta/a -> infinity.

    UNITS
        Same as A,K; sigma in energy/area (or energy for the 1D chain).

    BOUNDARY_CONDITIONS
        Pinned 180-degree wall, infinite system.

    LIMITATIONS
        See domain_wall_width: lattice corrections at small delta/a.
    """
    return float(4.0 * np.sqrt(float(A) * float(K)))


def domain_wall_profile(z, A, K, z0=0.0, wall_type="bloch", chirality=1.0):
    """Analytic 180-degree Bloch / Neel continuum magnetisation profile.

    PHYSICAL_MODEL
        A 1D chain along z with uniaxial easy axis e = z_hat.  A 180-degree
        wall m(z) = sech(u) (cos phi, sin phi, 0) + tanh(u) z_hat with
        u = (z - z0)/delta, delta = sqrt(A/K).
        'neel'  -> phi = 0       (transverse component along x_hat),
        'bloch' -> phi = pi/2    (transverse component along y_hat),
        and 'chirality' multiplies the transverse component by its sign.

    ASSUMPTIONS
        Uniaxial anisotropy, exchange stiffness A, no dipolar (demagnetising)
        interaction and no DMI.  Under these assumptions the energy is
        INDEPENDENT of phi, so Bloch and Neel walls are DEGENERATE: the
        distinction only acquires an energy splitting from dipolar shape
        anisotropy or DMI (both absent from this reference).  This is the
        physical reason a purely exchange+anisotropy model cannot select
        between Bloch and Neel walls.

    EQUATIONS
        delta = sqrt(A/K), sigma = 4 sqrt(A K), u = (z-z0)/delta,
        m_z = tanh(u), m_perp = sech(u).  Returns a dict with 'm' (N,3),
        'delta', 'sigma', 'u'.

    UNITS
        z in Angstrom, A,K as above, m dimensionless (unit magnetisation).

    BOUNDARY_CONDITIONS
        180-degree wall: m_z -> -1 as z->-inf, +1 as z->+inf.

    LIMITATIONS
        CONTINUUM profile, not an exact lattice solution.  Bloch/Neel degeneracy
        is a consequence of MISSING dipolar physics, not a bug.
    """
    zc = np.asarray(z, dtype=np.float64)
    delta = domain_wall_width(A, K)
    sigma = domain_wall_energy(A, K)
    u = (zc - z0) / delta
    phi = 0.0 if wall_type.lower().startswith("neel") else 0.5 * np.pi
    ct = np.cos(phi) * float(chirality)
    st = np.sin(phi) * float(chirality)
    m = np.zeros((zc.size, 3), dtype=np.float64)
    m[:, 0] = np.cosh(u) ** -1 * ct
    m[:, 1] = np.cosh(u) ** -1 * st
    m[:, 2] = np.tanh(u)
    return {"m": m, "delta": delta, "sigma": sigma, "u": u,
            "wall_type": wall_type, "chirality": float(chirality)}


def domain_wall_relax_lattice(N, A, K, a=1.0, wall_type="bloch",
                              chirality=1.0, lr=0.1, max_iter=40000,
                              tol=1e-11):
    """Discrete-lattice gradient-descent relaxation of a 1D 180-degree wall.

    PHYSICAL_MODEL
        Chain of N spins (spacing a) on z = n*a, easy axis e = z_hat.
        Discrete energy (midpoint discretisation of the continuum functional):
            E = sum_{n=0}^{N-2} 2A (1 - S_n.S_{n+1}) + K sum_n (1 - (S_n.e)^2)
        whose CONTINUUM limit (a->0) is integral[A (dS/dz)^2 + K sin^2theta] dz,
        giving delta = sqrt(A/K) and sigma = 4 sqrt(A K).
        Boundary spins are PINNED: S_0 = -z_hat, S_{N-1} = +z_hat (matching
        m_z = tanh(u) going -1 -> +1), forcing one 180-degree wall.

    ASSUMPTIONS
        Unit spins, damped (non-precessional) relaxation on the unit sphere:
        S_n <- normalize(S_n + lr * (B_n - (B_n.S_n) S_n)) with the effective
        field B_n = -dE/dS_n = 2A(S_{n-1}+S_{n+1}) + 2K (S_n.e) e.  Interior
        spins only are updated; pinned ends are held fixed.

    EQUATIONS
        Wall energy E_wall = E at convergence (the uniform bulk contributes 0).
        Three width definitions from the relaxed m_z(n) profile:
          (1) INTEGRAL  : delta_int  = (a/2) sum_n (1 - m_z(n)^2)  [=delta for tanh],
          (2) SLOPE     : delta_slope = 1 / max |dm_z/dz|  [=delta for tanh],
          (3) FWHM      : delta_FWHM = FWHM(|dm_z/dz|) / (2 arccosh(sqrt(2)))
                                        [=delta for tanh].

    UNITS
        Reduced (a, A, K in consistent units); widths in units of a.

    BOUNDARY_CONDITIONS
        Pinned ends (+z at left, -z at right); open chain.

    LIMITATIONS
        Explicitly a CONTINUUM-LIMIT comparison: for delta ~ a the lattice
        wall energy and widths DEVIATE from 4 sqrt(A K) and sqrt(A/K).  The
        relaxation is a numerical minimiser, not an exact lattice solution.
        Bloch/Neel initialisations relax to the SAME energy (degeneracy).
    """
    N = int(N)
    A = float(A)
    K = float(K)
    a = float(a)
    e = np.array([0.0, 0.0, 1.0])
    z = np.arange(N, dtype=np.float64) * a

    # initial profile from the analytic continuum wall
    init = domain_wall_profile(z, A, K, z0=0.5 * (N - 1) * a,
                               wall_type=wall_type, chirality=chirality)
    S = init["m"].copy()
    # continuum m_z = tanh((z-z0)/delta): -1 at the LEFT end, +1 at the RIGHT.
    # Pin the ends to match that orientation (a 180-degree wall -z -> +z).
    S[0] = -e.copy()
    S[-1] = e.copy()

    def energy_grad(SS):
        E = np.sum(2.0 * A * (1.0 - np.einsum('ij,ij->i', SS[:-1], SS[1:]))) \
            + K * np.sum(1.0 - (SS @ e) ** 2)
        B = np.zeros_like(SS)
        B[1:] += 2.0 * A * SS[:-1]
        B[:-1] += 2.0 * A * SS[1:]
        B += (2.0 * K * (SS @ e))[:, None] * e[None, :]
        return float(E), B

    prev_E = None
    for it in range(max_iter):
        E, B = energy_grad(S)
        Bt = B - (np.einsum('ij,ij->i', B, S))[:, None] * S  # tangential part
        Snew = S.copy()
        Snew[1:-1] = S[1:-1] + lr * Bt[1:-1]
        nrm = np.linalg.norm(Snew[1:-1], axis=1)
        Snew[1:-1] /= nrm[:, None]
        Snew[0] = -e
        Snew[-1] = e
        step = float(np.max(np.linalg.norm(Snew[1:-1] - S[1:-1], axis=1)))
        S = Snew
        if prev_E is not None and abs(prev_E - E) < tol and step < 1e-10:
            break
        prev_E = E

    E_wall, _ = energy_grad(S)
    mz = S @ e
    # (1) integral width
    delta_int = 0.5 * a * float(np.sum(1.0 - mz ** 2))
    # (2) slope width
    dmdz = np.gradient(mz, a)
    dmax = float(np.max(np.abs(dmdz)))
    delta_slope = 1.0 / dmax if dmax > 0 else float("inf")
    # (3) FWHM of |dm/dz|
    prof = np.abs(dmdz)
    peak = int(np.argmax(prof))
    half = 0.5 * prof[peak]
    lo = peak
    while lo > 0 and prof[lo] > half:
        lo -= 1
    hi = peak
    while hi < N - 1 and prof[hi] > half:
        hi += 1
    fwhm = (hi - lo) * a
    coef = 2.0 * np.arccosh(np.sqrt(2.0))
    delta_fwhm = fwhm / coef
    continuum_delta = domain_wall_width(A, K)
    continuum_sigma = domain_wall_energy(A, K)
    return {
        "energy": E_wall, "profile": S, "mz": mz, "z": z,
        "delta_int": delta_int, "delta_slope": delta_slope,
        "delta_fwhm": delta_fwhm, "fwhm_raw": fwhm,
        "continuum_delta": continuum_delta,
        "continuum_sigma": continuum_sigma,
        "E_over_sigma": E_wall / continuum_sigma if continuum_sigma > 0 else float("nan"),
        "delta_over_a": continuum_delta / a,
        "iterations": it + 1,
        "wall_type": wall_type, "chirality": float(chirality),
    }


# 7. Descriptor symmetry / invariance report
def _radial_f(r):
    """A simple smooth positive radial function used by pair descriptors."""
    return 1.0 / (1.0 + r * r)


def _candidate_descriptors():
    """Return {name: (callable, symmetry_class, explanation)}.

    Each callable takes (R, S, P, n) where R positions, S spins, P unordered
    pairs, n = a LATTICE-DERIVED unit axis (rotated with the lattice).  It
    returns a scalar (float) or a 3-vector (np.ndarray).
    """
    def sex(R, S, P, n):
        return float(np.sum(np.einsum('ij,ij->i', S[P[:, 0]], S[P[:, 1]])))

    def pair_radial(R, S, P, n):
        u = R[P[:, 1]] - R[P[:, 0]]
        d = np.linalg.norm(u, axis=1)
        f = _radial_f(d)
        return float(np.sum(f * np.einsum('ij,ij->i', S[P[:, 0]], S[P[:, 1]])))

    def vector_chiral(R, S, P, n):
        return np.sum(np.cross(S[P[:, 0]], S[P[:, 1]]), axis=0)

    def sia(R, S, P, n):
        return float(np.sum((S @ n) ** 2))

    def sdm(R, S, P, n):
        u = R[P[:, 1]] - R[P[:, 0]]
        d = np.linalg.norm(u, axis=1)
        d = np.where(d == 0.0, 1.0, d)
        rhat = u / d[:, None]
        f = _radial_f(d)
        c = np.cross(S[P[:, 0]], S[P[:, 1]])
        return float(np.sum(f * (rhat * c).sum(axis=1)))

    def san(R, S, P, n):
        u = R[P[:, 1]] - R[P[:, 0]]
        d = np.linalg.norm(u, axis=1)
        d = np.where(d == 0.0, 1.0, d)
        rhat = u / d[:, None]
        a = np.einsum('ij,ij->i', S[P[:, 0]], rhat)
        b = np.einsum('ij,ij->i', S[P[:, 1]], rhat)
        f = _radial_f(d)
        return float(np.sum(f * a * b))

    def senv(R, S, P, n):
        u = R[P[:, 1]] - R[P[:, 0]]
        d = np.linalg.norm(u, axis=1)
        d = np.where(d == 0.0, 1.0, d)
        rhat = u / d[:, None]
        f = _radial_f(d)
        return float(np.sum(f * (S[P[:, 0]] * rhat).sum(axis=1)))

    return {
        "Sex_scalar_SiSj": (sex, "no-SOC isotropic exchange",
                            "sum S_i.S_j : spin-rotation invariant (no SOC), P+, T+."),
        "pair_radial_x_SiSj": (pair_radial, "no-SOC radial pair scalar",
                               "f(r) S_i.S_j : still spin-rotation invariant (no SOC), P+, T+."),
        "vector_SixSj": (vector_chiral, "chiral vector (no SOC)",
                         "sum S_i x S_j : a VECTOR, rotates under spin rotations (not invariant), "
                         "T-even (bilinear in S), P-even (axial x axial)."),
        "SIA_onsite_(Si.n)^2": (sia, "SOC magnetocrystalline anisotropy",
                                "onsite (S_i.n)^2 with n a LATTICE axis: NOT invariant under "
                                "independent spin rotation at fixed lattice; invariant under "
                                "JOINT lattice-spin rotation. P+, T+."),
        "SDM_rhat_(SixSj)": (sdm, "SOC chiral / bond-parallel DMI",
                             "P-ODD pseudoscalar rhat.(S_i x S_j); joint-rotation invariant, "
                             "T+, flips sign under chirality reversal (parity)."),
        "SAN_(Si.r)(Sj.r)": (san, "SOC anisotropic exchange",
                             "(S_i.rhat)(S_j.rhat): joint-rotation invariant, P+, T+, "
                             "NOT invariant under independent spin rotation."),
        "SENV_Si.rhat": (senv, "SOC deprecated linear T-odd",
                         "T-ODD (linear in one spin) and P-odd; joint-rotation invariant. "
                         "A model built on it is not time-reversal symmetric."),
    }


def _transform(positions, spins, pairs, kind, R=None, perm=None):
    """Return (R', S', P', n') for a named symmetry transformation.

    n is a LATTICE-DERIVED unit axis (a specific bond direction) so that SOC
    descriptors are locked to the lattice and are invariant under JOINT
    rotations but not under independent spin rotations.
    """
    Rr = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    S = _as_spins(spins)
    P = _as_pairs(pairs)
    u0 = Rr[P[0, 1]] - Rr[P[0, 0]]
    n = _unit(u0)

    if kind == "global_spin_rotation":
        return Rr, S @ R.T, P, n
    if kind == "time_reversal":
        return Rr, -S, P, n
    if kind == "spin_inversion":
        # For a classical STATIC spin configuration S->-S coincides with
        # time reversal; both are S->-S. Documented explicitly.
        return Rr, -S, P, n
    if kind == "joint_lattice_spin_rotation":
        return Rr @ R.T, S @ R.T, P, _unit(R @ n)
    if kind == "atom_permutation":
        Rp = Rr[perm]
        Sp = S[perm]
        inv = np.argsort(perm)
        Pp = np.array([[inv[int(i)], inv[int(j)]] for i, j in P], dtype=np.int64)
        up = Rp[Pp[0, 1]] - Rp[Pp[0, 0]]
        return Rp, Sp, Pp, _unit(up)
    if kind == "chirality_reversal":
        # spatial inversion of the CONFIGURATION (parity P): polar r -> -r,
        # axial spins S -> S (unchanged). rhat -> -rhat.
        return -Rr, S.copy(), P, n
    raise ValueError("unknown transform kind: %r" % (kind,))


def _sign_of(q0, q1, tol):
    """Return +1 if q1~=q0, -1 if q1~=-q0, else 0 (covariant/rotated)."""
    d0 = np.asarray(q0, dtype=np.float64)
    d1 = np.asarray(q1, dtype=np.float64)
    if np.linalg.norm(d1 - d0) <= tol:
        return 1
    if np.linalg.norm(d1 + d0) <= tol:
        return -1
    return 0


def descriptor_invariance_report(positions, spins, pairs, seed=20240607,
                                 tol=1e-9):
    """Numerically evaluate descriptor invariance under symmetry operations.

    PHYSICAL_MODEL
        A set of candidate spin-dependent descriptors (Sex-like scalar,
        radial pair scalar, S_i x S_j vector, SIA-like onsite (S_i.n)^2,
        SDM-like bond chirality, SAN-like anisotropic exchange, SENV-like
        linear term) evaluated on a concrete noncollinear configuration and
        re-evaluated after each symmetry transformation.

    ASSUMPTIONS
        Spins are classical; P = inversion of the CONFIGURATION (polar r->-r,
        axial S->S); T = S->-S.  n is a LATTICE-DERIVED axis (a bond
        direction), so SOC descriptors rotate WITH the lattice.  'invariant'
        means unchanged; a reported sign -1 means ODD (flips sign), 0 means
        neither (e.g. a vector rotating to a different direction).

    EQUATIONS / TRANSFORMATIONS
        global_spin_rotation: S -> R S, lattice FIXED.  Only NO-SOC
            descriptors (Sex, pair_radial_x_SiSj, |S| channels) are invariant.
        time_reversal / spin_inversion: S -> -S.  Bilinear descriptors are
            EVEN (+1); the linear SENV is ODD (-1).  For classical static
            spins these two operations coincide (both S->-S).
        atom_permutation: consistent relabeling of atoms; sum-form
            descriptors are invariant by construction.
        joint_lattice_spin_rotation (SOC): BOTH lattice and spins -> R(.).
            SOC descriptors (SIA, SDM, SAN, SENV) ARE invariant here even
            though they are NOT invariant under independent spin rotation.
        chirality_reversal: parity r->-r with S fixed; P-ODD SDM and the
            P-ODD linear SENV flip sign (-1); P-even descriptors stay (+1).
        WARNING: requiring ARBITRARY INDEPENDENT spin-rotation invariance at
            FIXED lattice is WRONG for SOC systems.  With SOC the physical
            symmetry is the JOINT lattice-spin rotation; a model containing
            SIA / DMI / anisotropic exchange MUST change its energy under a
            spin-only rotation and that is NOT a bug.

    UNITS
        Descriptor values are dimensionless (spins O(1), radial f in 1/A^2).

    BOUNDARY_CONDITIONS
        Explicit finite unordered bond list; transformations applied to that
        list (atom permutation reindexes it consistently).

    LIMITATIONS
        Numerical evaluation on ONE configuration; an accidental zero of a
        descriptor can make an ODD descriptor appear invariant.  We use a
        generic noncollinear, non-coplanar, inversion-BROKEN configuration to
        avoid that.  This is an INDEPENDENT algebraic reference, not a check
        of any GPUMD kernel.
    """
    rng = np.random.default_rng(seed)
    Rr = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    S = _as_spins(spins)
    P = _as_pairs(pairs)
    Rrot = _rand_rot(rng)
    perm = rng.permutation(S.shape[0])

    kinds = ["global_spin_rotation", "time_reversal", "spin_inversion",
             "atom_permutation", "joint_lattice_spin_rotation",
             "chirality_reversal"]
    descs = _candidate_descriptors()
    # reference (original) evaluation
    u0 = Rr[P[0, 1]] - Rr[P[0, 0]]
    n0 = _unit(u0)
    base = {name: fn(Rr, S, P, n0) for name, (fn, _, _) in descs.items()}
    scale = {name: max(1.0, float(np.linalg.norm(np.asarray(v))))
             for name, v in base.items()}

    table = []
    for name, (fn, sym_class, explanation) in descs.items():
        q0 = base[name]
        inv = {}
        sign = {}
        for kind in kinds:
            Rp, Sp, Pp, np_ = _transform(Rr, S, P, kind, R=Rrot, perm=perm)
            q1 = fn(Rp, Sp, Pp, np_)
            sg = _sign_of(q0, q1, tol * scale[name])
            sign[kind] = sg
            inv[kind] = bool(sg == 1)
        table.append({
            "descriptor": name,
            "symmetry_class": sym_class,
            "invariance": inv,
            "sign": sign,
            "explanation": explanation,
        })

    warnings = [
        "Requiring arbitrary INDEPENDENT spin-rotation invariance at FIXED "
        "lattice is WRONG for SOC systems: with SOC the symmetry is the JOINT "
        "lattice-spin rotation. A spin-only rotation changing the energy is "
        "the EXPECTED behaviour, not a bug.",
        "For classical static spins, 'spin_inversion' (S->-S) and "
        "'time_reversal' (S->-S) are the same map; their columns agree here. "
        "In a quantum/dynamical treatment TR additionally complex-conjugates "
        "and reverses angular momenta, which is outside this static model.",
        "DMI/chiral terms require BROKEN inversion symmetry (Moriya rules); "
        "a P-odd descriptor such as SDM is identically zero on an "
        "inversion-symmetric configuration.",
        "Magnetocrystalline anisotropy (SIA) is SOC-locked to the lattice: "
        "it is NOT invariant under a spin-only rotation.",
    ]
    return {"table": table, "transformations": kinds,
            "warnings": warnings,
            "notes": ("Independent algebraic reference; no GPUMD kernel is "
                      "called or validated.")}


# 8. Berry curvature / topology feasibility checklist (NO Chern number)
_BERRY_REQUIREMENTS = [
    "stable magnetic ground state (a converged, physically correct spin "
    "configuration to expand about)",
    "second derivatives of the Hamiltonian (Hessian / spin-wave matrix "
    "d2E/dS_i dS_j evaluated at that ground state)",
    "correct local spin frames (a smooth tangent-plane basis for each "
    "sublattice, with the right gauge for noncollinear textures)",
    "multi-sublattice Bloch phases (correct q-dependent embedding / basis "
    "phases for every sublattice, not a single-site Fourier transform)",
    "paraunitary / bosonic BdG normalisation with the CORRECT metric "
    "(sigma_3 / J = diag(1,-1) indefinite metric; a bosonic BdG must NOT "
    "be solved as an ordinary Hermitian eigenproblem)",
    "band isolation (a finite, resolved gap separating the band(s) of "
    "interest so a Chern number is well defined)",
    "gauge-invariant discretisation (Fukui-Hatsugai-Suzuki link variables / "
    "Wilson loops, NOT a raw sum of per-k Berry connections)",
    "grid convergence (Berry curvature / Chern number converged with respect "
    "to the k-mesh, with degeneracy-subspace handling)",
]


def berry_curvature_feasibility(requirements=None, evidence=None):
    """DOCUMENTED feasibility checklist for magnon Berry curvature / Chern.

    PHYSICAL_MODEL
        A topological-magnon calculation would require linearised spin
        dynamics (a bosonic BdG / paraunitary eigenproblem) on a stable
        magnetic ground state, followed by a gauge-invariant Berry-curvature
        integration over the Brillouin zone.  THIS FUNCTION DOES NOT IMPLEMENT
        ANY OF THAT.  It returns an honest BLOCKED / PARTIAL status and the
        exact list of missing requirements.

    ASSUMPTIONS
        The caller may pass a dict of booleans 'requirements' (keys are the
        short requirement names below) and an optional 'evidence' dict.  Any
        requirement not supplied is assumed UNSATISFIED.  Even if every listed
        requirement were marked satisfied, the capability remains PARTIAL
        here because NO quantitative Chern number is implemented by design.

    EQUATIONS / CONCEPT DISTINCTIONS
        Noncollinear trajectories OR the mere presence of DMI DO NOT establish
        topological magnons.  The following are DISTINCT concepts:
          * chiral magnons         (finite circular polarisation / angular
                                    momentum of a mode - a LOCAL observable),
          * nonreciprocal magnons  (omega(q) != omega(-q), e.g. from DMI),
          * topological magnons    (nonzero integer Chern number of a band,
                                    requiring a gap + Berry curvature),
          * noncollinear excitations (spin waves about a noncollinear ground
                                    state, which need not be topological).
        Nonzero Berry curvature does NOT by itself imply a quantised Chern
        number in a gapless/multiband system; a Chern number requires band
        isolation.

    UNITS
        Not applicable (status / checklist function).

    BOUNDARY_CONDITIONS
        Would require periodic (Bloch) boundary conditions in the magnon
        problem; none are set up here.

    LIMITATIONS
        status is ONLY 'BLOCKED' (nothing satisfied) or 'PARTIAL' (some, but
        not all, requirements satisfied - still no Chern number).  It NEVER
        returns READY and NEVER computes a quantitative Chern number.  A
        bosonic BdG is deliberately NOT treated as a Hermitian eigenproblem.
    """
    short = ["stable_ground_state", "second_derivatives", "local_spin_frames",
             "sublattice_bloch_phases", "paraunitary_bdg_metric",
             "band_isolation", "fhs_link_variables", "grid_convergence"]
    req = {k: False for k in short}
    if requirements:
        for k, v in requirements.items():
            if k in req:
                req[k] = bool(v)
    satisfied = [k for k in short if req[k]]
    missing = [k for k in short if not req[k]]
    missing_full = [r for k, r in zip(short, _BERRY_REQUIREMENTS) if not req[k]]
    n_sat = len(satisfied)
    if n_sat == 0:
        status = "BLOCKED"
    else:
        status = "PARTIAL"
    return {
        "status": status,
        "quantitative_chern_number": "NOT_IMPLEMENTED",
        "capability": "BLOCKED",
        "satisfied_requirements": satisfied,
        "missing_requirements": missing,
        "missing_requirements_full": missing_full,
        "all_requirements": list(_BERRY_REQUIREMENTS),
        "n_satisfied": n_sat,
        "n_total": len(short),
        "concept_distinctions": {
            "chiral_magnons": "local circular polarisation / mode angular momentum",
            "nonreciprocal_magnons": "omega(q) != omega(-q), e.g. from DMI",
            "topological_magnons": "nonzero integer Chern number of an isolated band",
            "noncollinear_excitations": "spin waves about a noncollinear ground state",
            "dmi_mere_presence": "does NOT establish topological magnons",
            "noncollinear_trajectory_mere_presence": "does NOT establish topological magnons",
        },
        "warnings": [
            "The mere presence of DMI or of a noncollinear trajectory does "
            "NOT establish topological magnons.",
            "Chiral / nonreciprocal / topological magnons and noncollinear "
            "excitations are DISTINCT concepts.",
            "A bosonic BdG eigenproblem must use the paraunitary sigma_3 "
            "metric; treating it as an ordinary Hermitian eigenproblem gives "
            "unphysical eigenvalues.",
            "Nonzero Berry curvature is NOT by itself a quantised Chern "
            "number; band isolation is required.",
        ],
        "notes": ("Honest feasibility checklist only. No quantitative Chern "
                  "number is computed. Test PASSES when this honest status is "
                  "returned, but the CAPABILITY itself is BLOCKED."),
        "evidence": dict(evidence) if evidence else {},
    }
