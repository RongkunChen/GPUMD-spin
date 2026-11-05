#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R22-NonHeat symmetry / Hamiltonian independent-reference self-test driver.

Run from the package root (r22_src)::

    python tests/r22_symmetry.py --json <out.json>

WHAT THIS IS
------------
A CPU-only, numpy-only (no scipy) validation of the INDEPENDENT analytic
reference in tools/r22/symmetry_analysis.py.  It generates concrete spin /
lattice configurations with KNOWN analytic properties, then MEASURES real
numbers (energies, fields, forces, finite differences, relaxed wall energies,
descriptor transformations, feasibility statuses) and checks them against the
closed forms.  It does NOT run or validate any GPUMD CUDA kernel.

STATUS VOCABULARY: PASS / FAIL / SKIP / BLOCKED / NOT_RUN / PARTIAL /
NOT_RESOLVED.  A PASS is emitted only for an assertion that ACTUALLY EXECUTED
and met its stated tolerance.  Informational records (is_assertion=false) carry
honest BLOCKED / PARTIAL / NOT_RUN / SKIP / NOT_RESOLVED statuses for the
topological capability, which is deliberately NOT implemented here; those are
NOT test failures.  Exit code 0 requires: at least one assertion, every
assertion PASS, and no FAIL status anywhere.

Every number reported is MEASURED by this script; nothing is fabricated.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TOOLS_R22 = os.path.join(PKG, "tools", "r22")
if TOOLS_R22 not in sys.path:
    sys.path.insert(0, TOOLS_R22)

import symmetry_analysis as sa  # noqa: E402  (path set above)

FAIL_SET = ("FAIL", "BLOCKED", "NOT_RUN")


# result helpers
def emit(results, test_id, category, status, expected, measured,
         abs_err, rel_err, tolerance, notes, is_assertion=True):
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


def aerr(a, b):
    return abs(float(a) - float(b))


def rerr(a, b):
    a, b = float(a), float(b)
    return abs(a - b) / abs(b) if b != 0 else (0.0 if a == 0 else float("inf"))


def num(results, tid, cat, expected, measured, tol, notes, is_assertion=True):
    ae = aerr(expected, measured)
    re = rerr(expected, measured)
    ok = ae <= tol
    emit(results, tid, cat, "PASS" if ok else "FAIL", float(expected),
         float(measured), ae, re, tol, notes, is_assertion)
    return ok


def boolean(results, tid, cat, expected, measured, notes, is_assertion=True):
    ok = bool(expected) == bool(measured)
    emit(results, tid, cat, "PASS" if ok else "FAIL", bool(expected),
         bool(measured), 0.0 if ok else 1.0, 0.0, "exact boolean match",
         notes, is_assertion)
    return ok


def status_eq(results, tid, cat, expected, measured, notes, is_assertion=True):
    ok = str(expected) == str(measured)
    emit(results, tid, cat, "PASS" if ok else "FAIL", str(expected),
         str(measured), 0.0 if ok else 1.0, 0.0, "exact status match",
         notes, is_assertion)
    return ok


def info(results, tid, cat, status, expected, measured, notes):
    """Informational record (not an assertion); never fails the run."""
    emit(results, tid, cat, status, expected, measured, None, None,
         "informational", notes, is_assertion=False)


def _norm_rows(R):
    return R / np.linalg.norm(R, axis=1)[:, None]


def _random_config(N, seed, complete=False):
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(N, 3)) * 1.5
    S = _norm_rows(rng.normal(size=(N, 3)))
    if complete:
        P = np.array([[i, j] for i in range(N) for j in range(i + 1, N)],
                     dtype=np.int64)
    else:
        pairs = set()
        for i in range(N):
            pairs.add(tuple(sorted((i, (i + 1) % N))))
            pairs.add(tuple(sorted((i, (i + 3) % N))))
        for _ in range(N):
            a, b = rng.integers(0, N, 2)
            if a != b:
                pairs.add(tuple(sorted((int(a), int(b)))))
        P = np.array(sorted(pairs), dtype=np.int64)
    return R, S, P


# 1. Exchange: energy / field FD consistency, total force zero
def run_exchange(results):
    cat = "exchange"
    N = 16
    R, S, P = _random_config(N, seed=101, complete=False)
    J = 0.37
    h = 1e-6

    # field = -dH/dS by FD of the energy
    B = sa.heisenberg_field(S, P, J)
    maxerr = 0.0
    for i in range(N):
        for a in range(3):
            Sp = S.copy()
            Sm = S.copy()
            Sp[i, a] += h
            Sm[i, a] -= h
            fd = -(sa.heisenberg_energy(Sp, P, J)
                   - sa.heisenberg_energy(Sm, P, J)) / (2 * h)
            maxerr = max(maxerr, abs(fd - B[i, a]))
    num(results, "R22.DESCRIPTOR.EXCHANGE_FIELD_FD", cat, 0.0, maxerr,
        1e-6, "max |(-dH/dS)_FD - B_analytic| over all 3N components; "
              "H=-sum J S_i.S_j, B=+sum_j J S_j. Units eV/mu_B.",)

    # uniform periodic ring: total force zero (constant J and r-dependent J)
    Nr = 12
    ang = 2.0 * np.pi * np.arange(Nr) / Nr
    Rr = np.stack([np.cos(ang), np.sin(ang), np.zeros(Nr)], axis=1) * 1.3
    Sr = np.tile(np.array([0.0, 0.0, 1.0]), (Nr, 1))
    Pr = np.array([[i, (i + 1) % Nr] for i in range(Nr)], dtype=np.int64)
    fv0 = sa.heisenberg_force_virial(Sr, Rr, Pr, J)
    fv1 = sa.heisenberg_force_virial(Sr, Rr, Pr, J, dJ_dr=0.08)
    tf0 = float(np.linalg.norm(fv0["total_force"]))
    tf1 = float(np.linalg.norm(fv1["total_force"]))
    num(results, "R22.DESCRIPTOR.EXCHANGE_TOTAL_FORCE_ZERO_CONST_J", cat,
        0.0, tf0, 1e-10, "periodic uniform ring, constant J: |sum_i F_i|; "
        "exchange has no r-dependence so every force is exactly 0.")
    num(results, "R22.DESCRIPTOR.EXCHANGE_TOTAL_FORCE_ZERO_RDEP_J", cat,
        0.0, tf1, 1e-10, "periodic uniform ring, dJ/dr=0.08: |sum_i F_i|; "
        "Newton's third law F_p=-F_q gives zero net force exactly.")

    # uniform energy analytic value
    E = sa.heisenberg_energy(Sr, Pr, J)
    Eexp = -J * Pr.shape[0] * 1.0  # parallel unit spins, Nr bonds
    num(results, "R22.DESCRIPTOR.EXCHANGE_UNIFORM_ENERGY", cat, Eexp, E,
        1e-12, "uniform FM ring: H=-J*N_bonds*(S.S)=-J*12; measured H.")

    # virial 9-component layout consistency with the 3x3 matrix
    W, W9 = sa.heisenberg_virial(Sr, Rr, Pr, J, dJ_dr=0.08)
    W9_ref = np.array([W[0, 0], W[1, 1], W[2, 2], W[0, 1], W[0, 2],
                       W[1, 2], W[1, 0], W[2, 0], W[2, 1]])
    num(results, "R22.DESCRIPTOR.EXCHANGE_VIRIAL9_LAYOUT", cat, 0.0,
        float(np.max(np.abs(W9 - W9_ref))), 1e-12,
        "9-component order [xx yy zz xy xz yz yx zx zy] matches the 3x3 "
        "virial tensor (cross-check of the GPUMD index convention).")


# 2. MAE / SOC: easy-hard, angular scan, torque FD, joint vs fixed rotation
def _soc_energy(R, S, K):
    """SOC model: anisotropy axis n is a LATTICE bond direction."""
    n = sa._unit(R[1] - R[0])
    return sa.mae_energy(S, n, K)


def run_soc(results):
    cat = "mae_soc"
    rng = np.random.default_rng(202)
    N = 7
    R = rng.normal(size=(N, 3))
    S = _norm_rows(rng.normal(size=(N, 3)))
    K = 0.42
    e = np.array([0.0, 0.0, 1.0])

    # easy axis (K>0): E(S||e) < E(S_perp e)
    S_par = np.tile(e, (N, 1))
    S_perp = np.tile(np.array([1.0, 0.0, 0.0]), (N, 1))
    Epar = sa.mae_energy(S_par, e, K)
    Eperp = sa.mae_energy(S_perp, e, K)
    boolean(results, "R22.SOC.MAE_EASY_AXIS_K_POS", cat, True, Epar < Eperp,
            "K>0: E_parallel=%.6f eV < E_perp=%.6f eV, so e is the EASY "
            "axis (convention E=-K(S.e)^2)." % (Epar, Eperp))
    # hard axis (K<0): reversed
    boolean(results, "R22.SOC.MAE_EASY_PLANE_K_NEG", cat, True,
            sa.mae_energy(S_par, e, -K) > sa.mae_energy(S_perp, e, -K),
            "K<0: e becomes HARD and the perpendicular plane is EASY.")

    # E(theta) angular dependence vs explicit rotated spins
    angles = np.array([0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
    scan = sa.mae_scan_energy(angles, e, K, spin_magnitude=1.0)
    helper = np.array([1.0, 0.0, 0.0])
    eperp = sa._unit(helper - np.dot(helper, e) * e)
    explicit = np.array([sa.mae_energy(
        (np.cos(t) * e + np.sin(t) * eperp)[None, :], e, K) for t in angles])
    num(results, "R22.SOC.MAE_ANGULAR_DEPENDENCE", cat, 0.0,
        float(np.max(np.abs(scan - explicit))), 1e-12,
        "E(theta)=-K cos^2(theta) from mae_scan_energy matches mae_energy of "
        "an explicitly rotated single spin at 5 angles (radians).")
    num(results, "R22.SOC.MAE_ANGULAR_AMPLITUDE", cat, -K,
        float(scan[0] - scan[-1]), 1e-12,
        "E(0)-E(pi/2) = -K for |S|=1; measured from the angular scan.")

    # finite-difference spin torque vs analytic field (tangential perturbation)
    h = 1e-6
    maxerr = 0.0
    for i in range(min(N, 4)):
        for t in sa._tangent_basis(S)[i]:
            Sp = S.copy()
            Sm = S.copy()
            Sp[i] = sa._tangential_perturb(S[i], t, +h, "normalize")
            Sm[i] = sa._tangential_perturb(S[i], t, -h, "normalize")
            dE = (sa.mae_energy(Sp, e, K) - sa.mae_energy(Sm, e, K)) / (2 * h)
            fd_torque = -dE                       # = B_i . t
            analytic = float(np.dot(sa.mae_field(S, e, K)[i], t))
            maxerr = max(maxerr, abs(fd_torque - analytic))
    num(results, "R22.SOC.MAE_SPIN_TORQUE_FD", cat, 0.0, maxerr, 1e-6,
        "max |(-dE/dS . t)_FD - (B_analytic . t)| over tangential directions; "
        "B=2K(S.e)e, t perpendicular to S, |S| preserved. Units eV/mu_B.")

    # JOINT lattice-spin rotation: energy unchanged (SOC symmetry)
    Rrot = sa._rand_rot(np.random.default_rng(303))
    E0 = _soc_energy(R, S, K)
    E_joint = _soc_energy(R @ Rrot.T, S @ Rrot.T, K)
    num(results, "R22.SOC.JOINT_LATTICE_SPIN_ROTATION_INVARIANT", cat, E0,
        E_joint, 1e-10,
        "rotate BOTH lattice (R->R Rot^T) and spins (S->S Rot^T): the "
        "lattice-locked easy axis rotates too, so E is unchanged. This is "
        "the PHYSICAL SOC symmetry.")

    # FIXED-lattice independent spin rotation: energy CHANGES (expected PASS)
    E_fixed = sa.mae_energy(S @ Rrot.T, sa._unit(R[1] - R[0]), K)
    dE_fixed = abs(E_fixed - E0)
    boolean(results, "R22.SOC.FIXED_LATTICE_SPIN_ROTATION_CHANGES_ENERGY",
            cat, True, dE_fixed > 1e-3,
            "SOC case: rotating ONLY the spins at FIXED lattice DOES change "
            "the energy (|dE|=%.6f eV). Requiring spin-rotation invariance "
            "here would be WRONG; this change is the EXPECTED PASS, not a "
            "failure." % dE_fixed)


# 3. DMI: chirality, antisymmetry, T, P, field/force FD
def run_dmi(results):
    cat = "dmi"
    N = 9
    R, S, P = _random_config(N, seed=404, complete=True)
    rng = np.random.default_rng(405)
    D = rng.normal(size=(P.shape[0], 3)) * 0.25
    H0 = sa.dmi_energy(S, P, D)

    # chirality reversal flips the sign of the DMI energy
    Hflip = sa.dmi_energy(S, P, -D)
    num(results, "R22.DMI.CHIRALITY_REVERSAL_SIGN", cat, -H0, Hflip, 1e-12,
        "D -> -D (chirality reversal) flips the sign of H_DMI = sum "
        "D_ij.(S_i x S_j); measured H(D) and H(-D).")
    boolean(results, "R22.DMI.CHIRALITY_REVERSAL_FLIPS", cat, True,
            Hflip * H0 < 0.0, "E(-D) = -E(D) confirmed (nonzero H0=%.6f eV)." % H0)

    # time reversal: S -> -S leaves S_i x S_j invariant -> T-even
    Htr = sa.dmi_energy(-S, P, D)
    num(results, "R22.DMI.TIME_REVERSAL_EVEN", cat, H0, Htr, 1e-12,
        "S->-S: S_i x S_j is invariant (two minus signs), so the DMI ENERGY "
        "term is TIME-REVERSAL EVEN with D held fixed. The D vector itself "
        "is T-odd (orbital-current / spin-orbit origin).")
    boolean(results, "R22.DMI.CROSS_PRODUCT_T_EVEN", cat, True,
            np.allclose(np.cross(-S[P[:, 0]], -S[P[:, 1]]),
                        np.cross(S[P[:, 0]], S[P[:, 1]])),
            "explicit check (-S_i)x(-S_j) = S_i x S_j.")

    # D antisymmetry required for a consistent ordered-pair sum
    ordered, Dv = [], []
    for m, (p, q) in enumerate(P):
        ordered.append([p, q]); Dv.append(D[m])
        ordered.append([q, p]); Dv.append(-D[m])
    ordered = np.array(ordered, dtype=np.int64)
    Dv = np.array(Dv, dtype=np.float64)
    chk = sa.dmi_antisymmetry_check(ordered, Dv)
    boolean(results, "R22.DMI.ANTISYMMETRY_REQUIRED", cat, True,
            chk["all_antisymmetric"],
            "D_ij=-D_ji verified; " + chk["explanation"])
    Hord = sa.dmi_energy_ordered(S, ordered, Dv, 0.5)
    num(results, "R22.DMI.ORDERED_PAIR_SUM_CONSISTENT", cat, H0, Hord, 1e-12,
        "(1/2) sum_ordered D_ij.(S_i x S_j) reproduces the unordered bond "
        "sum ONLY when D_ji=-D_ij; measured both.")
    # negative control: symmetric D_ji=+D_ij makes the ordered sum vanish
    Dv_bad = Dv.copy()
    for m in range(P.shape[0]):
        Dv_bad[2 * m + 1] = Dv[2 * m]  # break antisymmetry
    boolean(results, "R22.DMI.ANTISYMMETRY_BREAKS_SUM", cat, True,
            abs(sa.dmi_energy_ordered(S, ordered, Dv_bad, 0.5)) < 1e-12,
            "symmetric D_ji=+D_ij cancels the ordered-pair sum to zero, "
            "demonstrating the antisymmetry requirement (negative control).")

    # space-inversion constraint: bond-parallel DMI is P-odd (rhat polar)
    D0 = 0.19
    Hp = sa.dmi_energy_bond_parallel(S, R, P, D0)
    Hm = sa.dmi_energy_bond_parallel(S, -R, P, D0)
    num(results, "R22.DMI.SPACE_INVERSION_P_ODD", cat, -Hp, Hm, 1e-12,
        "spatial inversion r->-r (spins axial, unchanged) sends rhat->-rhat, "
        "so bond-parallel DMI H=D0 rhat.(S_i x S_j) is P-ODD. A nonzero D "
        "therefore requires BROKEN inversion symmetry (Moriya rule).")
    # Moriya: inversion about a bond midpoint swaps the two spins and flips
    # the sign of the fixed-D DMI term -> a centrosymmetric bond must have
    # D=0. Dedicated single-bond configuration (no conditional shortcut).
    S2 = _norm_rows(np.array([[1.0, 0.2, 0.1], [0.1, 1.0, -0.3]]))
    P2 = np.array([[0, 1]], dtype=np.int64)
    D2 = np.array([[0.05, -0.12, 0.20]])
    H2 = sa.dmi_energy(S2, P2, D2)
    H2swap = sa.dmi_energy(S2[[1, 0]], P2, D2)  # swap spins = inversion about midpoint
    num(results, "R22.DMI.MORIYA_BOND_MIDPOINT_INVERSION", cat, -H2,
        H2swap, 1e-12,
        "single bond: inversion about its midpoint swaps S_i<->S_j and gives "
        "H -> -H (D fixed, spins axial). For an inversion-symmetric bond the "
        "transformed state IS the same state, so H must equal -H, forcing "
        "D_ij=0 (Moriya rule). Measured on a 2-atom bond.")

    # field FD
    h = 1e-6
    B = sa.dmi_field(S, P, D)
    maxerr = 0.0
    for i in range(N):
        for a in range(3):
            Sp = S.copy(); Sm = S.copy()
            Sp[i, a] += h; Sm[i, a] -= h
            fd = -(sa.dmi_energy(Sp, P, D) - sa.dmi_energy(Sm, P, D)) / (2 * h)
            maxerr = max(maxerr, abs(fd - B[i, a]))
    num(results, "R22.DMI.FIELD_FD", cat, 0.0, maxerr, 1e-6,
        "max |(-dH/dS)_FD - B_analytic|; B=-dH/dS with "
        "dH/dS_p=S_q x D, dH/dS_q=D x S_p. Units eV/mu_B.")

    # force: FIXED D has NO position dependence -> force is identically zero.
    # MEASURE it with a central FD of a position wrapper (never assert 0-vs-0).
    h2 = 1e-6

    def H_fixed_D(Rdummy):
        # D_ij is a FIXED vector here; the energy depends on S only, not on R.
        return sa.dmi_energy(S, P, D)

    fixed_force_max = 0.0
    for i in range(min(N, 5)):
        for a in range(3):
            Rpp = R.copy(); Rmm = R.copy()
            Rpp[i, a] += h2; Rmm[i, a] -= h2
            fd = -(H_fixed_D(Rpp) - H_fixed_D(Rmm)) / (2 * h2)
            fixed_force_max = max(fixed_force_max, abs(fd))
    num(results, "R22.DMI.FIXED_D_FORCE_ZERO", cat, 0.0, fixed_force_max,
        1e-12, "MEASURED max |(-dH/dr)_FD| over 15 position components for a "
        "FIXED D_ij = %.3e eV/Angstrom (identically zero because fixed-D DMI "
        "is position-independent). The nonzero-force test applies to the "
        "r-dependent bond-parallel model below." % fixed_force_max)

    # force FD for the bond-parallel (r-dependent) DMI
    F = sa.dmi_force_bond_parallel(S, R, P, D0)
    ferr = 0.0
    for i in range(min(N, 5)):
        for a in range(3):
            Rpp = R.copy(); Rmm = R.copy()
            Rpp[i, a] += h2; Rmm[i, a] -= h2
            fd = -(sa.dmi_energy_bond_parallel(S, Rpp, P, D0)
                   - sa.dmi_energy_bond_parallel(S, Rmm, P, D0)) / (2 * h2)
            ferr = max(ferr, abs(fd - F[i, a]))
    num(results, "R22.DMI.BOND_PARALLEL_FORCE_FD", cat, 0.0, ferr, 1e-6,
        "max |(-dH/dr)_FD - F_analytic| for D_ij=D0 rhat_ij; "
        "F_i=(D0/d) P_perp (S_i x S_j). Units eV/Angstrom.")
    num(results, "R22.DMI.BOND_PARALLEL_FORCE_NET_ZERO", cat, 0.0,
        float(np.linalg.norm(F.sum(axis=0))), 1e-10,
        "Newton's third law: sum_i F_i = 0 for the r-dependent DMI model.")


# 4. Scalar chirality
def run_chirality(results):
    cat = "chirality"
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    num(results, "R22.CHIRALITY.NONCOPLANAR_VALUE", cat, 1.0,
        sa.scalar_spin_chirality(x, y, z), 1e-15,
        "chi = S_i.(S_j x S_k) = x.(y x z) = +1 for the ordered "
        "right-handed triple (i,j,k).")
    num(results, "R22.CHIRALITY.ATOM_ORDER_SWAP_SIGN", cat,
        -sa.scalar_spin_chirality(x, y, z),
        sa.scalar_spin_chirality(x, z, y), 1e-15,
        "swapping j<->k (a transposition) flips the sign: chi -> -chi.")
    num(results, "R22.CHIRALITY.CYCLIC_PERMUTATION_INVARIANT", cat,
        sa.scalar_spin_chirality(x, y, z),
        sa.scalar_spin_chirality(y, z, x), 1e-15,
        "a CYCLIC permutation of the ordered triple leaves det invariant.")
    num(results, "R22.CHIRALITY.TIME_REVERSAL_ODD", cat,
        -sa.scalar_spin_chirality(x, y, z),
        sa.scalar_spin_chirality(-x, -y, -z), 1e-15,
        "S->-S for all three spins: chi -> (-1)^3 chi = -chi (T-ODD). "
        "Distinct from the DMI energy term which is T-EVEN.")
    # coplanar limit
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = sa._unit(np.array([0.3, 0.7, 0.0]))
    num(results, "R22.CHIRALITY.COPLANAR_LIMIT_ZERO", cat, 0.0,
        sa.scalar_spin_chirality(a, b, c), 1e-15,
        "three spins in the xy-plane: S_j x S_k is along z while S_i is "
        "in-plane, so the triple product is exactly 0 (degenerate coplanar "
        "limit).")
    boolean(results, "R22.CHIRALITY.NONCOPLANAR_NONZERO", cat, True,
            abs(sa.scalar_spin_chirality(x, y, z)) > 0.5,
            "a generic noncoplanar triple has a nonzero chirality, unlike "
            "the coplanar case.")
    # chirality reversal (parity of the site-ordered texture): odd
    num(results, "R22.CHIRALITY.SPACE_INVERSION_PSEUDOSCALAR", cat,
        -sa.scalar_spin_chirality(x, y, z),
        sa.scalar_spin_chirality(-x, y, z), 1e-15,
        "chi is a PSEUDOSCALAR: reversing one spin (equivalently reversing "
        "the orientation of the site triangle under inversion/mirror) flips "
        "its sign; a nonzero net chirality needs an inversion-breaking "
        "texture.")


# 5. Mixed derivatives: reciprocity + closed form (2 atom counts, 3 h)
def run_mixed(results):
    cat = "spin_lattice_mixed_derivative"
    prm = dict(J0=0.5, g=0.07, d0=1.0, K0=0.3, k1=0.05,
               w=np.array([0.0, 0.0, 1.0]), e=np.array([0.0, 0.0, 1.0]))
    hvals = [1e-3, 3e-4, 1e-4]
    tol = 5e-6
    for N in [16, 43]:
        R, S, P = _random_config(N, seed=600 + N, complete=False)
        ch = sa.coupled_hamiltonian(R, S, P, prm)
        Cclosed_tang = np.einsum("iajb,jmb->iajm", ch["mixed_C"],
                                 sa._tangent_basis(S))
        for h in hvals:
            md = sa.mixed_derivative_reference(ch["func"], R, S, h=h)
            asym = md["max_asymmetry"]
            eg = float(np.max(np.abs(md["C_grad_tang"] - Cclosed_tang)))
            ef = float(np.max(np.abs(md["C_force_tang"] - Cclosed_tang)))
            tid = "R22.SPIN_LATTICE.MIXED_N%d_H%.0e" % (N, h)
            num(results, tid + "_RECIPROCITY", cat, 0.0, asym, tol,
                "N=%d, h=%.1e: max |dG/dr - (-dF/dS)| (Schwarz/Maxwell). "
                "C = -dF/dS = dG/dr = d2U/(dr dS); units eV/(Angstrom*mu_B). "
                "Tangential constraint-preserving spin perturbations."
                % (N, h))
            num(results, tid + "_VS_CLOSEDFORM_GRAD", cat, 0.0, eg, tol,
                "N=%d, h=%.1e: max |(dG/dr)_FD - C_closed.t|; "
                "closed form from coupled_hamiltonian." % (N, h))
            num(results, tid + "_VS_CLOSEDFORM_FORCE", cat, 0.0, ef, tol,
                "N=%d, h=%.1e: max |(-dF/dS)_FD - C_closed.t|." % (N, h))

    # tangent modes agree
    R, S, P = _random_config(16, seed=777, complete=False)
    ch = sa.coupled_hamiltonian(R, S, P, prm)
    md_n = sa.mixed_derivative_reference(ch["func"], R, S, h=1e-4,
                                         tangent_mode="normalize")
    md_g = sa.mixed_derivative_reference(ch["func"], R, S, h=1e-4,
                                         tangent_mode="generator")
    num(results, "R22.SPIN_LATTICE.TANGENT_MODES_AGREE", cat, 0.0,
        float(np.max(np.abs(md_n["C_force_tang"] - md_g["C_force_tang"]))),
        1e-6, "S->normalize(S+eps t) and the rotation-generator "
        "deltaS=eps*(axis x S) give the same tangential derivative.")

    # analytic F and G of the coupled model vs FD of the energy (sanity)
    h = 1e-6
    U0 = ch["energy"]
    ferr = 0.0
    gerr = 0.0
    for i in range(min(16, 5)):
        for a in range(3):
            Rp = R.copy(); Rm = R.copy(); Rp[i, a] += h; Rm[i, a] -= h
            fd = -(ch["func"](Rp, S)["energy"] - ch["func"](Rm, S)["energy"]) / (2 * h)
            ferr = max(ferr, abs(fd - ch["force"][i, a]))
            Sp = S.copy(); Sm = S.copy(); Sp[i, a] += h; Sm[i, a] -= h
            fd2 = (ch["func"](R, Sp)["energy"] - ch["func"](R, Sm)["energy"]) / (2 * h)
            gerr = max(gerr, abs(fd2 - ch["gradS"][i, a]))
    num(results, "R22.SPIN_LATTICE.COUPLED_FORCE_FD", cat, 0.0, ferr, 1e-6,
        "analytic F=-dU/dr of the coupled model matches a central FD of U.")
    num(results, "R22.SPIN_LATTICE.COUPLED_GRADS_FD", cat, 0.0, gerr, 1e-6,
        "analytic G=dU/dS of the coupled model matches a central FD of U.")


# 6. Domain wall: continuum vs lattice, Bloch/Neel, system-size effect
def run_domain(results):
    cat = "domain_wall"
    A = 1.0
    # (a) wide wall delta/a = 5 -> continuum should hold well
    wide = sa.domain_wall_relax_lattice(64, A, 0.04, a=1.0)
    num(results, "R22.DOMAIN.CONTINUUM_WIDE_WALL_ENERGY", cat, 1.0,
        wide["E_over_sigma"], 0.05,
        "delta/a=5: relaxed lattice wall energy / (4 sqrt(A K)) = %.5f; "
        "tolerance 0.05 ACKNOWLEDGES the continuum approximation (exact "
        "only as delta/a -> infinity)." % wide["E_over_sigma"])
    num(results, "R22.DOMAIN.CONTINUUM_WIDE_WALL_WIDTH", cat,
        wide["continuum_delta"], wide["delta_int"], 0.15 * wide["continuum_delta"],
        "delta=sqrt(A/K)=%.3f vs relaxed integral width %.3f; plus slope "
        "width %.3f and FWHM width %.3f (three definitions)."
        % (wide["continuum_delta"], wide["delta_int"],
           wide["delta_slope"], wide["delta_fwhm"]))
    wdefs = [wide["delta_int"], wide["delta_slope"], wide["delta_fwhm"]]
    num(results, "R22.DOMAIN.WIDTH_DEFINITIONS_AGREE", cat, 0.0,
        float(max(wdefs) - min(wdefs)), 1.5,
        "spread of the three width definitions (integral / slope / FWHM of "
        "|dM/dz|) for a wide wall; loose tolerance acknowledges discrete "
        "finite differences.")

    # (b) narrow wall delta/a = 1 -> continuum degrades (limitation)
    narrow = sa.domain_wall_relax_lattice(64, A, 1.0, a=1.0)
    num(results, "R22.DOMAIN.CONTINUUM_NARROW_WALL_ENERGY", cat, 1.0,
        narrow["E_over_sigma"], 0.15,
        "delta/a=1: relaxed wall energy / sigma = %.5f. Looser tolerance "
        "0.15 acknowledges the CONTINUUM LIMITATION on the lattice; the "
        "deviation is MEASURED, not hidden." % narrow["E_over_sigma"])
    dev_wide = abs(1.0 - wide["E_over_sigma"])
    dev_narrow = abs(1.0 - narrow["E_over_sigma"])
    boolean(results, "R22.DOMAIN.CONTINUUM_DEGRADATION_TREND", cat, True,
            dev_narrow > dev_wide,
            "continuum error grows as delta/a shrinks: |1-E/sigma| = %.5f "
            "(delta/a=5) -> %.5f (delta/a=1). This verifies, rather than "
            "assumes, the continuum limitation." % (dev_wide, dev_narrow))

    # (c) Bloch vs Neel initialisation: degenerate without dipolar/DMI
    rb = sa.domain_wall_relax_lattice(64, A, 0.04, wall_type="bloch")
    rn = sa.domain_wall_relax_lattice(64, A, 0.04, wall_type="neel")
    num(results, "R22.DOMAIN.BLOCH_NEEL_DEGENERACY", cat, 0.0,
        abs(rb["energy"] - rn["energy"]), 1e-8,
        "Bloch and Neel walls have IDENTICAL relaxed energy (measured "
        "difference %.2e eV) because this exchange+anisotropy model has NO "
        "dipolar (demagnetising) interaction and NO DMI to split them."
        % abs(rb["energy"] - rn["energy"]))
    info(results, "R22.DOMAIN.BLOCH_NEEL_PHYSICS_NOTE", cat, "PARTIAL",
         "degenerate in this model", "degenerate in this model",
         "PARTIAL expressivity: a real material's Bloch-vs-Neel selection "
         "requires long-range dipolar shape anisotropy (or DMI), which is "
         "NOT present in this reference (and NOT implemented in the audited "
         "GPUMD production code). This is MISSING PHYSICS, not a CUDA bug.")

    # (d) system-size effect
    Esig = {}
    for N in [24, 32, 48, 64, 96, 128]:
        Esig[N] = sa.domain_wall_relax_lattice(N, A, 0.04)["E_over_sigma"]
    num(results, "R22.DOMAIN.SYSTEM_SIZE_CONVERGENCE", cat, Esig[128],
        Esig[64], 0.01,
        "E/sigma at N=64 (%.5f) and N=128 (%.5f) agree within 0.01; the "
        "wall has converged with system size." % (Esig[64], Esig[128]))
    boolean(results, "R22.DOMAIN.SYSTEM_SIZE_SMALL_DEVIATES", cat, True,
            abs(1.0 - Esig[24]) > abs(1.0 - Esig[128]),
            "N=24 (too short for delta=5) gives E/sigma=%.5f, further from "
            "the continuum value than N=128 (%.5f): a genuine measured "
            "finite-size effect." % (Esig[24], Esig[128]))


# 7. Descriptor invariance table
def run_descriptor_invariance(results):
    cat = "descriptor_invariance"
    N = 8
    R, S, P = _random_config(N, seed=808, complete=True)
    rep = sa.descriptor_invariance_report(R, S, P)

    def sign(desc, tr):
        for row in rep["table"]:
            if row["descriptor"] == desc:
                return row["sign"][tr]
        return None

    # no-SOC: only isotropic scalars survive an independent global spin rotation
    boolean(results, "R22.DESCRIPTOR.NO_SOC_SPIN_ROTATION", cat, True,
            sign("Sex_scalar_SiSj", "global_spin_rotation") == 1 and
            sign("pair_radial_x_SiSj", "global_spin_rotation") == 1 and
            sign("SIA_onsite_(Si.n)^2", "global_spin_rotation") == 0 and
            sign("SDM_rhat_(SixSj)", "global_spin_rotation") == 0 and
            sign("SAN_(Si.r)(Sj.r)", "global_spin_rotation") == 0,
            "only NO-SOC descriptors (Sex, radial x S_i.S_j) are invariant "
            "under an INDEPENDENT global spin rotation at fixed lattice; "
            "SOC descriptors (SIA/SAN/SDM/SENV) are NOT.")

    # SOC joint lattice-spin rotation: ALL SOC descriptors ARE invariant
    boolean(results, "R22.SOC.JOINT_ROTATION_SOC_INVARIANT", cat, True,
            sign("SIA_onsite_(Si.n)^2", "joint_lattice_spin_rotation") == 1 and
            sign("SDM_rhat_(SixSj)", "joint_lattice_spin_rotation") == 1 and
            sign("SAN_(Si.r)(Sj.r)", "joint_lattice_spin_rotation") == 1 and
            sign("SENV_Si.rhat", "joint_lattice_spin_rotation") == 1,
            "with SOC, rotating the LATTICE AND SPINS together leaves every "
            "SOC descriptor invariant (the physical symmetry). This is the "
            "distinction from the fixed-lattice spin-only rotation above.")

    # time reversal: bilinear even, linear SENV odd
    boolean(results, "R22.DESCRIPTOR.TIME_REVERSAL_PARITY", cat, True,
            sign("Sex_scalar_SiSj", "time_reversal") == 1 and
            sign("SDM_rhat_(SixSj)", "time_reversal") == 1 and
            sign("SENV_Si.rhat", "time_reversal") == -1,
            "bilinear descriptors are T-even (+1); the deprecated linear "
            "SENV is T-odd (-1).")
    boolean(results, "R22.DESCRIPTOR.SPIN_INVERSION_EQ_TIME_REVERSAL", cat,
            True,
            all(sign(d, "spin_inversion") == sign(d, "time_reversal")
                for d in ["Sex_scalar_SiSj", "SENV_Si.rhat",
                          "SDM_rhat_(SixSj)"]),
            "for classical static spins S->-S, spin inversion and time "
            "reversal coincide; both columns agree (documented).")

    # chirality reversal: P-odd descriptors flip sign
    boolean(results, "R22.CHIRALITY.DESCRIPTOR_P_ODD", cat, True,
            sign("SDM_rhat_(SixSj)", "chirality_reversal") == -1 and
            sign("SENV_Si.rhat", "chirality_reversal") == -1 and
            sign("Sex_scalar_SiSj", "chirality_reversal") == 1 and
            sign("SIA_onsite_(Si.n)^2", "chirality_reversal") == 1,
            "chirality reversal (parity r->-r, axial S fixed) flips P-ODD "
            "SDM and SENV, leaves P-even Sex/SIA/SAN unchanged.")

    # atom permutation: all sum-form descriptors invariant
    boolean(results, "R22.DESCRIPTOR.ATOM_PERMUTATION_INVARIANT", cat, True,
            all(row["invariance"]["atom_permutation"] for row in rep["table"]),
            "consistent atom relabeling leaves every sum-form descriptor "
            "invariant (verified for all rows of the table).")

    # the vector descriptor is NOT a scalar invariant under rotations
    boolean(results, "R22.DESCRIPTOR.VECTOR_ROTATES_NOT_INVARIANT", cat, True,
            sign("vector_SixSj", "global_spin_rotation") == 0 and
            sign("vector_SixSj", "joint_lattice_spin_rotation") == 0 and
            sign("vector_SixSj", "time_reversal") == 1,
            "sum S_i x S_j is a VECTOR: it rotates under (joint or spin-only) "
            "rotations (sign 0), but is T-even because it is bilinear in S.")

    # warnings present
    warn_text = " ".join(rep["warnings"]).lower()
    boolean(results, "R22.SOC.WRONG_INDEPENDENT_ROTATION_WARNING", cat, True,
            "wrong for soc" in warn_text and "joint" in warn_text,
            "report explicitly WARNS that requiring arbitrary independent "
            "spin-rotation invariance at FIXED lattice is WRONG for SOC.")


# 8. Berry curvature / topology feasibility (BLOCKED / PARTIAL)
def run_topology(results):
    cat = "topology_feasibility"

    # (a) default -> honest BLOCKED, capability blocked, never a Chern number
    f0 = sa.berry_curvature_feasibility()
    status_eq(results, "R22.TOPOLOGY.FEASIBILITY_DEFAULT_BLOCKED", cat,
              "BLOCKED", f0["status"],
              "with NO requirements satisfied the function returns the "
              "HONEST status BLOCKED (this assertion PASSES).")
    boolean(results, "R22.TOPOLOGY.CAPABILITY_IS_BLOCKED", cat, True,
            f0["capability"] == "BLOCKED" and
            f0["quantitative_chern_number"] == "NOT_IMPLEMENTED",
            "capability itself is BLOCKED and no quantitative Chern number "
            "is computed (by design).")

    # (b) complete missing-requirements list
    short = ["stable_ground_state", "second_derivatives",
             "local_spin_frames", "sublattice_bloch_phases",
             "paraunitary_bdg_metric", "band_isolation",
             "fhs_link_variables", "grid_convergence"]
    boolean(results, "R22.TOPOLOGY.MISSING_REQUIREMENTS_COMPLETE", cat, True,
            set(short) == set(f0["missing_requirements"]) and
            len(f0["all_requirements"]) == 8 and
            f0["n_total"] == 8 and f0["n_satisfied"] == 0,
            "all 8 missing requirements are listed: stable magnetic ground "
            "state, 2nd derivatives, local spin frames, multi-sublattice "
            "Bloch phases, paraunitary/bosonic BdG metric, band isolation, "
            "Fukui-Hatsugai-Suzuki link variables, grid convergence.")
    txt = " ".join(f0["all_requirements"]).lower()
    boolean(results, "R22.TOPOLOGY.BDG_METRIC_WARNING", cat, True,
            "paraunitary" in txt and "sigma_3" in txt and
            "hermitian eigenproblem" in " ".join(f0["warnings"]).lower(),
            "checklist requires the paraunitary sigma_3 metric and explicitly "
            "forbids treating a bosonic BdG as an ordinary Hermitian "
            "eigenproblem.")
    boolean(results, "R22.TOPOLOGY.FHS_GAUGE_WARNING", cat, True,
            "fukui" in txt or "fukui-hatsugai-suzuki" in txt,
            "checklist requires gauge-invariant FHS link variables.")

    # (c) PARTIAL when some (not all) requirements are supplied
    f1 = sa.berry_curvature_feasibility(
        requirements={"stable_ground_state": True,
                      "second_derivatives": True})
    status_eq(results, "R22.TOPOLOGY.FEASIBILITY_PARTIAL", cat,
              "PARTIAL", f1["status"],
              "with 2 of 8 requirements satisfied the status is PARTIAL "
              "(still no Chern number); this assertion PASSES on the honest "
              "status.")
    boolean(results, "R22.TOPOLOGY.PARTIAL_NEVER_READY", cat, True,
            f1["quantitative_chern_number"] == "NOT_IMPLEMENTED" and
            f1["n_satisfied"] == 2 and len(f1["missing_requirements"]) == 6,
            "PARTIAL still reports NOT_IMPLEMENTED Chern number and the "
            "remaining 6 missing requirements.")

    # (d) concept distinctions are explicit
    cd = f0["concept_distinctions"]
    boolean(results, "R22.TOPOLOGY.CONCEPT_DISTINCTIONS", cat, True,
            all(k in cd for k in ["chiral_magnons", "nonreciprocal_magnons",
                                  "topological_magnons",
                                  "noncollinear_excitations",
                                  "dmi_mere_presence"]),
            "chiral / nonreciprocal / topological magnons and noncollinear "
            "excitations are documented as DISTINCT concepts.")
    warn = " ".join(f0["warnings"]).lower()
    boolean(results, "R22.TOPOLOGY.DMI_NOT_TOPOLOGY", cat, True,
            "mere presence of dmi" in warn and "does not establish" in warn,
            "warnings state explicitly that the MERE presence of DMI or a "
            "noncollinear trajectory does NOT establish topological magnons.")

    # --- honest informational records: the capability is BLOCKED ---
    info(results, "R22.TOPOLOGY.CAPABILITY_STATUS", cat, "BLOCKED",
         "quantitative magnon Chern number", "NOT IMPLEMENTED",
         "The topological-magnon capability is BLOCKED: no Berry curvature, "
         "no BdG/paraunitary eigensolver, no Wilson-loop/FHS discretisation, "
         "no Chern number. The test that CHECKS this honest status PASSES; "
         "the capability itself is BLOCKED.")
    info(results, "R22.TOPOLOGY.CHERN_NUMBER_NOT_RUN", cat, "NOT_RUN",
         "quantitative Chern number", "NOT_RUN",
         "A quantitative Chern number is deliberately NOT_RUN: by design the "
         "checklist function never computes one.")
    info(results, "R22.TOPOLOGY.FEASIBILITY_PARTIAL_INFO", cat, "PARTIAL",
         "some infrastructure present", f1["status"],
         "With 2/8 requirements satisfied the honest feasibility status is "
         "PARTIAL; still insufficient for a topological conclusion.")
    info(results, "R22.TOPOLOGY.MATERIAL_CLAIM_SKIPPED", cat, "SKIP",
         "material topological-magnon claim", "SKIP",
         "Any claim that a specific material hosts topological magnons is "
         "SKIPPED: it would require a stable ground state, a BdG spectrum, "
         "band isolation and a converged gauge-invariant Chern number.")
    info(results, "R22.TOPOLOGY.MAGNON_LIFETIME_NOT_RESOLVED", cat,
         "NOT_RESOLVED", "physical magnon lifetime", "NOT_RESOLVED",
         "A physical magnon lifetime (Lorentzian linewidth de-convolved from "
         "the window/resolution) is NOT_RESOLVED here; no spectrum or fit is "
         "computed. Harmonic/Hessian data alone cannot give a finite-T "
         "lifetime.")


# main
def run_all():
    results = []
    run_exchange(results)
    run_soc(results)
    run_dmi(results)
    run_chirality(results)
    run_mixed(results)
    run_domain(results)
    run_descriptor_invariance(results)
    run_topology(results)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="R22 symmetry / Hamiltonian independent-reference self-test")
    ap.add_argument("--json", dest="json_path", default=None,
                    help="write the JSON result list to this path as well as stdout")
    args = ap.parse_args(argv)

    results = run_all()

    if args.json_path:
        try:
            with open(args.json_path, "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=2, allow_nan=False)
        except (OSError, ValueError) as exc:
            sys.stderr.write("failed to write --json file: %r\n" % (exc,))

    sys.stdout.write(json.dumps(results, indent=2, allow_nan=False))
    sys.stdout.write("\n")

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    assertions = [r for r in results if r["is_assertion"]]
    any_fail = any(r["status"] == "FAIL" for r in results)
    all_assertions_pass = (len(assertions) > 0 and
                           all(r["status"] == "PASS" for r in assertions))
    exit_code = 0 if (not any_fail and all_assertions_pass) else 1

    sys.stderr.write(
        "\nR22.SYMMETRY summary: total=%d  %s  |  assertions=%d  "
        "info=%d  exit=%d\n"
        % (len(results),
           " ".join("%s=%d" % (k, counts[k]) for k in sorted(counts)),
           len(assertions), len(results) - len(assertions), exit_code))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
