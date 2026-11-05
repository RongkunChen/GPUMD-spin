#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPU-only, numpy-only independent magnon / spin-dynamics reference for R22.

SCOPE / HONESTY STATEMENT
-------------------------
This module is an INDEPENDENT CPU REFERENCE derived from the spin Hamiltonian
and from the definition of the dynamical structure factor.  It does NOT run,
call, or validate any GPU kernel.  Validation of GPUMD GPU trajectories (the
spin trajectories produced by src/, run.cu, spin_external.cu, the LLG/SLLG
integrators, and the heat/coupling stages) is performed ELSEWHERE in stage
R22-NonHeat.  Here we only:

  * read extended-XYZ spin trajectories (a file format convention),
  * compute the discrete dynamical structure factor S(q, omega),
  * provide analytic linear-spin-wave dispersions,
  * fit Lorentzian linewidths with explicit resolution / broadening gating,
  * compare fixed-lattice vs coupled linewidths with error estimates.

No scipy is used (not installed).  Everything is float64 / numpy.

CONVENTIONS
-----------
  * Spin components in an extended-XYZ file are MAGNETIC MOMENTS in mu_B.
  * Positions are in Angstrom; q is in 1/Angstrom so that q.r is dimensionless.
  * Simulation time step dt_fs is in fs.  TIME_UNIT_CONVERSION = 10.18051 fs
    per GPUMD natural time unit (only relevant when converting GPUMD output).
  * Energy in eV; angular frequency omega in rad/fs.  1 THz = 2*pi*1e-3 rad/fs.
  * Magnetic field in eV/mu_B; gamma = g*mu_B/hbar = 3.0385 mu_B/(eV*fs) is
    the gyromagnetic ratio used by the LLG integrator.
  * hbar in eV*fs (HBAR_EV_FS) converts a magnon energy E [eV] to an angular
    frequency omega = E/hbar [rad/fs].

STATUS VOCABULARY (used by the test driver, not by this module's internals):
PASS / FAIL / SKIP / BLOCKED / NOT_RUN / NOT_RESOLVED.  This module returns
plain data; the driver in tests/r22_magnon.py assigns statuses.
"""

import numpy as np

# Physical constants (eV, fs, mu_B, Angstrom)
HBAR_EV_FS = 0.6582119          # hbar in eV*fs  (6.582119e-16 eV*s)
GAMMA_GYRO = 3.0385             # g*mu_B/hbar in mu_B/(eV*fs)
TIME_UNIT_CONVERSION = 10.18051 # fs per GPUMD natural time unit
THZ_TO_RAD_PER_FS = 2.0 * np.pi * 1.0e-3   # 1 THz -> rad/fs


# A. Extended-XYZ spin trajectory reader
def read_spin_trajectory(path):
    """Parse an extended-XYZ spin trajectory into float64 arrays.

    Each frame is::

        <natoms>                                          (integer line)
        <comment> Properties=species:S:1:pos:R:3:spin:R:3 ...  (comment line)
        <species> x y z  sx sy sz                        (natoms lines, possibly
                                                        with extra columns)

    The ``Properties=`` value is a colon-separated list of (name:type:count)
    triplets.  The ``spin:R:3`` field may appear ANYWHERE in that list and may
    be accompanied by extra fields (e.g. ``group:I:1``, ``vel:R:3``,
    ``Lattice=...``, ``pbc=...``).  Column offsets are computed from the
    cumulative triplet counts, so this is robust to reordered / augmented
    property lists.

    Parameters
    ----------
    path : str
        Path to the extended-XYZ file.

    Returns
    -------
    dict with keys
        'pos'       : (nframes, N, 3) float64 positions [Angstrom]
        'spin'      : (nframes, N, 3) float64 magnetic moments [mu_B]
        'species'   : (N,) array of species strings (from the first frame)
        'natoms'    : int
        'nframes'   : int
        'properties': the raw Properties= string of the first frame
        'spin_col'  : column index of the first spin component in an atom line
        'pos_col'   : column index of the first position component

    Raises
    ------
    ValueError / IOError
        Clear error if the file is malformed or if no ``spin`` property of
        width >= 3 is present.
    """
    with open(path, "r") as fh:
        lines = fh.read().splitlines()

    frames_pos = []
    frames_spin = []
    first_species = None
    first_props = None
    first_spin_col = None
    first_pos_col = None

    i = 0
    nlines = len(lines)
    while i < nlines:
        # skip blank separator lines between frames
        while i < nlines and lines[i].strip() == "":
            i += 1
        if i >= nlines:
            break
        try:
            natoms = int(lines[i].strip())
        except ValueError as exc:
            raise ValueError("frame %d: expected integer atom count, got %r"
                             % (len(frames_pos), lines[i])) from exc
        i += 1
        if i >= nlines:
            raise ValueError("truncated file: missing comment line")
        comment = lines[i]
        i += 1

        props_str, props = _parse_properties(comment)
        spin_col, spin_width = _find_property(props, "spin")
        pos_col, pos_width = _find_property(props, "pos")
        if spin_col is None:
            raise ValueError("no 'spin' property found in Properties=%s "
                             "(a spin:R:3 field is required)" % props_str)
        if spin_width < 3 or pos_width < 3:
            raise ValueError("spin/pos property width must be >= 3 "
                             "(got spin=%s pos=%s)" % (spin_width, pos_width))

        if first_props is None:
            first_props = props_str
            first_spin_col = spin_col
            first_pos_col = pos_col

        pos = np.empty((natoms, 3), dtype=np.float64)
        spin = np.empty((natoms, 3), dtype=np.float64)
        species = []
        for j in range(natoms):
            if i >= nlines:
                raise ValueError("truncated file: missing atom lines")
            toks = lines[i].split()
            i += 1
            if len(toks) < spin_col + 3 or len(toks) < pos_col + 3:
                raise ValueError("atom line has too few columns: %r" % lines[i - 1])
            species.append(toks[0])
            for k in range(3):
                pos[j, k] = float(toks[pos_col + k])
                spin[j, k] = float(toks[spin_col + k])

        frames_pos.append(pos)
        frames_spin.append(spin)
        if first_species is None:
            first_species = np.array(species, dtype=object)

    if not frames_pos:
        raise ValueError("no frames found in %r" % path)
    nframes = len(frames_pos)
    if any(f.shape[0] != frames_pos[0].shape[0] for f in frames_pos):
        raise ValueError("inconsistent atom count between frames")

    return {
        "pos": np.stack(frames_pos, axis=0),
        "spin": np.stack(frames_spin, axis=0),
        "species": first_species,
        "natoms": int(frames_pos[0].shape[0]),
        "nframes": int(nframes),
        "properties": first_props,
        "spin_col": int(first_spin_col),
        "pos_col": int(first_pos_col),
    }


def _parse_properties(comment):
    """Return (raw_properties_string, list_of_(name,dtype,count)_triplets)."""
    import re
    m = re.search(r"Properties=(\S+)", comment)
    if m is None:
        raise ValueError("comment line has no Properties= field: %r" % comment)
    raw = m.group(1)
    toks = raw.split(":")
    if len(toks) % 3 != 0:
        raise ValueError("malformed Properties list (not name:type:count "
                         "triplets): %r" % raw)
    props = []
    for k in range(0, len(toks), 3):
        try:
            cnt = int(toks[k + 2])
        except ValueError as exc:
            raise ValueError("bad property count in %r" % raw) from exc
        props.append((toks[k], toks[k + 1], cnt))
    return raw, props


def _find_property(props, name):
    """Return (start_column, width) of a named property, or (None, None)."""
    col = 0
    for (pname, dtype, cnt) in props:
        if pname == name:
            return col, cnt
        col += cnt
    return None, None


# Window utilities (coherent gain / ENBW)
def window_properties(name, T):
    """Return (w, CG, U, ENBW_bins) for a length-T spectral window.

    w           : window samples (float64, length T)
    CG          : coherent gain = mean(w)  (DC / narrowband amplitude gain)
    U           : mean(w**2)               (window power)
    ENBW_bins   : equivalent-noise bandwidth in FFT bins
                  = T*sum(w**2)/sum(w)**2 = U/CG**2

    For a rectangular window ENBW_bins = 1.0; for the periodic Hann window
    used here ENBW_bins = 1.5, i.e. a Hann window broadens a spectral line by
    an extra ~50% in noise bandwidth relative to rectangular.  The DSF divides
    the window by its coherent gain (CG) so that narrowband peak HEIGHTS remain
    window-independent; the ENBW (broadening) is reported separately and must
    be de-convolved before any physical linewidth is inferred.
    """
    n = np.arange(T, dtype=np.float64)
    key = str(name).lower()
    if key in ("rectangular", "boxcar", "none", "rect"):
        w = np.ones(T, dtype=np.float64)
    elif key in ("hann", "hanning", "hann_periodic"):
        # periodic (DFT-even) Hann: w[n] = 0.5 - 0.5 cos(2*pi*n/T)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * n / T)
    else:
        raise ValueError("unknown window %r (use 'hann' or 'rectangular')" % name)
    wsum = np.sum(w)
    if wsum == 0:
        raise ValueError("window has zero coherent gain")
    CG = wsum / T
    U = np.mean(w ** 2)
    ENBW = (T * np.sum(w ** 2)) / (wsum ** 2)
    return w, float(CG), float(U), float(ENBW)


# B. Dynamical structure factor
def dynamic_structure_factor(spin, pos, q_vectors, dt_fs, subtract_mean=True,
                             window="hann", transverse_axis=None,
                             local_frames=None, component="transverse"):
    """Discrete, explicitly normalised dynamical structure factor S(q,omega).

    PHYSICAL_MODEL
        Classical spin field S_j(t) (magnetic moment, mu_B) on sites r_j
        (Angstrom).  We measure spin fluctuations about the time-mean local
        moment and their space-time Fourier transform.  For a ferromagnet the
        transverse fluctuations are the magnon modes; for a noncollinear
        ground state one must project onto LOCAL transverse axes first.

    ASSUMPTIONS
        - Equally spaced samples t_n = n*dt_fs, n = 0..T-1 (T = nframes).
        - Linear response / small fluctuations about <S_j> (subtract_mean).
        - Positions are (near) equilibrium; we use their time mean so that
          q.r_j is the ACTUAL site position, including every sublattice offset
          (no primitive-cell folding, no "crystallographic" approximation).
        - Spins are classical vectors; no quantum Bose factor is applied.

    EQUATIONS
        delta S_j(t)   = S_j(t) - <S_j>                       (if subtract_mean)
        delta S_j^a(t) = v_j^a . delta S_j(t)                 (projection, see
                         component / local_frames / transverse_axis below)
        deltaS_a(q,t)  = sum_j exp(-i q.r_j) deltaS_j^a(t)
        S^{aa}(q,omega)= (dt / (2*pi*N*T)) * |FFT{ wn(t) deltaS_a(q,t) }|^2

        where FFT is the unnormalised numpy FFT, wn = w/CG is the
        coherent-gain-corrected window, N = number of spins, T = nframes.
        For 'transverse' we SUM the two linear projections:
            S_transverse = S^{e1e1} + S^{e2e2}.
        For 'full'/'trace' we sum all three, for 'longitudinal' only e3.

    FOURIER NORMALISATION
        omega = 2*pi*f with f = fftfreq(T, d=dt_fs)  [cycles/fs], i.e.
        omega in rad/fs.  We return a TWO-SIDED spectrum (both +omega and
        -omega), fftshift'd so omega runs from -omega_max to +omega_max.
        The prefactor dt/(2*pi*N*T) makes S an explicit periodogram density;
        peak POSITIONS are independent of this prefactor.  Dividing the window
        by CG (coherent-gain correction) keeps narrowband peak heights
        window-independent; the window ENBW (report via window_properties) is
        the residual broadening and is NOT removed from the line shape.

    UNITS
        spin [mu_B] -> S(q,omega) [mu_B^2 * fs / rad] (up to the stated prefactor);
        pos [Angstrom], q [1/Angstrom]; omega [rad/fs]; dt_fs [fs].
        Convert to THz via f_THz = omega / (2*pi*1e-3).

    BOUNDARY_CONDITIONS
        Periodic in time (discrete FFT); periodic in space via the plane-wave
        phase exp(-i q.r_j).  q need not lie on the reciprocal lattice of the
        simulation cell; arbitrary directions are accepted.

    LIMITATIONS
        - Finite T gives frequency resolution domega = 2*pi/(T*dt_fs) and
          spectral leakage (window ENBW).  Numerical/spectral broadening is
          NOT a physical lifetime; see lorentzian_fit().
        - A truncated/undamped trajectory produces sinc sidelobes; use a
          window and de-convolve the ENBW before quoting a linewidth.
        - Two-sided spectrum contains +omega and -omega mirror peaks for real
          (linearly polarised) fluctuations; fit one side at a time.
        - No multi-q interpolation, no quantum detailed balance, no
          instrumental resolution function beyond the time window.

    Parameters
    ----------
    spin : (T, N, 3) float array, magnetic moments [mu_B].
    pos  : (N, 3) or (T, N, 3) float array, positions [Angstrom].  If a full
           trajectory is given, the time-mean positions are used.
    q_vectors : (nq, 3) or (3,) array-like of wavevectors [1/Angstrom].
    dt_fs : float, time step [fs].
    subtract_mean : bool, subtract the per-site time-mean spin.
    window : 'hann' or 'rectangular' (see window_properties).
    transverse_axis : None or 3-vector / axis index.  When local_frames is
           None, this is the GLOBAL longitudinal (equilibrium) axis; the two
           perpendicular directions define the transverse plane.  Defaults to
           the z axis [0,0,1].
    local_frames : None or (N, 3, 3) float array of orthonormal local triads
           with rows (e1, e2, e3), e3 along the local equilibrium spin.  For
           noncollinear ground states e3 varies per site.  Fluctuations are
           projected onto (e1, e2) for 'transverse'.  The exp(-i q.r_j) phase
           ALWAYS uses the actual site position (sublattice offsets included),
           independent of the local frame.
    component : 'transverse' | 'longitudinal' | 'full'/'trace' | 0/1/2 | 'x'/
           'y'/'z'.  The last group selects a single GLOBAL Cartesian
           component (only meaningful when local_frames is None).

    Returns
    -------
    omega : (T,) float64 array, two-sided angular frequency [rad/fs].
    S     : (nq, T) float64 array; S[i] is S(q_i, omega).  Always 2-D; a
           single q is returned as shape (1, T).
    """
    spin = np.asarray(spin, dtype=np.float64)
    if spin.ndim != 3 or spin.shape[2] != 3:
        raise ValueError("spin must have shape (T, N, 3)")
    T, N, _ = spin.shape
    if dt_fs <= 0:
        raise ValueError("dt_fs must be positive")

    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim == 3:
        pos_eff = pos.mean(axis=0)          # time-mean ACTUAL site positions
    elif pos.ndim == 2:
        pos_eff = pos
    else:
        raise ValueError("pos must be (N,3) or (T,N,3)")
    if pos_eff.shape != (N, 3):
        raise ValueError("pos has %r but spin has N=%d sites" % (pos_eff.shape, N))

    q = np.atleast_2d(np.asarray(q_vectors, dtype=np.float64))
    if q.shape[1] != 3:
        raise ValueError("q_vectors must have shape (nq, 3) or (3,)")

    delta = spin - spin.mean(axis=0) if subtract_mean else spin.copy()
    P = _projection_vectors(N, local_frames, transverse_axis, component)
    # delta_proj[p, t, n] = sum_k P[p,n,k] * delta[t,n,k]  (complex allowed)
    delta_proj = np.einsum("pnk,tnk->ptn", P, delta)

    w, CG, U, ENBW = window_properties(window, T)
    wn = w / CG                              # coherent-gain-corrected window
    f = np.fft.fftfreq(T, d=dt_fs)
    omega = 2.0 * np.pi * f

    out = np.empty((q.shape[0], T), dtype=np.float64)
    for iq, qi in enumerate(q):
        phase = np.exp(-1j * (pos_eff @ qi))  # (N,), ACTUAL site positions
        dS = np.einsum("ptn,n->pt", delta_proj, phase)   # (nproj, T)
        F = np.fft.fft(dS * wn[None, :], axis=1)
        out[iq] = (dt_fs / (2.0 * np.pi * N * T)) * np.sum(np.abs(F) ** 2, axis=0)

    return np.fft.fftshift(omega), np.fft.fftshift(out, axes=1)


def _unit(v):
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size != 3:
        raise ValueError("expected a 3-vector, got %r" % (v,))
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError("zero-length axis")
    return a / n


def _perp_basis(axis):
    a = _unit(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = helper - np.dot(helper, a) * a
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    return u, v


def _projection_vectors(N, local_frames, transverse_axis, component):
    """Build (nproj, N, 3) complex projection vectors for each component."""
    idx_map = {"x": 0, "y": 1, "z": 2}
    single_idx = None
    if isinstance(component, (int, np.integer)) and component in (0, 1, 2):
        single_idx = int(component)
        component = "__global_index__"
    elif isinstance(component, str) and component in idx_map:
        single_idx = idx_map[component]
        component = "__global_index__"

    if local_frames is not None:
        lf = np.asarray(local_frames, dtype=np.float64)
        if lf.shape != (N, 3, 3):
            raise ValueError("local_frames must have shape (N, 3, 3)")
        e1, e2, e3 = lf[:, 0, :], lf[:, 1, :], lf[:, 2, :]   # (N,3) each
        if component == "transverse":
            return np.stack([e1, e2]).astype(np.complex128)
        if component == "longitudinal":
            return e3[None, :, :].astype(np.complex128)
        if component in ("full", "trace"):
            return np.stack([e1, e2, e3]).astype(np.complex128)
        raise ValueError("component %r not supported with local_frames" % component)

    # Global-Cartesian projections
    if component == "__global_index__":
        P = np.zeros((1, N, 3), dtype=np.complex128)
        P[0, :, single_idx] = 1.0
        return P
    if component in ("full", "trace"):
        P = np.zeros((3, N, 3), dtype=np.complex128)
        for k in range(3):
            P[k, :, k] = 1.0
        return P
    axis = _unit(transverse_axis if transverse_axis is not None else (0.0, 0.0, 1.0))
    u, v = _perp_basis(axis)
    if component == "transverse":
        U = np.broadcast_to(u, (N, 3)).astype(np.complex128)
        V = np.broadcast_to(v, (N, 3)).astype(np.complex128)
        return np.stack([U, V])
    if component == "longitudinal":
        A = np.broadcast_to(axis, (N, 3)).astype(np.complex128)
        return A[None, :, :]
    raise ValueError("unknown component %r" % component)


# C. Independent analytic references (from the Hamiltonian, not from GPUMD)
def _qarr(q):
    qa = np.atleast_2d(np.asarray(q, dtype=np.float64))
    if qa.shape[1] != 3:
        raise ValueError("q must have shape (3,) or (nq, 3)")
    return qa, (np.asarray(q).ndim == 1)


def _gamma_sc(qa, a):
    """Structure factor for a simple-cubic lattice, z = 6 nearest neighbours.

    gamma_q = (1/z) sum_{delta=+-a x,+-a y,+-a z} exp(i q.delta)
            = (1/3)[cos(q_x a) + cos(q_y a) + cos(q_z a)].
    """
    return (np.cos(qa[:, 0] * a) + np.cos(qa[:, 1] * a)
            + np.cos(qa[:, 2] * a)) / 3.0


def heisenberg_fm_dispersion(q, J, S, a, lattice="sc", z=6):
    """Ferromagnetic Heisenberg linear-spin-wave dispersion.

    PHYSICAL_MODEL
        Isotropic nearest-neighbour FM Heisenberg Hamiltonian
            H = -J sum_<ij> S_i . S_j ,  J > 0 (ferromagnetic),
        S = dimensionless spin quantum number, a = lattice constant.
    ASSUMPTIONS
        Holstein-Primakoff / classical linear spin wave about the fully
        polarised collinear FM ground state; large-S, harmonic (non-
        interacting magnon) limit; no field, no anisotropy, no SOC.
    EQUATIONS
        hbar*omega(q) = E(q) = 2*J*S*z*(1 - gamma_q)
        gamma_q = (1/3)[cos(q_x a)+cos(q_y a)+cos(q_z a)]   (sc, z=6)
        => omega(q) = 2*J*S*z*(1 - gamma_q) / hbar_eff,
        with hbar_eff = HBAR_EV_FS = 0.6582119 eV*fs.
    UNITS
        J [eV], S dimensionless, a [Angstrom], q [1/Angstrom], E [eV],
        omega [rad/fs] (omega = 2*pi*f).  Divide by 2*pi*1e-3 for THz.
        The Goldstone mode: omega(q=0) = 0 because the FM ground state
        spontaneously breaks the continuous spin-rotation symmetry.
    BOUNDARY_CONDITIONS
        Born-von Karman periodic lattice; q defined modulo a reciprocal
        lattice vector G (sc: G = (2*pi/a)(n_x,n_y,n_z)).  gamma_q is
        periodic under q -> q+G, hence omega(q+G) = omega(q).
    LIMITATIONS
        Only lattice='sc' is implemented (z=6).  No 1/S corrections, no
        magnon-magnon interactions, no finite-temperature renormalisation,
        no dipolar or anisotropic terms (use fm_dispersion_with_anisotropy_gap).
    """
    if lattice != "sc":
        raise NotImplementedError("only lattice='sc' is implemented (got %r)" % lattice)
    qa, single = _qarr(q)
    g = _gamma_sc(qa, a)
    omega = 2.0 * J * S * z * (1.0 - g) / HBAR_EV_FS
    return float(omega[0]) if single else omega


def heisenberg_afm_dispersion(q, J, S, a, z=6):
    """Two-sublattice bipartite AFM linear-spin-wave dispersion.

    PHYSICAL_MODEL
        Isotropic nearest-neighbour AFM Heisenberg Hamiltonian
            H = +J sum_<ij> S_i . S_j ,  J > 0 (antiferromagnetic),
        on a bipartite (A/B) lattice, e.g. simple cubic with NN A-B bonds.
    ASSUMPTIONS
        Collinear Neel ground state, linear spin wave (harmonic), isotropic
        exchange, no field, no anisotropy, no frustration; large-S limit.
    EQUATIONS
        hbar*omega(q) = E(q) = 2*J*S*z*sqrt(1 - gamma_q**2)
        gamma_q = (1/3)[cos(q_x a)+cos(q_y a)+cos(q_z a)]  (sc, z=6)
        Two magnon branches omega_+(q) and omega_-(q) exist; for the isotropic
        zero-field bipartite AFM they are DEGENERATE:
            omega_+ = omega_- = 2*J*S*z*sqrt(1 - gamma_q**2)/hbar_eff.
        (A finite field or easy-axis anisotropy would split them.)
    UNITS
        Same as heisenberg_fm_dispersion: omega [rad/fs].  Linear Goldstone
        mode near q=0: omega ~ c|q| (twofold degenerate), reflecting the
        broken continuous symmetry of the Neel order.
    BOUNDARY_CONDITIONS
        Periodic bipartite lattice; magnetic BZ is half the chemical BZ.
        Periodic under q -> q+G.
    LIMITATIONS
        Returns the two degenerate branches (shape (...,2)).  No anisotropy
        gap, no field splitting, no quantum 1/S renormalisation (Z_c factor),
        no 1/S corrections, no fermionic/spinon physics.
    """
    qa, single = _qarr(q)
    g = _gamma_sc(qa, a)
    pref = 2.0 * J * S * z / HBAR_EV_FS
    w = pref * np.sqrt(np.clip(1.0 - g ** 2, 0.0, None))
    branches = np.stack([w, w], axis=-1)        # (nq, 2), degenerate
    return branches[0] if single else branches


def fm_dispersion_with_anisotropy_gap(q, J, S, a, K, z=6):
    """FM dispersion with an explicit single-ion anisotropy gap.

    PHYSICAL_MODEL
        H = -J sum_<ij> S_i . S_j - K sum_i (S_i^z)^2 ,  J>0, K>0 easy axis.
    ASSUMPTIONS
        Collinear FM ground state along z, linear spin wave, harmonic limit.
        Single-ion (on-site) anisotropy of strength K [eV] adds a gap.
    EQUATIONS
        Delta = 2*K*S                      (gap energy [eV] at q=0)
        hbar*omega(q) = 2*J*S*z*(1 - gamma_q) + Delta
        omega(q) = [2*J*S*z*(1 - gamma_q) + 2*K*S] / hbar_eff
    UNITS
        omega [rad/fs]; K [eV]; gap omega(0) = 2*K*S/hbar_eff [rad/fs].
    BOUNDARY_CONDITIONS
        Periodic q -> q+G (same gamma_q as FM).
    LIMITATIONS
        Spin-orbit coupling / anisotropy BREAKS the continuous spin-rotation
        symmetry down to a discrete (Ising-like) symmetry, so q=0 is NOT
        required to be gapless: omega(0) = Delta/hbar_eff > 0 for K>0.  Only
        the exchange part retains the Goldstone-like (1-gamma_q) shape.  No
        1/S or finite-T corrections; K is a phenomenological gap parameter.
    """
    qa, single = _qarr(q)
    g = _gamma_sc(qa, a)
    Delta = 2.0 * K * S
    omega = (2.0 * J * S * z * (1.0 - g) + Delta) / HBAR_EV_FS
    return float(omega[0]) if single else omega


def multi_sublattice_example():
    """A small documented NONCOLLINEAR multi-sublattice test model.

    PHYSICAL_MODEL
        Triangular / 3-sublattice 120-degree noncollinear spin structure, the
        canonical example where a GLOBAL transverse/longitudinal decomposition
        fails and LOCAL reference frames are mandatory.
    ASSUMPTIONS
        Three sublattices with equilibrium directions e3_j separated by 120
        degrees in the xy-plane; e2 = z (out of plane), e1 = e2 x e3 so that
        (e1,e2,e3) is a right-handed orthonormal triad at every site.  Small
        transverse fluctuations precess in each LOCAL (e1,e2) plane.
    EQUATIONS
        e3_j = (cos(theta_j), sin(theta_j), 0), theta_j = 0, 120, 240 deg.
        e2_j = (0,0,1), e1_j = e2_j x e3_j = (-sin(theta_j), cos(theta_j), 0).
        S_j(t) = S0*e3_j + dS_j(t) with dS_j in span(e1_j, e2_j).
        DSF phase exp(-i q.r_j) uses ACTUAL positions r_j = (j*a, 0, 0),
        which include the sublattice offsets (no cell folding).
    UNITS
        a [Angstrom], J [eV], S dimensionless; positions [Angstrom].
    BOUNDARY_CONDITIONS
        Open three-site cluster used only to exercise the local-frame DSF.
    LIMITATIONS
        Illustrative data only; not a relaxed equilibrium, no real exchange
        matrix, no dynamics.  Provided as a documented fixture for local-frame
        and sublattice-phase handling tests.

    Returns
    -------
    dict with 'name', 'description', 'n_sublattices', 'positions' (N,3),
        'local_frames' (N,3,3), 'equilibrium_spin' (N,3), 'J', 'S', 'a',
        'notes'.
    """
    a = 2.5
    J = 0.010
    S = 1.0
    thetas = np.deg2rad([0.0, 120.0, 240.0])
    N = len(thetas)
    positions = np.zeros((N, 3), dtype=np.float64)
    local_frames = np.zeros((N, 3, 3), dtype=np.float64)
    equilibrium = np.zeros((N, 3), dtype=np.float64)
    for j, th in enumerate(thetas):
        positions[j] = (j * a, 0.0, 0.0)
        e3 = np.array([np.cos(th), np.sin(th), 0.0])
        e2 = np.array([0.0, 0.0, 1.0])
        e1 = np.cross(e2, e3)
        local_frames[j, 0, :] = e1
        local_frames[j, 1, :] = e2
        local_frames[j, 2, :] = e3
        equilibrium[j] = e3
    return {
        "name": "triangular_120_3sublattice_noncollinear",
        "description": "3-sublattice 120-degree noncollinear spin structure; "
                       "local triads (e1,e2,e3) with e3 along each local "
                       "equilibrium spin.",
        "n_sublattices": int(N),
        "positions": positions,
        "local_frames": local_frames,
        "equilibrium_spin": equilibrium,
        "J": J,
        "S": S,
        "a": a,
        "notes": "exp(-i q.r_j) uses ACTUAL site positions (j*a,0,0) "
                 "including sublattice offsets; local frames rotate the "
                 "fluctuations into (e1,e2) before the q-phase sum.",
    }


# F. Peak positions (parabolic interpolation on log spectrum)
def peak_positions(omega, S, n_peaks=1, omega_min=None, omega_max=None,
                   min_height_frac=0.0):
    """Refined peak positions via parabolic interpolation on log(S).

    Finds local maxima of S, then for each peak fits a parabola through
    (omega[k-1], log S[k-1]), (omega[k], log S[k]), (omega[k+1], log S[k+1])
    and returns the vertex.  Interpolating on the LOG makes the estimator
    accurate for Lorentzian-like peaks (whose log is locally quadratic near
    the top).  Requires a uniformly spaced omega axis (true for fftshift'd
    fftfreq output and any contiguous slice thereof).

    Returns an array of refined peak positions [same units as omega], sorted
    by descending peak height, limited to n_peaks.
    """
    omega = np.asarray(omega, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if omega.ndim != 1 or S.shape != omega.shape:
        raise ValueError("omega and S must be 1-D with equal length")
    mask = np.ones_like(omega, dtype=bool)
    if omega_min is not None:
        mask &= (omega >= omega_min)
    if omega_max is not None:
        mask &= (omega <= omega_max)
    om = omega[mask]
    sm = S[mask]
    if om.size < 3:
        return np.array([], dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    logS = np.log(np.clip(sm, tiny, None))
    hmax = sm.max()
    out = []
    for k in range(1, om.size - 1):
        if not (logS[k] > logS[k - 1] and logS[k] >= logS[k + 1]):
            continue
        if min_height_frac and (sm[k] < min_height_frac * hmax):
            continue
        y0, y1, y2 = logS[k - 1], logS[k], logS[k + 1]
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0.0 else 0.0
        domega = om[1] - om[0]
        out.append((sm[k], om[k] + delta * domega))
    out.sort(key=lambda p: -p[0])
    return np.array([p[1] for p in out[:n_peaks]], dtype=np.float64)


# D. Lorentzian linewidth fit (numpy-only nonlinear least squares)
def _lorentzian_model(p, w):
    A, bg, w0, G = p
    return bg + A * (G * G) / ((w - w0) ** 2 + G * G)


def lorentzian_fit(omega, S, dt_fs, n_frames=None, window="hann",
                   fit_range=None, snr_min=3.0, r2_min=0.80,
                   overlap_frac=0.35, resolution_factor=2.0):
    """Fit a Lorentzian + constant background with explicit resolution gating.

    Model (G is the HALF width at half maximum, i.e. HWHM):

        L(omega) = bg + A * G^2 / ((omega - omega0)^2 + G^2)

    so L(omega0) = bg + A and L(omega0 +/- G) = bg + A/2.
    FWHM = 2*G  (HWHM and FWHM are explicitly distinguished).

    PHYSICAL_MODEL
        A damped precession S(t) ~ exp(-Gamma t) exp(i omega0 t) has a
        Lorentzian power spectrum with HWHM Gamma (angular frequency).
    ASSUMPTIONS
        Single isolated Lorentzian line on a locally flat background; uniform
        omega grid; homoscedastic residual noise.
    EQUATIONS
        lifetime (angular-frequency convention): tau = 1/Gamma = 1/G.
        energy HWHM: Gamma_E = hbar_eff * Gamma [eV];
        tau = hbar_eff / Gamma_E = 1/Gamma (identical value, different units).
        finite-trajectory resolution: domega = 2*pi/(n_frames*dt_fs).
        window main-lobe HWHM ~ ENBW_bins*domega/2 (reported as
        window_broadening).
    UNITS
        omega [rad/fs] -> Gamma, omega0, domega [rad/fs]; tau [fs];
        Gamma_E [eV].
    BOUNDARY_CONDITIONS
        Fit is localised around the strongest peak in the provided spectrum;
        pass a single-peak (e.g. positive-frequency) slice for a clean fit.
    LIMITATIONS
        A lifetime is reported ONLY when resolved=True.  resolved is False
        (status NOT_RESOLVED) when Gamma <= resolution_factor*domega, or R^2
        is too low, or SNR is inadequate, or peaks overlap, or the peak sits
        on the fit boundary.  In that case NO quantitative lifetime is
        emitted (all lifetime fields are None).  CRITICAL WARNING: numerical
        / spectral broadening (finite T, window ENBW, FFT bins) is NOT a
        physical lifetime; it must be de-convolved before interpreting G as
        a physical damping rate.  No scipy: fit = grid search (linear
        least squares for A,bg at each (omega0,G)) + Gauss-Newton refinement.

    Returns
    -------
    dict with keys: status, resolved, omega0, HWHM_Gamma, FWHM, amplitude,
        background, R2, tau, Gamma_E, tau_energy, domega, ENBW_bins,
        window_broadening, snr, overlap, reasons, warnings, raw.
    """
    omega = np.asarray(omega, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    if n_frames is None:
        n_frames = len(omega)
    domega = 2.0 * np.pi / (n_frames * dt_fs)
    _, _, _, ENBW = window_properties(window, n_frames)
    window_broadening = ENBW * domega / 2.0

    warnings = [
        "Numerical/spectral broadening (finite trajectory + window ENBW) is "
        "NOT a physical lifetime; de-convolve resolution before quoting G.",
        "HWHM (Gamma) is HALF width; FWHM = 2*Gamma.",
    ]

    def unresolved(reasons, omega0=None, A=None, bg=None, R2=None, snr=None,
                   overlap=False):
        warnings.append("NOT_RESOLVED: " + "; ".join(reasons)
                        + ". No quantitative lifetime is reported.")
        return {
            "status": "NOT_RESOLVED", "resolved": False,
            "omega0": omega0, "HWHM_Gamma": None, "FWHM": None,
            "amplitude": A, "background": bg, "R2": R2,
            "tau": None, "Gamma_E": None, "tau_energy": None,
            "domega": float(domega), "ENBW_bins": float(ENBW),
            "window_broadening": float(window_broadening),
            "snr": snr, "overlap": bool(overlap),
            "reasons": reasons, "warnings": warnings, "raw": None,
        }

    # Optional frequency restriction (e.g. positive frequencies only)
    if fit_range is not None:
        lo, hi = fit_range
        m = (omega >= lo) & (omega <= hi)
        omega = omega[m]
        S = S[m]
    if omega.size < 8:
        return unresolved(["too few points for a fit"])

    bg0 = float(np.median(S))
    A_init = float(S.max() - bg0)
    if A_init <= 0:
        return unresolved(["no positive peak above background"])
    ip = int(np.argmax(S))
    w0_init = float(omega[ip])

    # Localise a single peak: expand while S stays above a small threshold.
    thr = bg0 + 0.05 * A_init
    i0 = ip
    while i0 > 0 and S[i0 - 1] > thr:
        i0 -= 1
    i1 = ip
    while i1 < len(S) - 1 and S[i1 + 1] > thr:
        i1 += 1
    min_half = 12
    i0 = max(0, min(i0, ip - min_half))
    i1 = min(len(S) - 1, max(i1, ip + min_half))
    wr = omega[i0:i1 + 1]
    sr = S[i0:i1 + 1]
    width = wr[-1] - wr[0]
    if width <= 0:
        return unresolved(["empty local fit region"])

    # --- grid search: linear LS for (A,bg) at each (omega0, Gamma) ---
    w0_grid = np.linspace(w0_init - 2.0 * domega * ENBW,
                          w0_init + 2.0 * domega * ENBW, 13)
    g_grid = np.geomspace(0.5 * domega, max(4.0 * domega, width), 18)
    best = None
    for w0c in w0_grid:
        for gc in g_grid:
            g = gc ** 2 / ((wr - w0c) ** 2 + gc ** 2)
            X = np.column_stack([g, np.ones_like(g)])
            coef, *_ = np.linalg.lstsq(X, sr, rcond=None)
            model = X @ coef
            sse = float(np.sum((model - sr) ** 2))
            if coef[0] > 0 and (best is None or sse < best[0]):
                best = (sse, np.array([coef[0], coef[1], w0c, gc], dtype=np.float64))

    if best is None:
        return unresolved(["grid search failed to find a positive-amplitude line"])
    p = best[1]

    # --- Gauss-Newton refinement (numerical Jacobian, numpy only) ---
    def sse_of(pp):
        return float(np.sum((_lorentzian_model(pp, wr) - sr) ** 2))

    cur = sse_of(p)
    for _ in range(40):
        r = _lorentzian_model(p, wr) - sr
        h = np.array([
            1e-4 * (abs(p[0]) + 1e-9),
            1e-4 * (abs(p[1]) + 1e-9),
            0.02 * p[3] + 0.1 * domega,
            0.02 * p[3] + 1e-9,
        ])
        J = np.empty((len(wr), 4), dtype=np.float64)
        for k in range(4):
            pp = p.copy(); pm = p.copy()
            pp[k] += h[k]; pm[k] -= h[k]
            J[:, k] = (_lorentzian_model(pp, wr) - _lorentzian_model(pm, wr)) / (2.0 * h[k])
        dp, *_ = np.linalg.lstsq(J, -r, rcond=None)
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            pn = p + alpha * dp
            pn[3] = max(pn[3], 0.1 * domega)
            pn[2] = np.clip(pn[2], wr[0] - 2.0 * domega, wr[-1] + 2.0 * domega)
            pn[0] = max(pn[0], 0.0)
            snew = sse_of(pn)
            if snew < cur:
                p, cur, accepted = pn, snew, True
                break
        if not accepted:
            break

    A, bg, w0, G = [float(x) for x in p]
    model = _lorentzian_model(p, wr)
    sse = float(np.sum((model - sr) ** 2))
    sst = float(np.sum((sr - np.mean(sr)) ** 2))
    R2 = 1.0 - sse / sst if sst > 0 else 0.0
    noise = float(np.std(sr - model))
    noise = max(noise, 1e-12 * max(1.0, float(np.max(np.abs(sr)))))
    snr = A / noise

    # --- overlap detection: significant local maxima, light smoothing to
    #     suppress noise wiggles, and merge maxima closer than ~2*HWHM (two
    #     peaks blended within one linewidth are not "resolved" as separate
    #     lines; a genuine blend also degrades R2, which is gated below). ---
    sm = np.convolve(sr, np.ones(3) / 3.0, mode="same")
    sig_peaks = []
    for k in range(1, len(sm) - 1):
        if sm[k] > sm[k - 1] and sm[k] >= sm[k + 1] \
                and (sm[k] - bg) > overlap_frac * A:
            sig_peaks.append(k)
    min_sep = max(3, int(round(2.0 * G / domega)))
    merged = []
    for k in sig_peaks:
        if not merged or k - merged[-1] > min_sep:
            merged.append(k)
    peak_count = len(merged)
    overlap = peak_count > 1

    boundary = (w0 <= wr[0] + domega) or (w0 >= wr[-1] - domega)

    reasons = []
    if G <= resolution_factor * domega:
        reasons.append("Gamma=%.4g <= %.1f*domega=%.4g (resolution limit)"
                       % (G, resolution_factor, resolution_factor * domega))
    if R2 < r2_min:
        reasons.append("R2=%.3f < %.2f" % (R2, r2_min))
    if snr < snr_min:
        reasons.append("snr=%.2f < %.2f" % (snr, snr_min))
    if overlap:
        reasons.append("overlapping peaks detected (%d)" % peak_count)
    if boundary:
        reasons.append("peak at fit boundary")

    if reasons:
        return unresolved(reasons, omega0=w0, A=A, bg=bg, R2=R2,
                          snr=snr, overlap=overlap)

    tau = 1.0 / G
    Gamma_E = HBAR_EV_FS * G
    warnings.append("Fit passed resolution/SNR/overlap gates; G=%.5g rad/fs, "
                    "FWHM=%.5g rad/fs, tau=1/Gamma=%.4g fs." % (G, 2.0 * G, tau))
    return {
        "status": "PASS", "resolved": True,
        "omega0": w0, "HWHM_Gamma": G, "FWHM": 2.0 * G,
        "amplitude": A, "background": bg, "R2": R2,
        "tau": tau, "Gamma_E": Gamma_E,
        "tau_energy": HBAR_EV_FS / Gamma_E,
        "domega": float(domega), "ENBW_bins": float(ENBW),
        "window_broadening": float(window_broadening),
        "snr": float(snr), "overlap": bool(overlap),
        "reasons": [], "warnings": warnings,
        "raw": {"omega0": w0, "HWHM_Gamma": G, "FWHM": 2.0 * G,
                "amplitude": A, "background": bg, "R2": R2, "tau": tau},
    }


# E. Fixed-lattice vs coupled linewidth comparison
def _mean_std_se(x):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 0 else 0.0
    return mean, std, float(se), n


def compare_fixed_vs_coupled(omega, S_fixed, S_coupled, dt_fs, n_frames=None,
                             window="hann", **fit_kwargs):
    """Compare DSF linewidths for fixed-lattice vs spin-lattice-coupled runs.

    Each of S_fixed / S_coupled is a collection of spectra (one per SEED),
    shape (n_seed, n_omega) or a list of 1-D arrays.  Each seed spectrum is
    fitted with lorentzian_fit; the resulting HWHM values are summarised by
    mean, std (sample, ddof=1) and standard error (std/sqrt(n)).

    Returns the difference D = mean(Gamma_coupled) - mean(Gamma_fixed) with an
    error estimate sigma_D = sqrt(se_coupled^2 + se_fixed^2), plus all
    per-condition statistics and an explicit warning.

    PHYSICAL_MODEL
        A larger magnon linewidth in the coupled run indicates ADDITIONAL
        relaxation channels opened by spin-lattice coupling.
    ASSUMPTIONS
        Independent seeds, same omega axis, same dt_fs and window; each seed
        spectrum is a single resolved Lorentzian line.
    EQUATIONS
        D = mean(G_c) - mean(G_f); sigma_D = sqrt(se_c^2 + se_f^2);
        tau = 1/G per seed.  A naive additive scattering rate would be
        1/tau_c - 1/tau_f = G_c - G_f, but see LIMITATIONS.
    UNITS
        G, omega, domega [rad/fs]; tau [fs]; D [rad/fs].
    BOUNDARY_CONDITIONS
        Unresolved seeds (NOT_RESOLVED) are excluded from the seed statistics
        and reported; if too few resolve, status NOT_RESOLVED is returned.
    LIMITATIONS
        EXPLICIT WARNING: a linewidth DIFFERENCE is NOT automatically an
        additive magnon-phonon scattering rate.  Linewidths also contain
        numerical/spectral broadening (finite T, window ENBW), Landau damping,
        magnon-magnon scattering, dephasing and inhomogeneous broadening, and
        Matthiessen's rule need not hold.  Only after subtracting the
        resolution floor and verifying rate additivity can a scattering rate
        be inferred.  No GPU trajectories are validated here.
    """
    Sf = np.atleast_2d(np.asarray(S_fixed, dtype=np.float64))
    Sc = np.atleast_2d(np.asarray(S_coupled, dtype=np.float64))

    def fit_seeds(arr):
        gammas, taus, statuses = [], [], []
        for row in arr:
            r = lorentzian_fit(omega, row, dt_fs, n_frames=n_frames,
                               window=window, **fit_kwargs)
            statuses.append(r["status"])
            if r["resolved"]:
                gammas.append(r["HWHM_Gamma"])
                taus.append(r["tau"])
        return np.array(gammas), np.array(taus), statuses

    gf, tf, sf = fit_seeds(Sf)
    gc, tc, sc = fit_seeds(Sc)

    base = {
        "omega": None,
        "fixed": {"n_seeds": int(Sf.shape[0]), "statuses": sf,
                  "mean_std_se": None, "gammas": gf.tolist(), "taus": tf.tolist()},
        "coupled": {"n_seeds": int(Sc.shape[0]), "statuses": sc,
                    "mean_std_se": None, "gammas": gc.tolist(), "taus": tc.tolist()},
        "difference": None, "sigma_difference": None,
        "warnings": [
            "A linewidth DIFFERENCE is NOT automatically an additive "
            "magnon-phonon scattering rate (resolution floor, window ENBW, "
            "Landau/magnon-magnon/dephasing channels, Matthiessen validity "
            "must all be checked first).",
            "CPU independent reference only; no GPU trajectory validation.",
        ],
    }
    if gf.size == 0 or gc.size == 0:
        base["status"] = "NOT_RESOLVED"
        base["warnings"].append("Too few resolved seeds to form a difference.")
        return base

    mf, sdf, sef, nf = _mean_std_se(gf)
    mc, sdc, sec, nc = _mean_std_se(gc)
    D = mc - mf
    sigma_D = float(np.sqrt(sef ** 2 + sec ** 2))
    base["fixed"]["mean_std_se"] = {"mean": mf, "std": sdf, "se": sef, "n": nf}
    base["coupled"]["mean_std_se"] = {"mean": mc, "std": sdc, "se": sec, "n": nc}
    base["difference"] = D
    base["sigma_difference"] = sigma_D
    base["status"] = "PASS"
    return base
