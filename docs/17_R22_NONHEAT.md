# R22-NonHeat: ME smooth cutoff and field-work ledger

This document is the in-tree technical note for `v4.3.2-r22-nonheat`.
The derivation it refers to, together with the audit of the earlier releases,
is in `PHYSICS_DERIVATION.md` at the top of the repository.

## 1. Magneto-electric Hamiltonian and the r21 discontinuity

PHYSICAL_MODEL
  Katsura-Nagaosa-Balatsky / LAMMPS `pair_style spin/magelec` interaction

    H_ME = sum_{i<j} kappa_ij (E x r_ij) . (S_i x S_j),
    r_ij = r_j - r_i (minimum image under periodic boundaries).

ASSUMPTIONS
  S in mu_B (magnetic moment), r in A, kappa in eV/(A mu_B^2), E a
  dimensionless screened-polarisation vector (the code never converts E).
  Classical fixed-length spins, deterministic torque, ordered i<j pairs.

EQUATIONS (r21)
  The r21 code applies `if (d2 >= rc*rc) continue;`, i.e. a Heaviside step.
  Define the bare pair energy h_ij(r) = kappa (E x r).(S_i x S_j), which is
  linear in r and therefore NONZERO at r = r_c for a generic (E, S_i, S_j):

    lim_{r->rc-} H_ME = h_ij(rc) != 0 = lim_{r->rc+} H_ME.

  Consequently all of the following are discontinuous at r_c:

    F_i     = -dH/dr_i  = kappa (S_i x S_j) x E        (finite jump)
    omega_i = -dH/dS_i  = -S_j x D_ij,
              D_ij = kappa (E x r_ij)                  (finite jump)
    W_ab    = -r_a F_b (per ordered visit)             (finite jump)

  A pair crossing r_c therefore injects/removes energy instantaneously, which
  is unsuitable for long conservative dynamics.

## 2. R22 C2 smooth switching (opt-in, versioned)

EQUATIONS

    x      = (r - r_on) / (r_c - r_on),   r_on = r_c (1 - smooth_width)
    f_c(x) = 1                                     r <= r_on
           = 1 - 10 x^3 + 15 x^4 - 6 x^5            r_on < r < r_c
           = 0                                     r >= r_c
    df_c/dr = (-30 x^2 + 60 x^3 - 30 x^4) / (r_c - r_on)

    f_c(0)=1, f_c(1)=0, f'_c(0)=f'_c(1)=f''_c(0)=f''_c(1)=0.
    The switch is C2 at BOTH endpoints (NOT C3: f''' is discontinuous there).

    H_ij^smooth = f_c(r) h_ij(r)

  Let rhat = r_ij / r.  The complete derivatives are

    F_i = f_c kappa (S_i x S_j) x E + (df_c/dr) h_ij(r) rhat
    omega_i = -f_c S_j x D_ij                    (f_c multiplies the torque)
    W_ab = -sum_ordered (1/2) r_ij,a (F_i)_b     (all 9 components)

  The second term of F_i is the radial `df_c/dr` contribution; omitting it
  breaks force/virial continuity even when the energy looks continuous.
  Spin gradients carry f_c but NOT df_c/dr (h_ij is linear in S).

BOUNDARY_CONDITIONS
  Minimum image under a general triclinic cell; ordered pairs i<j with a
  half-weight per visit for the per-atom virial/energy; periodic images are
  handled by the existing `apply_mic` path and the `nx,ny,nz` replica loops.

UNITS
  energy eV, length A, spin mu_B, force eV/A, spin effective field eV/mu_B,
  virial eV.  `stress = -W / V` (a DIFFERENT quantity, different index use).

SOURCE_LOCATION
  `src/spin/spin_external.cu` (`me_switch`, `me_accumulate`,
  `gpu_add_magelec_full`, `gpu_add_magelec_list`),
  `src/spin/spin_external.cuh` (`me_cutoff_mode`, `me_r_on`,
  `set_magelec_switch`), `src/main_gpumd/run.cu` (`magelec` option parser).

COMPATIBILITY
  `cutoff=legacy` (DEFAULT) reproduces r21 bit-for-bit.  `cutoff=smooth`
  CHANGES the Hamiltonian inside [r_on, r_c] and must be requested explicitly;
  it is the validated mode for long conservative dynamics.

## 3. External-field energy and work ledger

PHYSICAL_MODEL

    H(t) = U_NEP(r,S) + H_Z(t,S) + H_ME(r,S) + K(v)
    H_Z  = -sum_i S_i . B_i(t)
    B_i(t) = B_peak * env(t), env = 1 | sin(2 pi f t + phi) | cos(...)
    f in THz, t in fs, B in eV/mu_B, S in mu_B.

EQUATIONS
  Splitting the total derivative,

    dH/dt = P_drive + P_damping + P_bath + P_other
    P_drive = -sum_i S_i . dB_i/dt           (explicit time dependence)
    P_damping = -gamma alpha/(1+alpha^2) sum_i |S_i x B_eff_i|^2 / |S_i|
                (Gilbert dissipation, deterministic, T = 0)
    P_bath = stochastic Stratonovich/Wong-Zakai injection with FDT amplitude
             sqrt(2 alpha kB T / (gamma (1+alpha^2) |S| dt)); its net energy
             exchange is NOT instrumented in this stage.

  Discrete form actually implemented (midpoint-sampled field):

    W_drive = sum_steps -sum_i S_i(t_n) . [ B(t_{n+1/2}) - B(t_{n-1/2}) ]

  This is the EXACT chain-rule work of the piecewise-constant drive used by
  the integrator, not a continuum approximation.  The first sample only primes
  the reference field, because B(t_{1/2}) is already in the initial H.

  r21 F16 identity (UNCHANGED):

    W_rot + W_noise + W_ext = -sum_i B_det . dS_i = dU_spin

  For a COMPLETE ledger (NVE, T=0, no add_bfield, strang split):

    residual = Delta E_total - (W_drive + W_spin_internal)

  measures integrator drift.  It is NOT printed when any source/sink is
  missing.  `E_spin_related` and `W_bath` are written as literal `NA`.

UNITS / BOUNDARY_CONDITIONS
  energy eV, time fs, spin mu_B, field eV/mu_B, omega THz.
  `time_step` in run.in is in fs; `run.cu` divides by TIME_UNIT_CONVERSION
  (10.18051) to obtain natural units for `global_time`, and the ledger's
  `time_fs` column is `global_time * TIME_UNIT_CONVERSION` (fs).

SOURCE_LOCATION
  `src/main_gpumd/run.cu` (`R22_LEDGER_HEADER`,
  `r22_append_ledger_row`, `r22_energy_from_thermo`,
  `r22_ledger_completeness`, `r22_write_ledger_sample`),
  `src/spin/spin_external.cu` (`compute_applied_b`, `reset_ledger`,
  `record_drive_work`, `gpu_accumulate_drive`, `read_drive_work`).

LIMITATIONS
  Damping, noise and bath energy exchanges are reported through the r21
  proxies (`W_rot`, `W_noise`, `W_ext`) and are NOT decomposed into separate
  thermodynamic source terms.  No claim of conservative energy balance is
  made for damped, stochastic, thermostatted or explicitly driven systems.

## 4. Explicitly excluded

The general-NEP spin total-energy flux (`compute_heat.cu`), `force.spin_flux_A`,
Green-Kubo spin thermal conductivity, and magnon/phonon/cross thermal
conductivity are OUT OF SCOPE and remain BLOCKED with their r21 warnings.
