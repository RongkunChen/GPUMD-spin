# V4 additions: magnetic moments in the minimisers, external field, magneto-electric coupling

This page documents the three keywords added on top of v3.1.0-r19.  Everything
here is additive: an existing `run.in` produces the same numbers as before
(verified byte-for-byte, see `validation_v4/compat`).

| keyword | LAMMPS counterpart |
|---|---|
| `minimize ... spin=<on\|off\|auto> spin_tol= alpha_damp= discrete_factor=` | `min_style spin` + `min_modify alpha_damp discrete_factor` |
| `magfield <H> <nx> <ny> <nz> [unit=…] [wave=…] [freq=…] [phase=…]` | `fix precession/spin zeeman H x y z` |
| `magelec <cutoff> <kappa> <Ex> <Ey> <Ez>` / `magelec_pair <ti> <tj> <kappa>` | `pair_style spin/magelec` + `pair_coeff * * magelec` |

---

## 1. `minimize ... spin=`

### What it does

The minimiser used to call the 7-argument `Force::compute()`, which passes
`spin = nullptr`; the NEP potential then sets `spin_active = false` and the
magnetic blocks of the model never enter the potential at all.  With `spin=on`
the minimiser calls the 11-argument overload, so the potential now contains the
magnetic energy, `atom.torque` contains `-dE/dS`, and both the SD/FIRE
coordinate step and a spin step run in the same iteration.

### Spin step (LAMMPS `min_style spin`)

```
dS_i/dtau = lambda [ omega_i - S_hat_i (S_hat_i . omega_i) ]      (sphere gradient)
theta_i   = lambda * |omega_i,perp| * 2 pi / (kappa |omega|_max)
S_i'      = |S_i| [ cos(theta_i) S_hat_i + sin(theta_i) w_hat_i,perp ]
```

* `lambda` = `alpha_damp` (default 1.0), `kappa` = `discrete_factor` (default 10),
  so the largest single-step rotation is `2 pi lambda / kappa` = 0.628 rad, the
  LAMMPS default.
* The update is the exact geodesic of that angle: `|S_i|` is conserved to
  round-off (measured 3.7e-6 relative on FeGe), no renormalisation is applied.
* The angle is a trust region, not an Euler step.  Its scale is adapted once per
  iteration from the total energy: a step that raises the energy halves it, a
  step that lowers it grows it back towards 1.  Without this adaptation the
  LAMMPS default angle limit-cycles on stiff exchange (FeGe: `w_perp` oscillates
  between 0.038 and 0.236 eV/muB and never converges); with it the same case
  converges in ~100 steps.

### Convergence

A state is converged only when **both** hold:

| criterion | expression | unit | default |
|---|---|---|---|
| atomic force | `max_i |F_i| < <force_tol>` | eV/A | 2nd positional argument |
| transverse torque | `max_i |omega_i - S_hat_i (S_hat_i . omega_i)| <= spin_tol` | eV/muB | `spin_tol=1e-3` |

The *transverse* part is used because at fixed `|S|` the longitudinal part of
`-dE/dS` does no work and cannot be driven to zero.

### Options

```
spin=on|off|auto   auto (default) = on when `spin_llg` was read
spin_tol=<x>       transverse torque tolerance, eV/mu_B
alpha_damp=<x>     fictitious damping, LAMMPS min_modify name
discrete_factor=<x> adaptive step factor, LAMMPS min_modify name
```

`minimize` runs at parse time, so `potential`, `spin_llg` and any
`magfield`/`magelec` line must precede it.

---

## 2. `magfield` — Zeeman field

```
E_Z = -sum_i S_i . B_ext                (B in eV/mu_B, S in mu_B)
B[eV/mu_B] = mu_B H[T] = 5.7883818066e-5 * H[T]    (g = 1)

# Why g = 1 and not 2: the integrator's gamma = 3.0385 (fs^-1 per eV/mu_B) *is*
# g*mu_B/hbar (1.7588e11 rad/s/T divided by 1.7276e4 T per eV/mu_B, in fs), so
# the Lande factor already sits in gamma.  The spin column holds the moment in
# mu_B (Fe = 2.0).  Multiplying the field by g as well doubled every Zeeman
# energy and halved every field.  `magfield ... g=2` restores the old numbers
# for inputs that carry an angular momentum instead of a moment.
```

```
magfield <H> <nx> <ny> <nz> [unit=tesla|ev] [wave=static|sin|cos] [freq=<THz>] [phase=<rad>]
```

* `unit=tesla` (default): `H` is the magnitude in Tesla, `(nx,ny,nz)` is the
  direction and is normalised, exactly like LAMMPS.
* `unit=ev`: the three numbers are the field vector itself in eV/mu_B (the same
  unit as the legacy `add_bfield`).
* `wave`/`freq`/`phase` give `B(t) = B env(t)` with `env = sin/cos(2 pi f t + phi)`,
  `f` in THz, `t` in fs.  This is the FMR / spin-pumping drive.
* The energy `-S_i.B` is added to `potential_per_atom`, so a minimiser, a
  thermostat and the energy-conservation guard see the same number without any
  `fix_modify energy` equivalent.
* Uniform field only (LAMMPS zeeman is uniform too).  A field gradient would
  also exert a force `grad(S.B)` on the atoms; that is not implemented.
* `magfield` and the legacy `add_bfield` are mutually exclusive (input error) so
  the same field can never be counted twice.

---

## 3. `magelec` — magneto-electric coupling

```
H_ME  = sum_{i<j} kappa_ij (E x r_ij) . (S_i x S_j)        r_ij = r_j - r_i
D_ij  = kappa_ij (E x r_ij)
omega_i = -sum_j S_j x D_ij = -dH/dS_i
F_i     = +sum_j kappa_ij (S_i x S_j) x E = -dH/dr_i
```

This is the electric-polarisation-induced Dzyaloshinskii-Moriya term of
Katsura-Nagaosa-Balatsky (PRL 95, 057205 (2005)) in the form implemented by
Tranchida et al. (JCP 372, 406 (2018)) as `pair_style spin/magelec`.

Units (r21 correction): `kappa` in **eV/(A mu_B^2)**, `E` a **dimensionless**
screened-polarisation direction/weight vector, `r` in A, `S` in mu_B.  Then
`H_ME = kappa (E x r).(S_i x S_j)` has units
`eV/(A mu_B^2) * A * mu_B^2 = eV`.  Declaring `E` in eV (r20 docs) would give
eV^2 and is a dimensional error; all physical field strength is absorbed in
`kappa`, which is why `magelec_pair` may legitimately be negative or zero.
LAMMPS carries a unit spin vector, so `kappa_GPUMD = kappa_LAMMPS / |S|^2`.

```
magelec <cutoff_A> <kappa> <Ex> <Ey> <Ez>
magelec_pair <type_i> <type_j> <kappa>     # optional per-pair override
```

The term contributes energy (0.5 to each atom), torque, force and virial, all
written with `atomicAdd` because `potential[n2]` is also touched by the thread
that owns atom `n2`.

### Sign convention

With `r_ij = r_j - r_i` the force is `+kappa (S_i x S_j) x E`; the LAMMPS manual
prints the opposite sign because its `r_ij` points the other way.  The
implementation is fixed by requiring `F = -dH/dr` and `omega = -dH/dS`, which is
verified by finite differences (2.5e-13 eV/A, 7e-11 eV/muB).

---

## 4. Where the terms are applied

Both terms are added at the end of `Force::compute(...)` (the overload that
receives `spin` and `torque`), i.e. exactly where the potential is evaluated.
MD, SD, FIRE and the box-changing FIRE therefore all see them with no extra
plumbing.  `Force::set_spin_external()` is called once by `Run`; with no
`magfield`/`magelec` line the pointer is inert and the old path is untouched.

## 5. Validation summary (FeGe, 216 atoms)

| check | result |
|---|---|
| Zeeman energy vs `-sum S.B`, single point | < 1e-10 relative |
| magelec energy vs float64 reference, single point | 6e-11 relative |
| per-atom torque/force vs float64 reference | at the NEP float32 noise floor (2e-7 … 6e-7) |
| finite-difference self-consistency of H_ME | 2.5e-13 eV/A, 7e-11 eV/muB |
| minimisation: both criteria met | 100 steps, f_max 1.5e-4, w_perp 6.6e-4 |
| `|S_i|` conservation in the spin step | 3.7e-6 relative |
| M(B) saturation, 50 T | cos(M,B) = 0.9976, dE = -M.B to 1e-4 |
| E-field induced chirality, kappa <= 2e-4 | dchi/dkappa = 4.4-4.8, odd in kappa |
| backward compatibility | MD and lattice-only `minimize` byte-identical to r19 |

Details, tables and the reproduction commands are in
`V4_FIELD_MINIMIZATION_REPORT.md` at the top of this tree.

---

## 6. V4.1 fixes and validation upgrades

| issue | symptom | fix |
|---|---|---|
| magelec virial written at `n+k` instead of `n+k*N` | stress tensor had only xx non-zero, trace 6x off the energy | component-major index |
| pair energy accumulated with a non-atomic `potential[n2] +=` | total energy 4-20% low, forces and torques perfect | one thread per atom, writes only its own slots, no atomics |
| cell list used below 5 cells per direction | the same pair reached through two wrapped cells | O(N^2) fallback when `num_bins < 5` |
| torque summed with `atomicAdd(float*)` | ~1e-8 accumulation noise | double registers, one rounding |
| two host syncs per force evaluation | pipeline stall every step | energies stay on the device until `report()` |
| `minimize ... spin=off` silently ignored magfield/magelec | field had no effect | explicit WARNING |
| `add_bfield` + `magfield` only checked one way | same field counted twice | both directions checked |

Validation now reaches machine precision: the Zeeman and magneto-electric
per-atom energies agree with the float64 reference to 6.8e-18 / 6.9e-18 eV and
the stress to 2e-15 eV.  The torque bottoms out at 1.6e-8 eV/muB, which is the
float32 storage quantum of `atom.torque`, not an implementation error.

**Reproducibility requirement**: a spin run on a structure without a velocity
column gets time-seeded initial velocities, so two runs of the *same* binary
differ.  Add `velocity <T> seed <n>` to any run that has to be compared bit for
bit.  All regression cases do.

Regression: `python tests/v4_regression.py` (5 groups, 33 checks, ~14 s).
