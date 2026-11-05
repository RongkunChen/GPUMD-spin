/*
    Spin Route B, V4: spin degrees of freedom inside the energy minimizers.

    The algorithm is the one LAMMPS uses for `min_style spin`
    (min_spin.html): a damped spin dynamics with an adaptive timestep,

        ds_i/dt = lambda  s_i x (omega_i x s_i)  =  lambda  omega_i,perp

    which is gradient descent on the sphere of constant |S_i| -- the
    precession term is dropped, so the energy decreases monotonically for a
    small enough step.  LAMMPS sizes the step from the largest precession
    frequency in the system,

        dt_max = 2 pi / (kappa |omega|_max),      kappa = discrete_factor = 10

    and this file uses the same rule, so the per-step rotation angle is

        theta_i = alpha_damp * |omega_i,perp| * 2 pi / (kappa |omega|_max)
                <= 2 pi alpha_damp / kappa      (= 0.628 rad by default)

    The update itself is the exact geodesic of that angle,

        s' = |s| [ cos(theta) s_hat + sin(theta) omega_perp_hat ]

    so |S_i| is conserved to round-off and no renormalisation is needed.  The
    step is a trust-region move with a hard cap, not an Euler step: that is what
    keeps the minimizer stable when the exchange field is large.

    Convergence is judged on the *transverse* torque, because the longitudinal
    part of -dE/dS does no work at fixed |S| and can never be driven to zero:

        max_i |omega_i - s_hat_i (s_hat_i . omega_i)| <= spin_tolerance

    with spin_tolerance in eV/mu_B.  A spin run is converged only when the
    atomic force criterion AND this one are both satisfied.
*/

#pragma once
#include "utilities/gpu_vector.cuh"

// Applies one damped-spin relaxation step to all atoms.
//   spin   : 3N floats, component-major (modified in place)
//   torque : 3N floats, -dE/dS in eV/mu_B (not modified)
//   w_perp : 3N workspace, w_max : 1 double workspace
// Returns the largest transverse torque *before* the step (eV/mu_B).
double spin_relax_step(
  const int number_of_atoms,
  float* spin,
  const float* torque,
  const double alpha_damp,
  const double discrete_factor,
  GPU_Vector<float>& w_perp,
  GPU_Vector<double>& w_max);

// The two halves of the step, for callers that must test convergence before
// moving any degree of freedom.
double spin_torque_measure(
  const int number_of_atoms,
  float* spin,
  const float* torque,
  GPU_Vector<float>& w_perp,
  GPU_Vector<double>& w_max);

void spin_relax_apply(
  const int number_of_atoms,
  float* spin,
  const double alpha_damp,
  const double discrete_factor,
  const double w_max,
  GPU_Vector<float>& w_perp);
