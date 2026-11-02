/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include "utilities/gpu_vector.cuh"

void compute_heat(
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& heat_per_atom);

// R17 / round-8: same as above plus the CONVECTIVE spin-energy term
// e_i^spin = -1/2 S_i . B_i (B_i = torque = -dE/dS_i).  This is the only part
// of a magnon heat current that a NEP can supply -- see P72 in
// runs/spectra/REPORT_validation8.md.
void compute_heat(
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<float>& spin,
  const GPU_Vector<float>& torque,
  GPU_Vector<double>& heat_per_atom);

// Round 9 (task 48) -- the full spin-resolved heat current.
//   J   = J_p + J_m
//   J_p = sum_i W_i . v_i + sum_i e_i^spin v_i        (phonon / lattice)
//   J_m = sum_j A_j : dS_j/dt                          (magnon, conductive)
//   A_j = sum_{i != j} (r_i - r_j) (x) dU_i/dS_j       (9N, virial layout)
// dS_j/dt is the DETERMINISTIC Landau-Lifshitz-Gilbert rate
//   dS/dt = -gamma S x B - gamma alpha /|S| S x (S x B),
//   B     = torque (-dE/dS) + b_ext   (the external `add_bfield`, if any)
// evaluated from the spin and the torque of the current step.  The stochastic
// SLLG field is deliberately excluded: it is white noise, and its
// autocorrelation would add a delta(t) spike to the Green-Kubo integrand.
// The external field IS included, because it is a physical drive (FMR /
// spin pumping) and does real work on the spins; `b_ext` is the value that was
// actually applied on the current step, so a time-modulated field is handled.
// UNITS -- both channels are returned in the SAME time unit (GPUMD natural
// time), so that the caller can add them and apply a single conversion:
//   J_p : eV * A / natural-time   (W_i [eV] * v_i [A/natural-time])
//   J_m : A [A*eV/mu_B] * dS/dt [mu_B/natural-time]
// gamma is per femtosecond (see spin_llg), so the LLG rate comes out in
// mu_B/fs.  Since one natural time unit is TIME_UNIT_CONVERSION fs, a per-fs
// rate is converted with
//       dS/dt [mu_B/nattime] = dS/dt [mu_B/fs] * TIME_UNIT_CONVERSION
// (multiply, not divide -- a rate per longer unit is a LARGER number).
// Without that factor J_m is TIME_UNIT_CONVERSION (~10.18) times too small;
// that was the bug fixed in this revision.
// The Gilbert term  -sum_j r_j (B_j . dS_j/dt)  is a LOCAL source (spin ->
// lattice energy transfer), not a flux, and is therefore NOT included here; it
// is reported separately by the F16 spin-work accounting in spin_work.out.
// `spin_flux_A` may be empty (size 0), in which case J_m is zero.
// `heat_phonon` / `heat_magnon`, when non-null, receive the two channels
// separately (same 5N layout) so that Green-Kubo can build
//   kappa_pp ~ <J_p J_p>,  kappa_mm ~ <J_m J_m>,  kappa_pm ~ <J_p J_m>.
void compute_heat(
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& spin_flux_A,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<float>& spin,
  const GPU_Vector<float>& torque,
  const double spin_gamma,
  const double spin_alpha,
  const double b_ext[3],
  GPU_Vector<double>& heat_per_atom,
  GPU_Vector<double>* heat_phonon = nullptr,
  GPU_Vector<double>* heat_magnon = nullptr);

void compute_heat(
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential,
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& heat_per_atom);
