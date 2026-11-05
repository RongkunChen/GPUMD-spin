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
#include <vector>
class Box;
class Group;
class Force;
class Atom;

class Minimizer
{
public:
  Minimizer(
    const int fixed_group,
    const int fixed_grouping_method,
    const int number_of_atoms,
    const int number_of_steps,
    const double force_tolerance)
    : fixed_group_(fixed_group),
      fixed_grouping_method_(fixed_grouping_method),
      number_of_atoms_(number_of_atoms),
      number_of_steps_(number_of_steps),
      force_tolerance_(force_tolerance)
  {
    position_per_atom_temp_.resize(number_of_atoms * 3);
    potential_per_atom_temp_.resize(number_of_atoms);
    force_per_atom_temp_.resize(number_of_atoms * 3);

    force_square_max_.resize(1);
    total_potential_.resize(2);

    cpu_force_square_max_.resize(1);
    cpu_total_potential_.resize(2);
  }

  virtual ~Minimizer() = default;

  virtual void compute(
    Force& force,
    Box& box,
    Atom& atom,
    GPU_Vector<double>& position_per_atom,
    std::vector<Group>& group) = 0;

  // ---- V4: spin degrees of freedom -------------------------------------
  // When enabled, every minimizer also relaxes the magnetic moments with the
  // damped spin dynamics of LAMMPS `min_style spin` (see minimizer_spin.cuh),
  // and a state counts as converged only when the atomic force criterion AND
  // the transverse-torque criterion are both met.
  void set_spin_minimization(
    const bool enabled, const double tolerance, const double alpha_damp,
    const double discrete_factor)
  {
    spin_minimization_ = enabled;
    spin_tolerance_ = tolerance;
    alpha_damp_ = alpha_damp;
    discrete_factor_ = discrete_factor;
  }
  bool spin_minimization() const { return spin_minimization_; }
  // True when this minimizer will actually move magnetic moments: the keyword
  // asked for it and the run really carries spins.
  bool spin_active(Atom& atom) const;

protected:
  // Max transverse torque |w - s_hat(s_hat.w)| over all atoms, eV/mu_B.  Also
  // fills spin_w_perp_ so that spin_relax_apply() can move the spins without
  // recomputing it.  Returns 0 when the run carries no spins.
  double spin_torque_max(Atom& atom);
  // One geodesic step along the stored spin_w_perp_, sized by `w_max`.
  void spin_relax_apply(Atom& atom, const double w_max);
  // The spin step has no line search.  The trust region is adapted from the
  // total energy the caller has just evaluated: a step that raises the energy
  // halves it, a step that lowers it grows it back towards 1.  Without this the
  // LAMMPS default angle (2 pi/kappa = 0.63 rad) limit-cycles between two
  // configurations when the exchange field is stiff.
  void update_spin_step_scale(const double total_potential, const bool lattice_converged);
  double spin_step_scale() const { return spin_step_scale_; }
  // Forces (and, when spins are minimized, torques) at `position`.  The
  // spin-aware overload is what makes the magnetic energy visible to the
  // minimizer at all: the plain 7-argument Force::compute leaves the spin
  // blocks out of the potential.
  void compute_force_at(
    Force& force, Box& box, Atom& atom, std::vector<Group>& group,
    GPU_Vector<double>& position_per_atom);

  bool spin_minimization_ = false;
  double spin_tolerance_ = 1.0e-3; // eV/mu_B
  double alpha_damp_ = 1.0;        // LAMMPS min_modify alpha_damp
  double discrete_factor_ = 10.0;  // LAMMPS min_modify discrete_factor
  GPU_Vector<float> spin_w_perp_;
  GPU_Vector<double> spin_w_max_;
  double spin_step_scale_ = 1.0;
  double spin_last_energy_ = 0.0;
  bool spin_have_last_energy_ = false;

  void calculate_total_potential(const GPU_Vector<double>& potential_per_atom);

  void calculate_force_square_max(const GPU_Vector<double>& force_per_atom);

  int fixed_group_ = -1;
  int fixed_grouping_method_ = 0;
  int number_of_steps_ = 1000;
  int number_of_atoms_ = 0;
  double force_tolerance_ = 1.0e-6;

  GPU_Vector<double> position_per_atom_temp_;
  GPU_Vector<double> potential_per_atom_temp_;
  GPU_Vector<double> force_per_atom_temp_;
  GPU_Vector<double> force_square_max_;
  GPU_Vector<double> total_potential_;
  std::vector<double> cpu_force_square_max_;
  std::vector<double> cpu_total_potential_;
};
