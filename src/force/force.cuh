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

#include "model/box.cuh"
#include "model/group.cuh"
#include "potential.cuh"
#ifdef GPUMD_WPE_ENABLED
#include "wpe_stage.cuh"
#endif
#include "utilities/common.cuh"
#include <memory>
#include <stdio.h>
#include <string>
#include <vector>

class RunInput;

// Forward declaration: force.cu is the only place that needs the definition,
// and this keeps the spin headers out of every file that includes force.cuh.
class SpinExternal;

class Force
{
public:
  Force(void);

  void parse_potential(
    const std::vector<std::string>& tokens,
    const Box& box,
    const int number_of_atoms,
    const RunInput& run_input);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

  void compute(
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    GPU_Vector<double>& velocity_per_atom,
    GPU_Vector<double>& mass_per_atom,
    int* position_image = nullptr,
    const float* spin = nullptr,
    float* torque = nullptr);

  // Zeroes the given arrays and evaluates one potential alone into them. The positions and the
  // float box are read as they are.
  // V4: external magnetic field / magneto-electric coupling.  The pointer is
  // owned by Run; a null pointer (the default) reproduces the pre-V4 behaviour
  // exactly.  The terms are added at the end of compute(), i.e. at the same
  // place the potential is evaluated, so MD, both minimizers and the replica
  // build all see them without further plumbing.
  void set_spin_external(SpinExternal* external) { spin_external_ = external; }

  // F-10 (spin-aware non-MD drivers).  compute_phonon (Hessian::find_H12) and
  // compute_cohesive / compute_elastic (Cohesive -> Minimizer_SD) go through
  // the short 7-argument compute() overload, which cannot carry spin/torque as
  // arguments.  Before this registration that overload left NEP::spin_ ==
  // nullptr, so a nep4_spin potential silently evaluated its NON-MAGNETIC
  // Hamiltonian, and neither magfield nor magelec contributed to the
  // energy/force/virial.  Run binds the (already replicated) device buffers
  // with a scoped guard around each of those three commands only; the long MD
  // overload still overrides them explicitly on every call.  Both pointers
  // null (no binding) keeps the old behaviour, including `minimize spin=off`.
  void set_static_spin(const float* spin, float* torque)
  {
    static_spin_ = spin;
    static_torque_ = torque;
  }
  bool has_static_spin() const
  {
    return static_spin_ != nullptr && static_torque_ != nullptr;
  }

  void compute_one_potential(
    const int potential_index,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);

#ifdef GPUMD_WPE_ENABLED
  void wpe_process_command(
    const std::vector<std::string>& tokens,
    int current_number_of_atoms);
#endif

  void finalize();

  // Round 9 (task 48): the magnetic heat-current tensor
  //   A_j = sum_{i != j} (r_i - r_j) (x) dU_i/dS_j ,  9N doubles,
  // component layout identical to the per-atom virial
  //   (0..8 = xx yy zz xy xz yz yx zx zy).
  // Filled by the spin-aware NEP; empty (size 0) when the run has no spins.
  GPU_Vector<double> spin_flux_A;

  int get_number_of_types(FILE* fid_potential);
  void set_multiple_potentials_mode(std::string mode);
  void set_temperature_range(
    const double temperature1, const double temperature2, const int number_of_steps);
  void advance_temperature();
  int get_number_of_potentials() const;
  Potential& get_potential(const int index);

private:
#ifdef GPUMD_WPE_ENABLED
  WpeGpumdStageState wpe_stage_state_;
#endif
  std::unique_ptr<Potential> create_potential(
    const std::vector<std::string>& tokens,
    FILE* fid_potential,
    char* potential_name,
    const int num_types,
    const Box& box,
    const int number_of_atoms,
    const RunInput& run_input,
    bool& is_nep);

  double temperature = 0;
  double delta_T;
  SpinExternal* spin_external_ = nullptr; // V4, see set_spin_external()
  const float* static_spin_ = nullptr;    // F-10, see set_static_spin()
  float* static_torque_ = nullptr;        // F-10, see set_static_spin()
  std::vector<std::unique_ptr<Potential>> potentials;
  bool is_fcp = false;
  bool has_non_nep = false;
  std::string multiple_potentials_mode_ = "observe"; // "observe" or "average"
  std::string atom_types[NUM_ELEMENTS];

  void check_types(const std::string& file_potential);
  void prepare_compute(
    const int number_of_atoms,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom,
    int* position_image);
  void compute_potentials(
    const int number_of_atoms,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);
  void compute_single_potential(
    Potential& potential,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    const std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);
};
