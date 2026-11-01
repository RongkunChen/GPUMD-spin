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
#include "dftd3.cuh"
#include "neighbor.cuh"
#include "potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <vector>

class RunInput;

struct NEP_Data {
  GPU_Vector<float> f12x; // 3-body or manybody partial forces
  GPU_Vector<float> f12y; // 3-body or manybody partial forces
  GPU_Vector<float> f12z; // 3-body or manybody partial forces
  GPU_Vector<float> Fp;
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> descriptor_parameters_type_pair;
  GPU_Vector<int> NN_radial;    // radial neighbor list
  GPU_Vector<int> NL_radial;    // radial neighbor list
  GPU_Vector<int> NN_angular;   // angular neighbor list
  GPU_Vector<int> NL_angular;   // angular neighbor list
  GPU_Vector<float> parameters; // parameters to be optimized
  GPU_Vector<float> q_spin;     // spin descriptor components (N * dim_spin)
  GPU_Vector<int> use_spin;     // per-type spin switch (Route B)
  std::vector<int> cpu_NN_radial;
  std::vector<int> cpu_NN_angular;
};

class NEP : public Potential
{
public:
  NEP_Data nep_data;
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.0f;
    int model_type = 0; // 0=potential, 3=temperature-dependent free energy
    float rc_radial_max = 0.0f;
    float rc_radial_max_inv = 0.0f; 
    float rc_radial[NUM_ELEMENTS];     // radial cutoff
    float rc_angular[NUM_ELEMENTS];    // angular cutoff
    int MN_radial = 200;
    int MN_angular = 100;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 0, 1, 2, ..., L_max
    int dim_angular;
    int has_q_222 = 0;
    int has_q_1111 = 0;
    int has_q_112 = 0;
    int has_q_123 = 0;
    int has_q_233 = 0;
    int has_q_134 = 0;
    int num_L;
    int basis_size_radial = 8;
    int basis_size_angular = 8;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int num_types = 0;
    // spin (magnetic) extension
    bool enable_spin = false; // true for a spin-aware NEP model
    int dim_spin = 0;         // number of spin descriptor components
    int spin_layout_version = 1; // 1 = implicit, 2 = explicit (Senv), 3 = explicit (Sia)
    // true when the 4th spin block is the single-ion anisotropy "Sia" (layout 3)
    bool spin_use_sia = false;
    int spin_block_mask = 31;  // F26 ablation mask read from nep.txt
    float spin_mag_scale = 0.25f; // |S| -> u scaling of the Smag block
    std::vector<int> use_spin; // per atom type: 1 if the type carries a spin vector
  };

  struct ANN {
    int dim = 0;                   // dimension of the descriptor
    int num_neurons1 = 0;          // number of neurons in the 1st hidden layer
    int num_para = 0;              // number of parameters
    int num_para_ann = 0;          // number of parameters for the ANN part
    const float* w0[NUM_ELEMENTS]; // weight from the input layer to the hidden layer
    const float* b0[NUM_ELEMENTS]; // bias for the hidden layer
    const float* w1[NUM_ELEMENTS]; // weight from the hidden layer to the output layer
    const float* b1;               // bias for the output layer
    const float* c;
    const float* c_type_pair;
    const float* q_scaler;
  };

  struct ZBL {
    bool enabled = false;
    bool flexible = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
    int num_types;
  };

  struct ExpandedBox {
    int num_cells[3];
    float h[18];
  };

  struct Small_Box_Data {
        GPU_Vector<int> NN_radial;
        GPU_Vector<int> NL_radial;
        GPU_Vector<int> NN_angular;
        GPU_Vector<int> NL_angular;
        GPU_Vector<float> r12;
    } small_box_data;

  NEP(const char* file_potential, const int num_atoms, const RunInput& run_input);
  virtual ~NEP(void);

  // Spin Route B: receive the MD spin array and the torque output buffer.
  void set_spin(const float* spin, float* torque) override
  {
    spin_ = spin;
    torque_ = torque;
  }

  // Round 9 (task 48): optional output buffer for the magnetic heat-current
  // tensor  A_j = sum_{i != j} (r_i - r_j) (x) dU_i/dS_j,  laid out as 9N
  // doubles in the same component order as the per-atom virial:
  //   xx yy zz xy xz yz yx zx zy   ->   slots 0..8
  // A null pointer disables the (small) extra kernel work completely.
  void set_spin_flux_output(double* A) override { Aspin_out_ = A; }
  virtual void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  virtual void compute(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  const GPU_Vector<int>& get_NN_radial_ptr();

  const GPU_Vector<int>& get_NL_radial_ptr();

  // Stage03A.1: read-only view of the parameters this object actually read from
  // nep.txt.  A verification driver has to report the real configuration
  // (mask / layout / dimension / types / scaler) rather than echo the requested
  // one back.  Pure accessor: no behaviour change, nothing else uses it.
  const ParaMB& get_paramb() const { return paramb; }
  const ANN& get_ann() const { return annmb; }

private:
  ParaMB paramb;
  ANN annmb;
  ZBL zbl;
  ExpandedBox ebox;
  DFTD3 dftd3;
  NeighborManager neighbor_manager;
  const float* spin_ = nullptr; // MD spin vectors (3N), nullptr when not spin-aware
  float* torque_ = nullptr;     // magnetic torque output (3N)
  double* Aspin_out_ = nullptr; // magnetic heat-current tensor output (9N), optional

  // The neighbour part of the torque has to be accumulated with atomics.  In
  // single precision the result then depends on the order in which the threads
  // happen to run, which makes two runs of the very same input differ in the
  // last bits and destroys bitwise reproducibility.  Accumulating in double
  // precision -- like every other reduction in the NEP kernels -- pushes the
  // order dependence to ~1e-16, far below the float the integrator consumes.
  GPU_Vector<double> torque_accumulator_;

  void update_potential(float* parameters, ANN& ann);

  void compute_small_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_large_box(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_small_box(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  void compute_large_box(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& position,
    GPU_Vector<double>& potential,
    GPU_Vector<double>& force,
    GPU_Vector<double>& virial);

  bool has_dftd3 = false;
  void initialize_dftd3(const RunInput& run_input);
};
