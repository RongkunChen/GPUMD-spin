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
#include "potential.cuh"
#include "nep_compile.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <memory>
class Parameters;
class Dataset;

struct NEP_Data {
  GPU_Vector<float> descriptors; 
  GPU_Vector<float> Fp;          // gradient of descriptors
  GPU_Vector<float> sum_fxyz;
  GPU_Vector<float> parameters; // parameters to be optimized
};

class NEP : public Potential
{
public:
  struct ParaMB {
    bool use_typewise_cutoff_zbl = false;
    float typewise_cutoff_zbl_factor = 0.65f;
    float rc_radial[NUM_ELEMENTS];  // radial cutoff
    float rc_angular[NUM_ELEMENTS]; // angular cutoff
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int n_max_radial = 0;  // n_radial = 0, 1, 2, ..., n_max_radial
    int n_max_angular = 0; // n_angular = 0, 1, 2, ..., n_max_angular
    int L_max = 0;         // l = 1, 2, ..., L_max
    int dim_angular;
    int has_q_222;
    int has_q_1111;
    int has_q_112;
    int has_q_123;
    int has_q_233;
    int has_q_134;
    int num_L;
    int num_types = 0;
    int num_types_sq = 0;
    int num_c_radial = 0;
    int version = 4;
    // spin (magnetic) extension
    int enable_spin = 0;  // 1 if the model has spin descriptor components
    int dim_radial = 0;   // number of radial descriptor components
    int dim_spin = 0;     // number of spin descriptor components (5 * (n_max_angular + 1))
    float spin_mag_scale = 0.25f; // |S| -> u scaling of the Smag block (nep.txt)
    int spin_use_sia = 0; // 1 when the 4th spin block is Sia (layout 3), else Senv
    int spin_block_mask = 31;  // F26: which of the five spin blocks are active
  };

  struct ANN {
    int dim = 0;                    // dimension of the descriptor
    int num_neurons1 = 0;           // number of neurons in the hidden layer
    int num_neurons2 = 0;           // number of neurons in the second hidden layer
    int num_hidden_layers = 0;      // number of hidden layers
    int one_ann_no_bias = 0;        // number of parameters in the ANN without bias
    int num_para = 0;               // number of parameters
    const float* wb[NUM_ELEMENTS];  // weights and biases for the hidden layer
    const float* b;                 // bias for the output layer
    const float* c;                 // for elements in descriptor
    const float* rc;
  };

  struct ZBL {
    bool enabled = false;
    bool flexible = false;
    float rc_inner = 1.0f;
    float rc_outer = 2.0f;
    int num_types;
    float para[550];
    int atomic_numbers[NUM_ELEMENTS];
  };

  NEP(
    Parameters& para,
    int N,
    int version,
    int deviceCount);
  void find_force(
    Parameters& para,
    const float* parameters,
    std::vector<Dataset>& dataset,
    bool calculate_q_scaler,
    int deviceCount);

  // Read-only diagnostic accessor (Stage03A verification only).
  // Returns the descriptor array that the last find_force() produced on
  // `device`.  It exists so that a verification driver can compare the generic
  // trainer path against the specialized kernels on identical weights without
  // reimplementing any descriptor math.  It does not change any behaviour of
  // training.
  const float* get_descriptor_buffer(int device = 0) const
  {
    return nep_data[device].descriptors.data();
  }
  int get_descriptor_size(int device = 0) const
  {
    return static_cast<int>(nep_data[device].descriptors.size());
  }

  // Read-only diagnostic accessor (Stage03A verification only).
  // Exposes the neighbour data (NL, r12) that the last find_force() built on
  // `device`, so that a verification driver can feed *exactly* the same lists
  // to the specialized kernels.  Without this the two backends would each build
  // their own neighbour list and any difference would be ambiguous.
  NEP_Neighbor& get_neighbor(int device = 0) { return neighbor[device]; }

private:
  ParaMB paramb;
  ANN annmb[16];
  NEP_Data nep_data[16];
  ZBL zbl;
  std::unique_ptr<NEP_Compile> compiled_kernel_;
  void update_potential(Parameters& para, float* parameters, ANN& ann);
};
