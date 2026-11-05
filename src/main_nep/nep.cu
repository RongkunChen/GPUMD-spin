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

/*----------------------------------------------------------------------------80
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "dataset.cuh"
#include "mic.cuh"
#include "nep.cuh"
#include "nep_precision.h"
#include "parameters.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/nep_utilities.cuh"
#ifdef S3A2_DIAG
#include "s3a2_diag.cuh"
#endif
#include "../utilities/spin_descriptor.cuh"
#include <cstring>

static __global__ void find_descriptors_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    nep_acc_t q[MAX_NUM_N] = {0.0};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);

      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gn12 += fn12[k] * annmb.c[c_index];
        }
        q[n] += static_cast<nep_acc_t>(gn12);
      }
    }
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      g_descriptors[n1 + n * N] = static_cast<float>(q[n]);
    }
  }
}

static __global__ void find_descriptors_angular(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_descriptors,
  float* g_sum_fxyz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    int neighbor_number = g_NN[n1];
    float q[MAX_DIM_ANGULAR] = {0.0f};

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      nep_acc_t s_acc[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < neighbor_number; ++i1) {
        int index = g_NN_sum[n1] + i1;
        int n2 = g_NL[index];
        float x12 = g_x12[index];
        float y12 = g_y12[index];
        float z12 = g_z12[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s_acc);
      }
      float s[NUM_OF_ABC];
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        s[abc] = static_cast<float>(s_acc[abc]);
      }
      find_q(paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134, paramb.n_max_angular + 1, n, s, q);
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1] = s[abc];
      }
    }

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      for (int l = 0; l < paramb.num_L; ++l) {
        int ln = l * (paramb.n_max_angular + 1) + n;
        g_descriptors[n1 + ((paramb.n_max_radial + 1) + ln) * N] = q[ln];
      }
    }
  }
}

// Spin (magnetic) descriptor, Route B: the spin vector S_i enters as a rank-1
// equivariant feature. No virtual/pseudo atoms, no doubled type map.
//   Sex  q_n = sum_j g_n(r) (S_k . S_j)
//   San  q_n = sum_j g_n(r) (S_k.rhat)(S_j.rhat)
//   Sdm  q_n = sum_j g_n(r) rhat.(S_k x S_j)     (pseudoscalar, DM interaction)
//   Senv q_n = sqrt(3/4pi) sum_j g_n(r) (S_k.rhat)
// The radial functions g_n(r) share the trainable Chebyshev coefficients of the
// angular descriptor, so no extra descriptor parameters are introduced.
// The SPIN_ENV_FACTOR and SPIN_LAYOUT_VERSION constants now come from the
// shared descriptor core in utilities/spin_descriptor.cuh, which the MD kernel
// (force/nep.cu), the replica kernel (main_replica/force/nep.cu) and the JIT
// generator (nep_compile.cu) all use as well.
// On-site |S_i| channel (Smag).  See the comment in force/nep.cu: the cDFT
// magnetic-force label has a large longitudinal part that is almost a pure
// function of |S_i|, which none of Sex/San/Sdm/Senv can represent because they
// are all strictly linear in S_i.
// The scale itself is a model parameter: paramb.spin_mag_scale, written to
// nep.txt as `spin_mag_scale <value>` and read back by the MD side, so that a
// model trained with a different scale is evaluated with that same scale.

static __global__ void find_descriptors_spin(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_sx,
  const float* __restrict__ g_sy,
  const float* __restrict__ g_sz,
  const int* __restrict__ g_use_spin,
  float* g_descriptors)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    const int t1 = g_type[n1];
    const int neighbor_number = g_NN[n1];
    const int n_max = paramb.n_max_angular;
    const int nn1 = paramb.n_max_angular + 1;
    const int base = paramb.dim_radial + paramb.dim_angular;
    float skx = g_sx[n1];
    float sky = g_sy[n1];
    float skz = g_sz[n1];
    if (!g_use_spin[t1]) {
      skx = 0.0f;
      sky = 0.0f;
      skz = 0.0f;
    }

    float q_sex[MAX_NUM_N] = {0.0f};
    float q_san[MAX_NUM_N] = {0.0f};
    float q_sdm[MAX_NUM_N] = {0.0f};
    // Slot 4: SIA (layout 3) or legacy Senv (layout 1/2).  Both are computed;
    // paramb.spin_use_sia selects which one reaches the ANN input.
    float q_sia[MAX_NUM_N] = {0.0f};
    float q_senv[MAX_NUM_N] = {0.0f};
    const float s2 = skx * skx + sky * sky + skz * skz;

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float x12 = g_x12[index];
      float y12 = g_y12[index];
      float z12 = g_z12[index];
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      // B-8: the MD kernels skip a zero-length bond; the trainer must do the
      // same, otherwise one duplicate atom in the training set produces inf and
      // poisons every parameter while gpumd would silently run fine.
      if (d12 < 1.0e-8f) {
        continue;
      }
      float d12inv = 1.0f / d12;
      float rx = x12 * d12inv;
      float ry = y12 * d12inv;
      float rz = z12 * d12inv;
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);

      float gn12[MAX_NUM_N];
      for (int n = 0; n <= n_max; ++n) {
        float g = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          g += fn12[k] * annmb.c[c_index];
        }
        gn12[n] = g;
      }

      float sjx = g_sx[n2];
      float sjy = g_sy[n2];
      float sjz = g_sz[n2];
      // A type with use_spin == 0 carries no spin degree of freedom, so the
      // descriptor must not depend on its spin vector at all.  (The MD-side
      // kernels did this from the start; the trainer did not, which made the
      // two disagree for `spin 1 0` models.)
      if (!g_use_spin[t2]) {
        sjx = 0.0f;
        sjy = 0.0f;
        sjz = 0.0f;
      }
      float sksj = skx * sjx + sky * sjy + skz * sjz;
      float a = skx * rx + sky * ry + skz * rz;
      float b = sjx * rx + sjy * ry + sjz * rz;
      float cx = sky * sjz - skz * sjy; // S_k x S_j
      float cy = skz * sjx - skx * sjz;
      float cz = skx * sjy - sky * sjx;
      float p = rx * cx + ry * cy + rz * cz;

      for (int n = 0; n <= n_max; ++n) {
        q_sex[n] += gn12[n] * sksj;
        q_san[n] += gn12[n] * a * b;
        q_sdm[n] += gn12[n] * p;
        // single-ion anisotropy: sum_j g_n [3/2 (S_i.rhat)^2 - 1/2 |S_i|^2]
        q_sia[n] += gn12[n] * (1.5f * a * a - 0.5f * s2);
        // legacy Senv (layout <= 2), T-odd
        q_senv[n] += gn12[n] * a;
      }
    }

    // F26 ablation: the trainer keeps its OWN copy of the descriptor loop (it
    // does not call the shared kernels), so the mask has to be applied here as
    // well.  Missing this is exactly the failure the first version of the test
    // caught: training silently used all five blocks while gpumd evaluated two.
    const int blk = paramb.spin_block_mask;
    for (int n = 0; n <= n_max; ++n) {
      g_descriptors[n1 + (base + n) * N] =
        (blk & SPIN_BLOCK_BIT_SEX) ? q_sex[n] : 0.0f;
      g_descriptors[n1 + (base + nn1 + n) * N] =
        (blk & SPIN_BLOCK_BIT_SAN) ? q_san[n] : 0.0f;
      g_descriptors[n1 + (base + 2 * nn1 + n) * N] =
        (blk & SPIN_BLOCK_BIT_SDM) ? q_sdm[n] : 0.0f;
      g_descriptors[n1 + (base + 3 * nn1 + n) * N] =
        (blk & SPIN_BLOCK_BIT_SLOT4)
          ? (paramb.spin_use_sia ? q_sia[n] : (SPIN_ENV_FACTOR * q_senv[n]))
          : 0.0f;
    }

    // Smag: on-site |S| channel, u^(n+1) with u = |S| * SPIN_MAG_SCALE
    const float smag = sqrt(skx * skx + sky * sky + skz * skz);
    const float u_mag = smag * paramb.spin_mag_scale;
    float un_mag = 1.0f;
    for (int n = 0; n <= n_max; ++n) {
      un_mag *= u_mag;
      g_descriptors[n1 + (base + 4 * nn1 + n) * N] =
        (blk & SPIN_BLOCK_BIT_SMAG) ? un_mag : 0.0f;
    }
  }
}

NEP::NEP(
  Parameters& para,
  int N,
  int version,
  int deviceCount)
{
  paramb.version = version;
  paramb.use_typewise_cutoff_zbl = para.use_typewise_cutoff_zbl;
  paramb.typewise_cutoff_zbl_factor = para.typewise_cutoff_zbl_factor;
  paramb.num_types = para.num_types;
  for (int t = 0; t < paramb.num_types; ++t) {
    paramb.rc_radial[t] = para.rc_radial[t];
    paramb.rc_angular[t] = para.rc_angular[t];
  }
  paramb.n_max_radial = para.n_max_radial;
  paramb.n_max_angular = para.n_max_angular;
  paramb.L_max = para.L_max;
  paramb.has_q_222 = para.has_q_222;
  paramb.has_q_1111 = para.has_q_1111;
  paramb.has_q_112 = para.has_q_112;
  paramb.has_q_123 = para.has_q_123;
  paramb.has_q_233 = para.has_q_233;
  paramb.has_q_134 = para.has_q_134;
  paramb.num_L = paramb.L_max;
  paramb.dim_radial = para.dim_radial;
  paramb.dim_angular = para.dim_angular;
  paramb.dim_spin = para.dim_spin;
  paramb.enable_spin = para.enable_spin ? 1 : 0;
  paramb.spin_mag_scale = para.spin_mag_scale;
  paramb.spin_use_sia = para.spin_use_sia;
  paramb.spin_block_mask = para.spin_block_mask;
  if (para.has_q_222) {
    paramb.num_L += 1;
  }
  if (para.has_q_1111) {
    paramb.num_L += 1;
  }
  if (para.has_q_112) {
    paramb.num_L += 1;
  }
  if (para.has_q_123) {
    paramb.num_L += 1;
  }
  if (para.has_q_233) {
    paramb.num_L += 1;
  }
  if (para.has_q_134) {
    paramb.num_L += 1;
  }
  paramb.dim_angular = (para.n_max_angular + 1) * paramb.num_L;

  paramb.basis_size_radial = para.basis_size_radial;
  paramb.basis_size_angular = para.basis_size_angular;
  paramb.num_types_sq = para.num_types * para.num_types;
#ifdef USE_CJ
  paramb.num_c_radial =
    paramb.num_types * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#else
  paramb.num_c_radial =
    paramb.num_types_sq * (para.n_max_radial + 1) * (para.basis_size_radial + 1);
#endif

  zbl.enabled = para.enable_zbl;
  zbl.flexible = para.flexible_zbl;
  zbl.rc_inner = para.zbl_rc_inner;
  zbl.rc_outer = para.zbl_rc_outer;
  for (int n = 0; n < para.atomic_numbers.size(); ++n) {
    zbl.atomic_numbers[n] = para.atomic_numbers[n];        // starting from 1
  }
  if (zbl.flexible) {
    zbl.num_types = para.num_types;
    int num_type_zbl = (para.num_types * (para.num_types + 1)) / 2;
    for (int n = 0; n < num_type_zbl * 10; ++n) {
      zbl.para[n] = para.zbl_para[n];
    }
  }

  for (int device_id = 0; device_id < deviceCount; device_id++) {
    gpuSetDevice(device_id);
    annmb[device_id].dim = para.dim;
    annmb[device_id].num_neurons1 = para.num_neurons1;
    annmb[device_id].num_hidden_layers = para.num_hidden_layers;
    annmb[device_id].num_para = para.number_of_variables;
    if (para.num_hidden_layers == 2) {
      annmb[device_id].num_neurons2 = para.num_neurons2;
      annmb[device_id].one_ann_no_bias = (annmb[device_id].dim + 1) * annmb[device_id].num_neurons1 +
        (annmb[device_id].num_neurons1 + 2) * annmb[device_id].num_neurons2;
    } else {
      annmb[device_id].one_ann_no_bias = (annmb[device_id].dim + 2) * annmb[device_id].num_neurons1;
    }

    nep_data[device_id].descriptors.resize(N * annmb[device_id].dim);
    nep_data[device_id].Fp.resize(N * annmb[device_id].dim);
    nep_data[device_id].sum_fxyz.resize(N * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
    nep_data[device_id].parameters.resize(annmb[device_id].num_para);
  }
  // The runtime-specialized kernels implement the spin descriptor from
  // interface version 3 onwards; has_spin() reports whether they are present
  // and the generic kernels are used otherwise.
  if (para.nep_compile && para.prediction == 0) {
    CHECK(gpuSetDevice(0));
    compiled_kernel_.reset(new NEP_Compile(
      make_nep_compile_config(para, NEP_Compile_Mode::NEP)));
    if (!compiled_kernel_->is_valid()) {
      compiled_kernel_.reset();
    }
  }
}

void NEP::update_potential(Parameters& para, float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.wb[t] = pointer;
    pointer += ann.one_ann_no_bias;
  }
  ann.b = pointer;
  pointer += 1;
  ann.c = pointer;
}

static void __global__ find_max_min(const int N, const float* g_q, float* g_q_scaler, float* g_q_scaler_max, float* g_q_scaler_min)
{
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  __shared__ float s_max[1024];
  __shared__ float s_min[1024];
  s_max[tid] = -1000000.0f; // a small number
  s_min[tid] = +1000000.0f; // a large number
  const int stride = 1024;
  const int number_of_rounds = (N - 1) / stride + 1;
  for (int round = 0; round < number_of_rounds; ++round) {
    const int n = round * stride + tid;
    if (n < N) {
      const int m = n + N * bid;
      float q = g_q[m];
      if (q > s_max[tid]) {
        s_max[tid] = q;
      }
      if (q < s_min[tid]) {
        s_min[tid] = q;
      }
    }
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_max[tid] < s_max[tid + offset]) {
        s_max[tid] = s_max[tid + offset];
      }
      if (s_min[tid] > s_min[tid + offset]) {
        s_min[tid] = s_min[tid + offset];
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_q_scaler_max[bid] = max(g_q_scaler_max[bid], s_max[0]);
    g_q_scaler_min[bid] = min(g_q_scaler_min[bid], s_min[0]);
    // Guard against a degenerate (constant) descriptor component: this happens for
    // the spin components of non-magnetic types, and for Sdm in collinear configs.
    float range = g_q_scaler_max[bid] - g_q_scaler_min[bid];
    g_q_scaler[bid] = (range > 1.0e-12f) ? 1.0f / range : 1.0f;
  }
}

static __global__ void apply_ann(
  const int N,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  const float* __restrict__ g_q_scaler,
  float* g_pe,
  float* g_Fp)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int type = g_type[n1];

    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    const int neu1 = annmb.num_neurons1;
    const int neu1_dim = neu1 * annmb.dim;
#ifdef S3A2_DIAG
    if (s3a2_is_target(n1)) {
      for (int d = 0; d < annmb.dim; ++d) {
        g_s3a2_q_raw[d] = g_descriptors[n1 + d * N];
        g_s3a2_q_scaled[d] = q[d];
      }
      g_s3a2_type = type;
    }
#endif
    if (annmb.num_hidden_layers == 2) {
      const int neu2 = annmb.num_neurons2;
      apply_ann_two_layers(
        annmb.dim,
        neu1,
        neu2,
        annmb.wb[type],
        annmb.wb[type] + neu1_dim,
        annmb.wb[type] + neu1 * (annmb.dim + 1),
        annmb.wb[type] + neu1 * (annmb.dim + 1 + neu2),
        annmb.wb[type] + neu1 * (annmb.dim + 1 + neu2) + neu2,
        annmb.b,
        q,
        F,
        Fp);
    } else {
#ifdef S3A2_DIAG
      if (s3a2_is_target(n1)) {
        s3a2_ann_one_layer_diag(
          annmb.dim, neu1, annmb.wb[type], annmb.wb[type] + neu1_dim,
          annmb.wb[type] + neu1 * (annmb.dim + 1), annmb.b, q, F, Fp);
      } else
#endif
      apply_ann_one_layer(
        annmb.dim,
        neu1,
        annmb.wb[type],
        annmb.wb[type] + neu1_dim,
        annmb.wb[type] + neu1 * (annmb.dim + 1),
        annmb.b,
        q,
        F,
        Fp);
    }
    g_pe[n1] = F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void apply_ann_temperature(
  const int N,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_descriptors,
  float* __restrict__ g_q_scaler,
  const float* __restrict__ g_temperature,
  float* g_pe,
  float* g_Fp)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int type = g_type[n1];
    float temperature = g_temperature[n1];

    // get descriptors
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < annmb.dim - 1; ++d) {
      q[d] = g_descriptors[n1 + d * N] * g_q_scaler[d];
    }
    g_q_scaler[annmb.dim - 1] = 0.001; // temperature dimension scaler
    q[annmb.dim - 1] = temperature * g_q_scaler[annmb.dim - 1];
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};
    apply_ann_one_layer(
      annmb.dim,
      annmb.num_neurons1,
      annmb.wb[type],
      annmb.wb[type] + annmb.num_neurons1 * annmb.dim,
      annmb.wb[type] + annmb.num_neurons1 * (annmb.dim + 1),
      annmb.b,
      q,
      F,
      Fp);
    g_pe[n1] = F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[n1 + d * N] = Fp[d] * g_q_scaler[d];
    }
  }
}

static __global__ void zero_force(
  const int N, float* g_fx, float* g_fy, float* g_fz, float* g_vxx, float* g_vyy, float* g_vzz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_fx[n1] = 0.0f;
    g_fy[n1] = 0.0f;
    g_fz[n1] = 0.0f;
    g_vxx[n1] = 0.0f;
    g_vyy[n1] = 0.0f;
    g_vzz[n1] = 0.0f;
  }
}

static __global__ void find_force_radial(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    int neighbor_number = g_NN[n1];
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        float tmp12 = g_Fp[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);

      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] = s_virial_xy;
    g_virial[n1 + N * 4] = s_virial_yz;
    g_virial[n1 + N * 5] = s_virial_zx;
  }
}

static __global__ void find_force_angular(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {

    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] = 
          g_sum_fxyz[(n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) * N + n1];
      }
    }
    int neighbor_number = g_NN[n1];
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float f12[3] = {0.0f};

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134, 
          paramb.num_L, n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
      }

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);

      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }
    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

// Spin contribution to the atomic forces (-dE/dR) and to the magnetic torque
// (-dE/dS). Both are accumulated with the same partial-force pattern used by
// the radial and angular kernels:
//     dq/dR_k = -sum_j G_j,   dq/dR_j = +G_j     (G_j = dq/dd_j)
// The torque needs a similar cross-neighbour accumulation:
//     omega_i = -sum_k dU_k/dS_i     (k = i itself and every neighbour of i)
// because the descriptor of a neighbour also depends on S_i.
static __global__ void find_force_spin(
  const int N,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const NEP::ParaMB paramb,
  const NEP::ANN annmb,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sx,
  const float* __restrict__ g_sy,
  const float* __restrict__ g_sz,
  const int* __restrict__ g_use_spin,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_tx,
  float* g_ty,
  float* g_tz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    const int t1 = g_type[n1];
    if (!g_use_spin[t1]) {
      return;
    }
    const int neighbor_number = g_NN[n1];
    const int n_max = paramb.n_max_angular;
    const int base = paramb.dim_radial + paramb.dim_angular;
    const int nn1 = paramb.n_max_angular + 1;
    const float skx = g_sx[n1];
    const float sky = g_sy[n1];
    const float skz = g_sz[n1];
    const float s2 = skx * skx + sky * sky + skz * skz; // |S_i|^2 for the SIA block
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;

    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      // S06: the MD kernels and the descriptor above both skip a coincident
      // pair; this spin force kernel was the only one that divided by zero.
      if (d12 < 1.0e-8f) continue;
      float d12inv = 1.0f / d12;
      float r[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv}; // rhat
      float fc12, fcp12;
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);

      float gn12[MAX_NUM_N];
      float gnp12[MAX_NUM_N];
      for (int n = 0; n <= n_max; ++n) {
        float g = 0.0f;
        float gp = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
#ifdef USE_CJ
          int type_index = t2;
#else
          int type_index = t1 * paramb.num_types + t2;
#endif
          int c_index = get_c_index(
            type_index,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          g += fn12[k] * annmb.c[c_index];
          gp += fnp12[k] * annmb.c[c_index];
        }
        gn12[n] = g;
        gnp12[n] = gp;
      }

      float sj[3] = {g_sx[n2], g_sy[n2], g_sz[n2]};
      if (!g_use_spin[t2]) {
        sj[0] = 0.0f;
        sj[1] = 0.0f;
        sj[2] = 0.0f;
      }
      float sk[3] = {skx, sky, skz};
      float sksj = sk[0] * sj[0] + sk[1] * sj[1] + sk[2] * sj[2];
      float a = sk[0] * r[0] + sk[1] * r[1] + sk[2] * r[2]; // S_k . rhat
      float b = sj[0] * r[0] + sj[1] * r[1] + sj[2] * r[2]; // S_j . rhat
      float cs[3] = { // S_k x S_j
        sk[1] * sj[2] - sk[2] * sj[1],
        sk[2] * sj[0] - sk[0] * sj[2],
        sk[0] * sj[1] - sk[1] * sj[0]};
      float p = r[0] * cs[0] + r[1] * cs[1] + r[2] * cs[2]; // rhat . (S_k x S_j)
      float sjxr[3] = { // S_j x rhat
        sj[1] * r[2] - sj[2] * r[1],
        sj[2] * r[0] - sj[0] * r[2],
        sj[0] * r[1] - sj[1] * r[0]};
      float rxsk[3] = { // rhat x S_k
        r[1] * sk[2] - r[2] * sk[1],
        r[2] * sk[0] - r[0] * sk[2],
        r[0] * sk[1] - r[1] * sk[0]};

      float f12[3] = {0.0f};      // G_j = dq/dd_j (already weighted by Fp)
      float t_own[3] = {0.0f};    // dU_{n1}/dS_{n1}
      float t_nb[3] = {0.0f};     // dU_{n1}/dS_{n2}

      for (int n = 0; n <= n_max; ++n) {
        const int blk = paramb.spin_block_mask;
        float fp_sex = (blk & SPIN_BLOCK_BIT_SEX) ? g_Fp[n1 + (base + n) * N] : 0.0f;
        float fp_san = (blk & SPIN_BLOCK_BIT_SAN) ? g_Fp[n1 + (base + nn1 + n) * N] : 0.0f;
        float fp_sdm = (blk & SPIN_BLOCK_BIT_SDM) ? g_Fp[n1 + (base + 2 * nn1 + n) * N] : 0.0f;
        float fp_env = (blk & SPIN_BLOCK_BIT_SLOT4) ? g_Fp[n1 + (base + 3 * nn1 + n) * N] : 0.0f;
        float gn = gn12[n];
        float gnp = gnp12[n];

        // ---- dq/dd_j ----------------------------------------------------
        // Sex:  gnp * (S_k.S_j) * rhat
        float c_sex = fp_sex * gnp * sksj;
        // San:  gnp*a*b*rhat + (gn/r) * P (b*S_k + a*S_j)
        float vs[3] = {b * sk[0] + a * sj[0], b * sk[1] + a * sj[1], b * sk[2] + a * sj[2]};
        float rs = vs[0] * r[0] + vs[1] * r[1] + vs[2] * r[2];
        // Sdm:  gnp*p*rhat + (gn/r) * P (S_k x S_j)
        float rs_dm = cs[0] * r[0] + cs[1] * r[1] + cs[2] * r[2];
        // Senv: C * [gnp*a*rhat + (gn/r) * P S_k]
        // SIA : gnp*[3/2 a^2 - 1/2|S|^2]*rhat + (gn/r) * 3a * P S_k
        float rs_env = sk[0] * r[0] + sk[1] * r[1] + sk[2] * r[2];
        const float q_sia = 1.5f * a * a - 0.5f * s2;
        const float sia_scale = 3.0f * a;

        for (int d = 0; d < 3; ++d) {
          float san_d = fp_san * (gnp * a * b * r[d] + gn * d12inv * (vs[d] - rs * r[d]));
          float sdm_d = fp_sdm * (gnp * p * r[d] + gn * d12inv * (cs[d] - rs_dm * r[d]));
          float env_d;
          if (paramb.spin_use_sia) {
            env_d = fp_env * (gnp * q_sia * r[d] +
                              gn * d12inv * (sia_scale * sk[d] - sia_scale * a * r[d]));
          } else {
            env_d = fp_env * SPIN_ENV_FACTOR *
                    (gnp * a * r[d] + gn * d12inv * (sk[d] - rs_env * r[d]));
          }
          f12[d] += c_sex * r[d] + san_d + sdm_d + env_d;
        }

        // ---- dq/dS_k (own spin) -----------------------------------------
        // Sex: gn*S_j ; San: gn*b*rhat ; Sdm: gn*(S_j x rhat)
        // Senv: C*gn*rhat (S-independent) ; SIA: gn*(3a*rhat - S_k)
        for (int d = 0; d < 3; ++d) {
          const float t_env = paramb.spin_use_sia
                                ? fp_env * gn * (sia_scale * r[d] - sk[d])
                                : fp_env * SPIN_ENV_FACTOR * gn * r[d];
          t_own[d] += fp_sex * gn * sj[d] + fp_san * gn * b * r[d] +
                      fp_sdm * gn * sjxr[d] + t_env;
        }

        // ---- dq/dS_j (neighbour spin) -----------------------------------
        // Sex: gn*S_k ; San: gn*a*rhat ; Sdm: gn*(rhat x S_k) ; Senv: 0
        // A neighbour without a spin degree of freedom receives no torque.
        if (g_use_spin[t2]) {
          for (int d = 0; d < 3; ++d) {
            t_nb[d] += fp_sex * gn * sk[d] + fp_san * gn * a * r[d] + fp_sdm * gn * rxsk[d];
          }
        }
      }

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);

      // magnetic torque omega = -dE/dS
      atomicAdd(&g_tx[n1], -t_own[0]);
      atomicAdd(&g_ty[n1], -t_own[1]);
      atomicAdd(&g_tz[n1], -t_own[2]);
      atomicAdd(&g_tx[n2], -t_nb[0]);
      atomicAdd(&g_ty[n2], -t_nb[1]);
      atomicAdd(&g_tz[n2], -t_nb[2]);

      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
    }

    // Smag (on-site |S| channel):
    //   d q_smag[n] / dS = (n+1) u^n * SPIN_MAG_SCALE * S/|S|
    // It has no position dependence, hence no force / virial contribution.
    float t_mag[3] = {0.0f, 0.0f, 0.0f};
    {
      const float smag = sqrt(skx * skx + sky * sky + skz * skz);
      // S06-R2: exact derivative of q_n = (scale |S|)^(n+1) for every |S| > 0,
      // formed as c*S above the threshold so the value stays bit-identical,
      // and with a scale-invariant unit direction below it.  Identical to the
      // MD kernels and to nep_specialized.cu; see the long note there.
      float dir[3] = {skx, sky, skz};
      float denom = smag;
      if (smag < SPIN_MAG_MIN_MOMENT) {
        const float big = fmaxf(fabsf(skx), fmaxf(fabsf(sky), fabsf(skz)));
        if (big > 0.0f) {
          const float inv = 1.0f / big;
          const float ax = skx * inv, ay = sky * inv, az = skz * inv;
          const float f = 1.0f / sqrtf(ax * ax + ay * ay + az * az);
          dir[0] = ax * f;
          dir[1] = ay * f;
          dir[2] = az * f;
        } else {
          dir[0] = dir[1] = dir[2] = 0.0f;
        }
        denom = 1.0f;
      }
      {
        const float u_mag = smag * paramb.spin_mag_scale;
        float un_mag = 1.0f;
        for (int n = 0; n <= n_max; ++n) {
          const float fp_mag = (paramb.spin_block_mask & SPIN_BLOCK_BIT_SMAG)
            ? g_Fp[n1 + (base + 4 * nn1 + n) * N] : 0.0f;
          const float c = fp_mag * (n + 1) * un_mag * paramb.spin_mag_scale / denom;
          t_mag[0] += c * dir[0];
          t_mag[1] += c * dir[1];
          t_mag[2] += c * dir[2];
          un_mag *= u_mag;
        }
      }
    }
    atomicAdd(&g_tx[n1], -t_mag[0]);
    atomicAdd(&g_ty[n1], -t_mag[1]);
    atomicAdd(&g_tz[n1], -t_mag[2]);

    g_virial[n1] += s_virial_xx;
    g_virial[n1 + N] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
  }
}

static __global__ void
zero_torque(const int N, float* g_tx, float* g_ty, float* g_tz)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_tx[n1] = 0.0f;
    g_ty[n1] = 0.0f;
    g_tz[n1] = 0.0f;
  }
}

static __global__ void find_force_ZBL(
  const int N,
  const NEP::ParaMB paramb,
  const NEP::ZBL zbl,
  const int* g_NN_sum,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  float* g_fx,
  float* g_fy,
  float* g_fz,
  float* g_virial,
  float* g_pe)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    float s_pe = 0.0f;
    float s_virial_xx = 0.0f;
    float s_virial_yy = 0.0f;
    float s_virial_zz = 0.0f;
    float s_virial_xy = 0.0f;
    float s_virial_yz = 0.0f;
    float s_virial_zx = 0.0f;
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1]; // starting from 1
    float pow_zi = pow(float(zi), 0.23f);
    int neighbor_number = g_NN[n1];
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      int index = g_NN_sum[n1] + i1;
      int n2 = g_NL[index];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2]; // starting from 1
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexible) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};

      atomicAdd(&g_fx[n1], f12[0]);
      atomicAdd(&g_fy[n1], f12[1]);
      atomicAdd(&g_fz[n1], f12[2]);
      atomicAdd(&g_fx[n2], -f12[0]);
      atomicAdd(&g_fy[n2], -f12[1]);
      atomicAdd(&g_fz[n2], -f12[2]);
      s_virial_xx -= r12[0] * f12[0];
      s_virial_yy -= r12[1] * f12[1];
      s_virial_zz -= r12[2] * f12[2];
      s_virial_xy -= r12[0] * f12[1];
      s_virial_yz -= r12[1] * f12[2];
      s_virial_zx -= r12[2] * f12[0];
      s_pe += f * 0.5f;
    }
    g_virial[n1 + N * 0] += s_virial_xx;
    g_virial[n1 + N * 1] += s_virial_yy;
    g_virial[n1 + N * 2] += s_virial_zz;
    g_virial[n1 + N * 3] += s_virial_xy;
    g_virial[n1 + N * 4] += s_virial_yz;
    g_virial[n1 + N * 5] += s_virial_zx;
    g_pe[n1] += s_pe;
  }
}

void NEP::find_force(
  Parameters& para,
  const float* parameters,
  std::vector<Dataset>& dataset,
  bool calculate_q_scaler,
  int device_in_this_iter)
{

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    nep_data[device_id].parameters.copy_from_host(
      parameters + device_id * para.number_of_variables);
    update_potential(para, nep_data[device_id].parameters.data(), annmb[device_id]);
  }

  for (int device_id = 0; device_id < device_in_this_iter; ++device_id) {
    CHECK(gpuSetDevice(device_id));
    neighbor[device_id].prepare(para, dataset[device_id], device_id);
    const int block_size = 32;
    const int grid_size = (dataset[device_id].N - 1) / block_size + 1;

    if (compiled_kernel_) {
      compiled_kernel_->launch_descriptor_radial(
        dataset[device_id].N,
        dataset[device_id].NN_radial_sum.data(),
        dataset[device_id].NN_radial.data(),
        neighbor[device_id].NL_radial.data(),
        dataset[device_id].type.data(),
        neighbor[device_id].x12_radial.data(),
        neighbor[device_id].y12_radial.data(),
        neighbor[device_id].z12_radial.data(),
        nep_data[device_id].parameters.data(),
        nep_data[device_id].descriptors.data());
    } else {
      find_descriptors_radial<<<grid_size, block_size>>>(
        dataset[device_id].N,
        dataset[device_id].NN_radial_sum.data(),
        dataset[device_id].NN_radial.data(),
        neighbor[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        neighbor[device_id].x12_radial.data(),
        neighbor[device_id].y12_radial.data(),
        neighbor[device_id].z12_radial.data(),
        nep_data[device_id].descriptors.data());
      GPU_CHECK_KERNEL
    }

    if (compiled_kernel_) {
      compiled_kernel_->launch_descriptor_angular(
        dataset[device_id].N,
        dataset[device_id].NN_angular_sum.data(),
        dataset[device_id].NN_angular.data(),
        neighbor[device_id].NL_angular.data(),
        dataset[device_id].type.data(),
        neighbor[device_id].x12_angular.data(),
        neighbor[device_id].y12_angular.data(),
        neighbor[device_id].z12_angular.data(),
        nep_data[device_id].parameters.data(),
        nep_data[device_id].descriptors.data(),
        nep_data[device_id].sum_fxyz.data());
    } else {
      find_descriptors_angular<<<grid_size, block_size>>>(
        dataset[device_id].N,
        dataset[device_id].NN_angular_sum.data(),
        dataset[device_id].NN_angular.data(),
        neighbor[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        neighbor[device_id].x12_angular.data(),
        neighbor[device_id].y12_angular.data(),
        neighbor[device_id].z12_angular.data(),
        nep_data[device_id].descriptors.data(),
        nep_data[device_id].sum_fxyz.data());
      GPU_CHECK_KERNEL
    }

    if (paramb.enable_spin) {
      if (compiled_kernel_ && compiled_kernel_->has_spin()) {
        compiled_kernel_->launch_descriptor_spin(
          dataset[device_id].N,
          dataset[device_id].NN_angular_sum.data(),
          dataset[device_id].NN_angular.data(),
          neighbor[device_id].NL_angular.data(),
          dataset[device_id].type.data(),
          neighbor[device_id].x12_angular.data(),
          neighbor[device_id].y12_angular.data(),
          neighbor[device_id].z12_angular.data(),
          dataset[device_id].spin.data(),
          dataset[device_id].spin.data() + dataset[device_id].N,
          dataset[device_id].spin.data() + dataset[device_id].N * 2,
          dataset[device_id].use_spin_gpu.data(),
          nep_data[device_id].parameters.data(),
          nep_data[device_id].descriptors.data());
      } else {
        find_descriptors_spin<<<grid_size, block_size>>>(
          dataset[device_id].N,
          dataset[device_id].NN_angular_sum.data(),
          dataset[device_id].NN_angular.data(),
          neighbor[device_id].NL_angular.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          neighbor[device_id].x12_angular.data(),
          neighbor[device_id].y12_angular.data(),
          neighbor[device_id].z12_angular.data(),
          dataset[device_id].spin.data(),
          dataset[device_id].spin.data() + dataset[device_id].N,
          dataset[device_id].spin.data() + dataset[device_id].N * 2,
          dataset[device_id].use_spin_gpu.data(),
          nep_data[device_id].descriptors.data());
        GPU_CHECK_KERNEL
      }
    }

    if (para.prediction == 1 && para.output_descriptor >= 1) {
      FILE* fid_descriptor = my_fopen("descriptor.out", "a");
      std::vector<float> descriptor_cpu(nep_data[device_id].descriptors.size());
      nep_data[device_id].descriptors.copy_to_host(descriptor_cpu.data());
      for (int nc = 0; nc < dataset[device_id].Nc; ++nc) {
        float q_structure[MAX_DIM] = {0.0f};
        for (int na = 0; na < dataset[device_id].Na_cpu[nc]; ++na) {
          int n = dataset[device_id].Na_sum_cpu[nc] + na;
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            float q = descriptor_cpu[n + d * dataset[device_id].N] * para.q_scaler_cpu[d];
            q_structure[d] += q;
            if (para.output_descriptor == 2) {
              fprintf(fid_descriptor, NEP_FMT_OBS_BLANK, q);
            }
          }
          if (para.output_descriptor == 2) {
            fprintf(fid_descriptor, "\n");
          }
        }
        if (para.output_descriptor == 1) {
          for (int d = 0; d < annmb[device_id].dim; ++d) {
            fprintf(fid_descriptor, NEP_FMT_OBS_BLANK, q_structure[d] / dataset[device_id].Na_cpu[nc]);
          }
        }
        if (para.output_descriptor == 1) {
          fprintf(fid_descriptor, "\n");
        }
      }
      fclose(fid_descriptor);
    }

    if (calculate_q_scaler) {
      find_max_min<<<annmb[device_id].dim, 1024>>>(
        dataset[device_id].N,
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        para.q_scaler_max[device_id].data(),
        para.q_scaler_min[device_id].data());
      GPU_CHECK_KERNEL
    }

    zero_force<<<grid_size, block_size>>>(
      dataset[device_id].N,
      dataset[device_id].force.data(),
      dataset[device_id].force.data() + dataset[device_id].N,
      dataset[device_id].force.data() + dataset[device_id].N * 2,
      dataset[device_id].virial.data(),
      dataset[device_id].virial.data() + dataset[device_id].N,
      dataset[device_id].virial.data() + dataset[device_id].N * 2);
    GPU_CHECK_KERNEL

    if (paramb.enable_spin) {
      if (compiled_kernel_ && compiled_kernel_->has_spin()) {
        compiled_kernel_->launch_zero_torque(
          dataset[device_id].N,
          dataset[device_id].torque.data(),
          dataset[device_id].torque.data() + dataset[device_id].N,
          dataset[device_id].torque.data() + dataset[device_id].N * 2);
      } else {
        zero_torque<<<grid_size, block_size>>>(
          dataset[device_id].N,
          dataset[device_id].torque.data(),
          dataset[device_id].torque.data() + dataset[device_id].N,
          dataset[device_id].torque.data() + dataset[device_id].N * 2);
        GPU_CHECK_KERNEL
      }
    }

    if (para.model_type == 3) {
      if (compiled_kernel_) {
        compiled_kernel_->launch_ann_temperature(
          dataset[device_id].N,
          dataset[device_id].type.data(),
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].temperature_ref_gpu.data(),
          nep_data[device_id].parameters.data(),
          dataset[device_id].energy.data(),
          nep_data[device_id].Fp.data());
      } else {
        apply_ann_temperature<<<grid_size, block_size>>>(
          dataset[device_id].N,
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          nep_data[device_id].descriptors.data(),
          para.q_scaler_gpu[device_id].data(),
          dataset[device_id].temperature_ref_gpu.data(),
          dataset[device_id].energy.data(),
          nep_data[device_id].Fp.data());
        GPU_CHECK_KERNEL
      }
    } else if (compiled_kernel_) {
      compiled_kernel_->launch_ann(
        dataset[device_id].N,
        dataset[device_id].type.data(),
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        nep_data[device_id].parameters.data(),
        dataset[device_id].energy.data(),
        nep_data[device_id].Fp.data());
    } else {
      apply_ann<<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        nep_data[device_id].descriptors.data(),
        para.q_scaler_gpu[device_id].data(),
        dataset[device_id].energy.data(),
        nep_data[device_id].Fp.data());
      GPU_CHECK_KERNEL
    }

    if (compiled_kernel_) {
      compiled_kernel_->launch_force_radial(
        dataset[device_id].N,
        dataset[device_id].NN_radial_sum.data(),
        dataset[device_id].NN_radial.data(),
        neighbor[device_id].NL_radial.data(),
        dataset[device_id].type.data(),
        neighbor[device_id].x12_radial.data(),
        neighbor[device_id].y12_radial.data(),
        neighbor[device_id].z12_radial.data(),
        nep_data[device_id].parameters.data(),
        nep_data[device_id].Fp.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
    } else {
      find_force_radial<<<grid_size, block_size>>>(
        dataset[device_id].N,
        dataset[device_id].NN_radial_sum.data(),
        dataset[device_id].NN_radial.data(),
        neighbor[device_id].NL_radial.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        neighbor[device_id].x12_radial.data(),
        neighbor[device_id].y12_radial.data(),
        neighbor[device_id].z12_radial.data(),
        nep_data[device_id].Fp.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
      GPU_CHECK_KERNEL
    }

    if (compiled_kernel_) {
      compiled_kernel_->launch_force_angular(
        dataset[device_id].N,
        dataset[device_id].NN_angular_sum.data(),
        dataset[device_id].NN_angular.data(),
        neighbor[device_id].NL_angular.data(),
        dataset[device_id].type.data(),
        neighbor[device_id].x12_angular.data(),
        neighbor[device_id].y12_angular.data(),
        neighbor[device_id].z12_angular.data(),
        nep_data[device_id].parameters.data(),
        nep_data[device_id].Fp.data(),
        nep_data[device_id].sum_fxyz.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
    } else {
      find_force_angular<<<grid_size, block_size>>>(
        dataset[device_id].N,
        dataset[device_id].NN_angular_sum.data(),
        dataset[device_id].NN_angular.data(),
        neighbor[device_id].NL_angular.data(),
        paramb,
        annmb[device_id],
        dataset[device_id].type.data(),
        neighbor[device_id].x12_angular.data(),
        neighbor[device_id].y12_angular.data(),
        neighbor[device_id].z12_angular.data(),
        nep_data[device_id].Fp.data(),
        nep_data[device_id].sum_fxyz.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data());
      GPU_CHECK_KERNEL
    }

    if (paramb.enable_spin) {
      if (compiled_kernel_ && compiled_kernel_->has_spin()) {
        compiled_kernel_->launch_force_spin(
          dataset[device_id].N,
          dataset[device_id].NN_angular_sum.data(),
          dataset[device_id].NN_angular.data(),
          neighbor[device_id].NL_angular.data(),
          dataset[device_id].type.data(),
          neighbor[device_id].x12_angular.data(),
          neighbor[device_id].y12_angular.data(),
          neighbor[device_id].z12_angular.data(),
          nep_data[device_id].parameters.data(),
          nep_data[device_id].Fp.data(),
          dataset[device_id].spin.data(),
          dataset[device_id].spin.data() + dataset[device_id].N,
          dataset[device_id].spin.data() + dataset[device_id].N * 2,
          dataset[device_id].use_spin_gpu.data(),
          dataset[device_id].force.data(),
          dataset[device_id].force.data() + dataset[device_id].N,
          dataset[device_id].force.data() + dataset[device_id].N * 2,
          dataset[device_id].virial.data(),
          dataset[device_id].torque.data(),
          dataset[device_id].torque.data() + dataset[device_id].N,
          dataset[device_id].torque.data() + dataset[device_id].N * 2);
      } else {
        find_force_spin<<<grid_size, block_size>>>(
          dataset[device_id].N,
          dataset[device_id].NN_angular_sum.data(),
          dataset[device_id].NN_angular.data(),
          neighbor[device_id].NL_angular.data(),
          paramb,
          annmb[device_id],
          dataset[device_id].type.data(),
          neighbor[device_id].x12_angular.data(),
          neighbor[device_id].y12_angular.data(),
          neighbor[device_id].z12_angular.data(),
          nep_data[device_id].Fp.data(),
          dataset[device_id].spin.data(),
          dataset[device_id].spin.data() + dataset[device_id].N,
          dataset[device_id].spin.data() + dataset[device_id].N * 2,
          dataset[device_id].use_spin_gpu.data(),
          dataset[device_id].force.data(),
          dataset[device_id].force.data() + dataset[device_id].N,
          dataset[device_id].force.data() + dataset[device_id].N * 2,
          dataset[device_id].virial.data(),
          dataset[device_id].torque.data(),
          dataset[device_id].torque.data() + dataset[device_id].N,
          dataset[device_id].torque.data() + dataset[device_id].N * 2);
        GPU_CHECK_KERNEL
      }
    }

    if (zbl.enabled) {
      find_force_ZBL<<<grid_size, block_size>>>(
        dataset[device_id].N,
        paramb,
        zbl,
        dataset[device_id].NN_angular_sum.data(),
        dataset[device_id].NN_angular.data(),
        neighbor[device_id].NL_angular.data(),
        dataset[device_id].type.data(),
        neighbor[device_id].x12_angular.data(),
        neighbor[device_id].y12_angular.data(),
        neighbor[device_id].z12_angular.data(),
        dataset[device_id].force.data(),
        dataset[device_id].force.data() + dataset[device_id].N,
        dataset[device_id].force.data() + dataset[device_id].N * 2,
        dataset[device_id].virial.data(),
        dataset[device_id].energy.data());
      GPU_CHECK_KERNEL
    }
  }
}
