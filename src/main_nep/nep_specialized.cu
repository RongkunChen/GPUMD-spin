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

#ifdef NEP_SPECIALIZED_RUNTIME_BUILD
/*
 * Runtime-specialized kernels for NEP-family training.
 *
 * The normal nep executable compiles this file as an empty translation unit.
 * NEP_Compile invokes nvcc again at runtime with:
 *   -DNEP_SPECIALIZED_RUNTIME_BUILD
 * and a generated nep_special_config.cuh containing compile-time constants.
 *
 * Generic NEP-family kernels remain the reference implementation.
 */

#include <nep_special_config.cuh>
// S06: the |S| floor of the Smag torque.  The current generator writes it
// (see nep_compile.cu); the fallback keeps a config header produced before S06
// compilable, with the same value as utilities/spin_descriptor.cuh.
#ifndef SPIN_MAG_MIN_MOMENT_JIT
#define SPIN_MAG_MIN_MOMENT_JIT 1.0e-8f
#endif
#define NEP_SPECIALIZED_SOURCE_INTERFACE_VERSION 3
#if !defined(NEP_SPECIAL_CONFIG_INTERFACE_VERSION)
#error "Missing NEP specialization interface version."
#elif NEP_SPECIAL_CONFIG_INTERFACE_VERSION != NEP_SPECIALIZED_SOURCE_INTERFACE_VERSION
#error "Incompatible NEP specialization interface version."
#endif
#include "utilities/nep_utilities.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// F26 ablation mask.  The generated configuration always defines it; refusing
// to compile without it turns a silently-wrong specialization (all five blocks
// evaluated although the model trained fewer) into a build error.
#ifndef SPIN_BLOCK_MASK_JIT
#error "nep_special_config.cuh is missing SPIN_BLOCK_MASK_JIT."
#endif
// Bit values, copied from utilities/spin_descriptor.cuh so that this file keeps
// its "only the generated config and the utilities header" dependency.
#define SPIN_JIT_BIT_SEX 1
#define SPIN_JIT_BIT_SAN 2
#define SPIN_JIT_BIT_SDM 4
#define SPIN_JIT_BIT_SLOT4 8
#define SPIN_JIT_BIT_SMAG 16

__device__ __forceinline__ float get_rc_radial_jit(const int t1, const int t2)
{
#if RC_RADIAL_COMMON_JIT
  (void)t1;
  (void)t2;
  return RC_RADIAL_COMMON_VALUE_JIT;
#else
  return 0.5f * (RC_RADIAL_JIT[t1] + RC_RADIAL_JIT[t2]);
#endif
}

__device__ __forceinline__ float get_rc_angular_jit(const int t1, const int t2)
{
#if RC_ANGULAR_COMMON_JIT
  (void)t1;
  (void)t2;
  return RC_ANGULAR_COMMON_VALUE_JIT;
#else
  return 0.5f * (RC_ANGULAR_JIT[t1] + RC_ANGULAR_JIT[t2]);
#endif
}

__device__ __forceinline__ int descriptor_type_index_jit(const int t1, const int t2)
{
#if DESCRIPTOR_USE_CJ_JIT
  (void)t1;
  return t2;
#else
  return t1 * NUM_TYPES_JIT + t2;
#endif
}

__device__ __forceinline__ const float* get_c_jit(const float* parameters)
{
  return parameters + C_OFFSET_JIT;
}

__global__ void descriptor_radial_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  float* descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];
  nep_acc_t q[DIM_RADIAL_JIT] = {0.0};

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float x12 = x12_all[index];
    const float y12 = y12_all[index];
    const float z12 = z12_all[index];
    const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    const float rc = get_rc_radial_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[BASIS_SIZE_RADIAL_JIT + 1];
    find_fn(BASIS_SIZE_RADIAL_JIT, rcinv, d12, fc12, fn12);

    const int type_index = descriptor_type_index_jit(t1, t2);
#pragma unroll
    for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
      float gn12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_RADIAL_JIT; ++k) {
        const int c_index =
          get_c_index(type_index, n, k, N_MAX_RADIAL_JIT, BASIS_SIZE_RADIAL_JIT);
        gn12 += fn12[k] * c[c_index];
      }
      q[n] += static_cast<nep_acc_t>(gn12);
    }
  }

#pragma unroll
  for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
    descriptors[n1 + n * N] = static_cast<float>(q[n]);
  }
}

__global__ void descriptor_angular_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  float* descriptors,
  float* sum_fxyz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];
  float q[DIM_ANGULAR_JIT] = {0.0f};

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
    nep_acc_t s_acc[NUM_ABC_JIT] = {0.0};
    for (int i1 = 0; i1 < neighbor_number; ++i1) {
      const int index = NN_sum[n1] + i1;
      const int n2 = NL[index];
      const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
      const float x12 = x12_all[index];
      const float y12 = y12_all[index];
      const float z12 = z12_all[index];
      const float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      const float rc = get_rc_angular_jit(t1, t2);
      const float rcinv = 1.0f / rc;
      float fc12;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[BASIS_SIZE_ANGULAR_JIT + 1];
      find_fn(BASIS_SIZE_ANGULAR_JIT, rcinv, d12, fc12, fn12);

      const int type_index = descriptor_type_index_jit(t1, t2);
      float gn12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
        const int c_index = get_c_index(
          type_index,
          n,
          k,
          N_MAX_ANGULAR_JIT,
          BASIS_SIZE_ANGULAR_JIT,
          NUM_C_RADIAL_JIT);
        gn12 += fn12[k] * c[c_index];
      }
      accumulate_s(L_MAX_JIT, d12, x12, y12, z12, gn12, s_acc);
    }

    float s[NUM_ABC_JIT];
#pragma unroll
    for (int abc = 0; abc < NUM_ABC_JIT; ++abc) {
      s[abc] = static_cast<float>(s_acc[abc]);
    }
    find_q(
      L_MAX_JIT,
      HAS_Q_222_JIT,
      HAS_Q_1111_JIT,
      HAS_Q_112_JIT,
      HAS_Q_123_JIT,
      HAS_Q_233_JIT,
      HAS_Q_134_JIT,
      N_MAX_ANGULAR_JIT + 1,
      n,
      s,
      q);

#pragma unroll
    for (int abc = 0; abc < NUM_ABC_JIT; ++abc) {
      sum_fxyz[(n * NUM_ABC_JIT + abc) * N + n1] = s[abc];
    }
  }

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
#pragma unroll
    for (int l = 0; l < NUM_L_JIT; ++l) {
      const int ln = l * (N_MAX_ANGULAR_JIT + 1) + n;
      descriptors[n1 + (DIM_RADIAL_JIT + ln) * N] = q[ln];
    }
  }
}

// ---------------------------------------------------------------- spin (Route B)
#if ENABLE_SPIN_JIT

__global__ void descriptor_spin_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* sx_all,
  const float* sy_all,
  const float* sz_all,
  const int* use_spin,
  const float* parameters,
  float* descriptors)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  if (!use_spin[t1]) {
#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      descriptors[n1 + (DIM_RADIAL_JIT + DIM_ANGULAR_JIT + n) * N] = 0.0f;
      descriptors[n1 + (DIM_RADIAL_JIT + DIM_ANGULAR_JIT + NN1_JIT + n) * N] = 0.0f;
      descriptors[n1 + (DIM_RADIAL_JIT + DIM_ANGULAR_JIT + 2 * NN1_JIT + n) * N] = 0.0f;
      descriptors[n1 + (DIM_RADIAL_JIT + DIM_ANGULAR_JIT + 3 * NN1_JIT + n) * N] = 0.0f;
      descriptors[n1 + (DIM_RADIAL_JIT + DIM_ANGULAR_JIT + 4 * NN1_JIT + n) * N] = 0.0f;
    }
    return;
  }

  const float* c = get_c_jit(parameters);
  const int neighbor_number = NN[n1];
  const float sk[3] = {sx_all[n1], sy_all[n1], sz_all[n1]};
  float q_sex[N_MAX_ANGULAR_JIT + 1] = {0.0f};
  float q_san[N_MAX_ANGULAR_JIT + 1] = {0.0f};
  float q_sdm[N_MAX_ANGULAR_JIT + 1] = {0.0f};
  // Slot 4: SIA when SPIN_USE_SIA_JIT, else the legacy Senv block.  Both are
  // accumulated; the compiler removes the unused one (compile-time constant).
  float q_sia[N_MAX_ANGULAR_JIT + 1] = {0.0f};
  float q_senv[N_MAX_ANGULAR_JIT + 1] = {0.0f};
  const float s2 = sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2];

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float r12[3] = {
      x12_all[index], y12_all[index], z12_all[index]};
    const float d12 =
      sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    // S06: the MD kernels (utilities/spin_descriptor_kernels.cuh) skip a
    // coincident pair; these specialized copies used to divide by d12
    // unconditionally, so a duplicated position produced 1/0 = Inf here and
    // NaN downstream, while the MD path skipped the pair.  Same threshold and
    // same action, so the two implementations describe one function.
    if (d12 < 1.0e-8f) continue;
    const float d12inv = 1.0f / d12;
    const float rh[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

    const float rc = get_rc_angular_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[BASIS_SIZE_ANGULAR_JIT + 1];
    find_fn(BASIS_SIZE_ANGULAR_JIT, rcinv, d12, fc12, fn12);

    const int type_index = descriptor_type_index_jit(t1, t2);
    float gn12[N_MAX_ANGULAR_JIT + 1];
#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      float g = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
        const int c_index = get_c_index(
          type_index,
          n,
          k,
          N_MAX_ANGULAR_JIT,
          BASIS_SIZE_ANGULAR_JIT,
          NUM_C_RADIAL_JIT);
        g += fn12[k] * c[c_index];
      }
      gn12[n] = g;
    }

    // A type with use_spin == 0 carries no spin degree of freedom.
    const float sj[3] = {
      use_spin[t2] ? sx_all[n2] : 0.0f,
      use_spin[t2] ? sy_all[n2] : 0.0f,
      use_spin[t2] ? sz_all[n2] : 0.0f};
    const float sksj = sk[0] * sj[0] + sk[1] * sj[1] + sk[2] * sj[2];
    const float a = sk[0] * rh[0] + sk[1] * rh[1] + sk[2] * rh[2];
    const float b = sj[0] * rh[0] + sj[1] * rh[1] + sj[2] * rh[2];
    const float cs[3] = {
      sk[1] * sj[2] - sk[2] * sj[1],
      sk[2] * sj[0] - sk[0] * sj[2],
      sk[0] * sj[1] - sk[1] * sj[0]};
    const float p = rh[0] * cs[0] + rh[1] * cs[1] + rh[2] * cs[2];

#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      q_sex[n] += gn12[n] * sksj;
      q_san[n] += gn12[n] * a * b;
      q_sdm[n] += gn12[n] * p;
      q_sia[n] += gn12[n] * (1.5f * a * a - 0.5f * s2);
      q_senv[n] += gn12[n] * a;
    }
  }

  const int base = DIM_RADIAL_JIT + DIM_ANGULAR_JIT;
  // F26: a cleared bit writes an exact 0.0f into that block's q slots.  The
  // slots themselves stay where they are -- the mask only switches blocks on
  // and off, it never compacts the descriptor or moves parameters.
  const int mask = SPIN_BLOCK_MASK_JIT;
  const bool on_sex = (mask & SPIN_JIT_BIT_SEX) != 0;
  const bool on_san = (mask & SPIN_JIT_BIT_SAN) != 0;
  const bool on_sdm = (mask & SPIN_JIT_BIT_SDM) != 0;
  const bool on_slot4 = (mask & SPIN_JIT_BIT_SLOT4) != 0;
#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
    descriptors[n1 + (base + n) * N] = on_sex ? q_sex[n] : 0.0f;
    descriptors[n1 + (base + NN1_JIT + n) * N] = on_san ? q_san[n] : 0.0f;
    descriptors[n1 + (base + 2 * NN1_JIT + n) * N] = on_sdm ? q_sdm[n] : 0.0f;
    descriptors[n1 + (base + 3 * NN1_JIT + n) * N] =
      !on_slot4 ? 0.0f
                : ((SPIN_USE_SIA_JIT != 0) ? q_sia[n]
                                           : (SPIN_ENV_FACTOR_JIT * q_senv[n]));
  }

  // Smag: on-site |S| channel, u^(n+1) with u = |S| * SPIN_MAG_SCALE_JIT
  const float smag = sqrt(sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2]);
  const float u_mag = smag * SPIN_MAG_SCALE_JIT;
  const bool on_smag = (mask & SPIN_JIT_BIT_SMAG) != 0;
  float un_mag = 1.0f;
#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
    un_mag *= u_mag;
    descriptors[n1 + (base + 4 * NN1_JIT + n) * N] = on_smag ? un_mag : 0.0f;
  }
}

__global__ void zero_torque_jit(int N, float* tx, float* ty, float* tz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;
  tx[n1] = 0.0f;
  ty[n1] = 0.0f;
  tz[n1] = 0.0f;
}

__global__ void force_spin_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* Fp_in,
  const float* sx_all,
  const float* sy_all,
  const float* sz_all,
  const int* use_spin,
  float* fx,
  float* fy,
  float* fz,
  float* virial,
  float* tx,
  float* ty,
  float* tz)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  if (!use_spin[t1]) return;

  const float* c = get_c_jit(parameters);
  const int base = DIM_RADIAL_JIT + DIM_ANGULAR_JIT;
  const int neighbor_number = NN[n1];
  const float sk[3] = {sx_all[n1], sy_all[n1], sz_all[n1]};
  // F26: a cleared block must not contribute through the chain rule either.
  // Zeroing only q would leave q = 0 with fp != 0, which still moves F/g/virial
  // through the derivative of g_n and through the geometric prefactors; zeroing
  // only fp would leave the descriptor (and hence the training target) alive.
  // Both sides are masked, so the block's contribution to F, g and the virial
  // is exactly zero.  Note the ANN's raw dE/dq (Fp_in) is *not* required to be
  // zero at q = 0 -- only its effect through a switched-off block is.
  const int block_mask = SPIN_BLOCK_MASK_JIT;
  const bool on_sex = (block_mask & SPIN_JIT_BIT_SEX) != 0;
  const bool on_san = (block_mask & SPIN_JIT_BIT_SAN) != 0;
  const bool on_sdm = (block_mask & SPIN_JIT_BIT_SDM) != 0;
  const bool on_slot4 = (block_mask & SPIN_JIT_BIT_SLOT4) != 0;
  const bool on_smag = (block_mask & SPIN_JIT_BIT_SMAG) != 0;

  float sxx = 0.0f;
  float syy = 0.0f;
  float szz = 0.0f;
  float sxy = 0.0f;
  float syz = 0.0f;
  float szx = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float r12[3] = {
      x12_all[index], y12_all[index], z12_all[index]};
    const float d12 =
      sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    // S06: the MD kernels (utilities/spin_descriptor_kernels.cuh) skip a
    // coincident pair; these specialized copies used to divide by d12
    // unconditionally, so a duplicated position produced 1/0 = Inf here and
    // NaN downstream, while the MD path skipped the pair.  Same threshold and
    // same action, so the two implementations describe one function.
    if (d12 < 1.0e-8f) continue;
    const float d12inv = 1.0f / d12;
    const float rh[3] = {r12[0] * d12inv, r12[1] * d12inv, r12[2] * d12inv};

    const float rc = get_rc_angular_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[BASIS_SIZE_ANGULAR_JIT + 1];
    float fnp12[BASIS_SIZE_ANGULAR_JIT + 1];
    find_fn_and_fnp(
      BASIS_SIZE_ANGULAR_JIT, rcinv, d12, fc12, fcp12, fn12, fnp12);

    const int type_index = descriptor_type_index_jit(t1, t2);
    float gn12[N_MAX_ANGULAR_JIT + 1];
    float gnp12[N_MAX_ANGULAR_JIT + 1];
#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      float g = 0.0f;
      float gp = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
        const int c_index = get_c_index(
          type_index,
          n,
          k,
          N_MAX_ANGULAR_JIT,
          BASIS_SIZE_ANGULAR_JIT,
          NUM_C_RADIAL_JIT);
        const float c_value = c[c_index];
        g += fn12[k] * c_value;
        gp += fnp12[k] * c_value;
      }
      gn12[n] = g;
      gnp12[n] = gp;
    }

    // A type with use_spin == 0 carries no spin degree of freedom.
    const float sj[3] = {
      use_spin[t2] ? sx_all[n2] : 0.0f,
      use_spin[t2] ? sy_all[n2] : 0.0f,
      use_spin[t2] ? sz_all[n2] : 0.0f};
    const float sksj = sk[0] * sj[0] + sk[1] * sj[1] + sk[2] * sj[2];
    const float a = sk[0] * rh[0] + sk[1] * rh[1] + sk[2] * rh[2];
    const float b = sj[0] * rh[0] + sj[1] * rh[1] + sj[2] * rh[2];
    const float cs[3] = {
      sk[1] * sj[2] - sk[2] * sj[1],
      sk[2] * sj[0] - sk[0] * sj[2],
      sk[0] * sj[1] - sk[1] * sj[0]};
    const float p = rh[0] * cs[0] + rh[1] * cs[1] + rh[2] * cs[2];
    const float sjxr[3] = {
      sj[1] * rh[2] - sj[2] * rh[1],
      sj[2] * rh[0] - sj[0] * rh[2],
      sj[0] * rh[1] - sj[1] * rh[0]};
    const float rxsk[3] = {
      rh[1] * sk[2] - rh[2] * sk[1],
      rh[2] * sk[0] - rh[0] * sk[2],
      rh[0] * sk[1] - rh[1] * sk[0]};

    float f12[3] = {0.0f};
    float t_own[3] = {0.0f};
    float t_nb[3] = {0.0f};

#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      const float fp_sex =
        on_sex ? Fp_in[n1 + (base + n) * N] : 0.0f;
      const float fp_san =
        on_san ? Fp_in[n1 + (base + NN1_JIT + n) * N] : 0.0f;
      const float fp_sdm =
        on_sdm ? Fp_in[n1 + (base + 2 * NN1_JIT + n) * N] : 0.0f;
      const float fp_env =
        on_slot4 ? Fp_in[n1 + (base + 3 * NN1_JIT + n) * N] : 0.0f;
      const float gn = gn12[n];
      const float gnp = gnp12[n];

      const float c_sex = fp_sex * gnp * sksj;

      float vs[3] = {
        b * sk[0] + a * sj[0], b * sk[1] + a * sj[1], b * sk[2] + a * sj[2]};
      const float rs = vs[0] * rh[0] + vs[1] * rh[1] + vs[2] * rh[2];
      const float rs_dm = cs[0] * rh[0] + cs[1] * rh[1] + cs[2] * rh[2];
      const float rs_env = sk[0] * rh[0] + sk[1] * rh[1] + sk[2] * rh[2];
      const float s2 = sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2];
      const float q_sia = 1.5f * a * a - 0.5f * s2;
      const float sia_scale = 3.0f * a;

      for (int d = 0; d < 3; ++d) {
        const float san_d =
          fp_san * (gnp * a * b * rh[d] + gn * d12inv * (vs[d] - rs * rh[d]));
        const float sdm_d =
          fp_sdm * (gnp * p * rh[d] + gn * d12inv * (cs[d] - rs_dm * rh[d]));
        float env_d, t_env;
        if (SPIN_USE_SIA_JIT != 0) {
          env_d = fp_env * (gnp * q_sia * rh[d] +
                            gn * d12inv * (sia_scale * sk[d] - sia_scale * a * rh[d]));
          t_env = fp_env * gn * (sia_scale * rh[d] - sk[d]);
        } else {
          env_d = fp_env * SPIN_ENV_FACTOR_JIT *
                  (gnp * a * rh[d] + gn * d12inv * (sk[d] - rs_env * rh[d]));
          t_env = fp_env * SPIN_ENV_FACTOR_JIT * gn * rh[d];
        }
        f12[d] += c_sex * rh[d] + san_d + sdm_d + env_d;
        t_own[d] += fp_sex * gn * sj[d] + fp_san * gn * b * rh[d] +
                    fp_sdm * gn * sjxr[d] + t_env;
        // A neighbour without a spin degree of freedom receives no torque.
        if (use_spin[t2]) {
          t_nb[d] += fp_sex * gn * sk[d] + fp_san * gn * a * rh[d] +
                     fp_sdm * gn * rxsk[d];
        }
      }
    }

    atomicAdd(&fx[n1], f12[0]);
    atomicAdd(&fy[n1], f12[1]);
    atomicAdd(&fz[n1], f12[2]);
    atomicAdd(&fx[n2], -f12[0]);
    atomicAdd(&fy[n2], -f12[1]);
    atomicAdd(&fz[n2], -f12[2]);

    atomicAdd(&tx[n1], -t_own[0]);
    atomicAdd(&ty[n1], -t_own[1]);
    atomicAdd(&tz[n1], -t_own[2]);
    atomicAdd(&tx[n2], -t_nb[0]);
    atomicAdd(&ty[n2], -t_nb[1]);
    atomicAdd(&tz[n2], -t_nb[2]);

    sxx -= r12[0] * f12[0];
    syy -= r12[1] * f12[1];
    szz -= r12[2] * f12[2];
    sxy -= r12[0] * f12[1];
    syz -= r12[1] * f12[2];
    szx -= r12[2] * f12[0];
  }

  // Smag (on-site |S| channel):
  //   d q_smag[n] / dS = (n+1) u^n * SPIN_MAG_SCALE_JIT * S/|S|
  // S06-R2: exact derivative for every |S| > 0, formed as c*S above the
  // threshold (bit-identical to the previous build) and with a scale-invariant
  // unit direction below it, where 1/|S| is neither accurate nor
  // representable.  Same expression as the MD kernels; at |S| == 0 the
  // derivative does not exist and the code writes 0 by convention.
  float t_mag[3] = {0.0f, 0.0f, 0.0f};
  const float smag = sqrt(sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2]);
  float dir[3] = {sk[0], sk[1], sk[2]};
  float denom = smag;
  if (smag < SPIN_MAG_MIN_MOMENT_JIT) {
    const float big = fmaxf(fabsf(sk[0]), fmaxf(fabsf(sk[1]), fabsf(sk[2])));
    if (big > 0.0f) {
      const float inv = 1.0f / big;
      const float ax = sk[0] * inv, ay = sk[1] * inv, az = sk[2] * inv;
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
    const float u_mag = smag * SPIN_MAG_SCALE_JIT;
    float un_mag = 1.0f;
#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      const float fp_mag =
        on_smag ? Fp_in[n1 + (base + 4 * NN1_JIT + n) * N] : 0.0f;
      const float c = fp_mag * (n + 1) * un_mag * SPIN_MAG_SCALE_JIT / denom;
      t_mag[0] += c * dir[0];
      t_mag[1] += c * dir[1];
      t_mag[2] += c * dir[2];
      un_mag *= u_mag;
    }
  }
  atomicAdd(&tx[n1], -t_mag[0]);
  atomicAdd(&ty[n1], -t_mag[1]);
  atomicAdd(&tz[n1], -t_mag[2]);

  virial[n1] += sxx;
  virial[n1 + N] += syy;
  virial[n1 + 2 * N] += szz;
  virial[n1 + 3 * N] += sxy;
  virial[n1 + 4 * N] += syz;
  virial[n1 + 5 * N] += szx;
}

#endif // ENABLE_SPIN_JIT

#if ENABLE_SPIN_JIT
extern "C" int nep_train_launch_descriptor_spin(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* sx,
  const float* sy,
  const float* sz,
  const int* use_spin,
  const float* parameters,
  float* descriptors)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  descriptor_spin_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12, sx, sy, sz, use_spin,
    parameters, descriptors);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_zero_torque(
  int N, float* tx, float* ty, float* tz)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  zero_torque_jit<<<grid_size, block_size>>>(N, tx, ty, tz);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_force_spin(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  const float* Fp,
  const float* sx,
  const float* sy,
  const float* sz,
  const int* use_spin,
  float* fx,
  float* fy,
  float* fz,
  float* virial,
  float* tx,
  float* ty,
  float* tz)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  force_spin_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12, parameters, Fp, sx, sy, sz,
    use_spin, fx, fy, fz, virial, tx, ty, tz);
  return static_cast<int>(cudaGetLastError());
}
#endif // ENABLE_SPIN_JIT

__device__ __forceinline__ void apply_scalar_ann_jit(
  const float* parameters,
  const int set_offset,
  const int type,
  float* q,
  float& F,
  float* Fp)
{
  const float* wb = parameters + set_offset + type * ONE_ANN_NO_BIAS_JIT;
  const float* b =
    parameters + set_offset + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT;

  if (NUM_HIDDEN_LAYERS_JIT == 2) {
    apply_ann_two_layers(
      ANN_DIM_JIT,
      NUM_NEURONS1_JIT,
      NUM_NEURONS2_JIT,
      wb,
      wb + NUM_NEURONS1_JIT * ANN_DIM_JIT,
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1),
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1 + NUM_NEURONS2_JIT),
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1 + NUM_NEURONS2_JIT) +
        NUM_NEURONS2_JIT,
      b,
      q,
      F,
      Fp);
  } else {
    apply_ann_one_layer(
      ANN_DIM_JIT,
      NUM_NEURONS1_JIT,
      wb,
      wb + NUM_NEURONS1_JIT * ANN_DIM_JIT,
      wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1),
      b,
      q,
      F,
      Fp);
  }
}

__global__ void ann_nep_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  float F = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  apply_scalar_ann_jit(parameters, 0, t, q, F, Fp);
  pe[n1] = F;

#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
  }
}

__global__ void ann_temperature_jit(
  const int N,
  const int* type,
  const float* descriptors,
  float* q_scaler,
  const float* temperature,
  const float* parameters,
  float* pe,
  float* Fp_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT - 1; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }
  q_scaler[ANN_DIM_JIT - 1] = 0.001f;
  q[ANN_DIM_JIT - 1] = temperature[n1] * 0.001f;

  float F = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  const float* wb = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* b = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT;
  apply_ann_one_layer(
    ANN_DIM_JIT,
    NUM_NEURONS1_JIT,
    wb,
    wb + NUM_NEURONS1_JIT * ANN_DIM_JIT,
    wb + NUM_NEURONS1_JIT * (ANN_DIM_JIT + 1),
    b,
    q,
    F,
    Fp);

  pe[n1] = F;
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
  }
}

__global__ void ann_charge_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp_out,
  float* charge_out,
  float* charge_derivative_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  const float* block = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* w0 = block;
  const float* b0 = w0 + NUM_NEURONS1_JIT * ANN_DIM_JIT;
  const float* w1 = b0 + NUM_NEURONS1_JIT;
  const float* b1 = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT + 1;

  float F = 0.0f;
  float charge = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  float charge_derivative[ANN_DIM_JIT] = {0.0f};

  apply_ann_one_layer_charge(
    ANN_DIM_JIT,
    NUM_NEURONS1_JIT,
    w0,
    b0,
    w1,
    b1,
    q,
    F,
    Fp,
    charge,
    charge_derivative);

  pe[n1] = F;
  charge_out[n1] = charge;
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
    charge_derivative_out[n1 + d * N] =
      charge_derivative[d] * q_scaler[d];
  }
}

__global__ void ann_vdw_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp_out,
  float* C6_out,
  float* C6_derivative_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  const float* block = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* w0 = block;
  const float* b0 = w0 + NUM_NEURONS1_JIT * ANN_DIM_JIT;
  const float* w1 = b0 + NUM_NEURONS1_JIT;
  const float* b1 = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT;

  float F = 0.0f;
  float C6 = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  float C6_derivative[ANN_DIM_JIT] = {0.0f};

  apply_ann_one_layer_vdw(
    ANN_DIM_JIT,
    NUM_NEURONS1_JIT,
    w0,
    b0,
    w1,
    b1,
    q,
    F,
    Fp,
    C6,
    C6_derivative);

  pe[n1] = F;
  const float C6_exp =
    C6_REF_SQRT_JIT[t] * expf(C6 * C6_SCALING_FACTOR_JIT);
  C6_out[n1] = C6_exp;

#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
    C6_derivative_out[n1 + d * N] =
      C6_exp * C6_SCALING_FACTOR_JIT * C6_derivative[d] * q_scaler[d];
  }
}

__global__ void ann_charge_vdw_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp_out,
  float* charge_out,
  float* charge_derivative_out,
  float* C6_out,
  float* C6_derivative_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  const float* block = parameters + t * ONE_ANN_NO_BIAS_JIT;
  const float* w0 = block;
  const float* b0 = w0 + NUM_NEURONS1_JIT * ANN_DIM_JIT;
  const float* w1 = b0 + NUM_NEURONS1_JIT;
  const float* b1 = parameters + NUM_TYPES_JIT * ONE_ANN_NO_BIAS_JIT + 1;

  float F = 0.0f;
  float charge = 0.0f;
  float C6 = 0.0f;
  float Fp[ANN_DIM_JIT] = {0.0f};
  float charge_derivative[ANN_DIM_JIT] = {0.0f};
  float C6_derivative[ANN_DIM_JIT] = {0.0f};

  apply_ann_one_layer_charge_vdw(
    ANN_DIM_JIT,
    NUM_NEURONS1_JIT,
    w0,
    b0,
    w1,
    b1,
    q,
    F,
    Fp,
    charge,
    charge_derivative,
    C6,
    C6_derivative);

  pe[n1] = F;
  charge_out[n1] = charge;
  const float C6_exp =
    C6_REF_SQRT_JIT[t] * expf(C6 * C6_SCALING_FACTOR_JIT);
  C6_out[n1] = C6_exp;

#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
    charge_derivative_out[n1 + d * N] =
      charge_derivative[d] * q_scaler[d];
    C6_derivative_out[n1 + d * N] =
      C6_exp * C6_SCALING_FACTOR_JIT * C6_derivative[d] * q_scaler[d];
  }
}

__global__ void ann_tnep_pol_jit(
  const int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* virial,
  float* Fp_out)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const int t = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  float q[ANN_DIM_JIT] = {0.0f};
#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    q[d] = descriptors[n1 + d * N] * q_scaler[d];
  }

  // Generic TNEP model_type=2:
  // second ANN set -> scalar polarizability;
  // first ANN set -> tensor descriptor derivative.
  float F = 0.0f;
  float Fp_pol[ANN_DIM_JIT] = {0.0f};
  apply_scalar_ann_jit(
    parameters, ANN_SET_SIZE_JIT, t, q, F, Fp_pol);

  virial[n1] = F;
  virial[n1 + N] = F;
  virial[n1 + 2 * N] = F;

  float Fp[ANN_DIM_JIT] = {0.0f};
  apply_scalar_ann_jit(parameters, 0, t, q, F, Fp);

#pragma unroll
  for (int d = 0; d < ANN_DIM_JIT; ++d) {
    Fp_out[n1 + d * N] = Fp[d] * q_scaler[d];
  }
}

__global__ void bec_radial_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* charge_derivative,
  float* bec)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];

    const float r12[3] = {
      x12_all[index], y12_all[index], z12_all[index]};
    const float d12 =
      sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;

    const float rc = get_rc_radial_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    float fn12[BASIS_SIZE_RADIAL_JIT + 1];
    float fnp12[BASIS_SIZE_RADIAL_JIT + 1];
    find_fn_and_fnp(
      BASIS_SIZE_RADIAL_JIT,
      rcinv,
      d12,
      fc12,
      fcp12,
      fn12,
      fnp12);

    const int type_index = descriptor_type_index_jit(t1, t2);
    float f12[3] = {0.0f};

#pragma unroll
    for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
      float gnp12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_RADIAL_JIT; ++k) {
        const int c_index =
          get_c_index(type_index, n, k, N_MAX_RADIAL_JIT, BASIS_SIZE_RADIAL_JIT);
        gnp12 += fnp12[k] * c[c_index];
      }

      const float tmp12 =
        charge_derivative[n1 + n * N] * gnp12 * d12inv;
      f12[0] += tmp12 * r12[0];
      f12[1] += tmp12 * r12[1];
      f12[2] += tmp12 * r12[2];
    }

    const float bec_xx = 0.5f * r12[0] * f12[0];
    const float bec_xy = 0.5f * r12[0] * f12[1];
    const float bec_xz = 0.5f * r12[0] * f12[2];
    const float bec_yx = 0.5f * r12[1] * f12[0];
    const float bec_yy = 0.5f * r12[1] * f12[1];
    const float bec_yz = 0.5f * r12[1] * f12[2];
    const float bec_zx = 0.5f * r12[2] * f12[0];
    const float bec_zy = 0.5f * r12[2] * f12[1];
    const float bec_zz = 0.5f * r12[2] * f12[2];

    atomicAdd(&bec[n1], bec_xx);
    atomicAdd(&bec[n1 + N], bec_xy);
    atomicAdd(&bec[n1 + 2 * N], bec_xz);
    atomicAdd(&bec[n1 + 3 * N], bec_yx);
    atomicAdd(&bec[n1 + 4 * N], bec_yy);
    atomicAdd(&bec[n1 + 5 * N], bec_yz);
    atomicAdd(&bec[n1 + 6 * N], bec_zx);
    atomicAdd(&bec[n1 + 7 * N], bec_zy);
    atomicAdd(&bec[n1 + 8 * N], bec_zz);

    atomicAdd(&bec[n2], -bec_xx);
    atomicAdd(&bec[n2 + N], -bec_xy);
    atomicAdd(&bec[n2 + 2 * N], -bec_xz);
    atomicAdd(&bec[n2 + 3 * N], -bec_yx);
    atomicAdd(&bec[n2 + 4 * N], -bec_yy);
    atomicAdd(&bec[n2 + 5 * N], -bec_yz);
    atomicAdd(&bec[n2 + 6 * N], -bec_zx);
    atomicAdd(&bec[n2 + 7 * N], -bec_zy);
    atomicAdd(&bec[n2 + 8 * N], -bec_zz);
  }
}

__global__ void bec_angular_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* charge_derivative,
  const float* sum_fxyz_in,
  float* bec)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  float Fp[DIM_ANGULAR_JIT] = {0.0f};
  float sum_fxyz[(N_MAX_ANGULAR_JIT + 1) * NUM_ABC_JIT];

#pragma unroll
  for (int d = 0; d < DIM_ANGULAR_JIT; ++d) {
    Fp[d] = charge_derivative[(DIM_RADIAL_JIT + d) * N + n1];
  }

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
#pragma unroll
    for (int abc = 0; abc < NUM_ABC_JIT; ++abc) {
      sum_fxyz[n * NUM_ABC_JIT + abc] =
        sum_fxyz_in[(n * NUM_ABC_JIT + abc) * N + n1];
    }
  }

  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];

    const float r12[3] = {
      x12_all[index], y12_all[index], z12_all[index]};
    const float d12 =
      sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

    const float rc = get_rc_angular_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    float fn12[BASIS_SIZE_ANGULAR_JIT + 1];
    float fnp12[BASIS_SIZE_ANGULAR_JIT + 1];
    find_fn_and_fnp(
      BASIS_SIZE_ANGULAR_JIT,
      rcinv,
      d12,
      fc12,
      fcp12,
      fn12,
      fnp12);

    const int type_index = descriptor_type_index_jit(t1, t2);
    float f12[3] = {0.0f};

#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
        const int c_index = get_c_index(
          type_index,
          n,
          k,
          N_MAX_ANGULAR_JIT,
          BASIS_SIZE_ANGULAR_JIT,
          NUM_C_RADIAL_JIT);
        const float c_value = c[c_index];
        gn12 += fn12[k] * c_value;
        gnp12 += fnp12[k] * c_value;
      }

      accumulate_f12<NUM_ABC_JIT>(
        L_MAX_JIT,
        HAS_Q_222_JIT,
        HAS_Q_1111_JIT,
        HAS_Q_112_JIT,
        HAS_Q_123_JIT,
        HAS_Q_233_JIT,
        HAS_Q_134_JIT,
        NUM_L_JIT,
        n,
        N_MAX_ANGULAR_JIT + 1,
        d12,
        r12,
        gn12,
        gnp12,
        Fp,
        sum_fxyz,
        f12);
    }

    const float bec_xx = 0.5f * r12[0] * f12[0];
    const float bec_xy = 0.5f * r12[0] * f12[1];
    const float bec_xz = 0.5f * r12[0] * f12[2];
    const float bec_yx = 0.5f * r12[1] * f12[0];
    const float bec_yy = 0.5f * r12[1] * f12[1];
    const float bec_yz = 0.5f * r12[1] * f12[2];
    const float bec_zx = 0.5f * r12[2] * f12[0];
    const float bec_zy = 0.5f * r12[2] * f12[1];
    const float bec_zz = 0.5f * r12[2] * f12[2];

    atomicAdd(&bec[n1], bec_xx);
    atomicAdd(&bec[n1 + N], bec_xy);
    atomicAdd(&bec[n1 + 2 * N], bec_xz);
    atomicAdd(&bec[n1 + 3 * N], bec_yx);
    atomicAdd(&bec[n1 + 4 * N], bec_yy);
    atomicAdd(&bec[n1 + 5 * N], bec_yz);
    atomicAdd(&bec[n1 + 6 * N], bec_zx);
    atomicAdd(&bec[n1 + 7 * N], bec_zy);
    atomicAdd(&bec[n1 + 8 * N], bec_zz);

    atomicAdd(&bec[n2], -bec_xx);
    atomicAdd(&bec[n2 + N], -bec_xy);
    atomicAdd(&bec[n2 + 2 * N], -bec_xz);
    atomicAdd(&bec[n2 + 3 * N], -bec_yx);
    atomicAdd(&bec[n2 + 4 * N], -bec_yy);
    atomicAdd(&bec[n2 + 5 * N], -bec_yz);
    atomicAdd(&bec[n2 + 6 * N], -bec_zx);
    atomicAdd(&bec[n2 + 7 * N], -bec_zy);
    atomicAdd(&bec[n2 + 8 * N], -bec_zz);
  }
}

__device__ __forceinline__ float effective_fp_jit(
  const int index,
  const int atom,
  const float* Fp,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6)
{
  float value = Fp[index];

#if NEP_MODEL_MODE_JIT == NEP_MODEL_CHARGE || \
    NEP_MODEL_MODE_JIT == NEP_MODEL_CHARGE_VDW
  value += charge_derivative[index] * D_real[atom];
#else
  (void)charge_derivative;
  (void)D_real;
#endif

#if NEP_MODEL_MODE_JIT == NEP_MODEL_VDW || \
    NEP_MODEL_MODE_JIT == NEP_MODEL_CHARGE_VDW
  value += C6_derivative[index] * D_C6[atom];
#else
  (void)C6_derivative;
  (void)D_C6;
#endif

  return value;
}

__global__ void force_radial_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* Fp,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6,
  const int is_dipole,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];

  float sxx = 0.0f;
  float syy = 0.0f;
  float szz = 0.0f;
  float sxy = 0.0f;
  float syz = 0.0f;
  float szx = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float r12[3] = {
      x12_all[index], y12_all[index], z12_all[index]};
    const float d12 =
      sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
    const float d12inv = 1.0f / d12;

    const float rc = get_rc_radial_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    float fn12[BASIS_SIZE_RADIAL_JIT + 1];
    float fnp12[BASIS_SIZE_RADIAL_JIT + 1];
    find_fn_and_fnp(
      BASIS_SIZE_RADIAL_JIT,
      rcinv,
      d12,
      fc12,
      fcp12,
      fn12,
      fnp12);

    const int type_index = descriptor_type_index_jit(t1, t2);
    float f12[3] = {0.0f};

#pragma unroll
    for (int n = 0; n <= N_MAX_RADIAL_JIT; ++n) {
      float gnp12 = 0.0f;
#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_RADIAL_JIT; ++k) {
        const int c_index =
          get_c_index(type_index, n, k, N_MAX_RADIAL_JIT, BASIS_SIZE_RADIAL_JIT);
        gnp12 += fnp12[k] * c[c_index];
      }

      const int fp_index = n1 + n * N;
      float tmp12 = effective_fp_jit(
        fp_index,
        n1,
        Fp,
        charge_derivative,
        D_real,
        C6_derivative,
        D_C6);
      tmp12 *= gnp12 * d12inv;

      f12[0] += tmp12 * r12[0];
      f12[1] += tmp12 * r12[1];
      f12[2] += tmp12 * r12[2];
    }

    atomicAdd(&fx[n1], f12[0]);
    atomicAdd(&fy[n1], f12[1]);
    atomicAdd(&fz[n1], f12[2]);
    atomicAdd(&fx[n2], -f12[0]);
    atomicAdd(&fy[n2], -f12[1]);
    atomicAdd(&fz[n2], -f12[2]);

#if NEP_MODEL_MODE_JIT == NEP_MODEL_TNEP
    if (is_dipole) {
      const float r2 =
        r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      sxx -= r2 * f12[0];
      syy -= r2 * f12[1];
      szz -= r2 * f12[2];
    } else {
      sxx -= r12[0] * f12[0];
      syy -= r12[1] * f12[1];
      szz -= r12[2] * f12[2];
    }
#else
    (void)is_dipole;
    sxx -= r12[0] * f12[0];
    syy -= r12[1] * f12[1];
    szz -= r12[2] * f12[2];
#endif
    sxy -= r12[0] * f12[1];
    syz -= r12[1] * f12[2];
    szx -= r12[2] * f12[0];
  }

  virial[n1] += sxx;
  virial[n1 + N] += syy;
  virial[n1 + 2 * N] += szz;

#if NEP_MODEL_MODE_JIT == NEP_MODEL_NEP || \
    NEP_MODEL_MODE_JIT == NEP_MODEL_TNEP
  // These paths zero only the diagonal virial before radial force,
  // so radial force initializes the shear components.
  virial[n1 + 3 * N] = sxy;
  virial[n1 + 4 * N] = syz;
  virial[n1 + 5 * N] = szx;
#else
  // qNEP/vdW paths can already contain long-range virial contributions.
  virial[n1 + 3 * N] += sxy;
  virial[n1 + 4 * N] += syz;
  virial[n1 + 5 * N] += szx;
#endif
}

__global__ void force_angular_jit(
  const int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12_all,
  const float* y12_all,
  const float* z12_all,
  const float* parameters,
  const float* Fp_in,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6,
  const float* sum_fxyz_in,
  const int is_dipole,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 >= N) return;

  const float* c = get_c_jit(parameters);
  float Fp[DIM_ANGULAR_JIT] = {0.0f};
  float sum_fxyz[(N_MAX_ANGULAR_JIT + 1) * NUM_ABC_JIT];

#pragma unroll
  for (int d = 0; d < DIM_ANGULAR_JIT; ++d) {
    const int index = (DIM_RADIAL_JIT + d) * N + n1;
    Fp[d] = effective_fp_jit(
      index,
      n1,
      Fp_in,
      charge_derivative,
      D_real,
      C6_derivative,
      D_C6);
  }

#pragma unroll
  for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
#pragma unroll
    for (int abc = 0; abc < NUM_ABC_JIT; ++abc) {
      sum_fxyz[n * NUM_ABC_JIT + abc] =
        sum_fxyz_in[(n * NUM_ABC_JIT + abc) * N + n1];
    }
  }

  const int t1 = (NUM_TYPES_JIT == 1) ? 0 : type[n1];
  const int neighbor_number = NN[n1];

  float sxx = 0.0f;
  float syy = 0.0f;
  float szz = 0.0f;
  float sxy = 0.0f;
  float syz = 0.0f;
  float szx = 0.0f;

  for (int i1 = 0; i1 < neighbor_number; ++i1) {
    const int index = NN_sum[n1] + i1;
    const int n2 = NL[index];
    const int t2 = (NUM_TYPES_JIT == 1) ? 0 : type[n2];
    const float r12[3] = {
      x12_all[index], y12_all[index], z12_all[index]};
    const float d12 =
      sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);

    const float rc = get_rc_angular_jit(t1, t2);
    const float rcinv = 1.0f / rc;
    float fc12, fcp12;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

    float fn12[BASIS_SIZE_ANGULAR_JIT + 1];
    float fnp12[BASIS_SIZE_ANGULAR_JIT + 1];
    find_fn_and_fnp(
      BASIS_SIZE_ANGULAR_JIT,
      rcinv,
      d12,
      fc12,
      fcp12,
      fn12,
      fnp12);

    const int type_index = descriptor_type_index_jit(t1, t2);
    float f12[3] = {0.0f};

#pragma unroll
    for (int n = 0; n <= N_MAX_ANGULAR_JIT; ++n) {
      float gn12 = 0.0f;
      float gnp12 = 0.0f;

#pragma unroll
      for (int k = 0; k <= BASIS_SIZE_ANGULAR_JIT; ++k) {
        const int c_index = get_c_index(
          type_index,
          n,
          k,
          N_MAX_ANGULAR_JIT,
          BASIS_SIZE_ANGULAR_JIT,
          NUM_C_RADIAL_JIT);
        const float c_value = c[c_index];
        gn12 += fn12[k] * c_value;
        gnp12 += fnp12[k] * c_value;
      }

      accumulate_f12<NUM_ABC_JIT>(
        L_MAX_JIT,
        HAS_Q_222_JIT,
        HAS_Q_1111_JIT,
        HAS_Q_112_JIT,
        HAS_Q_123_JIT,
        HAS_Q_233_JIT,
        HAS_Q_134_JIT,
        NUM_L_JIT,
        n,
        N_MAX_ANGULAR_JIT + 1,
        d12,
        r12,
        gn12,
        gnp12,
        Fp,
        sum_fxyz,
        f12);
    }

    atomicAdd(&fx[n1], f12[0]);
    atomicAdd(&fy[n1], f12[1]);
    atomicAdd(&fz[n1], f12[2]);
    atomicAdd(&fx[n2], -f12[0]);
    atomicAdd(&fy[n2], -f12[1]);
    atomicAdd(&fz[n2], -f12[2]);

#if NEP_MODEL_MODE_JIT == NEP_MODEL_TNEP
    if (is_dipole) {
      const float r2 =
        r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      sxx -= r2 * f12[0];
      syy -= r2 * f12[1];
      szz -= r2 * f12[2];
    } else {
      sxx -= r12[0] * f12[0];
      syy -= r12[1] * f12[1];
      szz -= r12[2] * f12[2];
    }
#else
    (void)is_dipole;
    sxx -= r12[0] * f12[0];
    syy -= r12[1] * f12[1];
    szz -= r12[2] * f12[2];
#endif
    sxy -= r12[0] * f12[1];
    syz -= r12[1] * f12[2];
    szx -= r12[2] * f12[0];
  }

  virial[n1] += sxx;
  virial[n1 + N] += syy;
  virial[n1 + 2 * N] += szz;
  virial[n1 + 3 * N] += sxy;
  virial[n1 + 4 * N] += syz;
  virial[n1 + 5 * N] += szx;
}

extern "C" int nep_train_specialized_interface_version()
{
  return NEP_SPECIALIZED_SOURCE_INTERFACE_VERSION;
}

/*
 * Stage03A.1: the configuration this translation unit was actually compiled
 * with.
 *
 * Every value below is read from the compile-time constants in
 * nep_special_config.cuh, i.e. from the generated header the specialized
 * kernels themselves are built against -- not from anything the caller passes
 * in.  A verification driver can therefore ask the kernel object "what are
 * you?" and compare the answer against the model it is about to evaluate,
 * instead of echoing back the configuration that was requested.
 *
 * The fingerprint is FNV-1a 64 over a canonical text rendering of the same
 * constants, so a single number identifies the build.
 */
extern "C" int nep_train_specialized_config(
  int* out_mask,
  int* out_use_sia,
  int* out_dim,
  int* out_dim_radial,
  int* out_dim_angular,
  int* out_dim_spin,
  int* out_num_types,
  int* out_num_neurons1,
  int* out_n_max_radial,
  int* out_n_max_angular,
  int* out_basis_radial,
  int* out_basis_angular,
  int* out_l_max,
  int* out_enable_spin,
  float* out_spin_mag_scale,
  unsigned long long* out_fingerprint)
{
  const int dim = DIM_RADIAL_JIT + DIM_ANGULAR_JIT + DIM_SPIN_JIT;
  if (out_mask) *out_mask = SPIN_BLOCK_MASK_JIT;
  if (out_use_sia) *out_use_sia = SPIN_USE_SIA_JIT;
  if (out_dim) *out_dim = dim;
  if (out_dim_radial) *out_dim_radial = DIM_RADIAL_JIT;
  if (out_dim_angular) *out_dim_angular = DIM_ANGULAR_JIT;
  if (out_dim_spin) *out_dim_spin = DIM_SPIN_JIT;
  if (out_num_types) *out_num_types = NUM_TYPES_JIT;
  if (out_num_neurons1) *out_num_neurons1 = NUM_NEURONS1_JIT;
  if (out_n_max_radial) *out_n_max_radial = N_MAX_RADIAL_JIT;
  if (out_n_max_angular) *out_n_max_angular = N_MAX_ANGULAR_JIT;
  if (out_basis_radial) *out_basis_radial = BASIS_SIZE_RADIAL_JIT;
  if (out_basis_angular) *out_basis_angular = BASIS_SIZE_ANGULAR_JIT;
  if (out_l_max) *out_l_max = L_MAX_JIT;
  if (out_enable_spin) *out_enable_spin = ENABLE_SPIN_JIT;
  if (out_spin_mag_scale) *out_spin_mag_scale = SPIN_MAG_SCALE_JIT;
  if (out_fingerprint) {
    char text[512];
    std::snprintf(
      text,
      sizeof(text),
      "iface=%d;mask=%d;sia=%d;ty=%d;dimr=%d;dima=%d;dims=%d;neur=%d;"
      "nr=%d;na=%d;br=%d;ba=%d;lm=%d;spin=%d;mag=%.9e;cj=%d",
      NEP_SPECIALIZED_SOURCE_INTERFACE_VERSION,
      SPIN_BLOCK_MASK_JIT,
      SPIN_USE_SIA_JIT,
      NUM_TYPES_JIT,
      DIM_RADIAL_JIT,
      DIM_ANGULAR_JIT,
      DIM_SPIN_JIT,
      NUM_NEURONS1_JIT,
      N_MAX_RADIAL_JIT,
      N_MAX_ANGULAR_JIT,
      BASIS_SIZE_RADIAL_JIT,
      BASIS_SIZE_ANGULAR_JIT,
      L_MAX_JIT,
      ENABLE_SPIN_JIT,
      (double)SPIN_MAG_SCALE_JIT,
      DESCRIPTOR_USE_CJ_JIT);
    unsigned long long h = 0xcbf29ce484222325ULL;
    for (const char* p = text; *p != '\0'; ++p) {
      h ^= static_cast<unsigned long long>(static_cast<unsigned char>(*p));
      h *= 0x100000001b3ULL;
    }
    *out_fingerprint = h;
  }
  return 0;
}

extern "C" int nep_train_launch_descriptor_radial(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  float* descriptors)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  descriptor_radial_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12, parameters, descriptors);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_descriptor_angular(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  float* descriptors,
  float* sum_fxyz)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  descriptor_angular_jit<<<grid_size, block_size>>>(
    N,
    NN_sum,
    NN,
    NL,
    type,
    x12,
    y12,
    z12,
    parameters,
    descriptors,
    sum_fxyz);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_nep(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_nep_jit<<<grid_size, block_size>>>(
    N, type, descriptors, q_scaler, parameters, pe, Fp);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_temperature(
  int N,
  const int* type,
  const float* descriptors,
  float* q_scaler,
  const float* temperature,
  const float* parameters,
  float* pe,
  float* Fp)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_temperature_jit<<<grid_size, block_size>>>(
    N,
    type,
    descriptors,
    q_scaler,
    temperature,
    parameters,
    pe,
    Fp);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_charge(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp,
  float* charge,
  float* charge_derivative)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_charge_jit<<<grid_size, block_size>>>(
    N,
    type,
    descriptors,
    q_scaler,
    parameters,
    pe,
    Fp,
    charge,
    charge_derivative);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_vdw(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp,
  float* C6,
  float* C6_derivative)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_vdw_jit<<<grid_size, block_size>>>(
    N,
    type,
    descriptors,
    q_scaler,
    parameters,
    pe,
    Fp,
    C6,
    C6_derivative);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_charge_vdw(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* pe,
  float* Fp,
  float* charge,
  float* charge_derivative,
  float* C6,
  float* C6_derivative)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_charge_vdw_jit<<<grid_size, block_size>>>(
    N,
    type,
    descriptors,
    q_scaler,
    parameters,
    pe,
    Fp,
    charge,
    charge_derivative,
    C6,
    C6_derivative);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_ann_tnep_pol(
  int N,
  const int* type,
  const float* descriptors,
  const float* q_scaler,
  const float* parameters,
  float* virial,
  float* Fp)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  ann_tnep_pol_jit<<<grid_size, block_size>>>(
    N, type, descriptors, q_scaler, parameters, virial, Fp);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_bec_radial(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  const float* charge_derivative,
  float* bec)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  bec_radial_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12,
    parameters, charge_derivative, bec);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_bec_angular(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  const float* charge_derivative,
  const float* sum_fxyz,
  float* bec)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  bec_angular_jit<<<grid_size, block_size>>>(
    N, NN_sum, NN, NL, type, x12, y12, z12,
    parameters, charge_derivative, sum_fxyz, bec);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_force_radial(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  const float* Fp,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6,
  int is_dipole,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  force_radial_jit<<<grid_size, block_size>>>(
    N,
    NN_sum,
    NN,
    NL,
    type,
    x12,
    y12,
    z12,
    parameters,
    Fp,
    charge_derivative,
    D_real,
    C6_derivative,
    D_C6,
    is_dipole,
    fx,
    fy,
    fz,
    virial);
  return static_cast<int>(cudaGetLastError());
}

extern "C" int nep_train_launch_force_angular(
  int N,
  const int* NN_sum,
  const int* NN,
  const int* NL,
  const int* type,
  const float* x12,
  const float* y12,
  const float* z12,
  const float* parameters,
  const float* Fp,
  const float* charge_derivative,
  const float* D_real,
  const float* C6_derivative,
  const float* D_C6,
  const float* sum_fxyz,
  int is_dipole,
  float* fx,
  float* fy,
  float* fz,
  float* virial)
{
  const int block_size = 32;
  const int grid_size = (N - 1) / block_size + 1;
  force_angular_jit<<<grid_size, block_size>>>(
    N,
    NN_sum,
    NN,
    NL,
    type,
    x12,
    y12,
    z12,
    parameters,
    Fp,
    charge_derivative,
    D_real,
    C6_derivative,
    D_C6,
    sum_fxyz,
    is_dipole,
    fx,
    fy,
    fz,
    virial);
  return static_cast<int>(cudaGetLastError());
}

#endif // NEP_SPECIALIZED_RUNTIME_BUILD
