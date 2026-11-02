/*
 * Spin Route B descriptor core -- the CUDA kernels.
 *
 * These six kernels are the whole spin pipeline of a spin-aware NEP:
 *
 *   find_descriptor_spin             Sex, San, Sdm, Sia|Senv, Smag (large box)
 *   find_descriptor_spin_small_box   the same, from precomputed image vectors
 *   zero_torque_md                   clear the double-precision accumulator
 *   convert_torque_to_float          round it into the array the integrator reads
 *   find_force_spin                  spin contribution to force / torque / virial
 *   find_force_spin_small_box        the same, small box
 *
 * They used to exist twice, once in src/force/nep.cu and once -- with a
 * different macro name and an inlined coefficient index -- in the replica fork
 * src/main_replica/force/nep.cu.  The two copies were byte-identical apart
 * from those two details, which is exactly the situation in which a fix
 * applied to one copy silently fails to reach the other.  They are now defined
 * once, here, and both call sites instantiate them with their own NEP::ParaMB
 * and NEP::ANN types (hence the template parameters).
 *
 * The header must be included after nep.cuh / nep_small_box.cuh and
 * utilities/nep_utilities.cuh, because the kernels use Box, apply_mic() and
 * MAX_NUM_N.
 *
 * The declared symmetry and capability of each block (parity under inversion
 * and time reversal, whether it needs SOC, whether it is identically zero for
 * an inversion-symmetric configuration or for a locally cubic environment, and
 * whether it is analytic at |S| = 0) lives in spin_descriptor.cuh and is the
 * only place where it is written down.  S06 added it; nothing here may claim a
 * symmetry the table does not state.
 */
#pragma once

#include "spin_descriptor.cuh"

// The accumulator type has to be visible wherever this header is included.
// main_replica/ includes this file from the parent tree but keeps its own copy
// of utilities/nep_utilities.cuh, so the typedef cannot be assumed to come from
// there -- define it here once, guarded.
#ifndef NEP_ACC_T_DEFINED
#define NEP_ACC_T_DEFINED
typedef double nep_acc_t;
#endif



// Spin (Route B): Sex, San, Sdm, Senv and their analytic derivatives.
// The spin part of the descriptor is evaluated in a separate kernel so that
// both the small-box and the large-box code paths can use it unchanged.
// Smag channel: |S_i| enters the descriptor through the monomials of
// u = |S_i| * spin_mag_scale.  The cDFT magnetic-force label carries a large
// longitudinal (constraint) part that is almost a pure function of |S_i|
// (R^2 = 0.89 for a cubic fit), while Sex/San/Sdm/Senv are all strictly linear
// in S_i and therefore cannot represent it.  |S| is even under S -> -S and
// invariant under spin rotations, so it does not spoil any symmetry; it has no
// position dependence, so it contributes to the torque but not to the force or
// the virial.  The scale is a model parameter (paramb.spin_mag_scale), written
// to nep.txt as `spin_mag_scale <value>` so that a model trained with a
// different scaling is evaluated with that same scaling.

template <typename ParaMB, typename ANN>
__global__ void find_descriptor_spin(
  ParaMB paramb,
  ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_sx,
  const float* __restrict__ g_sy,
  const float* __restrict__ g_sz,
  const int* __restrict__ g_use_spin,
  float* g_q_spin)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const int t1 = g_type[n1];
  const int n_max = paramb.n_max_angular;
  const int nn1 = n_max + 1;
  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  float sk[3] = {g_sx[n1], g_sy[n1], g_sz[n1]};
  if (!g_use_spin[t1]) {
    sk[0] = 0.0f;
    sk[1] = 0.0f;
    sk[2] = 0.0f;
  }

  nep_acc_t q_sex[MAX_NUM_N] = {0.0};
  nep_acc_t q_san[MAX_NUM_N] = {0.0};
  nep_acc_t q_sdm[MAX_NUM_N] = {0.0};
  // Both slot-4 candidates are accumulated (spin_use_sia is uniform across the
  // grid) and only the selected one is written out at the tail of the kernel.
  nep_acc_t q_sia[MAX_NUM_N] = {0.0};
  nep_acc_t q_senv[MAX_NUM_N] = {0.0};
  const float s2 = sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2];

  for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
    int n2 = g_NL_angular[n1 + N * i1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    if (d12 < 1.0e-8f) continue;
    float d12inv = 1.0f / d12;
    float r[3] = {x12 * d12inv, y12 * d12inv, z12 * d12inv};
    int t2 = g_type[n2];
    float fc12;
    float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
    float rcinv = 1.0f / rc;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);

    float gn12[MAX_NUM_N];
    for (int n = 0; n <= n_max; ++n) {
      float g = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = spin_c_index(paramb, t1, t2, n, k);
        g += fn12[k] * annmb.c_type_pair[c_index];
      }
      gn12[n] = g;
    }

    float sj[3] = {g_sx[n2], g_sy[n2], g_sz[n2]};
    if (!g_use_spin[t2]) {
      sj[0] = 0.0f;
      sj[1] = 0.0f;
      sj[2] = 0.0f;
    }
    float sksj = sk[0] * sj[0] + sk[1] * sj[1] + sk[2] * sj[2];
    float a = sk[0] * r[0] + sk[1] * r[1] + sk[2] * r[2];
    float b = sj[0] * r[0] + sj[1] * r[1] + sj[2] * r[2];
    float cs[3] = {
      sk[1] * sj[2] - sk[2] * sj[1],
      sk[2] * sj[0] - sk[0] * sj[2],
      sk[0] * sj[1] - sk[1] * sj[0]};
    float p = r[0] * cs[0] + r[1] * cs[1] + r[2] * cs[2];

    for (int n = 0; n <= n_max; ++n) {
      q_sex[n] += gn12[n] * sksj;
      q_san[n] += gn12[n] * a * b;
      q_sdm[n] += gn12[n] * p;
      // SIA (see spin_descriptor.cuh): zero in a locally cubic environment,
      // non-zero -- and the source of the easy axis -- otherwise.
      q_sia[n] += gn12[n] * (1.5f * a * a - 0.5f * s2);
      // legacy Senv (layout <= 2): T-odd, kept for backward compatibility only
      q_senv[n] += gn12[n] * a;
    }
  }

  // F26: a masked-off block writes a literal zero (the matching fp_* is zeroed
  // in the force loops as well, see spin_descriptor.cuh).
  const int blk = paramb.spin_block_mask;
  for (int n = 0; n <= n_max; ++n) {
    g_q_spin[n1 + static_cast<size_t>(N) * n] =
      (blk & SPIN_BLOCK_BIT_SEX) ? static_cast<float>(q_sex[n]) : 0.0f;
    g_q_spin[n1 + static_cast<size_t>(N) * (nn1 + n)] =
      (blk & SPIN_BLOCK_BIT_SAN) ? static_cast<float>(q_san[n]) : 0.0f;
    g_q_spin[n1 + static_cast<size_t>(N) * (2 * nn1 + n)] =
      (blk & SPIN_BLOCK_BIT_SDM) ? static_cast<float>(q_sdm[n]) : 0.0f;
    g_q_spin[n1 + static_cast<size_t>(N) * (3 * nn1 + n)] =
      (blk & SPIN_BLOCK_BIT_SLOT4)
        ? static_cast<float>(paramb.spin_use_sia ? q_sia[n]
                                                  : (SPIN_ENV_FACTOR * q_senv[n]))
        : 0.0f;
  }

  // Smag: on-site |S| channel, u^(n+1) with u = |S| * paramb.spin_mag_scale
  const float smag = sqrt(sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2]);
  float u_mag = smag * paramb.spin_mag_scale;
  float un_mag = 1.0f;
  for (int n = 0; n <= n_max; ++n) {
    un_mag *= u_mag;
    g_q_spin[n1 + static_cast<size_t>(N) * (4 * nn1 + n)] =
      (blk & SPIN_BLOCK_BIT_SMAG) ? un_mag : 0.0f;
  }
}

// Small-box variant.  The small-box neighbour list enumerates *every* periodic
// image (the same neighbour index can appear several times with different image
// vectors), so the displacement vectors must be taken from the precomputed r12
// arrays instead of being rebuilt with apply_mic(), which would collapse all the
// images of a neighbour onto a single (wrong) vector.
template <typename ParaMB, typename ANN>
__global__ void find_descriptor_spin_small_box(
  ParaMB paramb,
  ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const int* __restrict__ g_type,
  const float* __restrict__ g_sx,
  const float* __restrict__ g_sy,
  const float* __restrict__ g_sz,
  const int* __restrict__ g_use_spin,
  float* g_q_spin)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const int t1 = g_type[n1];
  const int n_max = paramb.n_max_angular;
  const int nn1 = n_max + 1;
  float sk[3] = {g_sx[n1], g_sy[n1], g_sz[n1]};
  if (!g_use_spin[t1]) {
    sk[0] = sk[1] = sk[2] = 0.0f;
  }

  nep_acc_t q_sex[MAX_NUM_N] = {0.0};
  nep_acc_t q_san[MAX_NUM_N] = {0.0};
  nep_acc_t q_sdm[MAX_NUM_N] = {0.0};
  nep_acc_t q_sia[MAX_NUM_N] = {0.0};
  nep_acc_t q_senv[MAX_NUM_N] = {0.0};
  const float s2 = sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2];

  for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
    const int index = i1 * N + n1;
    int n2 = g_NL_angular[index];
    float x12 = g_x12[index];
    float y12 = g_y12[index];
    float z12 = g_z12[index];
    float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    if (d12 < 1.0e-8f) continue;
    float d12inv = 1.0f / d12;
    float r[3] = {x12 * d12inv, y12 * d12inv, z12 * d12inv};
    int t2 = g_type[n2];
    float fc12;
    float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
    float rcinv = 1.0f / rc;
    find_fc(rc, rcinv, d12, fc12);
    float fn12[MAX_NUM_N];
    find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);

    float gn12[MAX_NUM_N];
    for (int n = 0; n <= n_max; ++n) {
      float g = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = spin_c_index(paramb, t1, t2, n, k);
        g += fn12[k] * annmb.c_type_pair[c_index];
      }
      gn12[n] = g;
    }

    float sj[3] = {g_sx[n2], g_sy[n2], g_sz[n2]};
    if (!g_use_spin[t2]) {
      sj[0] = sj[1] = sj[2] = 0.0f;
    }
    float sksj = sk[0] * sj[0] + sk[1] * sj[1] + sk[2] * sj[2];
    float a = sk[0] * r[0] + sk[1] * r[1] + sk[2] * r[2];
    float b = sj[0] * r[0] + sj[1] * r[1] + sj[2] * r[2];
    float cs[3] = {
      sk[1] * sj[2] - sk[2] * sj[1],
      sk[2] * sj[0] - sk[0] * sj[2],
      sk[0] * sj[1] - sk[1] * sj[0]};
    float p = r[0] * cs[0] + r[1] * cs[1] + r[2] * cs[2];

    for (int n = 0; n <= n_max; ++n) {
      q_sex[n] += gn12[n] * sksj;
      q_san[n] += gn12[n] * a * b;
      q_sdm[n] += gn12[n] * p;
      // SIA (see spin_descriptor.cuh): zero in a locally cubic environment,
      // non-zero -- and the source of the easy axis -- otherwise.
      q_sia[n] += gn12[n] * (1.5f * a * a - 0.5f * s2);
      // legacy Senv (layout <= 2): T-odd, kept for backward compatibility only
      q_senv[n] += gn12[n] * a;
    }
  }

  // F26: a masked-off block writes a literal zero (the matching fp_* is zeroed
  // in the force loops as well, see spin_descriptor.cuh).
  const int blk = paramb.spin_block_mask;
  for (int n = 0; n <= n_max; ++n) {
    g_q_spin[n1 + static_cast<size_t>(N) * n] =
      (blk & SPIN_BLOCK_BIT_SEX) ? static_cast<float>(q_sex[n]) : 0.0f;
    g_q_spin[n1 + static_cast<size_t>(N) * (nn1 + n)] =
      (blk & SPIN_BLOCK_BIT_SAN) ? static_cast<float>(q_san[n]) : 0.0f;
    g_q_spin[n1 + static_cast<size_t>(N) * (2 * nn1 + n)] =
      (blk & SPIN_BLOCK_BIT_SDM) ? static_cast<float>(q_sdm[n]) : 0.0f;
    g_q_spin[n1 + static_cast<size_t>(N) * (3 * nn1 + n)] =
      (blk & SPIN_BLOCK_BIT_SLOT4)
        ? static_cast<float>(paramb.spin_use_sia ? q_sia[n]
                                                  : (SPIN_ENV_FACTOR * q_senv[n]))
        : 0.0f;
  }

  // Smag: on-site |S| channel, u^(n+1) with u = |S| * paramb.spin_mag_scale
  const float smag = sqrt(sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2]);
  float u_mag = smag * paramb.spin_mag_scale;
  float un_mag = 1.0f;
  for (int n = 0; n <= n_max; ++n) {
    un_mag *= u_mag;
    g_q_spin[n1 + static_cast<size_t>(N) * (4 * nn1 + n)] =
      (blk & SPIN_BLOCK_BIT_SMAG) ? un_mag : 0.0f;
  }
}

// The torque is accumulated in double precision (see torque_accumulator_) and
// only then rounded to the float array the integrator reads.
static __global__ void zero_torque_md(
  const int N, double* g_tx, double* g_ty, double* g_tz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) return;
  g_tx[n1] = 0.0;
  g_ty[n1] = 0.0;
  g_tz[n1] = 0.0;
}

// Round 9 (task 48): clear the 9N magnetic heat-current tensor A_j.  It is
// accumulated with atomicAdd (a neighbour index, not the owning thread index,
// is the target), so it has to be zeroed by a kernel, not by a `+= 0` write.
static __global__ void zero_Aspin(const int N, double* g_Aspin)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) return;
  for (int k = 0; k < 9; ++k) {
    g_Aspin[n1 + static_cast<size_t>(N) * k] = 0.0;
  }
}

static __global__ void convert_torque_to_float(
  const int N,
  const double* __restrict__ g_tx,
  const double* __restrict__ g_ty,
  const double* __restrict__ g_tz,
  float* g_tx_f,
  float* g_ty_f,
  float* g_tz_f)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) return;
  g_tx_f[n1] = static_cast<float>(g_tx[n1]);
  g_ty_f[n1] = static_cast<float>(g_ty[n1]);
  g_tz_f[n1] = static_cast<float>(g_tz[n1]);
}

template <typename ParaMB, typename ANN>
__global__ void find_force_spin(
  ParaMB paramb,
  ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sx,
  const float* __restrict__ g_sy,
  const float* __restrict__ g_sz,
  const int* __restrict__ g_use_spin,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_tx,
  double* g_ty,
  double* g_tz,
  double* g_Aspin = nullptr)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const int t1 = g_type[n1];
  if (!g_use_spin[t1]) return;

  const int n_max = paramb.n_max_angular;
  const int nn1 = n_max + 1;
  const int base = (paramb.n_max_radial + 1) + paramb.dim_angular;
  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  float sk[3] = {g_sx[n1], g_sy[n1], g_sz[n1]};
  const float s2 = sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2];  // |S_i|^2

  for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
    int n2 = g_NL_angular[n1 + N * i1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    if (d12 < 1.0e-8f) continue;
    float d12inv = 1.0f / d12;
    float r[3] = {x12 * d12inv, y12 * d12inv, z12 * d12inv};
    int t2 = g_type[n2];
    float fc12, fcp12;
    float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
    float rcinv = 1.0f / rc;
    find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
    float fn12[MAX_NUM_N];
    float fnp12[MAX_NUM_N];
    find_fn_and_fnp(
      paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);

    float gn12[MAX_NUM_N];
    float gnp12[MAX_NUM_N];
    for (int n = 0; n <= n_max; ++n) {
      float g = 0.0f;
      float gp = 0.0f;
      for (int k = 0; k <= paramb.basis_size_angular; ++k) {
        int c_index = spin_c_index(paramb, t1, t2, n, k);
        float c_value = annmb.c_type_pair[c_index];
        g += fn12[k] * c_value;
        gp += fnp12[k] * c_value;
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
    float sksj = sk[0] * sj[0] + sk[1] * sj[1] + sk[2] * sj[2];
    float a = sk[0] * r[0] + sk[1] * r[1] + sk[2] * r[2];
    float b = sj[0] * r[0] + sj[1] * r[1] + sj[2] * r[2];
    float cs[3] = {
      sk[1] * sj[2] - sk[2] * sj[1],
      sk[2] * sj[0] - sk[0] * sj[2],
      sk[0] * sj[1] - sk[1] * sj[0]};
    float p = r[0] * cs[0] + r[1] * cs[1] + r[2] * cs[2];
    float sjxr[3] = {
      sj[1] * r[2] - sj[2] * r[1],
      sj[2] * r[0] - sj[0] * r[2],
      sj[0] * r[1] - sj[1] * r[0]};
    float rxsk[3] = {
      r[1] * sk[2] - r[2] * sk[1],
      r[2] * sk[0] - r[0] * sk[2],
      r[0] * sk[1] - r[1] * sk[0]};

    float f12[3] = {0.0f};
    float t_own[3] = {0.0f};
    float t_nb[3] = {0.0f};

    for (int n = 0; n <= n_max; ++n) {
      const int blk = paramb.spin_block_mask;
      float fp_sex = (blk & SPIN_BLOCK_BIT_SEX)
        ? g_Fp[static_cast<size_t>(N) * (base + n) + n1] : 0.0f;
      float fp_san = (blk & SPIN_BLOCK_BIT_SAN)
        ? g_Fp[static_cast<size_t>(N) * (base + nn1 + n) + n1] : 0.0f;
      float fp_sdm = (blk & SPIN_BLOCK_BIT_SDM)
        ? g_Fp[static_cast<size_t>(N) * (base + 2 * nn1 + n) + n1] : 0.0f;
      float fp_env = (blk & SPIN_BLOCK_BIT_SLOT4)
        ? g_Fp[static_cast<size_t>(N) * (base + 3 * nn1 + n) + n1] : 0.0f;
      float gn = gn12[n];
      float gnp = gnp12[n];
      float c_sex = fp_sex * gnp * sksj;

      float vs[3] = {b * sk[0] + a * sj[0], b * sk[1] + a * sj[1], b * sk[2] + a * sj[2]};
      float rs = vs[0] * r[0] + vs[1] * r[1] + vs[2] * r[2];
      float rs_dm = cs[0] * r[0] + cs[1] * r[1] + cs[2] * r[2];
      float rs_env = a;
      // slot 4: SIA (layout 3) or the legacy Senv (layout 1/2).
      //   SIA : q = sum_j g_n [3/2 a^2 - 1/2 s2]
      //         dq/dr_i = gnp*q*rhat + (gn/d)*3a*(S_i - a*rhat)   [q here = q_sia]
      //         dq/dS_i = 3a*rhat - S_i      (the -S_i part is longitudinal)
      //   Senv: q = sum_j g_n a,  dq/dS_i = C*gn*rhat  (no S dependence)
      const float q_sia = 1.5f * a * a - 0.5f * s2;
      const float sia_scale = 3.0f * a;

      for (int d = 0; d < 3; ++d) {
        float san_d = fp_san * (gnp * a * b * r[d] + gn * d12inv * (vs[d] - rs * r[d]));
        float sdm_d = fp_sdm * (gnp * p * r[d] + gn * d12inv * (cs[d] - rs_dm * r[d]));
        float env_d, t_env;
        if (paramb.spin_use_sia) {
          env_d = fp_env * (gnp * q_sia * r[d] +
                            gn * d12inv * (sia_scale * sk[d] - sia_scale * a * r[d]));
          t_env = fp_env * gn * (sia_scale * r[d] - sk[d]);
        } else {
          env_d = fp_env * SPIN_ENV_FACTOR *
                  (gnp * a * r[d] + gn * d12inv * (sk[d] - rs_env * r[d]));
          t_env = fp_env * SPIN_ENV_FACTOR * gn * r[d];
        }
        f12[d] += c_sex * r[d] + san_d + sdm_d + env_d;
        t_own[d] += fp_sex * gn * sj[d] + fp_san * gn * b * r[d] +
                    fp_sdm * gn * sjxr[d] + t_env;
        // A neighbour without a spin degree of freedom receives no torque.
        if (g_use_spin[t2]) {
          t_nb[d] += fp_sex * gn * sk[d] + fp_san * gn * a * r[d] + fp_sdm * gn * rxsk[d];
        }
      }
    }

    atomicAdd(&g_fx[n1], (double)f12[0]);
    atomicAdd(&g_fy[n1], (double)f12[1]);
    atomicAdd(&g_fz[n1], (double)f12[2]);
    atomicAdd(&g_fx[n2], -(double)f12[0]);
    atomicAdd(&g_fy[n2], -(double)f12[1]);
    atomicAdd(&g_fz[n2], -(double)f12[2]);

    atomicAdd(&g_tx[n1], -t_own[0]);
    atomicAdd(&g_ty[n1], -t_own[1]);
    atomicAdd(&g_tz[n1], -t_own[2]);
    atomicAdd(&g_tx[n2], -t_nb[0]);
    atomicAdd(&g_ty[n2], -t_nb[1]);
    atomicAdd(&g_tz[n2], -t_nb[2]);

    // ---- per-atom virial, Route B (round 9: ownership + components fixed) --
    // The force this kernel produces for the pair is f12 = dU_{n1}/dr_{n2}:
    // it is added to atom n1 and subtracted from atom n2.  The GPUMD per-atom
    // virial convention (force/potential.cu, find_properties_many_body) is
    //     W_i = sum_{j != i} (r_j - r_i) (x) dU_j/dr_i ,
    // so the pair term generated here,
    //     (r_{n1} - r_{n2}) (x) dU_{n1}/dr_{n2} = -r12 (x) f12 ,
    // belongs to W_{n2}, NOT to W_{n1}.  It used to be written into
    // g_virial[n1] with a wrong component map (syz -> slot 4 = xz,
    // szx -> slot 5 = yz) and with three of the nine components missing
    // (xz, yx, zy were never produced).  The sum over all atoms was
    // unaffected by the ownership error, which is why the pressure and the
    // stress tensor looked right, but the per-atom virial -- and therefore
    // the phonon heat current J_p = sum_i W_i . v_i -- was wrong.
    {
      const double cx = -(double)x12, cy = -(double)y12, cz = -(double)z12;
      const double fx = (double)f12[0], fy = (double)f12[1], fz = (double)f12[2];
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 0], cx * fx); // xx
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 1], cy * fy); // yy
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 2], cz * fz); // zz
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 3], cx * fy); // xy
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 4], cx * fz); // xz
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 5], cy * fz); // yz
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 6], cy * fx); // yx
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 7], cz * fx); // zx
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 8], cz * fy); // zy
    }

    // ---- magnetic (magnon) conductive heat-current tensor, round 9 ---------
    //     A_j = sum_{i != j} (r_i - r_j) (x) dU_i/dS_j ,
    //     J_m = sum_j A_j : dS_j/dt .
    // t_nb is exactly dU_{n1}/dS_{n2}: the kernel adds -t_nb to the torque of
    // atom n2 and the torque is -dU/dS.  So this pair contributes
    // -r12 (x) t_nb to A_{n2}.  This is the term REPORT_validation8 declared
    // unreachable ("the NEP only gives dU/dS_i"); the neighbour derivative has
    // in fact been computed here all along and was simply thrown away after
    // being added into the total torque.
    if (g_Aspin != nullptr && g_use_spin[t2]) {
      const double cx = -(double)x12, cy = -(double)y12, cz = -(double)z12;
      const double tx = (double)t_nb[0], ty = (double)t_nb[1], tz = (double)t_nb[2];
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 0], cx * tx); // A_xx
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 1], cy * ty); // A_yy
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 2], cz * tz); // A_zz
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 3], cx * ty); // A_xy
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 4], cx * tz); // A_xz
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 5], cy * tz); // A_yz
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 6], cy * tx); // A_yx
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 7], cz * tx); // A_zx
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 8], cz * ty); // A_zy
    }
  }

  // Smag (on-site |S| channel): d q_smag[n] / dS = (n+1) u^n * scale * S/|S|.
  // No force and no virial contribution because q_smag does not depend on R.
  {
    const float smag = sqrt(sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2]);
    // S06-R2: the torque is the exact derivative of the energy the descriptor
    // defines, at every |S| > 0.  q_n = (scale |S|)^(n+1) gives
    //     dq_n/dS = (n+1) u^n scale * S/|S| ,
    // so the chain rule is  t = sum_n fp_n (n+1) u^n scale * S/|S|.  For
    // |S| >= SPIN_MAG_MIN_MOMENT the product is formed exactly as before
    // (c = .../|S|, then c*S), which keeps it bit-identical to the previous
    // build; below that the unit direction is formed by a scale-invariant
    // normalisation, because 1/|S| there is either inaccurate or not
    // representable while the direction is perfectly well defined.  Only at
    // |S| == 0 does the derivative genuinely not exist: the energy is |S|-like
    // there (a cone), its gradient has no limit, and the code writes 0.  That
    // zero is a convention at a non-differentiable point, not a derivative.
    {
      float dir[3] = {sk[0], sk[1], sk[2]};
      float denom = smag;
      if (smag < SPIN_MAG_MIN_MOMENT) {
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
      const float u_mag = smag * paramb.spin_mag_scale;
      float un_mag = 1.0f;
      float t_mag[3] = {0.0f, 0.0f, 0.0f};
      for (int n = 0; n <= n_max; ++n) {
        const float fp_mag = (paramb.spin_block_mask & SPIN_BLOCK_BIT_SMAG)
          ? g_Fp[static_cast<size_t>(N) * (base + 4 * nn1 + n) + n1] : 0.0f;
        const float c = fp_mag * (n + 1) * un_mag * paramb.spin_mag_scale / denom;
        t_mag[0] += c * dir[0];
        t_mag[1] += c * dir[1];
        t_mag[2] += c * dir[2];
        un_mag *= u_mag;
      }
      atomicAdd(&g_tx[n1], -t_mag[0]);
      atomicAdd(&g_ty[n1], -t_mag[1]);
      atomicAdd(&g_tz[n1], -t_mag[2]);
    }
  }

}

// Small-box variant: takes the precomputed image displacement vectors, see the
// comment on find_descriptor_spin_small_box.
template <typename ParaMB, typename ANN>
__global__ void find_force_spin_small_box(
  ParaMB paramb,
  ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const int* __restrict__ g_type,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sx,
  const float* __restrict__ g_sy,
  const float* __restrict__ g_sz,
  const int* __restrict__ g_use_spin,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_tx,
  double* g_ty,
  double* g_tz,
  double* g_Aspin = nullptr)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) return;

  const int t1 = g_type[n1];
  if (!g_use_spin[t1]) return;

  const int n_max = paramb.n_max_angular;
  const int nn1 = n_max + 1;
  const int base = (paramb.n_max_radial + 1) + paramb.dim_angular;
  float sk[3] = {g_sx[n1], g_sy[n1], g_sz[n1]};
  const float s2 = sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2];  // |S_i|^2

  for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
    const int index = i1 * N + n1;
    int n2 = g_NL_angular[index];
    float x12 = g_x12[index];
    float y12 = g_y12[index];
    float z12 = g_z12[index];
    float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
    if (d12 < 1.0e-8f) continue;
    float d12inv = 1.0f / d12;
    float r[3] = {x12 * d12inv, y12 * d12inv, z12 * d12inv};
    int t2 = g_type[n2];
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
        int c_index = spin_c_index(paramb, t1, t2, n, k);
        float c_value = annmb.c_type_pair[c_index];
        g += fn12[k] * c_value;
        gp += fnp12[k] * c_value;
      }
      gn12[n] = g;
      gnp12[n] = gp;
    }

    float sj[3] = {g_sx[n2], g_sy[n2], g_sz[n2]};
    if (!g_use_spin[t2]) {
      sj[0] = sj[1] = sj[2] = 0.0f;
    }
    float sksj = sk[0] * sj[0] + sk[1] * sj[1] + sk[2] * sj[2];
    float a = sk[0] * r[0] + sk[1] * r[1] + sk[2] * r[2];
    float b = sj[0] * r[0] + sj[1] * r[1] + sj[2] * r[2];
    float cs[3] = {
      sk[1] * sj[2] - sk[2] * sj[1],
      sk[2] * sj[0] - sk[0] * sj[2],
      sk[0] * sj[1] - sk[1] * sj[0]};
    float p = r[0] * cs[0] + r[1] * cs[1] + r[2] * cs[2];
    float sjxr[3] = {
      sj[1] * r[2] - sj[2] * r[1],
      sj[2] * r[0] - sj[0] * r[2],
      sj[0] * r[1] - sj[1] * r[0]};
    float rxsk[3] = {
      r[1] * sk[2] - r[2] * sk[1],
      r[2] * sk[0] - r[0] * sk[2],
      r[0] * sk[1] - r[1] * sk[0]};

    float f12[3] = {0.0f};
    float t_own[3] = {0.0f};
    float t_nb[3] = {0.0f};

    for (int n = 0; n <= n_max; ++n) {
      const int blk = paramb.spin_block_mask;
      float fp_sex = (blk & SPIN_BLOCK_BIT_SEX)
        ? g_Fp[static_cast<size_t>(N) * (base + n) + n1] : 0.0f;
      float fp_san = (blk & SPIN_BLOCK_BIT_SAN)
        ? g_Fp[static_cast<size_t>(N) * (base + nn1 + n) + n1] : 0.0f;
      float fp_sdm = (blk & SPIN_BLOCK_BIT_SDM)
        ? g_Fp[static_cast<size_t>(N) * (base + 2 * nn1 + n) + n1] : 0.0f;
      float fp_env = (blk & SPIN_BLOCK_BIT_SLOT4)
        ? g_Fp[static_cast<size_t>(N) * (base + 3 * nn1 + n) + n1] : 0.0f;
      float gn = gn12[n];
      float gnp = gnp12[n];
      float c_sex = fp_sex * gnp * sksj;

      float vs[3] = {b * sk[0] + a * sj[0], b * sk[1] + a * sj[1], b * sk[2] + a * sj[2]};
      float rs = vs[0] * r[0] + vs[1] * r[1] + vs[2] * r[2];
      float rs_dm = cs[0] * r[0] + cs[1] * r[1] + cs[2] * r[2];
      float rs_env = a;
      // slot 4: SIA (layout 3) or the legacy Senv (layout 1/2).
      //   SIA : q = sum_j g_n [3/2 a^2 - 1/2 s2]
      //         dq/dr_i = gnp*q*rhat + (gn/d)*3a*(S_i - a*rhat)   [q here = q_sia]
      //         dq/dS_i = 3a*rhat - S_i      (the -S_i part is longitudinal)
      //   Senv: q = sum_j g_n a,  dq/dS_i = C*gn*rhat  (no S dependence)
      const float q_sia = 1.5f * a * a - 0.5f * s2;
      const float sia_scale = 3.0f * a;

      for (int d = 0; d < 3; ++d) {
        float san_d = fp_san * (gnp * a * b * r[d] + gn * d12inv * (vs[d] - rs * r[d]));
        float sdm_d = fp_sdm * (gnp * p * r[d] + gn * d12inv * (cs[d] - rs_dm * r[d]));
        float env_d, t_env;
        if (paramb.spin_use_sia) {
          env_d = fp_env * (gnp * q_sia * r[d] +
                            gn * d12inv * (sia_scale * sk[d] - sia_scale * a * r[d]));
          t_env = fp_env * gn * (sia_scale * r[d] - sk[d]);
        } else {
          env_d = fp_env * SPIN_ENV_FACTOR *
                  (gnp * a * r[d] + gn * d12inv * (sk[d] - rs_env * r[d]));
          t_env = fp_env * SPIN_ENV_FACTOR * gn * r[d];
        }
        f12[d] += c_sex * r[d] + san_d + sdm_d + env_d;
        t_own[d] += fp_sex * gn * sj[d] + fp_san * gn * b * r[d] +
                    fp_sdm * gn * sjxr[d] + t_env;
        // A neighbour without a spin degree of freedom receives no torque.
        if (g_use_spin[t2]) {
          t_nb[d] += fp_sex * gn * sk[d] + fp_san * gn * a * r[d] + fp_sdm * gn * rxsk[d];
        }
      }
    }

    atomicAdd(&g_fx[n1], (double)f12[0]);
    atomicAdd(&g_fy[n1], (double)f12[1]);
    atomicAdd(&g_fz[n1], (double)f12[2]);
    atomicAdd(&g_fx[n2], -(double)f12[0]);
    atomicAdd(&g_fy[n2], -(double)f12[1]);
    atomicAdd(&g_fz[n2], -(double)f12[2]);

    atomicAdd(&g_tx[n1], -t_own[0]);
    atomicAdd(&g_ty[n1], -t_own[1]);
    atomicAdd(&g_tz[n1], -t_own[2]);
    atomicAdd(&g_tx[n2], -t_nb[0]);
    atomicAdd(&g_ty[n2], -t_nb[1]);
    atomicAdd(&g_tz[n2], -t_nb[2]);

    // ---- per-atom virial, Route B (round 9: ownership + components fixed) --
    // The force this kernel produces for the pair is f12 = dU_{n1}/dr_{n2}:
    // it is added to atom n1 and subtracted from atom n2.  The GPUMD per-atom
    // virial convention (force/potential.cu, find_properties_many_body) is
    //     W_i = sum_{j != i} (r_j - r_i) (x) dU_j/dr_i ,
    // so the pair term generated here,
    //     (r_{n1} - r_{n2}) (x) dU_{n1}/dr_{n2} = -r12 (x) f12 ,
    // belongs to W_{n2}, NOT to W_{n1}.  It used to be written into
    // g_virial[n1] with a wrong component map (syz -> slot 4 = xz,
    // szx -> slot 5 = yz) and with three of the nine components missing
    // (xz, yx, zy were never produced).  The sum over all atoms was
    // unaffected by the ownership error, which is why the pressure and the
    // stress tensor looked right, but the per-atom virial -- and therefore
    // the phonon heat current J_p = sum_i W_i . v_i -- was wrong.
    {
      const double cx = -(double)x12, cy = -(double)y12, cz = -(double)z12;
      const double fx = (double)f12[0], fy = (double)f12[1], fz = (double)f12[2];
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 0], cx * fx); // xx
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 1], cy * fy); // yy
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 2], cz * fz); // zz
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 3], cx * fy); // xy
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 4], cx * fz); // xz
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 5], cy * fz); // yz
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 6], cy * fx); // yx
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 7], cz * fx); // zx
      atomicAdd(&g_virial[n2 + static_cast<size_t>(N) * 8], cz * fy); // zy
    }

    // ---- magnetic (magnon) conductive heat-current tensor, round 9 ---------
    //     A_j = sum_{i != j} (r_i - r_j) (x) dU_i/dS_j ,
    //     J_m = sum_j A_j : dS_j/dt .
    // t_nb is exactly dU_{n1}/dS_{n2}: the kernel adds -t_nb to the torque of
    // atom n2 and the torque is -dU/dS.  So this pair contributes
    // -r12 (x) t_nb to A_{n2}.  This is the term REPORT_validation8 declared
    // unreachable ("the NEP only gives dU/dS_i"); the neighbour derivative has
    // in fact been computed here all along and was simply thrown away after
    // being added into the total torque.
    if (g_Aspin != nullptr && g_use_spin[t2]) {
      const double cx = -(double)x12, cy = -(double)y12, cz = -(double)z12;
      const double tx = (double)t_nb[0], ty = (double)t_nb[1], tz = (double)t_nb[2];
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 0], cx * tx); // A_xx
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 1], cy * ty); // A_yy
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 2], cz * tz); // A_zz
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 3], cx * ty); // A_xy
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 4], cx * tz); // A_xz
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 5], cy * tz); // A_yz
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 6], cy * tx); // A_yx
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 7], cz * tx); // A_zx
      atomicAdd(&g_Aspin[n2 + static_cast<size_t>(N) * 8], cz * ty); // A_zy
    }
  }

  // Smag (on-site |S| channel): d q_smag[n] / dS = (n+1) u^n * scale * S/|S|.
  // No force and no virial contribution because q_smag does not depend on R.
  {
    const float smag = sqrt(sk[0] * sk[0] + sk[1] * sk[1] + sk[2] * sk[2]);
    // S06-R2: the torque is the exact derivative of the energy the descriptor
    // defines, at every |S| > 0.  q_n = (scale |S|)^(n+1) gives
    //     dq_n/dS = (n+1) u^n scale * S/|S| ,
    // so the chain rule is  t = sum_n fp_n (n+1) u^n scale * S/|S|.  For
    // |S| >= SPIN_MAG_MIN_MOMENT the product is formed exactly as before
    // (c = .../|S|, then c*S), which keeps it bit-identical to the previous
    // build; below that the unit direction is formed by a scale-invariant
    // normalisation, because 1/|S| there is either inaccurate or not
    // representable while the direction is perfectly well defined.  Only at
    // |S| == 0 does the derivative genuinely not exist: the energy is |S|-like
    // there (a cone), its gradient has no limit, and the code writes 0.  That
    // zero is a convention at a non-differentiable point, not a derivative.
    {
      float dir[3] = {sk[0], sk[1], sk[2]};
      float denom = smag;
      if (smag < SPIN_MAG_MIN_MOMENT) {
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
      const float u_mag = smag * paramb.spin_mag_scale;
      float un_mag = 1.0f;
      float t_mag[3] = {0.0f, 0.0f, 0.0f};
      for (int n = 0; n <= n_max; ++n) {
        const float fp_mag = (paramb.spin_block_mask & SPIN_BLOCK_BIT_SMAG)
          ? g_Fp[static_cast<size_t>(N) * (base + 4 * nn1 + n) + n1] : 0.0f;
        const float c = fp_mag * (n + 1) * un_mag * paramb.spin_mag_scale / denom;
        t_mag[0] += c * dir[0];
        t_mag[1] += c * dir[1];
        t_mag[2] += c * dir[2];
        un_mag *= u_mag;
      }
      atomicAdd(&g_tx[n1], -t_mag[0]);
      atomicAdd(&g_ty[n1], -t_mag[1]);
      atomicAdd(&g_tz[n1], -t_mag[2]);
    }
  }

}

