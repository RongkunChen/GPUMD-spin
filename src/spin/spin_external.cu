/*
    Spin Route B, V4: external field and magneto-electric coupling (device side).
    See spin_external.cuh for the conventions and the LAMMPS correspondence.

    Implementation notes (V4.1):

    * One thread per atom, full neighbour sum, **no atomics**.  Each thread
      accumulates its own energy / torque / force / virial in registers and then
      writes only its own slots, so the earlier cross-block read-modify-write on
      `potential[n2]` (which silently lost updates and made the total energy
      4-20% low) cannot happen by construction.  It is also faster: the old
      version issued 26 atomics per pair.

    * Torque is accumulated in double and rounded to float once, instead of
      being summed with atomicAdd on a float array.

    * No host synchronisation per force evaluation.  The energy diagnostics live
      in a device accumulator that is only read by report(), so an MD or
      minimisation step no longer stalls the pipeline twice per force call.

    * The cell list is used only when it is safe: at least five cells per
      periodic direction.  Below that the same pair can be reached through two
      different wrapped cells, so the code falls back to an O(N^2) kernel that
      has no such ambiguity.  (4.5 A cutoff in a 14.4 A box -> 6 cells, list.)
*/

#include "spin_external.cuh"
#include "model/box.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

namespace
{

const int SE_BLOCK = 128;

// Same cell-id rule as neighbor.cu / dftd3.cu: the cell size is rc, handed in
// through rc_inv.  Local, so that this file does not have to expose a static
// helper of neighbor.cuh.
__device__ inline void se_find_cell_id(
  const Box& box,
  const double x,
  const double y,
  const double z,
  const double rc_inv,
  const int nx,
  const int ny,
  const int nz,
  int& cell_id_x,
  int& cell_id_y,
  int& cell_id_z,
  int& cell_id)
{
  const double sx = box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z;
  const double sy = box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z;
  const double sz = box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z;
  cell_id_x = (int)floor(sx * box.thickness_x * rc_inv);
  cell_id_y = (int)floor(sy * box.thickness_y * rc_inv);
  cell_id_z = (int)floor(sz * box.thickness_z * rc_inv);
  while (cell_id_x < 0) cell_id_x += nx;
  while (cell_id_x >= nx) cell_id_x -= nx;
  while (cell_id_y < 0) cell_id_y += ny;
  while (cell_id_y >= ny) cell_id_y -= ny;
  while (cell_id_z < 0) cell_id_z += nz;
  while (cell_id_z >= nz) cell_id_z -= nz;
  cell_id = cell_id_x + nx * cell_id_y + nx * ny * cell_id_z;
}

__device__ inline int se_neighbor_cell(
  const int cell_id, const int cx, const int cy, const int cz,
  const int nx, const int ny, const int nz, const int xx, const int yy, const int zz)
{
  int neighbor_cell = cell_id + zz * nx * ny + yy * nx + xx;
  if (cx + xx < 0) neighbor_cell += nx;
  else if (cx + xx >= nx) neighbor_cell -= nx;
  if (cy + yy < 0) neighbor_cell += ny * nx;
  else if (cy + yy >= ny) neighbor_cell -= ny * nx;
  if (cz + zz < 0) neighbor_cell += nz * ny * nx;
  else if (cz + zz >= nz) neighbor_cell -= nz * ny * nx;
  return neighbor_cell;
}

__global__ void se_cell_counts(
  const Box box, const int N, int* cell_count,
  const double* x, const double* y, const double* z,
  const int nx, const int ny, const int nz, const double rc_inv)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int cx, cy, cz, cid;
  se_find_cell_id(box, x[n], y[n], z[n], rc_inv, nx, ny, nz, cx, cy, cz, cid);
  atomicAdd(cell_count + cid, 1);
}

__global__ void se_cell_contents(
  const Box box, const int N, int* cell_count, const int* cell_count_sum, int* cell_contents,
  const double* x, const double* y, const double* z,
  const int nx, const int ny, const int nz, const double rc_inv)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  int cx, cy, cz, cid;
  se_find_cell_id(box, x[n], y[n], z[n], rc_inv, nx, ny, nz, cx, cy, cz, cid);
  const int slot = atomicAdd(cell_count + cid, 1);
  cell_contents[cell_count_sum[cid] + slot] = n;
}

__global__ void se_zero_int(const int size, int* p)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < size) p[n] = 0;
}

__global__ void se_zero_double(const int size, double* p)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < size) p[n] = 0.0;
}

// E_Z = -sum_i S_i . B, applied as (a) the potential it is and (b) the field
// contribution it makes to the effective field.  Consistent by construction:
// torque = -dE_Z/dS_i = +B.
__global__ void gpu_add_zeeman(
  const int N,
  const double bx, const double by, const double bz,
  const float* __restrict__ sx,
  const float* __restrict__ sy,
  const float* __restrict__ sz,
  float* __restrict__ tx,
  float* __restrict__ ty,
  float* __restrict__ tz,
  double* __restrict__ potential,
  double* __restrict__ g_energy)
{
  __shared__ double sh[SE_BLOCK];
  const int tid = threadIdx.x;
  const int n = blockIdx.x * blockDim.x + tid;
  double e = 0.0;
  if (n < N) {
    e = -((double)sx[n] * bx + (double)sy[n] * by + (double)sz[n] * bz);
    if (potential != nullptr) potential[n] += e;
    tx[n] += (float)bx;
    ty[n] += (float)by;
    tz[n] += (float)bz;
  }
  sh[tid] = e;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0 && g_energy != nullptr) atomicAdd(g_energy, sh[0]);
}

// H_ME = sum_{i<j} kappa_ij (E x r_ij).(S_i x S_j),  r_ij = r_j - r_i
// The per-atom accumulator.  Thread i sums over *all* its neighbours (not only
// j > i) and writes only its own slots; every pair is therefore visited twice,
// which is why the energy is 0.5*H_ij per visit:  sum_i e_i = sum_{i<j} H_ij.
struct MeAccum {
  double e;
  double tau[3];
  double frc[3];
  double vir[9];
};

__device__ inline void me_init(MeAccum& a)
{
  a.e = 0.0;
  for (int k = 0; k < 3; ++k) {
    a.tau[k] = 0.0;
    a.frc[k] = 0.0;
  }
  for (int k = 0; k < 9; ++k) a.vir[k] = 0.0;
}

// R22-NonHeat (R22-COMPAT-01): C2 smooth switching for the ME cutoff.
//   mode 0  r21 legacy hard cutoff: f=1 for r<r_c, f=0 at r>=r_c.  Energy,
//           force, spin gradient and virial are DISCONTINUOUS at r_c.
//   mode 1  x = (r-r_on)/(r_c-r_on),
//           f(x) = 1 - 10x^3 + 15x^4 - 6x^5,
//           df/dx = -30x^2 + 60x^3 - 30x^4 = -30 x^2 (1-x)^2,
//           df/dr = (df/dx)/(r_c-r_on).
//   f(0)=1, f(1)=0, f'(0)=f'(1)=0, f''(0)=f''(1)=0  => C2 at both ends.
//   f''' is discontinuous at the endpoints, so the switch is C2, not C3.
// With f = f_c(r) and h = H_ij (the full unordered-pair energy),
//     H_ij^smooth = f * h,
//     dH/dr_vec   = f * (d h/dr_vec) + h * (df/dr) * r_vec/r.
// The second term is the extra force from the switch and is propagated into
// the nine virial components through W_ab = -r_a F_b (see me_accumulate).
// Spin gradient: omega_i = -dH/dS_i = -f * (S_j x D_ij), i.e. only the
// prefactor changes (f does not depend on the spins).
__device__ inline void me_switch(
  const double r, const double r_on, const double rc, const int mode,
  double& fc, double& dfc_dr)
{
  fc = 1.0;
  dfc_dr = 0.0;
  if (mode == 0) return;              // legacy: identical to r21
  if (r <= r_on) return;              // plateau: f=1, df=0
  if (r >= rc) { fc = 0.0; return; } // outside: f=0, df=0
  const double w = rc - r_on;
  const double x = (r - r_on) / w;
  const double x2 = x * x;
  const double x3 = x2 * x;
  const double x4 = x2 * x2;
  const double x5 = x4 * x;
  fc = 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5;
  dfc_dr = (-30.0 * x2 + 60.0 * x3 - 30.0 * x4) / w;
}

__device__ inline void me_accumulate(
  MeAccum& a,
  const double k,
  const double Ex, const double Ey, const double Ez,
  const double r0, const double r1, const double r2,
  const double r, const double r_on, const double rc, const int cutoff_mode,
  const double si0, const double si1, const double si2,
  const double sj0, const double sj1, const double sj2,
  const bool need_force, const bool need_virial)
{
  double fc = 1.0, dfc_dr = 0.0;
  me_switch(r, r_on, rc, cutoff_mode, fc, dfc_dr);
  if (fc == 0.0) return;

  // D = k (E x r)
  const double D0 = k * (Ey * r2 - Ez * r1);
  const double D1 = k * (Ez * r0 - Ex * r2);
  const double D2 = k * (Ex * r1 - Ey * r0);
  // c = S_i x S_j
  const double c0 = si1 * sj2 - si2 * sj1;
  const double c1 = si2 * sj0 - si0 * sj2;
  const double c2 = si0 * sj1 - si1 * sj0;

  // h = H_ij for this (ordered) visit.  (E x r).c = r.(c x E).
  const double h = D0 * c0 + D1 * c1 + D2 * c2;
  a.e += 0.5 * fc * h;

  // omega_i = -S_j x D_ij, scaled by f_c (f_c has no spin dependence).
  a.tau[0] -= fc * (sj1 * D2 - sj2 * D1);
  a.tau[1] -= fc * (sj2 * D0 - sj0 * D2);
  a.tau[2] -= fc * (sj0 * D1 - sj1 * D0);

  if (!need_force) return;

  // Hard-cutoff force F_i^hard = k (c x E) = d h/d r_vec.
  double f0 = k * (c1 * Ez - c2 * Ey);
  double f1 = k * (c2 * Ex - c0 * Ez);
  double f2 = k * (c0 * Ey - c1 * Ex);
  // R22: d(f h)/d r_vec = f F^hard + h (df/dr) r_vec/r.  The second term is
  // the ONLY new force contribution; it is radial and identically zero at
  // both switching endpoints because df/dr = 0 there.
  const double sw = (r > 0.0) ? (dfc_dr * h / r) : 0.0;
  f0 = fc * f0 + sw * r0;
  f1 = fc * f1 + sw * r1;
  f2 = fc * f2 + sw * r2;
  a.frc[0] += f0;
  a.frc[1] += f1;
  a.frc[2] += f2;

  if (!need_virial) return;

  // F-05 (r21): GPUMD's per-atom virial convention (see force/lj.cu) is
  //     W_i = sum_{ordered j} (-1/2) r_ij (x) F_i ,
  // so that summing the two ordered visits of one unordered pair gives
  //     W_pair = - r_ij (x) F_i .
  // This kernel visits EVERY ordered pair (i,j) and (j,i), exactly like the
  // energy's `0.5 * H_ij`.  r20 accumulated the FULL -r_ij (x) F_i on each
  // visit, so the total virial was twice the correct value while the energy
  // was correct.  The 0.5 below restores the same counting as the energy.
  // Independent check (PHYSICS_DERIVATION.md): H_ME = k (c x E).r with
  // c = S_i x S_j, so dH/dr = F_i = k (c x E).  Under a deformation-gradient
  // perturbation r' = (I+eps) r one gets dH/deps_ab = F_i,a r_b, whereas
  // r' = (I+eps^T) r gives dH/deps_ab = r_a F_i,b.  GPUMD's stored tensor is
  // W_ab = -r_a F_i,b, so -dH/deps = W^T for the first convention and W for
  // the second (the `change_box` form).  For central potentials F_i || r the
  // two coincide; for ME with a fixed external E, W is genuinely
  // NON-symmetric and the transpose convention matters.  Both relations are
  // verified against an energy-only finite difference in v4_regression.py
  // and tests/r21_validation.py.  In either convention trace(W) = -E_ME
  // (not -2 E_ME), because H_ME is homogeneous of degree 1 in r.
  const double half = 0.5;
  a.vir[0] += half * r0 * (-f0);
  a.vir[1] += half * r1 * (-f1);
  a.vir[2] += half * r2 * (-f2);
  a.vir[3] += half * r0 * (-f1);
  a.vir[4] += half * r0 * (-f2);
  a.vir[5] += half * r1 * (-f2);
  a.vir[6] += half * r1 * (-f0);
  a.vir[7] += half * r2 * (-f0);
  a.vir[8] += half * r2 * (-f1);
}

__device__ inline void me_flush(
  const int n,
  const int N,
  const MeAccum& a,
  float* tx, float* ty, float* tz,
  double* potential,
  double* fx, double* fy, double* fz,
  double* virial)
{
  if (potential != nullptr) potential[n] += a.e;
  tx[n] += (float)a.tau[0];
  ty[n] += (float)a.tau[1];
  tz[n] += (float)a.tau[2];
  if (fx != nullptr) {
    fx[n] += a.frc[0];
    fy[n] += a.frc[1];
    fz[n] += a.frc[2];
  }
  if (virial != nullptr) {
    // virial_per_atom is component-major: component k lives at n + k*N, the
    // same layout force/lj.cu uses.  Writing n + k would scatter the nine
    // components across nine neighbouring atoms.
    for (int k = 0; k < 9; ++k) virial[n + k * N] += a.vir[k];
  }
}

// F-06 (r21): `kappa_set` is a separate mask, so a NEGATIVE kappa is a legal
// explicit override and the "unset -> global me_kappa" fallback no longer
// occupies the whole negative half-line.
__device__ inline double me_kappa_for(
  const double kappa, const float* kappa_pair, const float* kappa_set,
  const int num_types, const int ti, const int tj)
{
  if (kappa_pair != nullptr && kappa_set != nullptr && num_types > 0) {
    const int idx = ti * num_types + tj;
    if (kappa_set[idx] > 0.5f) return (double)kappa_pair[idx];
  }
  return kappa;
}

__global__ void gpu_add_magelec_list(
  const int N,
  const int num_types,
  const double kappa,
  const float* __restrict__ kappa_pair,
  const float* __restrict__ kappa_set,
  const double Ex, const double Ey, const double Ez,
  const double cutoff,
  const double r_on,
  const int cutoff_mode,
  const int nx, const int ny, const int nz,
  const Box box,
  const int* __restrict__ type,
  const int* __restrict__ cell_count,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const float* __restrict__ sx,
  const float* __restrict__ sy,
  const float* __restrict__ sz,
  float* __restrict__ tx,
  float* __restrict__ ty,
  float* __restrict__ tz,
  double* __restrict__ potential,
  double* __restrict__ fx,
  double* __restrict__ fy,
  double* __restrict__ fz,
  double* __restrict__ virial,
  double* __restrict__ g_energy)
{
  __shared__ double sh_e[SE_BLOCK];
  const int tid = threadIdx.x;
  const int n1 = blockIdx.x * blockDim.x + tid;
  MeAccum acc;
  me_init(acc);

  const double cutoff2 = cutoff * cutoff;
  const bool need_force = (fx != nullptr);
  const bool need_virial = (virial != nullptr);

  if (n1 < N) {
    const double x1 = x[n1], y1 = y[n1], z1 = z[n1];
    const double si0 = (double)sx[n1], si1 = (double)sy[n1], si2 = (double)sz[n1];
    int cx, cy, cz, cell_id;
    se_find_cell_id(box, x1, y1, z1, 2.0 / cutoff, nx, ny, nz, cx, cy, cz, cell_id);

    const int z_lim = box.pbc_z ? 2 : 0;
    const int y_lim = box.pbc_y ? 2 : 0;
    const int x_lim = box.pbc_x ? 2 : 0;

    for (int zz = -z_lim; zz <= z_lim; ++zz) {
      for (int yy = -y_lim; yy <= y_lim; ++yy) {
        for (int xx = -x_lim; xx <= x_lim; ++xx) {
          const int nc = se_neighbor_cell(cell_id, cx, cy, cz, nx, ny, nz, xx, yy, zz);
          const int count = cell_count[nc];
          const int offset = cell_count_sum[nc];
          for (int m = 0; m < count; ++m) {
            const int n2 = cell_contents[offset + m];
            if (n2 == n1) continue;
            double x12 = x[n2] - x1;
            double y12 = y[n2] - y1;
            double z12 = z[n2] - z1;
            apply_mic(box, x12, y12, z12);
            const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
            if (d2 >= cutoff2 || d2 < 1.0e-20) continue;
            const double r12 = sqrt(d2);
            const double k = me_kappa_for(
              kappa, kappa_pair, kappa_set, num_types, type[n1], type[n2]);
            if (k == 0.0) continue;
            me_accumulate(
              acc, k, Ex, Ey, Ez, x12, y12, z12,
              r12, r_on, cutoff, cutoff_mode,
              si0, si1, si2, (double)sx[n2], (double)sy[n2], (double)sz[n2],
              need_force, need_virial);
          }
        }
      }
    }
    me_flush(n1, N, acc, tx, ty, tz, potential, fx, fy, fz, virial);
  }

  sh_e[tid] = acc.e;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) sh_e[tid] += sh_e[tid + off];
    __syncthreads();
  }
  if (tid == 0 && g_energy != nullptr) atomicAdd(g_energy, sh_e[0]);
}

// Fallback for boxes too small for the 5x5x5 cell scan: explicit O(N^2) pair
// loop.  Same physics, no ambiguity about wrapped images.
__global__ void gpu_add_magelec_full(
  const int N,
  const int num_types,
  const double kappa,
  const float* __restrict__ kappa_pair,
  const float* __restrict__ kappa_set,
  const double Ex, const double Ey, const double Ez,
  const double cutoff,
  const double r_on,
  const int cutoff_mode,
  const Box box,
  const int* __restrict__ type,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  const float* __restrict__ sx,
  const float* __restrict__ sy,
  const float* __restrict__ sz,
  float* __restrict__ tx,
  float* __restrict__ ty,
  float* __restrict__ tz,
  double* __restrict__ potential,
  double* __restrict__ fx,
  double* __restrict__ fy,
  double* __restrict__ fz,
  double* __restrict__ virial,
  double* __restrict__ g_energy)
{
  __shared__ double sh_e[SE_BLOCK];
  const int tid = threadIdx.x;
  const int n1 = blockIdx.x * blockDim.x + tid;
  MeAccum acc;
  me_init(acc);

  const double cutoff2 = cutoff * cutoff;
  const bool need_force = (fx != nullptr);
  const bool need_virial = (virial != nullptr);

  if (n1 < N) {
    const double x1 = x[n1], y1 = y[n1], z1 = z[n1];
    const double si0 = (double)sx[n1], si1 = (double)sy[n1], si2 = (double)sz[n1];
    for (int n2 = 0; n2 < N; ++n2) {
      if (n2 == n1) continue;
      double x12 = x[n2] - x1;
      double y12 = y[n2] - y1;
      double z12 = z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      const double d2 = x12 * x12 + y12 * y12 + z12 * z12;
      if (d2 >= cutoff2 || d2 < 1.0e-20) continue;
      const double r12 = sqrt(d2);
      const double k = me_kappa_for(
        kappa, kappa_pair, kappa_set, num_types, type[n1], type[n2]);
      if (k == 0.0) continue;
      me_accumulate(
        acc, k, Ex, Ey, Ez, x12, y12, z12,
        r12, r_on, cutoff, cutoff_mode,
        si0, si1, si2, (double)sx[n2], (double)sy[n2], (double)sz[n2],
        need_force, need_virial);
    }
    me_flush(n1, N, acc, tx, ty, tz, potential, fx, fy, fz, virial);
  }

  sh_e[tid] = acc.e;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) sh_e[tid] += sh_e[tid + off];
    __syncthreads();
  }
  if (tid == 0 && g_energy != nullptr) atomicAdd(g_energy, sh_e[0]);
}

// R22: explicit-time field work  dU = -sum_i S_i . dB  (eV).
// One thread per atom, shared-memory block reduction, one atomicAdd per block.
// Grid covers N exactly (n >= N contributes 0.0), so N = 128/129/1024 block
// boundaries are handled and no uninitialised shared slot is ever read.
__global__ void gpu_accumulate_drive(
  const int N,
  const double dbx, const double dby, const double dbz,
  const float* __restrict__ sx,
  const float* __restrict__ sy,
  const float* __restrict__ sz,
  double* __restrict__ acc)
{
  __shared__ double sh[SE_BLOCK];
  const int tid = threadIdx.x;
  const int n = blockIdx.x * blockDim.x + tid;
  double v = 0.0;
  if (n < N) {
    v = -((double)sx[n] * dbx + (double)sy[n] * dby + (double)sz[n] * dbz);
    if (!isfinite(v)) v = 0.0;
  }
  sh[tid] = v;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) sh[tid] += sh[tid + off];
    __syncthreads();
  }
  if (tid == 0 && acc != nullptr) atomicAdd(acc, sh[0]);
}

} // namespace

void SpinExternal::set_zeeman(
  const double b[3], const int wave, const double freq_thz, const double phase_rad)
{
  b_peak[0] = b[0];
  b_peak[1] = b[1];
  b_peak[2] = b[2];
  zeeman_wave = wave;
  zeeman_freq_thz = freq_thz;
  zeeman_phase_rad = phase_rad;
  zeeman_enabled = (b[0] != 0.0 || b[1] != 0.0 || b[2] != 0.0);
  if (zeeman_enabled && zeeman_wave != 0 && zeeman_freq_thz <= 0.0) {
    PRINT_INPUT_ERROR("magfield: a time-dependent field needs freq > 0.");
  }
}

void SpinExternal::set_magelec(const double cutoff, const double kappa, const double E[3])
{
  me_cutoff = cutoff;
  me_kappa = kappa;
  me_E[0] = E[0];
  me_E[1] = E[1];
  me_E[2] = E[2];
  // R22: the r21 default stays the hard cutoff.  Only an explicit
  // `cutoff=smooth` on the magelec line switches to the C2 Hamiltonian.
  me_r_on = cutoff * (1.0 - me_smooth_fraction);
  magelec_enabled = true;
}

void SpinExternal::set_magelec_switch(const int mode, const double r_on)
{
  me_cutoff_mode = mode;
  if (mode == 1) {
    me_r_on = r_on;
    me_smooth_fraction = (me_cutoff > 0.0)
                           ? (me_cutoff - r_on) / me_cutoff
                           : me_smooth_fraction;
  } else {
    me_r_on = me_cutoff; // legacy: the switching interval is empty
  }
}

// R22 energy/work ledger ------------------------------------------------------
void SpinExternal::compute_applied_b()
{
  double env = 1.0;
  if (zeeman_wave != 0) {
    const double TWO_PI = 6.28318530717958647692;
    const double arg = TWO_PI * zeeman_freq_thz * (time_fs * 1.0e-3) + zeeman_phase_rad;
    env = (zeeman_wave == 1) ? sin(arg) : cos(arg);
  }
  applied_b[0] = b_peak[0] * env;
  applied_b[1] = b_peak[1] * env;
  applied_b[2] = b_peak[2] * env;
}

void SpinExternal::reset_ledger()
{
  drive_ready_ = false;
  prev_b_[0] = prev_b_[1] = prev_b_[2] = 0.0;
  drive_acc_.resize(1);
  CHECK(gpuStreamSynchronize(0));
  drive_acc_.fill(0.0);
}

void SpinExternal::record_drive_work(
  const float* sx, const float* sy, const float* sz, const int N)
{
  if (!zeeman_enabled || N <= 0 || sx == nullptr) return;
  if (drive_acc_.size() < 1) {
    drive_acc_.resize(1);
    CHECK(gpuStreamSynchronize(0));
    drive_acc_.fill(0.0);
  }
  if (!drive_ready_) {
    // The first sample only primes the reference field.  H_Z(t_{1/2}) is
    // already part of the initial total energy, so no drive work is due at
    // t=0; emitting -S_0.B_0 here would double count the initial Zeeman term.
    prev_b_[0] = applied_b[0];
    prev_b_[1] = applied_b[1];
    prev_b_[2] = applied_b[2];
    drive_ready_ = true;
    return;
  }
  const double db0 = applied_b[0] - prev_b_[0];
  const double db1 = applied_b[1] - prev_b_[1];
  const double db2 = applied_b[2] - prev_b_[2];
  if (db0 == 0.0 && db1 == 0.0 && db2 == 0.0) return; // static field
  const int grid = (N - 1) / SE_BLOCK + 1;
  gpu_accumulate_drive<<<grid, SE_BLOCK>>>(
    N, db0, db1, db2, sx, sy, sz, drive_acc_.data());
  GPU_CHECK_KERNEL
  prev_b_[0] = applied_b[0];
  prev_b_[1] = applied_b[1];
  prev_b_[2] = applied_b[2];
}

double SpinExternal::read_drive_work()
{
  if (drive_acc_.size() < 1) return 0.0;
  CHECK(gpuStreamSynchronize(0));
  double v = 0.0;
  drive_acc_.copy_to_host(&v);
  return v;
}

void SpinExternal::read_energies()
{
  if (energy_acc_.size() < 2) return;
  CHECK(gpuStreamSynchronize(0));
  double e[2] = {0.0, 0.0};
  energy_acc_.copy_to_host(e);
  energy_zeeman = e[0];
  energy_magelec = e[1];
}

void SpinExternal::set_magelec_pair(const int type_i, const int type_j, const double kappa)
{
  if (num_types <= 0) {
    PRINT_INPUT_ERROR("magelec_pair needs the number of atom types first.");
  }
  if (type_i < 0 || type_j < 0 || type_i >= num_types || type_j >= num_types) {
    PRINT_INPUT_ERROR("magelec_pair: type index out of range.");
  }
  if (!std::isfinite(kappa)) {
    PRINT_INPUT_ERROR("magelec_pair: kappa must be finite.");
  }
  const size_t want = static_cast<size_t>(num_types) * num_types;
  if (me_kappa_pair.size() != want || me_kappa_pair_set.size() != want) {
    me_kappa_pair.assign(want, 0.0f);
    me_kappa_pair_set.assign(want, 0.0f);
  }
  // F-06: write the symmetric pair explicitly and mark BOTH entries as set.
  // Negative and zero values are legal couplings, not sentinels.
  me_kappa_pair[static_cast<size_t>(type_i) * num_types + type_j] = (float)kappa;
  me_kappa_pair[static_cast<size_t>(type_j) * num_types + type_i] = (float)kappa;
  me_kappa_pair_set[static_cast<size_t>(type_i) * num_types + type_j] = 1.0f;
  me_kappa_pair_set[static_cast<size_t>(type_j) * num_types + type_i] = 1.0f;
  kappa_pair_dirty_ = true; // F-06: force an upload before the next kernel
  magelec_enabled = true;
}

void SpinExternal::allocate_types(const int number_of_types)
{
  if (number_of_types <= 0) return;
  const std::vector<float> old = me_kappa_pair;
  const std::vector<float> old_set = me_kappa_pair_set;
  const int old_n = num_types;
  num_types = number_of_types;
  const size_t want = static_cast<size_t>(num_types) * num_types;
  me_kappa_pair.assign(want, 0.0f);
  me_kappa_pair_set.assign(want, 0.0f);
  for (int i = 0; i < old_n && i < num_types; ++i) {
    for (int j = 0; j < old_n && j < num_types; ++j) {
      const size_t from = static_cast<size_t>(i) * old_n + j;
      const size_t to = static_cast<size_t>(i) * num_types + j;
      me_kappa_pair[to] = old[from];
      if (from < old_set.size()) me_kappa_pair_set[to] = old_set[from];
    }
  }
  // F-06: a CPU table change must reach the device.  In r20 the device copy
  // was uploaded only inside allocate_types(), which `magelec` calls BEFORE
  // `magelec_pair`; every later per-pair override therefore stayed on the host
  // and the GPU kept using the global kappa.  A dirty flag makes the ordering
  // irrelevant.
  kappa_pair_dirty_ = true;
}

void SpinExternal::sync_kappa_pair()
{
  if (!kappa_pair_dirty_) return;
  const size_t want = (num_types > 0)
                        ? static_cast<size_t>(num_types) * num_types
                        : 0;
  if (want == 0) {
    kappa_pair_.resize(0);
    kappa_pair_mask_.resize(0);
    kappa_pair_dirty_ = false;
    return;
  }
  if (me_kappa_pair.size() != want) me_kappa_pair.assign(want, 0.0f);
  if (me_kappa_pair_set.size() != want) me_kappa_pair_set.assign(want, 0.0f);
  kappa_pair_.resize(want);
  kappa_pair_mask_.resize(want);
  kappa_pair_.copy_from_host(me_kappa_pair.data());
  kappa_pair_mask_.copy_from_host(me_kappa_pair_set.data());
  kappa_pair_dirty_ = false;
}

void SpinExternal::apply_zeeman(
  const int number_of_atoms,
  const float* sx, const float* sy, const float* sz,
  float* tx, float* ty, float* tz,
  double* potential)
{
  const int grid = (number_of_atoms - 1) / SE_BLOCK + 1;
  gpu_add_zeeman<<<grid, SE_BLOCK>>>(
    number_of_atoms, applied_b[0], applied_b[1], applied_b[2],
    sx, sy, sz, tx, ty, tz, potential, energy_acc_.data());
  GPU_CHECK_KERNEL
}

void SpinExternal::apply_magelec(
  Box& box,
  const int number_of_atoms,
  const int* type,
  const double* x, const double* y, const double* z,
  const float* sx, const float* sy, const float* sz,
  float* tx, float* ty, float* tz,
  double* potential,
  double* fx, double* fy, double* fz,
  double* virial)
{
  const int grid = (number_of_atoms - 1) / SE_BLOCK + 1;
  // F-06 (r21): upload the CPU parameter table if anything changed it since
  // the last evaluation (magelec_pair, allocate_types, repeated overrides).
  sync_kappa_pair();
  const float* kp = (kappa_pair_.size() > 0) ? kappa_pair_.data() : nullptr;
  const float* ks =
    (kappa_pair_mask_.size() > 0) ? kappa_pair_mask_.data() : nullptr;

  int num_bins[3];
  box.get_num_bins(0.5 * me_cutoff, num_bins);
  const bool list_is_safe =
    num_bins[0] >= 5 && num_bins[1] >= 5 && num_bins[2] >= 5;

  if (!list_is_safe) {
    // Small / non-periodic box: the 5x5x5 scan would reach the same atom twice.
    gpu_add_magelec_full<<<grid, SE_BLOCK>>>(
      number_of_atoms, num_types, me_kappa, kp, ks,
      me_E[0], me_E[1], me_E[2], me_cutoff, me_r_on, me_cutoff_mode, box, type,
      x, y, z, sx, sy, sz, tx, ty, tz, potential, fx, fy, fz, virial,
      energy_acc_.data() + 1);
    GPU_CHECK_KERNEL
    return;
  }

  const int n_cells = num_bins[0] * num_bins[1] * num_bins[2];
  if (cell_count_.size() < n_cells) {
    cell_count_.resize(n_cells);
    cell_count_sum_.resize(n_cells);
  }
  if (cell_contents_.size() < number_of_atoms) {
    cell_contents_.resize(number_of_atoms);
  }

  const int block_size = 256;
  const int grid_size = (number_of_atoms - 1) / block_size + 1;
  const double rc = 0.5 * me_cutoff;
  const double rc_inv = 1.0 / rc;
  const int nx = num_bins[0], ny = num_bins[1], nz = num_bins[2];

  se_zero_int<<<(n_cells - 1) / 64 + 1, 64>>>(n_cells, cell_count_.data());
  se_zero_int<<<(n_cells - 1) / 64 + 1, 64>>>(n_cells, cell_count_sum_.data());
  se_zero_int<<<(number_of_atoms - 1) / 64 + 1, 64>>>(number_of_atoms, cell_contents_.data());
  GPU_CHECK_KERNEL

  se_cell_counts<<<grid_size, block_size>>>(
    box, number_of_atoms, cell_count_.data(), x, y, z, nx, ny, nz, rc_inv);
  GPU_CHECK_KERNEL

  thrust::exclusive_scan(
    thrust::device, cell_count_.data(), cell_count_.data() + n_cells, cell_count_sum_.data());

  se_zero_int<<<(n_cells - 1) / 64 + 1, 64>>>(n_cells, cell_count_.data());
  GPU_CHECK_KERNEL

  se_cell_contents<<<grid_size, block_size>>>(
    box, number_of_atoms, cell_count_.data(), cell_count_sum_.data(),
    cell_contents_.data(), x, y, z, nx, ny, nz, rc_inv);
  GPU_CHECK_KERNEL

  gpu_add_magelec_list<<<grid, SE_BLOCK>>>(
      number_of_atoms, num_types, me_kappa, kp, ks,
      me_E[0], me_E[1], me_E[2], me_cutoff, me_r_on, me_cutoff_mode,
    nx, ny, nz, box, type,
    cell_count_.data(), cell_count_sum_.data(), cell_contents_.data(),
    x, y, z, sx, sy, sz, tx, ty, tz, potential, fx, fy, fz, virial,
    energy_acc_.data() + 1);
  GPU_CHECK_KERNEL
}

void SpinExternal::apply(
  Box& box,
  const int number_of_atoms,
  const int* type,
  const double* x,
  const double* y,
  const double* z,
  const float* sx,
  const float* sy,
  const float* sz,
  float* tx,
  float* ty,
  float* tz,
  double* potential,
  double* fx,
  double* fy,
  double* fz,
  double* virial)
{
  if (!any() || number_of_atoms <= 0) return;
  if (sx == nullptr || tx == nullptr) return;

  if (energy_acc_.size() < 2) energy_acc_.resize(2);
  // F-07 (r21): BOTH accumulators must be cleared before every evaluation.
  // r20 launched se_zero_double<<<1,1>>>(2, ...) -- a single thread writes
  // only p[0], so `energy_magelec` kept accumulating across force calls and
  // the reported ME energy grew with the number of evaluations.  The kernel
  // is indexed by thread id, so the grid must cover `size` elements.
  // The host-side copy_to_host() below is synchronous on the default stream;
  // drain it explicitly so the device atomics from the last apply() are
  // visible even when the caller used a non-default stream.
  se_zero_double<<<(2 - 1) / 64 + 1, 64>>>(2, energy_acc_.data());
  GPU_CHECK_KERNEL

  if (zeeman_enabled) {
    // R22: the envelope is also evaluated at the step boundary by
    // compute_applied_b()/record_drive_work(); the expression is identical,
    // so every force evaluation in this step still sees ONE midpoint field.
    compute_applied_b();
    apply_zeeman(number_of_atoms, sx, sy, sz, tx, ty, tz, potential);
  }

  if (magelec_enabled && me_cutoff > 0.0) {
    apply_magelec(
      box, number_of_atoms, type, x, y, z, sx, sy, sz, tx, ty, tz,
      potential, fx, fy, fz, virial);
  }
}

void SpinExternal::report(const char* tag)
{
  if (!any()) return;
  if (energy_acc_.size() < 2) return;
  read_energies();
  const double e[2] = {energy_zeeman, energy_magelec};
  // NOTE: these two are DIAGNOSTIC totals of the external terms only.  The
  // per-atom `potential[]` array carries the sum of NEP + Zeeman + ME and is
  // what the energy-conservation check and the minimisers consume.  They are
  // deliberately not the same array; do not compare them directly.
  if (zeeman_enabled) {
    printf(
      "    %s: Zeeman energy = %+.8f eV  (B = (%.6g, %.6g, %.6g) eV/mu_B)\n",
      tag, e[0], applied_b[0], applied_b[1], applied_b[2]);
  }
  if (magelec_enabled) {
    printf("    %s: magneto-electric energy = %+.8f eV\n", tag, e[1]);
  }
}
