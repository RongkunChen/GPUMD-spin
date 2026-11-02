#include "spin.cuh"
#include "utilities/error.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define SPIN_BLOCK_SIZE 128

// Boltzmann constant in eV/K
#define SPIN_KB_EV_PER_K 8.617333262e-5

// F13: geometric LLG integrator.
// The previous update was explicit Euler followed by a renormalisation of |S|:
//     s' = s + dt * (-gamma s x B) + dt * (-gamma alpha/|s|) s x (s x B)
//     s  = s' * |s| / |s'|
// It has two defects that were measured, not guessed:
//  (1) The renormalised rotation angle is arctan(theta) instead of theta, with
//      theta = gamma |B| dt, so every step lags by ~theta^3/3.  The lag has a
//      fixed sign, so it accumulates linearly instead of averaging out:
//      measured +8.0e-9 eV/atom/step of *energy pumped in* at alpha = 0, where
//      the exact dynamics is strictly energy conserving.  That is ~2000x worse
//      than the bare lattice (3.7e-12 eV/atom/step).
//  (2) The error grows as |B|^3.  Heating disorders the spins, which raises
//      |B|, which raises the error: a positive feedback loop.  It is what makes
//      long NVE + SLLG runs diverge (R1): theta = gamma|B|dt reaches O(1) and
//      explicit Euler leaves the stability region entirely.
// The replacement below is *exact* for a field frozen over one step:
//   * precession  -> Rodrigues rotation about B-hat by gamma |B| dt;
//   * damping     -> analytic relaxation of the angle theta between S and B,
//                    tan(theta/2) -> tan(theta/2) exp(-gamma alpha |B| dt),
//                    which is the closed-form solution of dtheta/dt =
//                    -gamma alpha |B| sin(theta).
// Both maps are isometries, so |S| is conserved to round-off (no systematic
// radial error), both are unconditionally stable in theta, and the O(theta^3)
// energy pump disappears.  A Strang split (half precession, damping, half
// precession) keeps the scheme second order in dt.

// Rodrigues rotation: rotate v about the *unit* axis k by the angle a.
__device__ inline void spin_rotate(
  const float v[3], const float k[3], const float ca, const float sa, float out[3])
{
  const float kvx = k[1] * v[2] - k[2] * v[1];
  const float kvy = k[2] * v[0] - k[0] * v[2];
  const float kvz = k[0] * v[1] - k[1] * v[0];
  const float kd = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];
  const float one_m_c = 1.0f - ca;
  out[0] = v[0] * ca + kvx * sa + k[0] * kd * one_m_c;
  out[1] = v[1] * ca + kvy * sa + k[1] * kd * one_m_c;
  out[2] = v[2] * ca + kvz * sa + k[2] * kd * one_m_c;
}

// Exact Gilbert relaxation of the angle between S and the unit axis k:
// tan(theta/2) *= exp(-rate), rate = gamma*alpha*|B|*dt.  |S| is preserved.
__device__ inline void spin_relax(
  const float v[3], const float k[3], const float rate, float out[3])
{
  const float sn = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const float sc = k[0] * v[0] + k[1] * v[1] + k[2] * v[2];  // = |v| cos(theta)
  if (sn < 1.0e-12f) {
    out[0] = v[0];
    out[1] = v[1];
    out[2] = v[2];
    return;
  }
  const float c = fminf(1.0f, fmaxf(-1.0f, sc / sn));
  const float sin_th = sqrt(fmaxf(0.0f, 1.0f - c * c));
  // theta = 0 or pi: the damping has no tangential direction to act on, and
  // theta = pi is an unstable fixed point.  Leave the spin untouched.
  if (sin_th < 1.0e-7f) {
    out[0] = v[0];
    out[1] = v[1];
    out[2] = v[2];
    return;
  }
  float t = sqrt(fmaxf(0.0f, (1.0f - c) / (1.0f + c)));  // tan(theta/2)
  t *= expf(-rate);
  const float t2 = t * t;
  const float cos_new = (1.0f - t2) / (1.0f + t2);
  const float sin_new = 2.0f * t / (1.0f + t2);
  // out = |v| cos(theta') k + (sin(theta')/sin(theta)) (v - |v| cos(theta) k)
  const float f = sin_new / sin_th;
  out[0] = sn * cos_new * k[0] + f * (v[0] - sc * k[0]);
  out[1] = sn * cos_new * k[1] + f * (v[1] - sc * k[1]);
  out[2] = sn * cos_new * k[2] + f * (v[2] - sc * k[2]);
}

static __global__ void gpu_llg_step(
  const int N,
  const float* g_sx,
  const float* g_sy,
  const float* g_sz,
  const float* g_tx,
  const float* g_ty,
  const float* g_tz,
  const float* g_bx,
  const float* g_by,
  const float* g_bz,
  const float bex,
  const float bey,
  const float bez,
  const float gamma,
  const float alpha,
  const float dt,
  float* g_sx_out,
  float* g_sy_out,
  float* g_sz_out,
  // ---- F16: spin work/energy accounting (R12) --------------------------
  // U(r, S) is a single function, so rotating the spins by dS changes the
  // potential energy by  dU = (dU/dS).dS = -B_det.dS  with B_det = w = -dU/dS
  // (the deterministic torque).  The kernel rotates about the *total* field
  // b = B_det + b_noise + b_ext, so split the same quantity exactly:
  //     dU = -b.dS  +  b_noise.dS  +  b_ext.dS
  //          W_rot       W_noise       W_ext
  // W_rot is the part coming from the geometric maps: a Rodrigues rotation
  // about b-hat leaves b.dS = 0, so W_rot is entirely due to the Gilbert
  // relaxation and is <= 0 (a one-way energy sink).  W_noise is the work done
  // by the stochastic field -- this is the term that makes the SLLG thermal
  // field a net energy SOURCE (risk R12) and it is what was previously only
  // inferred by differencing two 300k-step runs.
  // Each block reduces its three contributions in shared memory and issues one
  // atomicAdd per block, so the diagnostic costs ~3 atomics per block per step
  // (not per atom).
  double* g_w_rot,
  double* g_w_noise,
  double* g_w_ext)
{
  const int tid = threadIdx.x;
  const int n = blockIdx.x * blockDim.x + tid;

  __shared__ double sh_rot[SPIN_BLOCK_SIZE];
  __shared__ double sh_noise[SPIN_BLOCK_SIZE];
  __shared__ double sh_ext[SPIN_BLOCK_SIZE];

  double loc_rot = 0.0, loc_noise = 0.0, loc_ext = 0.0;

  float s[3] = {0.0f, 0.0f, 0.0f};
  if (n < N) {
    s[0] = g_sx[n];
    s[1] = g_sy[n];
    s[2] = g_sz[n];
  }
  // effective field = deterministic torque + (optional) thermal field
  //                   + (optional) external field from `add_bfield`
  float b[3] = {0.0f, 0.0f, 0.0f};
  if (n < N) {
    b[0] = g_tx[n] + ((g_bx == nullptr) ? 0.0f : g_bx[n]) + bex;
    b[1] = g_ty[n] + ((g_by == nullptr) ? 0.0f : g_by[n]) + bey;
    b[2] = g_tz[n] + ((g_bz == nullptr) ? 0.0f : g_bz[n]) + bez;
  }
  const float sn = sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  const float bnorm = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);

  // F16: the deterministic part of the field, i.e. what defines U.  b_det is
  // needed to split dU; see the comment on the kernel signature.
  const float bdet[3] = {
    b[0] - ((g_bx == nullptr || n >= N) ? 0.0f : g_bx[n]) - bex,
    b[1] - ((g_by == nullptr || n >= N) ? 0.0f : g_by[n]) - bey,
    b[2] - ((g_bz == nullptr || n >= N) ? 0.0f : g_bz[n]) - bez};

  // Both early exits must fall through to the F16 reduction (with a zero
  // contribution) instead of returning, otherwise the block reduction below
  // would be left with uninitialised lanes.
  const bool active = (n < N) && (sn >= 1.0e-12f) && (bnorm >= 1.0e-12f);

  float v[3] = {s[0], s[1], s[2]};
  float w[3] = {s[0], s[1], s[2]};

  if (active) {
    const float k[3] = {b[0] / bnorm, b[1] / bnorm, b[2] / bnorm};

    // half precession angle and the damping exponent over one step
    const float theta_half = 0.5f * gamma * bnorm * dt;
    const float rate = gamma * alpha * bnorm * dt;
    const float ca = cosf(theta_half);
    const float sa = sinf(theta_half);

    spin_rotate(v, k, ca, sa, w);      // half precession
    spin_relax(w, k, rate, v);         // exact Gilbert relaxation
    spin_rotate(v, k, ca, sa, w);      // half precession
  }

  // |S| is conserved to round-off by both maps; this only removes the
  // accumulated float error and is a no-op at the 1e-7 level.
  const float sn_new = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  const float scale = (sn_new > 1.0e-12f && active) ? (sn / sn_new) : 1.0f;
  const float wx = w[0] * scale, wy = w[1] * scale, wz = w[2] * scale;
  if (n < N) {
    g_sx_out[n] = wx;
    g_sy_out[n] = wy;
    g_sz_out[n] = wz;
  }

  // ---- F16: accumulate the three work terms ----------------------------
  if (active) {
    const double ds[3] = {(double)wx - (double)s[0],
                          (double)wy - (double)s[1],
                          (double)wz - (double)s[2]};
    // dU = -b_det . dS, split as (-b.dS) + (b - b_det).dS
    loc_rot = -((double)b[0] * ds[0] + (double)b[1] * ds[1] + (double)b[2] * ds[2]);
    loc_ext = (double)bex * ds[0] + (double)bey * ds[1] + (double)bez * ds[2];
    loc_noise = -((double)bdet[0] * ds[0] + (double)bdet[1] * ds[1] +
                  (double)bdet[2] * ds[2]) - loc_rot - loc_ext;
  }

  sh_rot[tid] = loc_rot;
  sh_noise[tid] = loc_noise;
  sh_ext[tid] = loc_ext;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      sh_rot[tid] += sh_rot[tid + offset];
      sh_noise[tid] += sh_noise[tid + offset];
      sh_ext[tid] += sh_ext[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (g_w_rot) atomicAdd(g_w_rot, sh_rot[0]);
    if (g_w_noise) atomicAdd(g_w_noise, sh_noise[0]);
    if (g_w_ext) atomicAdd(g_w_ext, sh_ext[0]);
  }
}

// P22: one curand state per atom, seeded once in initialize().
static __global__ void gpu_init_spin_rng(gpurandState* g_state, const int N, const int seed)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  gpurand_init(seed, n, 0, &g_state[n]);
}

// P22: b_i = sqrt(prefactor / |S_i|) * xi_i, drawn on the device.
// `prefactor` is 2 alpha kB T / (gamma dt (1 + alpha^2)); the per-atom |S_i| is
// read here so that no host round trip is needed.  The 1/(1 + alpha^2) is the
// fluctuation-dissipation factor of the Landau-Lifshitz form used in this file
// (damping coefficient gamma*alpha), see the header comment in spin.cuh.
static __global__ void gpu_fill_thermal_field(
  const int N,
  const float* __restrict__ g_sx,
  const float* __restrict__ g_sy,
  const float* __restrict__ g_sz,
  const double prefactor,
  gpurandState* g_state,
  float* g_bx,
  float* g_by,
  float* g_bz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const float sn = sqrt(
    g_sx[n] * g_sx[n] + g_sy[n] * g_sy[n] + g_sz[n] * g_sz[n]);
  float sigma = 0.0f;
  if (sn > 1.0e-12f) {
    const double var = prefactor / (double)sn;
    sigma = (var > 0.0) ? (float)sqrt(var) : 0.0f;
  }
  gpurandState state = g_state[n];
  g_bx[n] = sigma * gpurand_normal(&state);
  g_by[n] = sigma * gpurand_normal(&state);
  g_bz[n] = sigma * gpurand_normal(&state);
  g_state[n] = state;
}

// S05A: conservative implicit midpoint, fixed lattice, alpha = 0, T = 0.
//     S_mid = (S_old + S_new)/2
//     B_mid = w(S_mid)                      (whole-system field, re-evaluated)
//     S_new = S_old - gamma*dt (S_mid x B_mid)
// iterated until S_new stops moving.  This is the implicit midpoint rule: it is
// time-symmetric and second order, and |S| is conserved exactly in exact
// arithmetic -- with dS = S_new - S_old = -gamma dt S_mid x B_mid and
// S_old = S_mid - dS/2 one gets S_old.(S_mid x B_mid) = (dS/2).(S_mid x B_mid),
// hence |S_new|^2 = |S_old|^2.  No renormalisation is applied: a scheme that
// has to rescale |S| every step has not earned the right to be called
// conservative, and it would hide exactly the drift this stage must measure.
// The field handed to propagate() is NOT used here: it is the field at the
// beginning of the step, which is the defect this branch removes.
static __global__ void gpu_midpoint_average(
  const int N,
  const float* s_ox, const float* s_oy, const float* s_oz,
  const float* s_nx, const float* s_ny, const float* s_nz,
  float* m_x, float* m_y, float* m_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  m_x[n] = 0.5f * (s_ox[n] + s_nx[n]);
  m_y[n] = 0.5f * (s_oy[n] + s_ny[n]);
  m_z[n] = 0.5f * (s_oz[n] + s_nz[n]);
}

// S_new = S_old - gamma*dt*(S_mid x B_mid), with B_mid = deterministic field at
// S_mid plus the uniform `add_bfield` field.
// No bookkeeping here: this kernel runs once per trial iteration, and only the
// update that is finally accepted may be counted as physical work (S05B item 3;
// before, every trial iteration added to W_rot/W_ext and the F16 budget was
// inflated by the iteration count).  gpu_midpoint_work() below does the
// accounting once, after convergence.
static __global__ void gpu_midpoint_update(
  const int N,
  const float* s_ox, const float* s_oy, const float* s_oz,
  const float* m_x, const float* m_y, const float* m_z,
  const float* f_x, const float* f_y, const float* f_z,
  const double bex, const double bey, const double bez,
  const double gamma_dt,
  float* o_x, float* o_y, float* o_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const double mx = (double)m_x[n], my = (double)m_y[n], mz = (double)m_z[n];
  const double bx = (double)f_x[n] + bex;
  const double by = (double)f_y[n] + bey;
  const double bz = (double)f_z[n] + bez;
  const double cx = my * bz - mz * by;
  const double cy = mz * bx - mx * bz;
  const double cz = mx * by - my * bx;
  o_x[n] = (float)((double)s_ox[n] - gamma_dt * cx);
  o_y[n] = (float)((double)s_oy[n] - gamma_dt * cy);
  o_z[n] = (float)((double)s_oz[n] - gamma_dt * cz);
}

// F16 accounting for the *accepted* update only:
//     W_rot = -b_det(S_mid).(S_new - S_old)      O(dt^3) per step here
//     W_ext = +b_ext.(S_new - S_old)
static __global__ void gpu_midpoint_work(
  const int N,
  const float* s_ox, const float* s_oy, const float* s_oz,
  const float* s_nx, const float* s_ny, const float* s_nz,
  const float* f_x, const float* f_y, const float* f_z,
  const double bex, const double bey, const double bez,
  double* g_w_rot, double* g_w_ext)
{
  const int tid = threadIdx.x;
  const int n = blockIdx.x * blockDim.x + tid;
  __shared__ double sh_rot[SPIN_BLOCK_SIZE];
  __shared__ double sh_ext[SPIN_BLOCK_SIZE];
  double loc_rot = 0.0, loc_ext = 0.0;
  if (n < N) {
    const double dx = (double)s_nx[n] - (double)s_ox[n];
    const double dy = (double)s_ny[n] - (double)s_oy[n];
    const double dz = (double)s_nz[n] - (double)s_oz[n];
    loc_rot = -((double)f_x[n] * dx + (double)f_y[n] * dy + (double)f_z[n] * dz);
    loc_ext = bex * dx + bey * dy + bez * dz;
  }
  sh_rot[tid] = loc_rot;
  sh_ext[tid] = loc_ext;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) {
      sh_rot[tid] += sh_rot[tid + off];
      sh_ext[tid] += sh_ext[tid + off];
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (g_w_rot) atomicAdd(g_w_rot, sh_rot[0]);
    if (g_w_ext) atomicAdd(g_w_ext, sh_ext[0]);
  }
}

// S08A: stochastic implicit-midpoint update.
//     S_new = S_old - gdt (m x H) - (gdt alpha/|m|) m x (m x H),   m = midway
//     H     = B_det(m) + b_noise(step) + b_ext
// b_noise is drawn once per step and held fixed for every iteration: letting it
// depend on the iterate would turn a fixed-point iteration into a different
// (and non-reproducible) stochastic scheme.
// In exact arithmetic (S_new - S_old) is perpendicular to m, and
// (S_new + S_old) = 2m, so |S_new| = |S_old| exactly.  The residual norm error
// is therefore pure float32 rounding, which is why it is measured instead of
// assumed: gpu_project_anchor() records it before it touches the spin.
static __global__ void gpu_midpoint_llg_update(
  const int N,
  const float* s_ox, const float* s_oy, const float* s_oz,
  const float* m_x, const float* m_y, const float* m_z,
  const float* f_x, const float* f_y, const float* f_z,
  const float* b_x, const float* b_y, const float* b_z,
  const double bex, const double bey, const double bez,
  const double gamma_dt, const double alpha,
  float* o_x, float* o_y, float* o_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const double mx = (double)m_x[n], my = (double)m_y[n], mz = (double)m_z[n];
  const double hx = (double)f_x[n] + (b_x == nullptr ? 0.0 : (double)b_x[n]) + bex;
  const double hy = (double)f_y[n] + (b_y == nullptr ? 0.0 : (double)b_y[n]) + bey;
  const double hz = (double)f_z[n] + (b_z == nullptr ? 0.0 : (double)b_z[n]) + bez;
  // precession part  p = m x H
  const double px = my * hz - mz * hy;
  const double py = mz * hx - mx * hz;
  const double pz = mx * hy - my * hx;
  // damping part  d = m x p = m x (m x H)
  const double dx = my * pz - mz * py;
  const double dy = mz * px - mx * pz;
  const double dz = mx * py - my * px;
  const double sm = sqrt(mx * mx + my * my + mz * mz);
  const double k = (sm > 1.0e-12) ? gamma_dt * alpha / sm : 0.0;
  o_x[n] = (float)((double)s_ox[n] - gamma_dt * px - k * dx);
  o_y[n] = (float)((double)s_oy[n] - gamma_dt * py - k * dy);
  o_z[n] = (float)((double)s_oz[n] - gamma_dt * pz - k * dz);
}

// S08A: project each spin onto its magnitude anchor, and report the largest
// relative |S| error seen *before* the projection.  That number is the honest
// measure of whether the integrator preserves the norm; the projected value is
// the stored trajectory.  An atom with |S| = 0 keeps its zero (the thermal
// field is exactly zero there, so the update is exactly zero as well).
static __global__ void gpu_project_anchor(
  const int N,
  float* g_sx, float* g_sy, float* g_sz,
  float* anchor,
  double* g_max_err)
{
  __shared__ double sh[SPIN_BLOCK_SIZE];
  const int tid = threadIdx.x;
  const int n = blockIdx.x * blockDim.x + tid;
  double err = 0.0;
  if (n < N) {
    double sx = (double)g_sx[n], sy = (double)g_sy[n], sz = (double)g_sz[n];
    const double sn = sqrt(sx * sx + sy * sy + sz * sz);
    double an = (double)anchor[n];
    // Refresh only on a real change of the moment (restart / replicate), never
    // on round-off: the tolerance is 1e-4, i.e. 3 orders of magnitude above the
    // float32 jitter this projection is meant to remove.
    if (an <= 0.0 || fabs(sn - an) > 1.0e-4 * fmax(an, 1.0e-30)) {
      an = sn;
      anchor[n] = (float)an;
    }
    if (an > 0.0) {
      err = fabs(sn - an) / an;
      const double scale = (sn > 0.0) ? an / sn : 1.0;
      g_sx[n] = (float)(sx * scale);
      g_sy[n] = (float)(sy * scale);
      g_sz[n] = (float)(sz * scale);
    }
  }
  sh[tid] = err;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off && sh[tid + off] > sh[tid]) sh[tid] = sh[tid + off];
    __syncthreads();
  }
  if (tid == 0 && sh[0] > 0.0) {
    atomicMax(reinterpret_cast<unsigned long long*>(g_max_err),
              static_cast<unsigned long long>(__double_as_longlong(sh[0])));
  }
}

// F16 accounting for the stochastic midpoint, with the same split as
// gpu_llg_step: dU = W_rot + W_noise + W_ext, W_rot = -H.dS with H the total
// field used by the accepted update, W_ext = +b_ext.dS, W_noise = +b_noise.dS.
static __global__ void gpu_midpoint_work_stoch(
  const int N,
  const float* s_ox, const float* s_oy, const float* s_oz,
  const float* s_nx, const float* s_ny, const float* s_nz,
  const float* f_x, const float* f_y, const float* f_z,
  const float* b_x, const float* b_y, const float* b_z,
  const double bex, const double bey, const double bez,
  double* g_w_rot, double* g_w_noise, double* g_w_ext)
{
  const int tid = threadIdx.x;
  const int n = blockIdx.x * blockDim.x + tid;
  __shared__ double sh_rot[SPIN_BLOCK_SIZE];
  __shared__ double sh_noise[SPIN_BLOCK_SIZE];
  __shared__ double sh_ext[SPIN_BLOCK_SIZE];
  double lr = 0.0, ln = 0.0, le = 0.0;
  if (n < N) {
    const double dx = (double)s_nx[n] - (double)s_ox[n];
    const double dy = (double)s_ny[n] - (double)s_oy[n];
    const double dz = (double)s_nz[n] - (double)s_oz[n];
    const double nb = (b_x == nullptr ? 0.0 : (double)b_x[n]);
    const double nby = (b_y == nullptr ? 0.0 : (double)b_y[n]);
    const double nbz = (b_z == nullptr ? 0.0 : (double)b_z[n]);
    const double hx = (double)f_x[n] + nb + bex;
    const double hy = (double)f_y[n] + nby + bey;
    const double hz = (double)f_z[n] + nbz + bez;
    lr = -(hx * dx + hy * dy + hz * dz);
    le = bex * dx + bey * dy + bez * dz;
    ln = nb * dx + nby * dy + nbz * dz;
  }
  sh_rot[tid] = lr;
  sh_noise[tid] = ln;
  sh_ext[tid] = le;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off) {
      sh_rot[tid] += sh_rot[tid + off];
      sh_noise[tid] += sh_noise[tid + off];
      sh_ext[tid] += sh_ext[tid + off];
    }
    __syncthreads();
  }
  if (tid == 0) {
    if (g_w_rot) atomicAdd(g_w_rot, sh_rot[0]);
    if (g_w_noise) atomicAdd(g_w_noise, sh_noise[0]);
    if (g_w_ext) atomicAdd(g_w_ext, sh_ext[0]);
  }
}

static __global__ void gpu_zero_one(double* p) { *p = 0.0; }

static __global__ void gpu_zero_float(const int N, float* p)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) p[i] = 0.0f;
}

// max |a - b| into a single double.  The values are non-negative, so their IEEE
// bit patterns order like the numbers and an integer atomicMax is exact.
static __global__ void gpu_max_abs_diff(
  const int n, const float* a, const float* b, double* out)
{
  __shared__ double sh[SPIN_BLOCK_SIZE];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  double v = 0.0;
  if (i < n) {
    v = fabs((double)a[i] - (double)b[i]);
  }
  sh[tid] = v;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off && sh[tid + off] > sh[tid]) sh[tid] = sh[tid + off];
    __syncthreads();
  }
  if (tid == 0 && sh[0] > 0.0) {
    atomicMax(reinterpret_cast<unsigned long long*>(out),
              static_cast<unsigned long long>(__double_as_longlong(sh[0])));
  }
}

void Spin::parse(const std::vector<std::string>& tokens)
{
  // F21 (round 17): named parameters are accepted in addition to the positional
  // form.  The positional form is a live hazard: a template that substituted
  // values in the wrong order silently put T into the gamma slot
  // (`spin_llg 0.01 300 300 4242`), which ran 3e4x too fast and produced a batch
  // of void M(T) data before anyone noticed.  Both spellings mean the same
  // thing, so old inputs keep working:
  //     spin_llg 3.04 0.5 300 4242
  //     spin_llg gamma=3.04 alpha=0.5 T=300 seed=4242
  bool named = false;
  for (size_t i = 1; i < tokens.size(); ++i) {
    if (tokens[i].find('=') != std::string::npos) {
      named = true;
    }
  }
  if (named) {
    int n_gamma = 0, n_alpha = 0;
    for (size_t i = 1; i < tokens.size(); ++i) {
      const std::string& t = tokens[i];
      const size_t eq = t.find('=');
      if (eq == std::string::npos) {
        PRINT_INPUT_ERROR(
          "spin_llg: cannot mix named (key=value) and positional arguments.\n");
      }
      const std::string key = t.substr(0, eq);
      const std::string val = t.substr(eq + 1);
      if (val.empty()) {
        PRINT_INPUT_ERROR("spin_llg: a key=value argument has an empty value.\n");
      }
      if (key == "gamma") {
        gamma = atof(val.c_str());
        ++n_gamma;
      } else if (key == "alpha") {
        alpha = atof(val.c_str());
        ++n_alpha;
      } else if (key == "T" || key == "temperature") {
        temperature = atof(val.c_str());
      } else if (key == "seed") {
        seed = atoi(val.c_str());
      } else if (key == "scheme") {
        // S08A: choose the finite-temperature integrator explicitly.  Both
        // schemes are measured by the same acceptance battery; this switch
        // exists so that a run states which one it used instead of relying on
        // a build-time default.
        if (val == "geometric") {
          sllg_scheme = 0;
        } else if (val == "midpoint") {
          sllg_scheme = 1;
        } else {
          PRINT_INPUT_ERROR(
            "spin_llg: scheme= must be 'geometric' or 'midpoint'.\n");
        }
        sllg_scheme_set = true;
      } else if (key == "restart") {
        // S08B: opt in to loading the magnetic part of a checkpoint.  It is an
        // explicit keyword, not an automatic pick-up of a file that happens to
        // be in the working directory: a stale spin_restart.bin from an earlier
        // run must never be applied silently.
        restart_file_ = val;
      } else {
        PRINT_INPUT_ERROR(
          "spin_llg: unknown key.  Use gamma=, alpha=, T= (or temperature=), "
          "seed=, scheme=, restart=.\n");
      }
    }
    if (n_gamma != 1 || n_alpha != 1) {
      PRINT_INPUT_ERROR(
        "spin_llg: gamma= and alpha= are both required exactly once.\n");
    }
  } else {
    if (tokens.size() != 3 && tokens.size() != 4 && tokens.size() != 5) {
      PRINT_INPUT_ERROR(
        "spin_llg should have 2 to 4 parameters: "
        "<gamma> <alpha> [<temperature> [<seed>]].\n");
    }
    gamma = atof(tokens[1].c_str());
    alpha = atof(tokens[2].c_str());
    if (tokens.size() >= 4) {
      temperature = atof(tokens[3].c_str());
    }
    if (tokens.size() >= 5) {
      seed = atoi(tokens[4].c_str());
    }
  }
  enabled = true;
  // F21: warn when gamma is not the physical value, because then the noise
  // field dominates the exchange field and every dynamical quantity (magnon
  // lifetime, kappa_mm, magnetisation relaxation) is wrong by orders of
  // magnitude, even though the equilibrium distribution is still correct.
  {
    const double gamma_phys = 3.0385;   // g*mu_B/hbar in mu_B/(eV fs), g = 2
    if (gamma > 0.0 && gamma < 0.5 * gamma_phys) {
      printf(
        "NOTE spin_llg: gamma = %g is %.0fx below the physical %.4f "
        "mu_B/(eV fs).\n"
        "      Equilibrium quantities (M, chi) are unaffected, but the spin "
        "dynamics is\n"
        "      then noise dominated (sigma_SLLG/rms|B| ~ 1/sqrt(gamma)) and any "
        "transport\n"
        "      or relaxation time is meaningless.  Use gamma = %.4f with a "
        "smaller\n"
        "      time_step for dynamical results.\n",
        gamma, gamma_phys / gamma, gamma_phys, gamma_phys);
    }
  }
  if (alpha < 0.0) {
    PRINT_INPUT_ERROR("The Gilbert damping in spin_llg must be >= 0.\n");
  }
  if (temperature < 0.0) {
    PRINT_INPUT_ERROR("The temperature in spin_llg must be >= 0.\n");
  }
  if (temperature > 0.0 && alpha <= 0.0) {
    PRINT_INPUT_ERROR(
      "A finite spin_llg temperature requires alpha > 0 "
      "(the noise amplitude is proportional to alpha).\n");
  }
  printf("Spin dynamics (LLG): gamma = %g, alpha = %g", gamma, alpha);
  if (temperature > 0.0) {
    printf(", temperature = %g K (stochastic LLG, seed %d)", temperature, seed);
  }
  printf(".\n");
}

void Spin::check_discretisation(double time_step_fs_in)
{
  // Round 11 (P86): remember dt so that report() can print the SLLG thermal
  // field amplitude.  Non-const on purpose (was const before).
  // The SLLG amplitude divides by dt, so a non-positive step is an input error
  // rather than a division by zero that would turn the thermal field into
  // inf/nan on the device.
  if (!(time_step_fs_in > 0.0)) {
    PRINT_INPUT_ERROR(
      "The time step used with spin_llg must be > 0 (it sets the SLLG "
      "thermal-field amplitude).\n");
  }
  time_step_fs = time_step_fs_in;
  if (!enabled) {
    return;
  }
  // S08B: the checkpoint is validated against the run's own parameters, and dt
  // is only known here (run.in reads `time_step` before `spin_llg`).
  if (!restart_file_.empty()) {
    load_restart_state(restart_file_.c_str());
  }
  // A-6: there used to be a `gamma * dt > 0.05` "stability" warning here.  That
  // criterion is dimensionally empty (gamma*dt carries units of mu_B/eV, not
  // radians) and it contradicted the F13 integrator, which is unconditionally
  // stable in the rotation angle.  The meaningful number is the per-step
  // rotation angle theta = gamma*|B_perp|*dt, which needs the effective field
  // and is therefore checked in report() as
  //     x = gamma*|B_perp|*dt/alpha   (C24: x <= ~2).
  if (temperature > 0.0 && alpha < 1e-3) {
    printf("NOTE: alpha = %g is far below any physical Gilbert damping.\n", alpha);
    printf("      Round 15 (P110/P111) showed that alpha does NOT change the\n");
    printf("      SLLG equilibrium -- it only scales the relaxation RATE, so the\n");
    printf("      equilibration time grows as 1/alpha.  At alpha = %g with\n", alpha);
    printf("      gamma*|B| ~ 0.08 /fs the Gilbert time is ~%.3g ps, so a run\n",
           1.0 / (alpha * 0.08) * 1.0e-3);
    printf("      shorter than ~50x that value will keep whatever magnetic order\n");
    printf("      it started from (round 13's alpha = 0.003 Fe run stayed at\n");
    printf("      M/sum|S| = 0.03 for exactly this reason).\n");
  }
}

void Spin::set_external_field(double bx, double by, double bz, double omega_thz)
{
  b_ext_x = bx;
  b_ext_y = by;
  b_ext_z = bz;
  b_ext_omega = omega_thz;
}

void Spin::initialize(const int number_of_atoms_in, const bool has_spin_in)
{
  number_of_atoms = number_of_atoms_in;
  has_spin = has_spin_in;
  if (enabled && !has_spin) {
    PRINT_INPUT_ERROR(
      "spin_llg requires a 'spin' column in model.xyz.\n");
  }
  if (enabled) {
    printf("Spin dynamics is active for %d atoms.\n", number_of_atoms);
  }
  if (enabled) {
    // The ping-pong buffer is allocated once here, not in propagate():
    // GPU_Vector::resize() frees and reallocates unconditionally, and a
    // cudaFree issued while the slot stream still has work in flight is what
    // made cudaMemcpyAsync fail with "invalid argument" in the replica path.
    spin_new_.resize(static_cast<size_t>(number_of_atoms) * 3);
    // F16: three accumulators (W_rot, W_noise, W_ext) in eV.  Allocated here
    // and not in propagate(): GPU_Vector::resize() frees and reallocates, and a
    // cudaFree while the slot stream still has work in flight is what made
    // cudaMemcpyAsync fail with "invalid argument" in the replica path.
    work_.resize(3);
    reset_work();
  }
  if (enabled && temperature > 0.0) {
    noise_.resize(static_cast<size_t>(number_of_atoms) * 3);
    states_.resize(number_of_atoms);
    const int grid_size = (number_of_atoms - 1) / SPIN_BLOCK_SIZE + 1;
    gpu_init_spin_rng<<<grid_size, SPIN_BLOCK_SIZE>>>(
      states_.data(), number_of_atoms, seed);
    GPU_CHECK_KERNEL
  }
}

void Spin::propagate(
  const double time_step,
  float* g_spin,
  const float* g_torque,
  const int number_of_atoms_in,
  gpuStream_t stream,
  SpinFieldProvider* provider)
{
  number_of_atoms = number_of_atoms_in;
  if (!enabled || !has_spin) return;
  const int N = number_of_atoms;
  if (N <= 0) return;

  // A ping-pong buffer is needed because the kernel reads the spin of the
  // neighbouring atoms while writing the new value of its own atom.  It is a
  // member (not a function-local static) so that it is released together with
  // the Spin object: a static would be destroyed after the CUDA context is
  // gone, which makes gpuFree fail with "invalid argument" at exit.
  if (spin_new_.size() != static_cast<size_t>(N) * 3) {
    spin_new_.resize(static_cast<size_t>(N) * 3);
  }
  // B-14: the thermal-field buffers are allocated in initialize().  If the atom
  // count changed afterwards (a `replicate` after `spin_llg`, or a replica slot
  // with a different cell) the noise kernel would write past their end, so the
  // size is re-asserted here and the RNG re-seeded to match.
  if (temperature > 0.0 &&
      (noise_.size() != static_cast<size_t>(N) * 3 || states_.size() != static_cast<size_t>(N))) {
    noise_.resize(static_cast<size_t>(N) * 3);
    states_.resize(static_cast<size_t>(N));
    const int grid_size_rng = (N - 1) / SPIN_BLOCK_SIZE + 1;
    gpu_init_spin_rng<<<grid_size_rng, SPIN_BLOCK_SIZE, 0, stream>>>(
      states_.data(), N, seed);
    GPU_CHECK_KERNEL
  }
  float* sx_new = spin_new_.data();
  float* sy_new = spin_new_.data() + N;
  float* sz_new = spin_new_.data() + N * 2;

  // ---- stochastic (SLLG) thermal field ----------------------------------
  float* bx = nullptr;
  float* by = nullptr;
  float* bz = nullptr;
  const int grid_size = (N - 1) / SPIN_BLOCK_SIZE + 1;
  if (temperature > 0.0) {
    // P22: the noise is drawn on the device.  No host round trip, no stream
    // synchronize, and the draw is enqueued on `stream` together with the LLG
    // kernel, so the replica slot stream stays self-consistent.
    // FDT for the deterministic equation this file actually integrates:
    //     dS/dt = -gamma S x B - (gamma alpha / |S|) S x (S x B)
    // is the Landau-Lifshitz form, i.e. the Gilbert damping alpha with
    // gamma_G = gamma (1 + alpha^2).  The noise correlator of the Gilbert form
    // is 2 alpha kB T / (gamma_G |S|), so the amplitude here carries an extra
    // 1/(1 + alpha^2).  Without it the thermostat sits at T (1 + alpha^2), a
    // 2x error in the effective temperature at alpha = 1.
    const double alpha_sq = 1.0 + alpha * alpha;
    const double prefactor = 2.0 * alpha * SPIN_KB_EV_PER_K * temperature /
                             (gamma * time_step * alpha_sq);
    gpu_fill_thermal_field<<<grid_size, SPIN_BLOCK_SIZE, 0, stream>>>(
      N,
      g_spin,
      g_spin + N,
      g_spin + N * 2,
      prefactor,
      states_.data(),
      noise_.data(),
      noise_.data() + N,
      noise_.data() + N * 2);
    GPU_CHECK_KERNEL
    bx = noise_.data();
    by = noise_.data() + N;
    bz = noise_.data() + N * 2;

    // F14 / R11: the amplitude sigma = sqrt(prefactor/|S|) diverges as |S| -> 0,
    // so an atom with a numerically tiny rather than exactly zero spin gets a
    // thermal field far larger than a magnetic one.  The LLG increment scales
    // as sqrt(|S| <b^2>) and therefore still vanishes, so nothing is changed
    // here -- the note only explains an otherwise alarming <b^2>.
    if (!warned_small_spin_) {
      warned_small_spin_ = true;
      std::vector<float> sv(static_cast<size_t>(N) * 3);
      CHECK(gpuMemcpy(
        sv.data(), g_spin, sizeof(float) * static_cast<size_t>(N) * 3,
        gpuMemcpyDeviceToHost));
      double smax = 0.0;
      std::vector<double> sm(N, 0.0);
      for (int i = 0; i < N; ++i) {
        const double a = sv[i], b = sv[i + N], c = sv[i + 2 * N];
        sm[i] = sqrt(a * a + b * b + c * c);
        if (sm[i] > smax) {
          smax = sm[i];
        }
      }
      int nsmall = 0;
      double msmall = 0.0;
      for (int i = 0; i < N; ++i) {
        if (sm[i] < 0.01 * smax) {
          ++nsmall;
          if (sm[i] > msmall) {
            msmall = sm[i];
          }
        }
      }
      if (nsmall > 0 && msmall > 0.0) {
        printf(
          "NOTE: %d of %d atoms carry |S| < 1%% of the maximum (%.4g vs %.4g).\n"
          "      Their SLLG thermal field is sqrt(smax/|S|) = %.1fx that of a magnetic\n"
          "      atom, which is what the FDT amplitude 2 alpha kB T/(gamma |S|) gives.\n"
          "      The LLG increment scales as sqrt(|S| <b^2>), so those spins still stay\n"
          "      put; this note only explains the large <b^2> seen in diagnostics.\n",
          nsmall, N, msmall, smax, sqrt(smax / msmall));
      }
    }

    // field, the spin and the torque *before* the update so that the LLG step
    // can be re-derived offline and the noise amplitude checked.
    if (getenv("SPIN_NOISE_DEBUG") != nullptr) {
      CHECK(gpuStreamSynchronize(stream));
      std::vector<float> s(static_cast<size_t>(N) * 3);
      CHECK(gpuMemcpy(
        s.data(), g_spin, sizeof(float) * static_cast<size_t>(N) * 3, gpuMemcpyDeviceToHost));
      std::vector<float> b(static_cast<size_t>(N) * 3);
      noise_.copy_to_host(b.data());
      std::vector<float> t(static_cast<size_t>(N) * 3);
      CHECK(gpuMemcpy(
        t.data(), g_torque, sizeof(float) * static_cast<size_t>(N) * 3, gpuMemcpyDeviceToHost));
      FILE* fp = fopen("spin_noise.out", "a");
      if (fp != nullptr) {
        fprintf(fp, "step %lld dt %.10g N %d\n", debug_step_, time_step, N);
        for (int c = 0; c < 3; ++c) {
          for (int n = 0; n < N; ++n) {
            fprintf(fp, " %.9g", (double)b[n + c * N]);
          }
          fprintf(fp, "\n");
        }
        for (int c = 0; c < 3; ++c) {
          for (int n = 0; n < N; ++n) {
            fprintf(fp, " %.9g", (double)s[n + c * N]);
          }
          fprintf(fp, "\n");
        }
        for (int c = 0; c < 3; ++c) {
          for (int n = 0; n < N; ++n) {
            fprintf(fp, " %.9g", (double)t[n + c * N]);
          }
          fprintf(fp, "\n");
        }
        fclose(fp);
      }
      ++debug_step_;
    }
  }

  // ---- external driving field (NEMD / FMR) --------------------------------
  // B(t) = B sin(2 pi omega t) with omega in THz (1/ps) and t in fs.  It is
  // evaluated on the host once per step and broadcast as three scalars, so
  // there is no extra memory traffic per atom.
  float bex = 0.0f, bey = 0.0f, bez = 0.0f;
  if (has_external_field()) {
    double amp = 1.0;
    if (b_ext_omega != 0.0) {
      const double TWO_PI = 6.28318530717958647692;
      amp = sin(TWO_PI * b_ext_omega * (elapsed_time_fs * 1.0e-3));
    }
    bex = (float)(b_ext_x * amp);
    bey = (float)(b_ext_y * amp);
    bez = (float)(b_ext_z * amp);
  }
  b_ext_applied[0] = bex;
  b_ext_applied[1] = bey;
  b_ext_applied[2] = bez;
  elapsed_time_fs += time_step;

  // ---- S05A: conservative branch (alpha = 0, T = 0) ----------------------
  // Same equation, different quadrature: the effective field is re-evaluated at
  // the midpoint spins instead of being frozen at the beginning of the step.
  // The finite-alpha / finite-temperature branch below is untouched.
  if (alpha == 0.0 && temperature == 0.0) {
    propagate_conservative(
      time_step, g_spin, N, bex, bey, bez, provider, stream);
    return;
  }

  // S08A: the stochastic implicit midpoint, when selected, replaces the
  // frozen-field map for the finite-temperature branch only.  alpha = 0 with
  // T = 0 keeps the S05A conservative solver above; alpha > 0 with T = 0 keeps
  // the deterministic map below (no noise, no stochastic ordering issue, and
  // every previously accepted deterministic trajectory stays reproducible).
  if (temperature > 0.0 && sllg_scheme == 1) {
    propagate_stochastic_midpoint(time_step, g_spin, N, bex, bey, bez, bx, by, bz,
                                  provider, stream);
    return;
  }

  gpu_llg_step<<<grid_size, SPIN_BLOCK_SIZE, 0, stream>>>(
    N,
    g_spin,
    g_spin + N,
    g_spin + N * 2,
    g_torque,
    g_torque + N,
    g_torque + N * 2,
    bx,
    by,
    bz,
    bex,
    bey,
    bez,
    (float)gamma,
    (float)alpha,
    (float)time_step,
    sx_new,
    sy_new,
    sz_new,
    // F16: spin work accounting (R12).  nullptr-safe, so a build that never
    // calls reset_work() still runs.
    work_.size() >= 3 ? work_.data() : nullptr,
    work_.size() >= 3 ? work_.data() + 1 : nullptr,
    work_.size() >= 3 ? work_.data() + 2 : nullptr);
  GPU_CHECK_KERNEL

  // The copy must be ordered on the same stream: the next step's force kernel
  // writes the torque and reads the spin on `stream`, so a default-stream copy
  // would let those two race with this update.
  if (stream != 0) {
    CHECK(gpuMemcpyAsync(
      g_spin,
      spin_new_.data(),
      sizeof(float) * static_cast<size_t>(N) * 3,
      gpuMemcpyDeviceToDevice,
      stream));
  } else {
    CHECK(gpuMemcpy(
      g_spin,
      spin_new_.data(),
      sizeof(float) * static_cast<size_t>(N) * 3,
      gpuMemcpyDeviceToDevice));
  }

  // S08A: both stochastic schemes end on the same constraint.  The stored
  // trajectory is what every downstream measurement reads, so |S| must not be
  // allowed to random-walk away on float32 round-off (that is the 1.06e-5 drift
  // Stage04 recorded at alpha = 1).  The kernel also reports how far the raw
  // map moved the norm, which is the number that says whether the integrator
  // itself conserves it.
  project_onto_norm_anchor(g_spin, N, stream);
  ++sllg_steps;
}

// S05A: implicit-midpoint solver.  Runs on the calling stream's ordering: the
// caller's stream is drained once (the solver needs a host-visible residual
// every iteration, so some synchronisation is unavoidable), after which all
// work is issued on the default stream.
void Spin::propagate_conservative(
  const double time_step,
  float* g_spin,
  const int N,
  const float bex,
  const float bey,
  const float bez,
  SpinFieldProvider* provider,
  gpuStream_t stream)
{
  if (provider == nullptr) {
    fprintf(stderr,
            "Spin Error://n"
            "    The conservative spin update (alpha = 0, T = 0) needs a field\n"
            "    provider to re-evaluate -dE/dS at the midpoint spins; without\n"
            "    one the update would silently fall back to the field frozen at\n"
            "    the beginning of the step, which is the error this scheme\n"
            "    exists to remove.\n");
    std::exit(4);
  }
  if (!(time_step > 0.0)) {
    fprintf(stderr, "Spin Error://n    non-positive time step in the conservative "
                    "spin update: %g\n", time_step);
    std::exit(4);
  }
  CHECK(gpuStreamSynchronize(stream));

  const size_t n3 = static_cast<size_t>(N) * 3;
  if (spin_new_.size() != n3) spin_new_.resize(n3);
  if (spin_prev_.size() != n3) spin_prev_.resize(n3);
  if (spin_mid_.size() != n3) spin_mid_.resize(n3);
  if (field_mid_.size() != n3) field_mid_.resize(n3);
  if (mid_residual_.size() != 1) mid_residual_.resize(1);

  float* s_new = spin_new_.data();
  float* s_prev = spin_prev_.data();
  float* s_mid = spin_mid_.data();
  float* f_mid = field_mid_.data();
  const double gamma_dt = gamma * time_step;
  const int grid = (N - 1) / SPIN_BLOCK_SIZE + 1;

  // First iterate: S_new = S_old (so S_mid = S_old, i.e. the scheme starts from
  // the field at the beginning of the step and then corrects it).
  CHECK(gpuMemcpy(s_new, g_spin, sizeof(float) * n3, gpuMemcpyDeviceToDevice));

  int iterations = 0;
  double residual = 0.0;
  bool converged = false;
  for (int it = 1; it <= midpoint_max_iterations; ++it) {
    iterations = it;
    CHECK(gpuMemcpy(s_prev, s_new, sizeof(float) * n3, gpuMemcpyDeviceToDevice));
    gpu_midpoint_average<<<grid, SPIN_BLOCK_SIZE>>>(
      N, g_spin, g_spin + N, g_spin + 2 * N, s_new, s_new + N, s_new + 2 * N,
      s_mid, s_mid + N, s_mid + 2 * N);
    GPU_CHECK_KERNEL

    // Whole-system field at the midpoint: every atom has moved, so the field of
    // every atom must be recomputed before any spin is advanced again.
    provider->compute_field(s_mid, f_mid, N);
    ++midpoint_field_evaluations;
    GPU_CHECK_KERNEL

    gpu_midpoint_update<<<grid, SPIN_BLOCK_SIZE>>>(
      N, g_spin, g_spin + N, g_spin + 2 * N,
      s_mid, s_mid + N, s_mid + 2 * N,
      f_mid, f_mid + N, f_mid + 2 * N,
      (double)bex, (double)bey, (double)bez, gamma_dt,
      s_new, s_new + N, s_new + 2 * N);
    GPU_CHECK_KERNEL

    gpu_zero_one<<<1, 1>>>(mid_residual_.data());
    gpu_max_abs_diff<<<grid, SPIN_BLOCK_SIZE>>>(static_cast<int>(n3), s_new, s_prev,
                                                mid_residual_.data());
    GPU_CHECK_KERNEL
    CHECK(gpuStreamSynchronize(0));
    mid_residual_.copy_to_host(&residual);
    if (residual <= midpoint_tolerance) {
      converged = true;
      break;
    }
  }

  if (!converged) {
    fprintf(stderr,
            "Spin Error://n"
            "    implicit-midpoint spin update did not converge: residual %.3e\n"
            "    after %d iterations (tolerance %.3e).  Refusing to fall back to\n"
            "    the frozen field -- that would silently drop the whole point of\n"
            "    the scheme.  Either the fixed point is contracting too slowly\n"
            "    (smaller time step / more iterations) or the tolerance is below\n"
            "    the float32 floor of the iterate itself (~1.2e-7 at |S| = 1).\n",
            residual, midpoint_max_iterations, midpoint_tolerance);
    std::exit(4);
  }

  // Only now, with the accepted iterate in hand, is the F16 work budget
  // advanced -- and only for that update, paired with the midpoint field that
  // produced it.  Trial iterations leave W_rot/W_ext untouched.
  if (work_.size() >= 3) {
    gpu_midpoint_work<<<grid, SPIN_BLOCK_SIZE>>>(
      N, g_spin, g_spin + N, g_spin + 2 * N,
      s_new, s_new + N, s_new + 2 * N,
      f_mid, f_mid + N, f_mid + 2 * N,
      (double)bex, (double)bey, (double)bez,
      work_.data(), work_.data() + 2);
    GPU_CHECK_KERNEL
  }

  CHECK(gpuMemcpy(g_spin, s_new, sizeof(float) * n3, gpuMemcpyDeviceToDevice));
  ++midpoint_steps;
  if (iterations > midpoint_max_iterations_used) {
    midpoint_max_iterations_used = iterations;
  }
  if (residual > midpoint_max_residual) midpoint_max_residual = residual;
}

// S08A: stochastic implicit midpoint (finite T, alpha > 0).
// One noise draw per step, held fixed for every iteration.  The deterministic
// field is recomputed at the midpoint iterate on every pass, exactly like the
// S05A conservative solver, because that is the whole point of the scheme: the
// field frozen at the beginning of the step is a first-order approximation that
// the midpoint rule is supposed to remove.
// Failure policy: the iteration is not allowed to fall back to the frozen
// field.  If it does not reach `midpoint_tolerance` within
// `midpoint_max_iterations` the last iterate is kept (it is still a midpoint
// consistent update, not the old field), the event is counted in
// sllg_nonconverged_steps and the worst residual is kept in sllg_max_residual.
// A stochastic run must not be killed by one stiff step; a *systematic* failure
// shows up as a non-zero counter, which is what the acceptance test reads.
void Spin::propagate_stochastic_midpoint(
  const double time_step,
  float* g_spin,
  const int N,
  const float bex, const float bey, const float bez,
  const float* bx, const float* by, const float* bz,
  SpinFieldProvider* provider,
  gpuStream_t stream)
{
  if (provider == nullptr) {
    fprintf(stderr,
            "Spin Error://n"
            "    The stochastic implicit-midpoint update needs a field provider\n"
            "    to re-evaluate -dE/dS at the midpoint spins.  Without one the\n"
            "    update would silently use the field frozen at the beginning of\n"
            "    the step, which is the error this scheme exists to remove.\n");
    std::exit(4);
  }
  if (!(time_step > 0.0)) {
    fprintf(stderr, "Spin Error://n    non-positive time step in the stochastic "
                    "spin update: %g\n", time_step);
    std::exit(4);
  }
  CHECK(gpuStreamSynchronize(stream));

  const size_t n3 = static_cast<size_t>(N) * 3;
  if (spin_new_.size() != n3) spin_new_.resize(n3);
  if (spin_prev_.size() != n3) spin_prev_.resize(n3);
  if (spin_mid_.size() != n3) spin_mid_.resize(n3);
  if (field_mid_.size() != n3) field_mid_.resize(n3);
  if (mid_residual_.size() != 1) mid_residual_.resize(1);

  float* s_new = spin_new_.data();
  float* s_prev = spin_prev_.data();
  float* s_mid = spin_mid_.data();
  float* f_mid = field_mid_.data();
  const double gamma_dt = gamma * time_step;
  const int grid = (N - 1) / SPIN_BLOCK_SIZE + 1;

  CHECK(gpuMemcpy(s_new, g_spin, sizeof(float) * n3, gpuMemcpyDeviceToDevice));

  int iterations = 0;
  double residual = 0.0;
  bool converged = false;
  for (int it = 1; it <= midpoint_max_iterations; ++it) {
    iterations = it;
    CHECK(gpuMemcpy(s_prev, s_new, sizeof(float) * n3, gpuMemcpyDeviceToDevice));
    gpu_midpoint_average<<<grid, SPIN_BLOCK_SIZE, 0, stream>>>(
      N, g_spin, g_spin + N, g_spin + 2 * N, s_new, s_new + N, s_new + 2 * N,
      s_mid, s_mid + N, s_mid + 2 * N);
    GPU_CHECK_KERNEL

    provider->compute_field(s_mid, f_mid, N);
    ++sllg_field_evaluations;
    GPU_CHECK_KERNEL

    gpu_midpoint_llg_update<<<grid, SPIN_BLOCK_SIZE, 0, stream>>>(
      N, g_spin, g_spin + N, g_spin + 2 * N,
      s_mid, s_mid + N, s_mid + 2 * N,
      f_mid, f_mid + N, f_mid + 2 * N,
      bx, by, bz,
      (double)bex, (double)bey, (double)bez, gamma_dt, alpha,
      s_new, s_new + N, s_new + 2 * N);
    GPU_CHECK_KERNEL

    gpu_zero_one<<<1, 1, 0, stream>>>(mid_residual_.data());
    gpu_max_abs_diff<<<grid, SPIN_BLOCK_SIZE, 0, stream>>>(
      static_cast<int>(n3), s_new, s_prev, mid_residual_.data());
    GPU_CHECK_KERNEL
    CHECK(gpuStreamSynchronize(stream));
    mid_residual_.copy_to_host(&residual);
    if (residual <= midpoint_tolerance) {
      converged = true;
      break;
    }
  }

  ++sllg_steps;
  if (iterations > sllg_max_iterations_used) sllg_max_iterations_used = iterations;
  if (residual > sllg_max_residual) sllg_max_residual = residual;
  if (!converged) {
    ++sllg_nonconverged_steps;
    if (sllg_nonconverged_steps == 1) {
      printf("NOTE: the stochastic midpoint iteration reached its iteration cap "
             "(residual %.3e > tolerance %.3e) at step %lld.  The last iterate "
             "is kept -- the frozen-field map is NOT used as a fallback.  "
             "Further occurrences are only counted.\n",
             residual, midpoint_tolerance, sllg_steps);
    }
  }

  if (work_.size() >= 3) {
    gpu_midpoint_work_stoch<<<grid, SPIN_BLOCK_SIZE, 0, stream>>>(
      N, g_spin, g_spin + N, g_spin + 2 * N,
      s_new, s_new + N, s_new + 2 * N,
      f_mid, f_mid + N, f_mid + 2 * N,
      bx, by, bz,
      (double)bex, (double)bey, (double)bez,
      work_.data(), work_.data() + 1, work_.data() + 2);
    GPU_CHECK_KERNEL
  }

  CHECK(gpuMemcpy(g_spin, s_new, sizeof(float) * n3, gpuMemcpyDeviceToDevice));
  project_onto_norm_anchor(g_spin, N, stream);
}

// S08A: constraint projection plus the diagnostic that says whether the map
// itself conserves |S|.  Runs on the caller's stream and never synchronises, so
// the production hot loop pays one kernel and no host round trip; the
// accumulated maximum is read on demand by sllg_pre_projection_norm_error().
void Spin::project_onto_norm_anchor(float* g_spin, const int N, gpuStream_t stream)
{
  const int grid = (N - 1) / SPIN_BLOCK_SIZE + 1;
  if (norm_anchor_.size() != static_cast<size_t>(N)) {
    norm_anchor_.resize(static_cast<size_t>(N));
    gpu_zero_float<<<grid, SPIN_BLOCK_SIZE, 0, stream>>>(N, norm_anchor_.data());
    GPU_CHECK_KERNEL
  }
  if (sllg_scalar_.size() != 1) {
    sllg_scalar_.resize(1);
    gpu_zero_one<<<1, 1, 0, stream>>>(sllg_scalar_.data());
    GPU_CHECK_KERNEL
  }
  gpu_project_anchor<<<grid, SPIN_BLOCK_SIZE, 0, stream>>>(
    N, g_spin, g_spin + N, g_spin + 2 * N, norm_anchor_.data(),
    sllg_scalar_.data());
  GPU_CHECK_KERNEL
}

double Spin::sllg_pre_projection_norm_error(gpuStream_t stream)
{
  if (sllg_scalar_.size() != 1) return 0.0;
  double v = 0.0;
  CHECK(gpuStreamSynchronize(stream));
  sllg_scalar_.copy_to_host(&v);
  return v;
}

// ---- S08B: restart state for the magnetic subsystem -----------------------
// restart.xyz is an xyz file: it can carry (R, v, S) and nothing else.  Two
// pieces of state that decide the next step are not in it:
//   * the per-atom curand state.  The thermal field is a stateful stream: after
//     k steps the stream is at a position that is *not* recoverable from
//     `seed` (re-seeding starts at step 0).  A restart that only re-seeds
//     therefore produces a different trajectory with the right distribution --
//     which is exactly the failure mode an "average magnetisation looks close"
//     test would miss;
//   * the per-atom |S| anchor used by the constraint projection.
// The file also stores identity fields so that a stale or mismatched file is
// refused instead of silently accepted: natoms, a hash of nep.txt, gamma,
// alpha, T, dt, scheme and the step counters.  Writing is atomic (tmp + rename)
// so a crash mid-dump cannot leave a half file that a later run would load.
namespace {
const char SPIN_RESTART_MAGIC[16] = {'G', 'P', 'U', 'M', 'D', 'S', 'P', 'I',
                                     'N', 'R', 'S', 'T', '0', '1', 0, 0};
const uint32_t SPIN_RESTART_VERSION = 1u;

uint64_t fnv1a_64(const void* data, size_t n, uint64_t h)
{
  const unsigned char* p = static_cast<const unsigned char*>(data);
  for (size_t i = 0; i < n; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 1099511628211ULL;
  }
  return h;
}

uint64_t file_fnv1a_64(const char* path)
{
  FILE* f = fopen(path, "rb");
  if (f == nullptr) return 0;
  uint64_t h = 1469598103934665603ULL;
  std::vector<unsigned char> buf(1 << 16);
  size_t n = 0;
  while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) {
    h = fnv1a_64(buf.data(), n, h);
  }
  fclose(f);
  return h;
}

template <typename T>
void append_bytes(std::vector<unsigned char>& out, const T& value)
{
  const unsigned char* p = reinterpret_cast<const unsigned char*>(&value);
  out.insert(out.end(), p, p + sizeof(T));
}

bool read_bytes(const std::vector<unsigned char>& in, size_t& off, void* dst, size_t n)
{
  if (off + n > in.size()) return false;
  std::memcpy(dst, in.data() + off, n);
  off += n;
  return true;
}
} // namespace

bool Spin::save_restart_state(const char* path, const long long physics_step,
                              const double global_time)
{
  if (!enabled || !has_spin || number_of_atoms <= 0) return false;
  const int N = number_of_atoms;
  const bool has_rng = temperature > 0.0 &&
                       states_.size() == static_cast<size_t>(N);

  std::vector<float> anchor(static_cast<size_t>(N), 1.0f);
  bool has_anchor = (norm_anchor_.size() == static_cast<size_t>(N));
  if (has_anchor) {
    CHECK(gpuStreamSynchronize(0));
    norm_anchor_.copy_to_host(anchor.data());
  }
  std::vector<gpurandState> rng;
  if (has_rng) {
    rng.resize(static_cast<size_t>(N));
    CHECK(gpuStreamSynchronize(0));
    states_.copy_to_host(rng.data());
  }

  const uint64_t model_hash = file_fnv1a_64("nep.txt");
  std::vector<unsigned char> buf;
  buf.reserve(256 + 4 * N + rng.size() * sizeof(gpurandState));
  buf.insert(buf.end(), SPIN_RESTART_MAGIC, SPIN_RESTART_MAGIC + 16);
  append_bytes(buf, SPIN_RESTART_VERSION);
  append_bytes(buf, static_cast<uint32_t>(N));
  append_bytes(buf, model_hash);
  append_bytes(buf, gamma);
  append_bytes(buf, alpha);
  append_bytes(buf, temperature);
  append_bytes(buf, time_step_fs);
  append_bytes(buf, static_cast<int32_t>(sllg_scheme));
  append_bytes(buf, static_cast<int32_t>(has_rng ? 1 : 0));
  append_bytes(buf, static_cast<int32_t>(has_anchor ? 1 : 0));
  append_bytes(buf, static_cast<int64_t>(sllg_steps));
  append_bytes(buf, static_cast<int64_t>(physics_step));
  append_bytes(buf, global_time);
  buf.insert(buf.end(), reinterpret_cast<const unsigned char*>(anchor.data()),
             reinterpret_cast<const unsigned char*>(anchor.data()) +
               anchor.size() * sizeof(float));
  if (has_rng) {
    buf.insert(buf.end(), reinterpret_cast<const unsigned char*>(rng.data()),
               reinterpret_cast<const unsigned char*>(rng.data()) +
                 rng.size() * sizeof(gpurandState));
  }
  const uint64_t payload = fnv1a_64(buf.data(), buf.size(), 1469598103934665603ULL);
  append_bytes(buf, payload);

  const std::string tmp = std::string(path) + ".tmp";
  FILE* f = fopen(tmp.c_str(), "wb");
  if (f == nullptr) {
    printf("S08B: cannot write %s\n", tmp.c_str());
    return false;
  }
  const size_t written = fwrite(buf.data(), 1, buf.size(), f);
  fflush(f);
  fclose(f);
  if (written != buf.size()) {
    printf("S08B: short write on %s\n", tmp.c_str());
    return false;
  }
  remove(path);
  if (rename(tmp.c_str(), path) != 0) {
    printf("S08B: cannot rename %s -> %s\n", tmp.c_str(), path);
    return false;
  }
  char hex[17];
  snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(payload));
  restart_digest_ = hex;
  restart_physics_step_ = physics_step;
  return true;
}

bool Spin::load_restart_state(const char* path)
{
  if (!enabled || !has_spin) return false;
  FILE* f = fopen(path, "rb");
  if (f == nullptr) {
    printf("Spin Error:\n    spin_llg restart= was given but '%s' cannot be "
           "opened.\n", path);
    std::exit(4);
  }
  std::vector<unsigned char> in;
  unsigned char chunk[1 << 16];
  size_t n = 0;
  while ((n = fread(chunk, 1, sizeof chunk, f)) > 0) in.insert(in.end(), chunk, chunk + n);
  fclose(f);

  size_t off = 0;
  char magic[16];
  uint32_t version = 0, natoms = 0;
  uint64_t model_hash = 0, payload = 0;
  double s_gamma = 0, s_alpha = 0, s_temp = 0, s_dt = 0, s_time = 0;
  int32_t scheme = 0, has_rng = 0, has_anchor = 0;
  int64_t s_steps = 0, s_physics_step = 0;
  bool ok = read_bytes(in, off, magic, 16) &&
            read_bytes(in, off, &version, sizeof version) &&
            read_bytes(in, off, &natoms, sizeof natoms) &&
            read_bytes(in, off, &model_hash, sizeof model_hash) &&
            read_bytes(in, off, &s_gamma, sizeof s_gamma) &&
            read_bytes(in, off, &s_alpha, sizeof s_alpha) &&
            read_bytes(in, off, &s_temp, sizeof s_temp) &&
            read_bytes(in, off, &s_dt, sizeof s_dt) &&
            read_bytes(in, off, &scheme, sizeof scheme) &&
            read_bytes(in, off, &has_rng, sizeof has_rng) &&
            read_bytes(in, off, &has_anchor, sizeof has_anchor) &&
            read_bytes(in, off, &s_steps, sizeof s_steps) &&
            read_bytes(in, off, &s_physics_step, sizeof s_physics_step) &&
            read_bytes(in, off, &s_time, sizeof s_time);
  if (!ok) {
    printf("Spin Error:\n    '%s' is too short to be a spin restart file.\n", path);
    std::exit(4);
  }
  const size_t expect_payload = off;
  const size_t need = expect_payload + sizeof(float) * natoms +
                      (has_rng ? sizeof(gpurandState) * natoms : 0);
  if (in.size() < need + sizeof(uint64_t)) {
    printf("Spin Error:\n    '%s' is truncated (need %zu bytes, have %zu).\n",
           path, need + sizeof(uint64_t), in.size());
    std::exit(4);
  }
  const uint64_t stored =
    *reinterpret_cast<const uint64_t*>(in.data() + need);
  const uint64_t recomputed = fnv1a_64(in.data(), need, 1469598103934665603ULL);

  auto fail = [&](const char* what) {
    printf("Spin Error:\n    spin restart file '%s' rejected: %s\n", path, what);
    std::exit(4);
  };
  if (std::memcmp(magic, SPIN_RESTART_MAGIC, 16) != 0) fail("bad magic");
  if (version != SPIN_RESTART_VERSION) fail("unsupported schema version");
  if (stored != recomputed) fail("payload checksum mismatch (file corrupted)");
  if (static_cast<int>(natoms) != number_of_atoms) fail("different atom count");
  const double tol = 1e-12;
  if (std::fabs(s_gamma - gamma) > tol * std::fmax(1.0, std::fabs(gamma)))
    fail("gamma differs from the run");
  if (std::fabs(s_alpha - alpha) > tol * std::fmax(1.0, std::fabs(alpha)))
    fail("alpha differs from the run");
  if (std::fabs(s_temp - temperature) > tol * std::fmax(1.0, temperature))
    fail("temperature differs from the run");
  if (std::fabs(s_dt - time_step_fs) > 1e-12 * std::fmax(1.0, time_step_fs))
    fail("time step differs from the run");
  if (scheme != sllg_scheme) fail("integrator scheme differs from the run");
  const uint64_t model_now = file_fnv1a_64("nep.txt");
  if (model_hash != 0 && model_now != 0 && model_hash != model_now)
    fail("nep.txt differs from the one that wrote the checkpoint");

  // ---- apply -----------------------------------------------------------
  if (has_anchor) {
    std::vector<float> anchor(static_cast<size_t>(natoms));
    size_t o = expect_payload;
    read_bytes(in, o, anchor.data(), anchor.size() * sizeof(float));
    norm_anchor_.resize(static_cast<size_t>(natoms));
    norm_anchor_.copy_from_host(anchor.data());
  }
  if (has_rng) {
    std::vector<gpurandState> rng(static_cast<size_t>(natoms));
    size_t o = expect_payload + sizeof(float) * natoms;
    read_bytes(in, o, rng.data(), rng.size() * sizeof(gpurandState));
    states_.resize(static_cast<size_t>(natoms));
    states_.copy_from_host(rng.data());
  }
  sllg_steps = s_steps;
  restart_physics_step_ = s_physics_step;
  char hex[17];
  snprintf(hex, sizeof hex, "%016llx",
           static_cast<unsigned long long>(recomputed));
  restart_digest_ = hex;
  printf("S08B: spin restart state loaded from %s\n", path);
  printf("      physics step %lld, %lld SLLG steps, RNG %s, anchor %s, "
         "digest %016llx\n", static_cast<long long>(s_physics_step),
         static_cast<long long>(s_steps), has_rng ? "restored" : "n/a",
         has_anchor ? "restored" : "n/a",
         static_cast<unsigned long long>(recomputed));
  return true;
}

// ---- F16: spin work / energy accounting -----------------------------------
void Spin::reset_work(gpuStream_t stream)
{
  if (work_.size() < 3) return;
  // Deliberately the *synchronous* fill: src/spin/spin.cu is a shared
  // translation unit and the main-tree GPU_Vector has no fill_async(), so
  // calling it here would break the gpumd build.  reset_work() is only called
  // once per run (never per step), so the stream drain costs nothing.
  CHECK(gpuStreamSynchronize(stream));
  work_.fill(0.0);
}

void Spin::read_work(double w[3], gpuStream_t stream)
{
  w[0] = w[1] = w[2] = 0.0;
  if (work_.size() < 3) return;
  // The accumulators are written with device-side atomics on `stream`, so the
  // stream must be drained before the host reads them.
  CHECK(gpuStreamSynchronize(stream));
  std::vector<double> tmp(3, 0.0);
  work_.copy_to_host(tmp.data());
  w[0] = tmp[0];
  w[1] = tmp[1];
  w[2] = tmp[2];
}

void Spin::print_work(const char* tag, gpuStream_t stream)
{
  if (!enabled || !has_spin) return;
  double w[3];
  read_work(w, stream);
  printf(
    "    spin work budget [%s]:  W_rot = %+.6g eV  W_noise = %+.6g eV  "
    "W_ext = %+.6g eV  ->  dU = %+.6g eV\n",
    tag, w[0], w[1], w[2], w[0] + w[1] + w[2]);
  // Round 15 (P110): W_noise/|W_rot| is NOT an equilibrium criterion.
  // |W_rot| contains a geometric part gamma*alpha*dt*|B|^2 that grows with dt
  // and does not depend on T, while W_noise is the (smaller) part of the same
  // relaxation that is driven by the thermal field.  Their ratio therefore
  // approaches 1 only as dt -> 0 or alpha -> infinity, and it *looks* like a
  // fluctuation-dissipation deficit at small alpha even when the dynamics is
  // exact.  Verified on an interacting Heisenberg model against Metropolis MC:
  // the exact Hamiltonian energy drift is 58x smaller than this diagnostic's
  // bias, and M matches the MC for every alpha.
  if (w[0] != 0.0) {
    printf("    spin work: W_noise/|W_rot| = %.3f  (diagnostic only -- biased by "
           "O(gamma dt |B|/alpha); judge equilibrium by a stationary potential "
           "energy, not by this ratio)\n",
           w[1] / fabs(w[0]));
  }
  fflush(stdout);
}

void Spin::report(const float* g_spin, const float* g_torque, const int number_of_atoms_in)
{
  const int N = number_of_atoms_in;
  if (!enabled || !has_spin || N <= 0) return;
  std::vector<float> s(static_cast<size_t>(N) * 3, 0.0f);
  CHECK(gpuMemcpy(
    s.data(), g_spin, sizeof(float) * static_cast<size_t>(N) * 3, gpuMemcpyDeviceToHost));
  std::vector<float> t(static_cast<size_t>(N) * 3, 0.0f);
  CHECK(gpuMemcpy(
    t.data(), g_torque, sizeof(float) * static_cast<size_t>(N) * 3, gpuMemcpyDeviceToHost));
  double s_sum = 0.0;
  double t_sum = 0.0;
  double m[3] = {0.0, 0.0, 0.0};
  for (int n = 0; n < N; ++n) {
    s_sum += sqrt(
      (double)s[n] * s[n] + (double)s[n + N] * s[n + N] + (double)s[n + 2 * N] * s[n + 2 * N]);
    t_sum += sqrt(
      (double)t[n] * t[n] + (double)t[n + N] * t[n + N] + (double)t[n + 2 * N] * t[n + 2 * N]);
    m[0] += s[n];
    m[1] += s[n + N];
    m[2] += s[n + 2 * N];
  }
  const double mnorm = sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
  const double mean_s = s_sum / N;
  const double mean_b = t_sum / N;
  // Only the TRANSVERSE part of the effective field drives the dynamics: the
  // longitudinal part is annihilated by S x (. ) in both LLG terms.  It is also
  // where the fitted |S| channel (Smag) and the constraint part of a cDFT
  // magnetic-force label live, so it can dominate the raw |torque| and would
  // make every time-scale diagnostic below look wrong.  Use
  //     |B_perp| = |S x B| / |S| .
  double t2_sum = 0.0;
  for (int n = 0; n < N; ++n) {
    const double sx = (double)s[n], sy = (double)s[n + N], sz = (double)s[n + 2 * N];
    const double bx = (double)t[n], by = (double)t[n + N], bz = (double)t[n + 2 * N];
    const double cx = sy * bz - sz * by;
    const double cy = sz * bx - sx * bz;
    const double cz = sx * by - sy * bx;
    const double sn = sqrt(sx * sx + sy * sy + sz * sz);
    const double bperp = (sn > 1.0e-12) ? sqrt(cx * cx + cy * cy + cz * cz) / sn : 0.0;
    t2_sum += bperp * bperp;
  }
  const double rms_b = sqrt(t2_sum / N);

  printf(
    "    spin: mean |S| = %.6f, mean |torque| = %.6e, rms |B_perp| = %.6e, "
    "|sum S| / sum|S| = %.6f\n",
    mean_s, mean_b, rms_b, mnorm / std::max(s_sum, 1e-30));

  // ---- Round 11 (P86): magnon time-scale diagnostics ----------------------
  // Why this exists: kappa_mm measured with SLLG came out ~0 for both Fe and
  // CrN, in contradiction with the literature (Zhou et al., PRB 101, 224303
  // (2020) find a sizeable magnon conductivity in bcc Fe).  Two mechanisms
  // can suppress it, and both are pure numbers that can be printed here:
  //  (a) the precession period T_prec = 2 pi / (gamma |B|) grows like 1/gamma.
  //      With gamma = 0.01 (300x below the physical 3.04) the spin subsystem
  //      evolves 300x more slowly than reality, so a ns trajectory samples
  //      ~300x fewer magnon periods than it should.
  //  (b) the SLLG thermal field has amplitude
  //          sigma = sqrt(2 alpha kB T / (gamma dt |S| (1 + alpha^2)))
  //          (see gpu_fill_thermal_field),
  //          and dS/dt_noise ~ gamma |S| sigma ~ sqrt(gamma), whereas the
  //          physical precession rate
  //          dS/dt_prec ~ gamma |S| |B| ~ gamma.  For gamma << 1 the noise
  //          therefore DOMINATES dS/dt by a factor 1/sqrt(gamma): it is white,
  //          so it inflates rms|J_m| but contributes nothing to the
  //          Green-Kubo integral.  That is the mechanism that drives
  //          kappa_mm -> 0 while rms|J_m| stays finite (and heavy tailed).
  if (gamma > 0.0 && rms_b > 0.0) {
    const double t_prec_fs = 6.28318530717958647692 / (gamma * rms_b);
    printf("    spin dynamics: T_prec = 2 pi/(gamma |B|) = %.4g fs "
           "(gamma = %g, rms|B_perp| = %.4g eV/uB)\n",
           t_prec_fs, gamma, rms_b);
  }
  if (temperature > 0.0 && alpha > 0.0 && gamma > 0.0 && mean_s > 0.0) {
    const double sigma = sqrt(2.0 * alpha * SPIN_KB_EV_PER_K * temperature
                              / (gamma * time_step_fs * mean_s
                                 * (1.0 + alpha * alpha)));
    printf("    spin dynamics: SLLG field rms sigma = %.4g eV/uB "
           "= %.3g x rms|B_perp|\n",
           sigma, sigma / std::max(rms_b, 1e-30));
    // F21: the F16 budget splits dU into -b.dS + b_noise.dS.  When the noise
    // field dominates the exchange field both terms are ~sigma^2/|B|^2 times the
    // true magnetic energy change and their difference loses its digits (Fe,
    // gamma=0.01: W_rot -4.47e6, W_noise +4.47e6, w_net -2.3e3 eV, dU +5.9 eV).
    if (sigma / std::max(rms_b, 1e-30) > 3.0) {
      printf("    spin dynamics: NOTE sigma/rms|B_perp| = %.2g > 3, so the F16 "
             "spin-work budget\n"
             "                   (spin_work.out, columns 2-4) is dominated by "
             "the damping of the\n"
             "                   noise field: use only its SIGN and TREND, "
             "never its absolute\n"
             "                   value.  Raise gamma towards 3.04 to make the "
             "splitting meaningful.\n",
             sigma / std::max(rms_b, 1e-30));
    }
    // Round 15 (P111): the Gilbert time sets the *equilibration* time, and it
    // is the only number that decides whether a run has reached equilibrium.
    // It is NOT an equilibrium parameter: alpha cancels out of the stationary
    // distribution (proved on an interacting Heisenberg model against
    // Metropolis MC, p110_sllg_interacting.py: M/M_MC = 1.007 at alpha = 0.01
    // and 0.997 at alpha = 0.03).
    if (rms_b > 0.0) {
      const double t_gil = 1.0 / (gamma * alpha * rms_b * time_step_fs);
      printf("    spin dynamics: T_Gilbert = 1/(gamma alpha |B| dt) = %.4g steps "
             "= %.4g ps\n",
             t_gil, t_gil * time_step_fs * 1.0e-3);
      // Round 15 (P111b): the magnetisation relaxes TEN times more slowly than
      // the single-spin Gilbert time, tau_M ~ 10 tau_G.  Quoting 50 tau_G only
      // buys 3-10 independent samples, so the protocol for any magnetisation
      // average is >= 50 tau_M, i.e. >= 500 tau_G.  (For non-magnetic
      // observables 50 tau_G remains the right figure.)
      printf("    spin dynamics:   equilibration: N steps cover N/%.4g Gilbert times; "
             "a magnetisation\n"
             "                     average needs >= 50 tau_M = >= %.4g steps "
             "(tau_M ~ 10 tau_G, P111b)\n",
             t_gil, 500.0 * t_gil);
      // Round 15 (C24): accuracy is governed by x = gamma|B|dt/alpha, not by a
      // bare alpha threshold.  Model-side calibration (P110) gives <= 1.2 % error
      // for x <= 1.8, ~5 % at x = 3, >= 36 % for x >= 4.5.
      const double x_disc = gamma * rms_b * time_step_fs / alpha;
      printf("    spin dynamics:   discretisation x = gamma|B|dt/alpha = %.4g  "
             "(need x <~ 2; C24)  %s\n",
             x_disc,
             x_disc <= 2.0 ? "[OK]" : "[TOO COARSE -- reduce dt or raise alpha]");
    }
  }
}

void Spin::magnetization(const float* g_spin, const int number_of_atoms_in, double m[3]) const
{
  m[0] = m[1] = m[2] = 0.0;
  if (!enabled || !has_spin) return;
  const int N = number_of_atoms_in;
  if (N <= 0) return;
  std::vector<float> s(static_cast<size_t>(N) * 3, 0.0f);
  CHECK(gpuMemcpy(
    s.data(), g_spin, sizeof(float) * static_cast<size_t>(N) * 3, gpuMemcpyDeviceToHost));
  for (int n = 0; n < N; ++n) {
    m[0] += s[n];
    m[1] += s[n + N];
    m[2] += s[n + 2 * N];
  }
}
