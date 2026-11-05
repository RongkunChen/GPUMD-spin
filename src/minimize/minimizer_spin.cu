/*
    Spin Route B, V4: spin degrees of freedom inside the energy minimizers.
    See minimizer_spin.cuh for the algorithm and the convergence criterion.
*/

#include "minimizer_spin.cuh"
#include "force/force.cuh"
#include "minimizer.cuh"
#include "model/atom.cuh"
#include "utilities/gpu_macro.cuh"
#include <cmath>

namespace
{

const int MS_BLOCK = 128;

__global__ void gpu_zero_max(double* p) { *p = 0.0; }

// Pass 1: tangential torque w_perp = w - s_hat (s_hat.w) and its maximum norm.
__global__ void gpu_spin_transverse_torque(
  const int N,
  const float* __restrict__ sx,
  const float* __restrict__ sy,
  const float* __restrict__ sz,
  const float* __restrict__ tx,
  const float* __restrict__ ty,
  const float* __restrict__ tz,
  float* __restrict__ wpx,
  float* __restrict__ wpy,
  float* __restrict__ wpz,
  double* __restrict__ g_max)
{
  __shared__ double sh[MS_BLOCK];
  const int tid = threadIdx.x;
  const int n = blockIdx.x * blockDim.x + tid;
  double wn = 0.0;
  if (n < N) {
    const double s0 = (double)sx[n], s1 = (double)sy[n], s2 = (double)sz[n];
    const double w0 = (double)tx[n], w1 = (double)ty[n], w2 = (double)tz[n];
    const double sn2 = s0 * s0 + s1 * s1 + s2 * s2;
    double p0 = w0, p1 = w1, p2 = w2;
    if (sn2 > 1.0e-24) {
      const double c = (w0 * s0 + w1 * s1 + w2 * s2) / sn2;
      p0 = w0 - c * s0;
      p1 = w1 - c * s1;
      p2 = w2 - c * s2;
    } else {
      p0 = p1 = p2 = 0.0;
    }
    wpx[n] = (float)p0;
    wpy[n] = (float)p1;
    wpz[n] = (float)p2;
    wn = sqrt(p0 * p0 + p1 * p1 + p2 * p2);
  }
  sh[tid] = wn;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off && sh[tid + off] > sh[tid]) sh[tid] = sh[tid + off];
    __syncthreads();
  }
  if (tid == 0 && sh[0] > 0.0) {
    atomicMax(
      reinterpret_cast<unsigned long long*>(g_max),
      static_cast<unsigned long long>(__double_as_longlong(sh[0])));
  }
}

// Pass 2: exact geodesic move along w_perp by the adaptive angle.
// theta = gain * |w_perp|, gain = alpha_damp * 2 pi / (kappa * |w|_max).
__global__ void gpu_spin_relax(
  const int N,
  const float* __restrict__ wpx,
  const float* __restrict__ wpy,
  const float* __restrict__ wpz,
  const double gain,
  float* __restrict__ sx,
  float* __restrict__ sy,
  float* __restrict__ sz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const double s0 = (double)sx[n], s1 = (double)sy[n], s2 = (double)sz[n];
  const double p0 = (double)wpx[n], p1 = (double)wpy[n], p2 = (double)wpz[n];
  const double sn = sqrt(s0 * s0 + s1 * s1 + s2 * s2);
  const double pn = sqrt(p0 * p0 + p1 * p1 + p2 * p2);
  if (sn < 1.0e-12 || pn < 1.0e-14) return;

  double theta = gain * pn;
  const double theta_cap = 0.5 * 3.14159265358979323846;
  if (theta > theta_cap) theta = theta_cap;
  if (theta < 0.0) return;

  const double c = cos(theta), s = sin(theta);
  const double u0 = p0 / pn, u1 = p1 / pn, u2 = p2 / pn;
  const double sh0 = s0 / sn, sh1 = s1 / sn, sh2 = s2 / sn;
  sx[n] = (float)(sn * (c * sh0 + s * u0));
  sy[n] = (float)(sn * (c * sh1 + s * u1));
  sz[n] = (float)(sn * (c * sh2 + s * u2));
}

} // namespace

double spin_torque_measure(
  const int number_of_atoms,
  float* spin,
  const float* torque,
  GPU_Vector<float>& w_perp,
  GPU_Vector<double>& w_max)
{
  if (number_of_atoms <= 0 || spin == nullptr || torque == nullptr) return 0.0;
  const size_t n3 = static_cast<size_t>(number_of_atoms) * 3;
  if (w_perp.size() != n3) w_perp.resize(n3);
  if (w_max.size() != 1) w_max.resize(1);

  const int grid = (number_of_atoms - 1) / MS_BLOCK + 1;
  gpu_zero_max<<<1, 1>>>(w_max.data());
  gpu_spin_transverse_torque<<<grid, MS_BLOCK>>>(
    number_of_atoms,
    spin, spin + number_of_atoms, spin + 2 * number_of_atoms,
    torque, torque + number_of_atoms, torque + 2 * number_of_atoms,
    w_perp.data(), w_perp.data() + number_of_atoms, w_perp.data() + 2 * number_of_atoms,
    w_max.data());
  GPU_CHECK_KERNEL

  double wmax = 0.0;
  w_max.copy_to_host(&wmax);
  return wmax;
}

void spin_relax_apply(
  const int number_of_atoms,
  float* spin,
  const double alpha_damp,
  const double discrete_factor,
  const double w_max,
  GPU_Vector<float>& w_perp)
{
  if (number_of_atoms <= 0 || spin == nullptr) return;
  if (w_max <= 0.0) return;
  const double kappa = (discrete_factor > 0.0) ? discrete_factor : 10.0;
  const double TWO_PI = 6.28318530717958647692;
  const double gain = alpha_damp * TWO_PI / (kappa * w_max);
  const int grid = (number_of_atoms - 1) / MS_BLOCK + 1;
  gpu_spin_relax<<<grid, MS_BLOCK>>>(
    number_of_atoms,
    w_perp.data(), w_perp.data() + number_of_atoms, w_perp.data() + 2 * number_of_atoms,
    gain, spin, spin + number_of_atoms, spin + 2 * number_of_atoms);
  GPU_CHECK_KERNEL
}

double spin_relax_step(
  const int number_of_atoms,
  float* spin,
  const float* torque,
  const double alpha_damp,
  const double discrete_factor,
  GPU_Vector<float>& w_perp,
  GPU_Vector<double>& w_max)
{
  const double wmax = spin_torque_measure(number_of_atoms, spin, torque, w_perp, w_max);
  spin_relax_apply(
    number_of_atoms, spin, alpha_damp, discrete_factor, wmax, w_perp);
  return wmax;
}

// Minimizer members (shared with the replica build, which never enables them).

void Minimizer::compute_force_at(
  Force& force, Box& box, Atom& atom, std::vector<Group>& group,
  GPU_Vector<double>& position_per_atom)
{
  const bool with_spin = spin_active(atom);
  if (with_spin) {
    force.compute(
      box, position_per_atom, atom.type, group,
      atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom,
      atom.velocity_per_atom, atom.mass,
      atom.position_image.size() > 0 ? atom.position_image.data() : nullptr,
      atom.spin.data(), atom.torque.data());
  } else {
    force.compute(
      box, position_per_atom, atom.type, group,
      atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
  }
}

bool Minimizer::spin_active(Atom& atom) const
{
  return spin_minimization_ && atom.has_spin &&
         atom.spin.size() == static_cast<size_t>(number_of_atoms_) * 3 &&
         atom.torque.size() == static_cast<size_t>(number_of_atoms_) * 3;
}

double Minimizer::spin_torque_max(Atom& atom)
{
  if (!spin_active(atom)) return 0.0;
  return spin_torque_measure(
    number_of_atoms_, atom.spin.data(), atom.torque.data(), spin_w_perp_, spin_w_max_);
}

void Minimizer::spin_relax_apply(Atom& atom, const double w_max)
{
  if (!spin_active(atom)) return;
  ::spin_relax_apply(
    number_of_atoms_, atom.spin.data(), alpha_damp_ * spin_step_scale_, discrete_factor_,
    w_max, spin_w_perp_);
}

void Minimizer::update_spin_step_scale(
  const double total_potential, const bool lattice_converged)
{
  if (!spin_minimization_) return;
  if (spin_have_last_energy_) {
    if (total_potential > spin_last_energy_ + 1.0e-12) {
      // A step that raised the energy was outside the trust region.
      spin_step_scale_ *= 0.5;
      if (spin_step_scale_ < 1.0e-3) spin_step_scale_ = 1.0e-3;
    } else if (lattice_converged) {
      // Grow only once the lattice has stopped moving: while it is still
      // relaxing, the energy drop is mostly the lattice's, and reading it as
      // "the spin step was fine" would ramp the gain back to 1 too early.
      spin_step_scale_ *= 1.1;
      if (spin_step_scale_ > 1.0) spin_step_scale_ = 1.0;
    }
  }
  spin_last_energy_ = total_potential;
  spin_have_last_energy_ = true;
}
