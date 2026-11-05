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
The FIRE (fast inertial relaxation engine) minimizer
Reference: PhysRevLett 97, 170201 (2006)
           Computational Materials Science 175 (2020) 109584
------------------------------------------------------------------------------*/

#include "minimizer_fire.cuh"
#include "model/atom.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

namespace
{
__global__ void gpu_multiply(const int size, double a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = b[n] * a;
}

__global__ void gpu_vector_sum(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] + b[n];
}

__global__ void gpu_pairwise_product(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] * b[n];
}

void pairwise_product(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_pairwise_product<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}

__global__ void gpu_sum(const int size, double* a, double* result)
{
  int number_of_batches = (size - 1) / 1024 + 1;
  int tid = threadIdx.x;
  int n, batch;
  __shared__ double data[1024];
  data[tid] = 0.0;
  for (batch = 0; batch < number_of_batches; ++batch) {
    n = tid + batch * 1024;
    if (n < size)
      data[tid] += a[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      data[tid] += data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0)
    *result = data[0];
}

double sum(GPU_Vector<double>& a)
{
  double ret;
  GPU_Vector<double> result(1);
  gpu_sum<<<1, 1024>>>(a.size(), a.data(), result.data());
  result.copy_to_host(&ret);
  return ret;
}

double dot(GPU_Vector<double>& a, GPU_Vector<double>& b)
{
  GPU_Vector<double> temp(a.size());
  pairwise_product(a, b, temp);
  return sum(temp);
}

void scalar_multiply(const double& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = b.size();
  gpu_multiply<<<(size - 1) / 128 + 1, 128>>>(size, a, b.data(), c.data());
}

void vector_sum(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_vector_sum<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}
} // namespace

void Minimizer_FIRE::compute(
  Force& force,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& position_per_atom,
  std::vector<Group>& group)
{
  double next_dt;
  const int size = number_of_atoms_ * 3;
  int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;
  // create a velocity vector in GPU
  GPU_Vector<double> v(size, 0);
  GPU_Vector<double> temp1(size);
  GPU_Vector<double> temp2(size);
  bool spin_warned = false;

  printf("\nEnergy minimization started.\n");
  if (spin_minimization_) {
    printf("    magnetic moments are minimized together with the positions\n"
           "    (damped spin dynamics, torque tolerance %g eV/mu_B,\n"
           "     alpha_damp = %g, discrete_factor = %g).\n",
           spin_tolerance_, alpha_damp_, discrete_factor_);
  }

  for (int step = 0; step < number_of_steps_; ++step) {
    compute_force_at(force, box, atom, group, position_per_atom);
    calculate_force_square_max(atom.force_per_atom);
    const double force_max = sqrt(cpu_force_square_max_[0]);
    calculate_total_potential(atom.potential_per_atom);
    // V4: the same iteration also relaxes the magnetic moments; the convergence
    // test below requires both criteria.
    const double torque_max = spin_torque_max(atom);
    const bool force_ok = force_max < force_tolerance_;
    const bool torque_ok = !spin_minimization_ || torque_max <= spin_tolerance_;
    update_spin_step_scale(cpu_total_potential_[0], force_ok);

    // The pre-V4 print condition is kept verbatim when the moments are not
    // minimised, so that a lattice-only run stays byte-identical to r19.
    if (step == 0 || (step + 1) % base == 0 || (force_ok && torque_ok)) {
      if (spin_active(atom)) {
        printf(
          "    step %d: total_potential = %.10f eV, f_max = %.10f eV/A, "
          "w_perp_max = %.10f eV/mu_B.\n",
          step == 0 ? 0 : (step + 1),
          cpu_total_potential_[0],
          force_max,
          torque_max);
      } else {
        printf(
          "    step %d: total_potential = %.10f eV, f_max = %.10f eV/A.\n",
          step == 0 ? 0 : (step + 1),
          cpu_total_potential_[0],
          force_max);
      }
      if (force_ok && torque_ok) {
        if (spin_active(atom)) {
          printf(
            "  Converged at step %d: f_max = %.3e eV/A < %.1e AND "
            "w_perp_max = %.3e eV/mu_B < %.1e, E = %.10f eV.\n",
            step + 1, force_max, force_tolerance_, torque_max, spin_tolerance_,
            cpu_total_potential_[0]);
        }
        break;
      }
    }
    if (spin_active(atom) && force_ok && !torque_ok && !spin_warned) {
      printf(
        "  NOTE: the forces have converged but the magnetic moments have not\n"
        "        (w_perp_max = %.3e eV/mu_B > %.1e).  Continuing with the spin\n"
        "        relaxation; the lattice will keep moving while it does.\n",
        torque_max, spin_tolerance_);
      spin_warned = true;
    }

    P = dot(v, atom.force_per_atom);

    if (P > 0) {
      if (N_neg > N_min) {
        next_dt = dt * f_inc;
        if (next_dt < dt_max)
          dt = next_dt;
        alpha *= f_alpha;
      }
      N_neg++;
    } else {
      next_dt = dt * f_dec;
      if (next_dt > dt_min)
        dt = next_dt;
      alpha = alpha_start;
      // move position back
      scalar_multiply(-0.5 * dt, v, temp1);
      vector_sum(position_per_atom, temp1, position_per_atom);
      v.fill(0);
      N_neg = 0;
    }

    // md step
    // implicit Euler integration
    double F_modulus = sqrt(dot(atom.force_per_atom, atom.force_per_atom));
    double v_modulus = sqrt(dot(v, v));
    // dv = F/m*dt
    scalar_multiply(dt / m, atom.force_per_atom, temp2);
    vector_sum(v, temp2, v);
    scalar_multiply(1 - alpha, v, temp1);
    scalar_multiply(alpha * v_modulus / F_modulus, atom.force_per_atom, temp2);
    vector_sum(temp1, temp2, v);
    // dx = v*dt
    scalar_multiply(dt, v, temp1);
    vector_sum(position_per_atom, temp1, position_per_atom);

    // V4: one damped-spin step with the torque that was tested above.  It uses
    // the same trust-region angle all minimizers use, so the spin and the
    // lattice converge in one loop rather than in two nested ones.
    spin_relax_apply(atom, torque_max);
  }

  printf("Energy minimization finished.\n");
}