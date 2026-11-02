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

/*-----------------------------------------------------------------------------------------------100
Calculate the potential part of the per-atom heat current

Reference:

Zheyong Fan, Luiz Felipe C. Pereira, Hui-Qiong Wang, Jin-Cheng Zheng, Davide Donadio, and Ari Harju.
Force and heat current formulas for many-body potentials in molecular dynamics simulations with
applications to thermal conductivity calculations. Phys. Rev. B 92, 094301, (2015).
https://doi.org/10.1103/PhysRevB.92.094301
--------------------------------------------------------------------------------------------------*/

#include "compute_heat.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <cstring>

namespace
{
// Round 9 (task 48): the spin-resolved heat current.
// Layout of both `virial_per_atom` and `spin_flux_A` (9N doubles):
//   0=xx 1=yy 2=zz 3=xy 4=xz 5=yz 6=yx 7=zx 8=zy
// Layout of the 5N output (unchanged GPUMD convention):
//   0=jx_in 1=jx_out 2=jy_in 3=jy_out 4=jz
static __global__ void gpu_compute_heat_spin(
  const int N,
  const double* __restrict__ g_virial,
  const double* __restrict__ g_Aspin,
  const double* __restrict__ vx,
  const double* __restrict__ vy,
  const double* __restrict__ vz,
  const float* __restrict__ g_spin,
  const float* __restrict__ g_torque,
  const double gamma,
  const double alpha,
  const double b_ex,
  const double b_ey,
  const double b_ez,
  const double fs_to_natural,
  double* jx_in,
  double* jx_out,
  double* jy_in,
  double* jy_out,
  double* jz,
  double* jpx_in,
  double* jpx_out,
  double* jpy_in,
  double* jpy_out,
  double* jpz,
  double* jmx_in,
  double* jmx_out,
  double* jmy_in,
  double* jmy_out,
  double* jmz)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n >= N) return;

  const size_t Ns = (size_t)N;
  const double v_x = vx[n], v_y = vy[n], v_z = vz[n];

  // ---- phonon (lattice) channel: J_p = sum_i W_i . v_i + e_i^spin v_i ----
  const double wxx = g_virial[n + Ns * 0];
  const double wyy = g_virial[n + Ns * 1];
  const double wzz = g_virial[n + Ns * 2];
  const double wxy = g_virial[n + Ns * 3];
  const double wxz = g_virial[n + Ns * 4];
  const double wyz = g_virial[n + Ns * 5];
  const double wyx = g_virial[n + Ns * 6];
  const double wzx = g_virial[n + Ns * 7];
  const double wzy = g_virial[n + Ns * 8];

  // e_i^spin = -1/2 S_i . B_i  (B_i = torque = -dE/dS_i).  For a Heisenberg
  // Hamiltonian this per-atom split sums to the exchange energy exactly.
  double espin = 0.0;
  double sx = 0.0, sy = 0.0, sz = 0.0;
  double bx = 0.0, by = 0.0, bz = 0.0;
  if (g_spin && g_torque) {
    sx = (double)g_spin[n];
    sy = (double)g_spin[n + N];
    sz = (double)g_spin[n + 2 * N];
    // e_i^spin uses the NEP torque only, i.e. the model energy U(R,S); the
    // Zeeman energy of `add_bfield` is an external potential and is not part
    // of the per-atom NEP energy that this convective term decomposes.
    bx = (double)g_torque[n];
    by = (double)g_torque[n + N];
    bz = (double)g_torque[n + 2 * N];
    espin = -0.5 * (sx * bx + sy * by + sz * bz);
  }

  const double px_in = (wxx + espin) * v_x + wxy * v_y;
  const double px_out = wxz * v_z;
  const double py_in = wyx * v_x + (wyy + espin) * v_y;
  const double py_out = wyz * v_z;
  const double pz = wzx * v_x + wzy * v_y + (wzz + espin) * v_z;

  // ---- magnon channel: J_m = sum_j A_j : dS_j/dt ------------------------
  double mx_in = 0.0, mx_out = 0.0, my_in = 0.0, my_out = 0.0, mz = 0.0;
  if (g_Aspin) {
    const double axx = g_Aspin[n + Ns * 0];
    const double ayy = g_Aspin[n + Ns * 1];
    const double azz = g_Aspin[n + Ns * 2];
    const double axy = g_Aspin[n + Ns * 3];
    const double axz = g_Aspin[n + Ns * 4];
    const double ayz = g_Aspin[n + Ns * 5];
    const double ayx = g_Aspin[n + Ns * 6];
    const double azx = g_Aspin[n + Ns * 7];
    const double azy = g_Aspin[n + Ns * 8];

    // deterministic LLG rate
    //   dS/dt = -gamma S x B - gamma alpha/|S| S x (S x B),  B = w + b_ext
    // B is the field the spins actually see, so the external drive of
    // `add_bfield` is included (it does real work); the SLLG noise is not.
    double rdx = 0.0, rdy = 0.0, rdz = 0.0;
    const double sn = sqrt(sx * sx + sy * sy + sz * sz);
    if (sn > 1.0e-12) {
      const double Bx = bx + b_ex;
      const double By = by + b_ey;
      const double Bz = bz + b_ez;
      // c1 = S x B
      const double c1x = sy * Bz - sz * By;
      const double c1y = sz * Bx - sx * Bz;
      const double c1z = sx * By - sy * Bx;
      // c2 = S x c1 = S x (S x B)
      const double c2x = sy * c1z - sz * c1y;
      const double c2y = sz * c1x - sx * c1z;
      const double c2z = sx * c1y - sy * c1x;
      const double pref = gamma * alpha / sn;
      // gamma is per femtosecond, so the rate above is mu_B/fs.  One natural
      // time unit is TIME_UNIT_CONVERSION fs, so a per-fs rate must be
      // MULTIPLIED by it to become a per-natural-time rate:
      //     dS/dt [mu_B/nattime] = dS/dt [mu_B/fs] * TIME_UNIT_CONVERSION .
      // After that, A:(dS/dt) is in eV*A/nattime, the same unit as the phonon
      // channel (W_i . v_i), and one single conversion factor in
      // dump_heat_flux.cu serves both channels.  See compute_heat.cuh.
      rdx = (-gamma * c1x - pref * c2x) * fs_to_natural;
      rdy = (-gamma * c1y - pref * c2y) * fs_to_natural;
      rdz = (-gamma * c1z - pref * c2z) * fs_to_natural;
    }
    mx_in = axx * rdx + axy * rdy;
    mx_out = axz * rdz;
    my_in = ayx * rdx + ayy * rdy;
    my_out = ayz * rdz;
    mz = azx * rdx + azy * rdy + azz * rdz;
  }

  jx_in[n] = px_in + mx_in;
  jx_out[n] = px_out + mx_out;
  jy_in[n] = py_in + my_in;
  jy_out[n] = py_out + my_out;
  jz[n] = pz + mz;

  if (jpx_in) {
    jpx_in[n] = px_in;
    jpx_out[n] = px_out;
    jpy_in[n] = py_in;
    jpy_out[n] = py_out;
    jpz[n] = pz;
  }
  if (jmx_in) {
    jmx_in[n] = mx_in;
    jmx_out[n] = mx_out;
    jmy_in[n] = my_in;
    jmy_out[n] = my_out;
    jmz[n] = mz;
  }
}
} // namespace

void compute_heat(
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& spin_flux_A,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<float>& spin,
  const GPU_Vector<float>& torque,
  const double spin_gamma,
  const double spin_alpha,
  const double b_ext[3],
  GPU_Vector<double>& heat_per_atom,
  GPU_Vector<double>* heat_phonon,
  GPU_Vector<double>* heat_magnon)
{
  const int N = velocity_per_atom.size() / 3;
  if (N <= 0) {
    return;
  }
  // An empty A buffer means "this run has no spin-resolved flux": the magnon
  // channel is then identically zero.
  const double* Aspin = (spin_flux_A.size() >= (size_t)N * 9) ? spin_flux_A.data() : nullptr;
  const double b_ex = (b_ext != nullptr) ? b_ext[0] : 0.0;
  const double b_ey = (b_ext != nullptr) ? b_ext[1] : 0.0;
  const double b_ez = (b_ext != nullptr) ? b_ext[2] : 0.0;
  // TIME_UNIT_CONVERSION is a host-side const, so it is read here and handed
  // to the kernel as a value.  It is used to multiply a per-fs spin rate and
  // turn it into a per-natural-time rate; see the unit note in
  // compute_heat.cuh.
  const double fs_to_natural = TIME_UNIT_CONVERSION;

  gpu_compute_heat_spin<<<(N - 1) / 128 + 1, 128>>>(
    N,
    virial_per_atom.data(),
    Aspin,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + 2 * N,
    spin.data(),
    torque.data(),
    spin_gamma,
    spin_alpha,
    b_ex,
    b_ey,
    b_ez,
    fs_to_natural,
    heat_per_atom.data(),
    heat_per_atom.data() + N,
    heat_per_atom.data() + N * 2,
    heat_per_atom.data() + N * 3,
    heat_per_atom.data() + N * 4,
    heat_phonon ? heat_phonon->data() : nullptr,
    heat_phonon ? heat_phonon->data() + N : nullptr,
    heat_phonon ? heat_phonon->data() + N * 2 : nullptr,
    heat_phonon ? heat_phonon->data() + N * 3 : nullptr,
    heat_phonon ? heat_phonon->data() + N * 4 : nullptr,
    heat_magnon ? heat_magnon->data() : nullptr,
    heat_magnon ? heat_magnon->data() + N : nullptr,
    heat_magnon ? heat_magnon->data() + N * 2 : nullptr,
    heat_magnon ? heat_magnon->data() + N * 3 : nullptr,
    heat_magnon ? heat_magnon->data() + N * 4 : nullptr);
  GPU_CHECK_KERNEL
}

namespace
{
static __global__ void gpu_compute_heat(
  const int N,
  const double* sxx,
  const double* sxy,
  const double* sxz,
  const double* syx,
  const double* syy,
  const double* syz,
  const double* szx,
  const double* szy,
  const double* szz,
  const double* vx,
  const double* vy,
  const double* vz,
  const float* g_spin,
  const float* g_torque,
  double* jx_in,
  double* jx_out,
  double* jy_in,
  double* jy_out,
  double* jz)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    // R17 / round-8: the CONVECTIVE part of the spin-energy flux.
    //   e_i^spin = -1/2 S_i . B_i,      B_i = torque = -dE/dS_i
    // For a quadratic (Heisenberg) spin Hamiltonian this per-atom decomposition
    // sums exactly to the total exchange energy; for the NEP it is the local
    // generalisation of it.  Like the (e_k + e_p) term of the convective
    // overload below, it enters the flux on the diagonal of the virial tensor.
    // NOTE: this is only the part of the magnon heat current that is reachable
    // from the quantities a NEP provides.  The CONDUCTIVE part -- the energy
    // handed from spin i to spin j -- needs dE_i/dS_j, which the NEP does not
    // expose (it only gives the total gradient dU/dS_i).  See REPORT
    // validation8, stage P72 / risk R17.
    double espin = 0.0;
    if (g_spin && g_torque) {
      espin = -0.5 * (double(g_spin[n]) * double(g_torque[n]) +
                      double(g_spin[n + N]) * double(g_torque[n + N]) +
                      double(g_spin[n + 2 * N]) * double(g_torque[n + 2 * N]));
    }
    jx_in[n] = (sxx[n] + espin) * vx[n] + sxy[n] * vy[n];
    jx_out[n] = sxz[n] * vz[n];
    jy_in[n] = syx[n] * vx[n] + (syy[n] + espin) * vy[n];
    jy_out[n] = syz[n] * vz[n];
    jz[n] = szx[n] * vx[n] + szy[n] * vy[n] + (szz[n] + espin) * vz[n];
  }
}
} // namespace

void compute_heat(
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& heat_per_atom)
{
  const int N = velocity_per_atom.size() / 3;

  // the virial tensor:
  // xx xy xz    0 3 4
  // yx yy yz    6 1 5
  // zx zy zz    7 8 2
  gpu_compute_heat<<<(N - 1) / 128 + 1, 128>>>(
    N,
    virial_per_atom.data(),
    virial_per_atom.data() + N * 3,
    virial_per_atom.data() + N * 4,
    virial_per_atom.data() + N * 6,
    virial_per_atom.data() + N * 1,
    virial_per_atom.data() + N * 5,
    virial_per_atom.data() + N * 7,
    virial_per_atom.data() + N * 8,
    virial_per_atom.data() + N * 2,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + 2 * N,
    nullptr, // no spin-energy convective term (no spin in this run)
    nullptr,
    heat_per_atom.data(),
    heat_per_atom.data() + N,
    heat_per_atom.data() + N * 2,
    heat_per_atom.data() + N * 3,
    heat_per_atom.data() + N * 4);
  GPU_CHECK_KERNEL
}

void compute_heat(
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  const GPU_Vector<float>& spin,
  const GPU_Vector<float>& torque,
  GPU_Vector<double>& heat_per_atom)
{
  const int N = velocity_per_atom.size() / 3;

  // the virial tensor:
  // xx xy xz    0 3 4
  // yx yy yz    6 1 5
  // zx zy zz    7 8 2
  gpu_compute_heat<<<(N - 1) / 128 + 1, 128>>>(
    N,
    virial_per_atom.data(),
    virial_per_atom.data() + N * 3,
    virial_per_atom.data() + N * 4,
    virial_per_atom.data() + N * 6,
    virial_per_atom.data() + N * 1,
    virial_per_atom.data() + N * 5,
    virial_per_atom.data() + N * 7,
    virial_per_atom.data() + N * 8,
    virial_per_atom.data() + N * 2,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + 2 * N,
    spin.data(),
    torque.data(),
    heat_per_atom.data(),
    heat_per_atom.data() + N,
    heat_per_atom.data() + N * 2,
    heat_per_atom.data() + N * 3,
    heat_per_atom.data() + N * 4);
  GPU_CHECK_KERNEL
}

namespace
{
static __global__ void gpu_compute_heat(
  const int N,
  const double* mass,
  const double* potential,
  const double* sxx,
  const double* sxy,
  const double* sxz,
  const double* syx,
  const double* syy,
  const double* syz,
  const double* szx,
  const double* szy,
  const double* szz,
  const double* vx,
  const double* vy,
  const double* vz,
  double* jx,
  double* jy,
  double* jz)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    double v_x = vx[n];
    double v_y = vy[n];
    double v_z = vz[n];
    double energy = mass[n] * (v_x * v_x + v_y * v_y + v_z * v_z) * 0.5 + potential[n];
    jx[n] = (energy + sxx[n]) * v_x + sxy[n] * v_y + sxz[n] * v_z;
    jy[n] = syx[n] * v_x + (energy + syy[n]) * v_y + syz[n] * v_z;
    jz[n] = szx[n] * v_x + szy[n] * v_y + (energy + szz[n]) * v_z;
  }
}
} // namespace

void compute_heat(
  const GPU_Vector<double>& mass,
  const GPU_Vector<double>& potential,
  const GPU_Vector<double>& virial_per_atom,
  const GPU_Vector<double>& velocity_per_atom,
  GPU_Vector<double>& heat_per_atom)
{
  const int N = velocity_per_atom.size() / 3;

  // the virial tensor:
  // xx xy xz    0 3 4
  // yx yy yz    6 1 5
  // zx zy zz    7 8 2
  gpu_compute_heat<<<(N - 1) / 128 + 1, 128>>>(
    N,
    mass.data(),
    potential.data(),
    virial_per_atom.data(),
    virial_per_atom.data() + N * 3,
    virial_per_atom.data() + N * 4,
    virial_per_atom.data() + N * 6,
    virial_per_atom.data() + N * 1,
    virial_per_atom.data() + N * 5,
    virial_per_atom.data() + N * 7,
    virial_per_atom.data() + N * 8,
    virial_per_atom.data() + N * 2,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + 2 * N,
    heat_per_atom.data(),
    heat_per_atom.data() + N,
    heat_per_atom.data() + N * 2);
  GPU_CHECK_KERNEL
}
