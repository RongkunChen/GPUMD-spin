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
Calculate the heat current autocorrelation (HAC) function.
------------------------------------------------------------------------------*/

#include "compute_heat.cuh"
#include "force/force.cuh"
#include "hac.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <vector>

#define NUM_OF_HEAT_COMPONENTS 5
#define FILE_NAME_LENGTH 200
#define DIM 3

// Allocate memory for recording heat current data
void HAC::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (compute) {
    int number_of_frames = number_of_steps / sample_interval;
    if (Nc > number_of_frames) {
      PRINT_INPUT_ERROR(
        "The number of HAC correlation steps should not exceed the number of sampled frames.\n");
    }
    heat_all.resize(NUM_OF_HEAT_COMPONENTS * number_of_frames);
    atom.heat_per_atom.resize(atom.number_of_atoms * 5);
  }
}

// sum up the per-atom heat current to get the total heat current
static __global__ void
gpu_sum_heat(const int N, const int Nd, const int nd, const double* g_heat, double* g_heat_all)
{
  // <<<NUM_OF_HEAT_COMPONENTS, 1024>>>
  const int tid = threadIdx.x;
  const int number_of_batches = (N - 1) / 1024 + 1;

  __shared__ double s_data[1024];
  s_data[tid] = 0.0;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    const int n = tid + batch * 1024;
    if (n < N) {
      s_data[tid] += g_heat[n + N * blockIdx.x];
    }
  }

  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_data[tid] += s_data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    g_heat_all[nd + Nd * blockIdx.x] = s_data[0];
  }
}

// sample heat current data for HAC calculations.
void HAC::end_of_step(
  const int number_of_steps,
  int step,
  const int fixed_group,
  const int move_group,
  const double global_time,
  const double temperature,
  Integrate& integrate,
  Box& box,
  std::vector<Group>& group,
  GPU_Vector<double>& thermo,
  Atom& atom,
  Force& force)
{
  if (!compute)
    return;
  if ((step + 1) % sample_interval != 0)
    return;

  const int N = atom.number_of_atoms;

  // Round 9 (task 48): HAC used to call the spin-less overload, so the Green-
  // Kubo route and the HNEMD route silently disagreed on spin systems.  Both
  // now use the full J = J_p + J_m.
  if (atom.has_spin) {
    compute_heat(
      atom.virial_per_atom,
      force.spin_flux_A,
      atom.velocity_per_atom,
      atom.spin,
      atom.torque,
      atom.spin_gamma,
      atom.spin_alpha,
      atom.spin_b_ext,
      atom.heat_per_atom);
  } else {
    compute_heat(atom.virial_per_atom, atom.velocity_per_atom, atom.heat_per_atom);
  }

  int nd = (step + 1) / sample_interval - 1;
  int Nd = number_of_steps / sample_interval;
  gpu_sum_heat<<<NUM_OF_HEAT_COMPONENTS, 1024>>>(N, Nd, nd, atom.heat_per_atom.data(), heat_all.data());
  GPU_CHECK_KERNEL
}

// Calculate the Heat current Auto-Correlation function (HAC)
static __global__ void gpu_find_hac(const int Nc, const int Nd, const double* g_heat, double* g_hac)
{
  //<<<Nc, 128>>>

  __shared__ double s_hac_xi[128];
  __shared__ double s_hac_xo[128];
  __shared__ double s_hac_yi[128];
  __shared__ double s_hac_yo[128];
  __shared__ double s_hac_z[128];

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int number_of_batches = (Nd - 1) / 128 + 1;
  int number_of_data = Nd - bid;

  s_hac_xi[tid] = 0.0;
  s_hac_xo[tid] = 0.0;
  s_hac_yi[tid] = 0.0;
  s_hac_yo[tid] = 0.0;
  s_hac_z[tid] = 0.0;

  for (int batch = 0; batch < number_of_batches; ++batch) {
    int index = tid + batch * 128;
    if (index + bid < Nd) {
      s_hac_xi[tid] += g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 0] +
                       g_heat[index + Nd * 0] * g_heat[index + bid + Nd * 1];
      s_hac_xo[tid] += g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 1] +
                       g_heat[index + Nd * 1] * g_heat[index + bid + Nd * 0];
      s_hac_yi[tid] += g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 2] +
                       g_heat[index + Nd * 2] * g_heat[index + bid + Nd * 3];
      s_hac_yo[tid] += g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 3] +
                       g_heat[index + Nd * 3] * g_heat[index + bid + Nd * 2];
      s_hac_z[tid] += g_heat[index + Nd * 4] * g_heat[index + bid + Nd * 4];
    }
  }
  __syncthreads();


  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_hac_xi[tid] += s_hac_xi[tid + offset];
      s_hac_xo[tid] += s_hac_xo[tid + offset];
      s_hac_yi[tid] += s_hac_yi[tid + offset];
      s_hac_yo[tid] += s_hac_yo[tid + offset];
      s_hac_z[tid] += s_hac_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_hac[bid + Nc * 0] = s_hac_xi[0] / number_of_data;
    g_hac[bid + Nc * 1] = s_hac_xo[0] / number_of_data;
    g_hac[bid + Nc * 2] = s_hac_yi[0] / number_of_data;
    g_hac[bid + Nc * 3] = s_hac_yo[0] / number_of_data;
    g_hac[bid + Nc * 4] = s_hac_z[0] / number_of_data;
  }
}

// Calculate the Running Thermal Conductivity (RTC) from the HAC
static void find_rtc(const int Nc, const double factor, const double* hac, double* rtc)
{
  for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++) {
    for (int nc = 1; nc < Nc; nc++) {
      const int index = Nc * k + nc;
      rtc[index] = rtc[index - 1] + (hac[index - 1] + hac[index]) * factor;
    }
  }
}

// Calculate HAC (heat current auto-correlation function)
// and RTC (running thermal conductivity)
void HAC::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute)
    return;
  print_line_1();
  printf("Start to calculate HAC and related quantities.\n");
  // F-12 (r21): cross-correlation is ALREADY implemented in gpu_find_hac --
  // both J_in(t)J_out(t+tau) and J_out(t)J_in(t+tau) are accumulated for x and
  // y.  The old documentation complaint ("HAC lacks the in/out cross term") is
  // NOT a defect of this version; see AUDIT_AND_FIX_REPORT.md, excluded clue.
  // The Green-Kubo prefactor below is derived from
  //   kappa_ab = V/(kB T^2) int_0^inf <J_a(0) J_b(t)> dt  with j = J/V,
  // and is NOT tuned to match any material (no empirical /98 rescaling).
  if (atom.has_spin) {
    printf(
      "WARNING [BLOCKED]: compute_hac on a spin system uses an UNVALIDATED\n"
      "  spin-resolved heat current (see PHYSICS_DERIVATION.md section 4).\n"
      "  hac.out / the running kappa are DIAGNOSTIC ONLY.  In particular\n"
      "  kappa_pm = kappa_mp does NOT follow from both channels being T-odd:\n"
      "  Onsager-Casimir requires a field-odd transformation and holds only at\n"
      "  B = 0 (or with the appropriate B -> -B exchange).\n");
  }
  printf(
    "    using T = %.4f K and V = %.4f A^3 for the Green-Kubo prefactor.\n",
    temperature,
    box.get_volume());

  const int Nd = number_of_steps / sample_interval;
  const double dt = time_step * sample_interval;
  const double dt_in_ps = dt * TIME_UNIT_CONVERSION / 1000.0; // ps

  // major data
  std::vector<double> rtc(Nc * NUM_OF_HEAT_COMPONENTS, 0.0);
  GPU_Vector<double> hac_gpu(Nc * NUM_OF_HEAT_COMPONENTS);
  std::vector<double> hac_cpu(Nc * NUM_OF_HEAT_COMPONENTS);

  // Subtract the sample mean of the heat current before correlating.
  // Green-Kubo needs the autocorrelation of the *fluctuation* delta-J =
  // J - <J>.  Over a finite trajectory the sample mean is not zero (its
  // standard error is sigma/sqrt(N_indep)), and its square adds a spurious
  // ramp <J>^2 * tau / (kB T^2 V) to the running integral.  For a low-kappa
  // material such as CrN (kappa ~ 1 W/(m K)) the ramp already exceeds the
  // physical plateau after a few ps and destroys it.  See
  // REPORT_validation.md, fix F10.
  {
    std::vector<double> heat_cpu(Nd * NUM_OF_HEAT_COMPONENTS);
    heat_all.copy_to_host(heat_cpu.data(), heat_cpu.size());
    for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; ++k) {
      double mean = 0.0;
      for (int i = 0; i < Nd; ++i) {
        mean += heat_cpu[i + Nd * k];
      }
      mean /= double(Nd);
      for (int i = 0; i < Nd; ++i) {
        heat_cpu[i + Nd * k] -= mean;
      }
    }
    heat_all.copy_from_host(heat_cpu.data(), heat_cpu.size());
  }

  // Here, the block size is fixed to 128, which is a good choice
  gpu_find_hac<<<Nc, 128>>>(Nc, Nd, heat_all.data(), hac_gpu.data());
  GPU_CHECK_KERNEL

  hac_gpu.copy_to_host(hac_cpu.data());

  if (!(temperature > 0.0)) {
    PRINT_INPUT_ERROR(
      "compute_hac needs a positive temperature; the NVE ensemble does not set\n"
      "        one, so put a thermostat block before it (or `ensemble nvt_*`) "
      "in run.in.\n");
  }

  double factor = dt * 0.5 / (K_B * temperature * temperature * box.get_volume());
  factor *= KAPPA_UNIT_CONVERSION;

  find_rtc(Nc, factor, hac_cpu.data(), rtc.data());

  FILE* fid = fopen("hac.out", "a");
  const int number_of_output_data = Nc / output_interval;
  for (int nd = 0; nd < number_of_output_data; nd++) {
    const int nc = nd * output_interval;
    double hac_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
    double rtc_ave[NUM_OF_HEAT_COMPONENTS] = {0.0};
    for (int k = 0; k < NUM_OF_HEAT_COMPONENTS; k++) {
      for (int m = 0; m < output_interval; m++) {
        const int count = Nc * k + nc + m;
        hac_ave[k] += hac_cpu[count];
        rtc_ave[k] += rtc[count];
      }
    }
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      hac_ave[m] /= output_interval;
      rtc_ave[m] /= output_interval;
    }
    fprintf(fid, "%25.15e", (nc + output_interval * 0.5) * dt_in_ps);
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", hac_ave[m]);
    }
    for (int m = 0; m < NUM_OF_HEAT_COMPONENTS; m++) {
      fprintf(fid, "%25.15e", rtc_ave[m]);
    }
    fprintf(fid, "\n");
  }
  fflush(fid);
  fclose(fid);

  printf("HAC and related quantities are calculated.\n");
  print_line_2();

  compute = 0;
}

void HAC::parse(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  compute = 1;

  printf("Compute HAC.\n");

  if (num_param != 4) {
    PRINT_INPUT_ERROR("compute_hac should have 3 parameters.\n");
  }

  if (!is_valid_int(tokens[1], &sample_interval)) {
    PRINT_INPUT_ERROR("sample interval for HAC should be an integer number.\n");
  }
  if (sample_interval <= 0) {
    PRINT_INPUT_ERROR("sample interval for HAC should be positive.\n");
  }
  printf("    sample interval is %d.\n", sample_interval);

  if (!is_valid_int(tokens[2], &Nc)) {
    PRINT_INPUT_ERROR("Nc for HAC should be an integer number.\n");
  }
  if (Nc <= 0) {
    PRINT_INPUT_ERROR("Nc for HAC should be positive.\n");
  }
  printf("    Nc is %d\n", Nc);

  if (!is_valid_int(tokens[3], &output_interval)) {
    PRINT_INPUT_ERROR("output_interval for HAC should be an integer number.\n");
  }
  if (output_interval <= 0) {
    PRINT_INPUT_ERROR("output_interval for HAC should be positive.\n");
  }
  printf("    output_interval is %d\n", output_interval);
}

HAC::HAC(const std::vector<std::string>& tokens)
{
  parse(tokens);
  action_name = "compute_hac";
}
