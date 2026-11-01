/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License,
    or (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "sed.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/error.cuh"
#ifdef USE_HIP
  // hipFFT is not part of this build; the time series is still written and the
  // spectrum can be produced by postprocessing/sed_spectrum.py.
  #define GPUMD_SED_HAS_FFT 0
#else
  #include <cuda_runtime.h>
  #include <cufft.h>
  #define GPUMD_SED_HAS_FFT 1
#endif
#include <cmath>
#include <cstring>

#define SED_BLOCK_SIZE 128

// One block per (q_k, Cartesian component a).  Accumulates
//   sum_i sqrt(m_i) u_{i,a} exp(i q_k x_i^0)
// with u = velocity (phonon mode) or spin (magnon mode).
static __global__ void gpu_project_sed(
  const int N,
  const int dir,
  const double q_step,
  const double q_dir_x,
  const double q_dir_y,
  const double q_dir_z,
  const int use_mass,
  const double* g_x0,
  const double* g_y0,
  const double* g_z0,
  const double* g_mass,
  const double* g_ux,
  const double* g_uy,
  const double* g_uz,
  double* g_out)
{
  const int idx = blockIdx.x;       // 0 .. 3*num_q-1
  const int k = idx / 3;
  const int a = idx % 3;
  const int tid = threadIdx.x;

  const double q = q_step * (double)k;
  // q_k = k * 2 pi / |a| is taken along the box vector a, so the phase must be
  // q_k * (a_hat . r).  For an orthogonal cell a_hat is a Cartesian axis and
  // this reduces to q * x; for a sheared cell the old `q * x` form was wrong
  // (the code only warned about it).
  const double* g_r0 = (dir == 0) ? g_x0 : ((dir == 1) ? g_y0 : g_z0);
  const double* g_u = (a == 0) ? g_ux : ((a == 1) ? g_uy : g_uz);

  __shared__ double s_re[SED_BLOCK_SIZE];
  __shared__ double s_im[SED_BLOCK_SIZE];

  double re = 0.0;
  double im = 0.0;
  for (int i = tid; i < N; i += SED_BLOCK_SIZE) {
    const double w = use_mass ? sqrt(g_mass[i]) * g_u[i] : g_u[i];
    const double arg = q * (q_dir_x * g_x0[i] + q_dir_y * g_y0[i] + q_dir_z * g_z0[i]);
    re += w * cos(arg);
    im += w * sin(arg);
  }
  s_re[tid] = re;
  s_im[tid] = im;
  __syncthreads();

  for (int offset = SED_BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_re[tid] += s_re[tid + offset];
      s_im[tid] += s_im[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_out[idx * 2 + 0] = s_re[0];
    g_out[idx * 2 + 1] = s_im[0];
  }
}

// Same as above but reading a float source (atom.spin is float), so the magnon
// mode never has to widen the whole spin array on the host per frame.
static __global__ void gpu_project_sed_f(
  const int N,
  const int dir,
  const double q_step,
  const double q_dir_x,
  const double q_dir_y,
  const double q_dir_z,
  const double* g_x0,
  const double* g_y0,
  const double* g_z0,
  const float* g_ux,
  const float* g_uy,
  const float* g_uz,
  double* g_out)
{
  const int idx = blockIdx.x;
  const int k = idx / 3;
  const int a = idx % 3;
  const int tid = threadIdx.x;

  const double q = q_step * (double)k;
  const double* g_r0 = (dir == 0) ? g_x0 : ((dir == 1) ? g_y0 : g_z0);
  const float* g_u = (a == 0) ? g_ux : ((a == 1) ? g_uy : g_uz);

  __shared__ double s_re[SED_BLOCK_SIZE];
  __shared__ double s_im[SED_BLOCK_SIZE];

  double re = 0.0;
  double im = 0.0;
  for (int i = tid; i < N; i += SED_BLOCK_SIZE) {
    const double w = (double)g_u[i];
    const double arg = q * (q_dir_x * g_x0[i] + q_dir_y * g_y0[i] + q_dir_z * g_z0[i]);
    re += w * cos(arg);
    im += w * sin(arg);
  }
  s_re[tid] = re;
  s_im[tid] = im;
  __syncthreads();

  for (int offset = SED_BLOCK_SIZE >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_re[tid] += s_re[tid + offset];
      s_im[tid] += s_im[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_out[idx * 2 + 0] = s_re[0];
    g_out[idx * 2 + 1] = s_im[0];
  }
}

SED::SED(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param < 4 || num_param > 6) {
    PRINT_INPUT_ERROR(
      "compute_sed should have 3 to 5 parameters:\n"
      "  <sample_interval> <num_q> <direction> [<mode>] [<series>]\n"
      "  direction = x | y | z, mode = velocity | spin, series = 0 | 1\n");
  }
  if (!is_valid_int(tokens[1], &sample_interval)) {
    PRINT_INPUT_ERROR("compute_sed: sample_interval should be an integer.\n");
  }
  if (!is_valid_int(tokens[2], &num_q)) {
    PRINT_INPUT_ERROR("compute_sed: num_q should be an integer.\n");
  }
  if (sample_interval <= 0) {
    PRINT_INPUT_ERROR("compute_sed: sample_interval should be positive.\n");
  }
  if (num_q <= 0) {
    PRINT_INPUT_ERROR("compute_sed: num_q should be positive.\n");
  }
  if (tokens[3] == "x") {
    direction = 0;
  } else if (tokens[3] == "y") {
    direction = 1;
  } else if (tokens[3] == "z") {
    direction = 2;
  } else {
    PRINT_INPUT_ERROR("compute_sed: direction should be x, y or z.\n");
  }
  spin_mode = false;
  dump_series = true;
  if (num_param >= 5) {
    if (tokens[4] == "velocity") {
      spin_mode = false;
    } else if (tokens[4] == "spin") {
      spin_mode = true;
    } else {
      PRINT_INPUT_ERROR("compute_sed: mode should be velocity or spin.\n");
    }
  }
  if (num_param >= 6) {
    if (tokens[5] == "0") {
      dump_series = false;
    } else if (tokens[5] == "1") {
      dump_series = true;
    } else {
      PRINT_INPUT_ERROR("compute_sed: series should be 0 or 1.\n");
    }
  }
  compute = true;
  action_name = "compute_sed";
  printf(
    "    compute the spectral energy density: %d q points along %c, "
    "every %d steps, %s mode.\n",
    num_q,
    "xyz"[direction],
    sample_interval,
    spin_mode ? "spin" : "velocity");
}

void SED::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!compute)
    return;

  // (1) The sampling run has to be plain NVE.  Any thermostat/barostat adds a
  //     non-Hamiltonian force and the spectra stop being the normal modes.
  if (integrate.get_type() != EnsembleType::NVE) {
    PRINT_INPUT_ERROR(
      "compute_sed requires `ensemble nve` in the run that samples the\n"
      "        trajectory.  Equilibrate with a thermostat in a previous run\n"
      "        block and sample here.");
  }

  // (2) Magnon mode needs the spin array.
  if (spin_mode && !atom.has_spin) {
    PRINT_INPUT_ERROR(
      "compute_sed: `spin` mode needs a 'spin:R:3' column in model.xyz.");
  }

  // (3) The run must be long enough to hold at least two samples.
  number_of_frames = number_of_steps / sample_interval;
  if (number_of_frames < 4) {
    PRINT_INPUT_ERROR(
      "compute_sed: the run is too short for the requested sample interval.");
  }

  // (4) q grid: q_k = k * 2pi / L_dir.  L_dir is the length of the box vector
  //     along `direction`, which is what makes q_k commensurate with the
  //     periodic images.
  double a_dir[3];
  if (direction == 0) {
    a_dir[0] = box.cpu_h[0];
    a_dir[1] = box.cpu_h[3];
    a_dir[2] = box.cpu_h[6];
  } else if (direction == 1) {
    a_dir[0] = box.cpu_h[1];
    a_dir[1] = box.cpu_h[4];
    a_dir[2] = box.cpu_h[7];
  } else {
    a_dir[0] = box.cpu_h[2];
    a_dir[1] = box.cpu_h[5];
    a_dir[2] = box.cpu_h[8];
  }
  const double L_dir = sqrt(a_dir[0] * a_dir[0] + a_dir[1] * a_dir[1] + a_dir[2] * a_dir[2]);
  if (!(L_dir > 0.0)) {
    PRINT_INPUT_ERROR("compute_sed: the box length along the q direction is zero.");
  }
  // Warn (do not fail) for a sheared cell: q along a box vector then mixes the
  // Cartesian directions and the branch assignment gets ambiguous.
  const bool sheared =
    (box.cpu_h[1] != 0 || box.cpu_h[2] != 0 || box.cpu_h[3] != 0 ||
     box.cpu_h[5] != 0 || box.cpu_h[6] != 0 || box.cpu_h[7] != 0);
  if (sheared) {
    printf("    WARNING: compute_sed with a triclinic box -- q is taken along the\n");
    printf("             box vector, which is not a pure Cartesian direction.\n");
  }
  q_step = 2.0 * PI / L_dir;
  // unit vector along the box vector, for the phase q_k * (a_hat . r)
  q_dir_[0] = a_dir[0] / L_dir;
  q_dir_[1] = a_dir[1] / L_dir;
  q_dir_[2] = a_dir[2] / L_dir;
  dt_sample_ps = time_step * sample_interval * TIME_UNIT_CONVERSION / 1000.0;

  // (5) Reference positions.  Wrapped positions would jump by a lattice
  //     vector whenever an atom crosses the boundary and destroy the phase,
  //     so the unwrapped positions are used.  They are snapshotted here and
  //     never updated, which is exactly the r_i^0 of the SED formula.
  if (atom.unwrapped_position.size() == 0) {
    atom.enable_unwrapped_position();
    atom.update_unwrapped_position(box);
  }
  const int N = atom.number_of_atoms;
  if (r0_.size() != static_cast<size_t>(N) * 3) {
    r0_.resize(static_cast<size_t>(N) * 3);
  }
  {
    std::vector<double> cpu_r0(static_cast<size_t>(N) * 3);
    atom.unwrapped_position.copy_to_host(cpu_r0.data());
    r0_.copy_from_host(cpu_r0.data());
  }

  // (6) Allocate the series.  Guard against an accidental out-of-memory: the
  //     user picks num_q and the run length, and the product is easy to
  //     underestimate.
  const size_t per_frame = static_cast<size_t>(num_q) * 3 * 2;
  const double bytes = (double)number_of_frames * (double)per_frame * sizeof(double);
  if (bytes > 2.0e9) {
    PRINT_INPUT_ERROR(
      "compute_sed: the time series would need more than 2 GB of device\n"
      "        memory.  Increase sample_interval or decrease num_q.");
  }
  frame_gpu_.resize(per_frame);
  frame_cpu_.resize(per_frame);
  series_gpu_.resize(static_cast<size_t>(number_of_frames) * per_frame);
  frame_index = 0;

  const double nu_max = 0.5 / dt_sample_ps; // Nyquist, THz
  printf(
    "    %d frames, dt = %.6f ps, frequency resolution %.5f THz, "
    "Nyquist %.3f THz.\n",
    number_of_frames,
    dt_sample_ps,
    1.0 / ((double)number_of_frames * dt_sample_ps),
    nu_max);
  printf("    series buffer: %.1f MB on the device.\n", bytes / 1.0e6);
}

void SED::end_of_step(
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
  if (frame_index >= number_of_frames)
    return;

  const int N = atom.number_of_atoms;
  if (spin_mode) {
    gpu_project_sed_f<<<num_q * 3, SED_BLOCK_SIZE>>>(
      N,
      direction,
      q_step,
      q_dir_[0],
      q_dir_[1],
      q_dir_[2],
      r0_.data(),
      r0_.data() + N,
      r0_.data() + N * 2,
      atom.spin.data(),
      atom.spin.data() + N,
      atom.spin.data() + N * 2,
      frame_gpu_.data());
    GPU_CHECK_KERNEL
  } else {
    gpu_project_sed<<<num_q * 3, SED_BLOCK_SIZE>>>(
      N,
      direction,
      q_step,
      q_dir_[0],
      q_dir_[1],
      q_dir_[2],
      1,
      r0_.data(),
      r0_.data() + N,
      r0_.data() + N * 2,
      atom.mass.data(),
      atom.velocity_per_atom.data(),
      atom.velocity_per_atom.data() + N,
      atom.velocity_per_atom.data() + N * 2,
      frame_gpu_.data());
    GPU_CHECK_KERNEL
  }

  const size_t per_frame = static_cast<size_t>(num_q) * 3 * 2;
  frame_gpu_.copy_to_host(frame_cpu_.data());
  // Append the frame to the frame-major device buffer.  GPU_Vector's
  // copy_from_host() takes an `int` offset, which cannot address the tail of a
  // buffer that is allowed to reach 2 GB, so the copy is done directly.
  gpuMemcpy(
    series_gpu_.data() + (size_t)frame_index * per_frame,
    frame_gpu_.data(),
    per_frame * sizeof(double),
    gpuMemcpyDeviceToDevice);
  ++frame_index;
}

void SED::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!compute)
    return;
  compute = false;

  const size_t per_frame = static_cast<size_t>(num_q) * 3 * 2;
  const int Nf = frame_index;
  if (Nf < 4) {
    printf("    compute_sed: only %d frames were sampled, nothing to transform.\n", Nf);
    return;
  }

  std::vector<double> series_cpu((size_t)Nf * per_frame);
  // Size-limited: series_gpu_ holds number_of_frames frames but only Nf were
  // sampled, and copy_to_host() without a size would overwrite past the end of
  // the host vector.
  series_gpu_.copy_to_host(series_cpu.data(), (size_t)Nf * per_frame);

  // ---- optional raw time series ------------------------------------------
  if (dump_series) {
    FILE* fid = fopen("sed_qv.out", "w");
    if (fid == nullptr) {
      PRINT_INPUT_ERROR("compute_sed: cannot open sed_qv.out for writing.");
    }
    fprintf(fid, "# q-space projection time series written by `compute_sed`\n");
    fprintf(fid, "# mode = %s, direction = %c, num_q = %d, frames = %d\n",
            spin_mode ? "spin" : "velocity", "xyz"[direction], num_q, Nf);
    fprintf(fid, "# dt = %.15e ps\n", dt_sample_ps);
    fprintf(fid, "# q_k (1/A):");
    for (int k = 0; k < num_q; ++k) {
      fprintf(fid, " %.10e", q_step * k);
    }
    fprintf(fid, "\n# columns: t(ps) then, for each (q_k, a=x,y,z), Re and Im\n");
    for (int f = 0; f < Nf; ++f) {
      fprintf(fid, "%25.15e", (double)f * dt_sample_ps);
      for (size_t c = 0; c < per_frame; ++c) {
        fprintf(fid, " %25.15e", series_cpu[(size_t)f * per_frame + c]);
      }
      fprintf(fid, "\n");
    }
    fclose(fid);
    printf("    wrote %d frames of the q-space time series to sed_qv.out.\n", Nf);
  }

#if GPUMD_SED_HAS_FFT
  // ---- spectrum -----------------------------------------------------------
  // cuFFT batched 1D transform.  The series lives frame-major on the device
  // ([frame][channel][re,im], which is what the sampling kernel produces), but
  // cufftPlanMany with inembed == NULL falls back to the *default* layout where
  // each of the `batch` signals is contiguous.  Passing an explicit inembed
  // (or, as here, transposing into the channel-major layout) is mandatory --
  // with the frame-major buffer and a NULL inembed cuFFT silently transforms
  // the wrong data and the spectrum is garbage.
  const int batch = num_q * 3;
  int nfft = Nf;                 // cufftPlanMany takes int*
  cufftHandle plan;
  cufftResult res = cufftPlanMany(
    &plan,
    1,
    &nfft,
    NULL,
    1,
    nfft, // input:  contiguous per channel, distance nfft
    NULL,
    1,
    nfft, // output: same
    CUFFT_Z2Z,
    batch);
  if (res != CUFFT_SUCCESS) {
    printf("    compute_sed: cufftPlanMany failed (%d), no spectrum written.\n", (int)res);
    return;
  }

  std::vector<cufftDoubleComplex> in((size_t)Nf * batch);
  std::vector<cufftDoubleComplex> out((size_t)Nf * batch);
  for (int b = 0; b < batch; ++b) {
    for (int f = 0; f < Nf; ++f) {
      const double* p = &series_cpu[(size_t)f * per_frame + (size_t)b * 2];
      in[(size_t)b * Nf + f].x = p[0];
      in[(size_t)b * Nf + f].y = p[1];
    }
  }

  cufftDoubleComplex* d_in = nullptr;
  cufftDoubleComplex* d_out = nullptr;
  cudaMalloc(&d_in, (size_t)Nf * batch * sizeof(cufftDoubleComplex));
  cudaMalloc(&d_out, (size_t)Nf * batch * sizeof(cufftDoubleComplex));
  cudaMemcpy(d_in, in.data(), (size_t)Nf * batch * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice);
  res = cufftExecZ2Z(plan, d_in, d_out, CUFFT_FORWARD);
  if (res != CUFFT_SUCCESS) {
    printf("    compute_sed: cufftExecZ2Z failed (%d), no spectrum written.\n", (int)res);
    cufftDestroy(plan);
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }
  cudaMemcpy(out.data(), d_out, (size_t)Nf * batch * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
  cufftDestroy(plan);
  cudaFree(d_in);
  cudaFree(d_out);

  const int n_nu = Nf / 2; // positive frequencies only
  const double nu_res = 1.0 / ((double)Nf * dt_sample_ps);
  FILE* fid = fopen("sed.out", "w");
  if (fid == nullptr) {
    PRINT_INPUT_ERROR("compute_sed: cannot open sed.out for writing.");
  }
  fprintf(fid, "# spectral energy density written by `compute_sed`\n");
  fprintf(fid, "# mode = %s, direction = %c, frames = %d, dt = %.15e ps\n",
          spin_mode ? "spin" : "velocity", "xyz"[direction], Nf, dt_sample_ps);
  fprintf(fid, "# resolution = %.10e THz, Nyquist = %.10e THz\n", nu_res, 0.5 / dt_sample_ps);
  fprintf(fid, "# columns: nu(THz) then Phi(q_k, nu) = sum_a |FFT[V_a(q_k,t)]|^2");
  fprintf(fid, " (arbitrary units, peak positions are the physical content)\n");
  fprintf(fid, "# q_k (1/A):");
  for (int k = 0; k < num_q; ++k) {
    fprintf(fid, " %.10e", q_step * k);
  }
  fprintf(fid, "\n");
  for (int m = 0; m < n_nu; ++m) {
    fprintf(fid, "%25.15e", (double)m * nu_res);
    for (int k = 0; k < num_q; ++k) {
      double phi = 0.0;
      for (int a = 0; a < 3; ++a) {
        const cufftDoubleComplex& c = out[((size_t)k * 3 + a) * Nf + m];
        phi += c.x * c.x + c.y * c.y;
      }
      fprintf(fid, " %25.15e", phi * dt_sample_ps / (double)Nf);
    }
    fprintf(fid, "\n");
  }
  fclose(fid);
  printf("    wrote the spectral energy density (%d frequencies x %d q points) "
         "to sed.out.\n", n_nu, num_q);
#else
  printf("    compute_sed: built without cuFFT; use postprocessing/sed_spectrum.py\n"
         "                 on sed_qv.out to obtain the spectrum.\n");
#endif
}
