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

#include "dump_heat_flux.cuh"
#include "compute_heat.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include "utilities/error.cuh"
#include <cmath>
#include <cstring>

#define HF_BLOCK 128

// Pack the five-component per-atom heat current produced by `compute_heat`
// into the three Cartesian components used by the binned output:
//     jx = jx_in + jx_out,  jy = jy_in + jy_out,  jz = jz
static __global__ void gpu_pack_heat(
  const int N,
  const double* g_jx_in,
  const double* g_jx_out,
  const double* g_jy_in,
  const double* g_jy_out,
  const double* g_jz,
  double* g_jx,
  double* g_jy,
  double* g_jz_out)
{
  const int n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n < N) {
    g_jx[n] = g_jx_in[n] + g_jx_out[n];
    g_jy[n] = g_jy_in[n] + g_jy_out[n];
    g_jz_out[n] = g_jz[n];
  }
}

// Sum the per-atom heat current of one slab along `axis`.  One block per bin;
// each block walks all atoms and keeps the ones inside its slab.  N is small
// (a few hundred to a few thousand) and nbin is a few tens, so the redundant
// walk is far cheaper than a sort or an atomicAdd-based scatter.
static __global__ void gpu_bin_heat(
  const int N,
  const int axis,
  const double inv_bin_length,
  const double box_length,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const double* g_jx,
  const double* g_jy,
  const double* g_jz,
  double* g_out)
{
  const int b = blockIdx.x;
  const int nb = gridDim.x;
  const int tid = threadIdx.x;
  const double* g_r = (axis == 0) ? g_x : ((axis == 1) ? g_y : g_z);

  __shared__ double s_x[HF_BLOCK];
  __shared__ double s_y[HF_BLOCK];
  __shared__ double s_z[HF_BLOCK];

  double ax = 0.0, ay = 0.0, az = 0.0;
  for (int i = tid; i < N; i += HF_BLOCK) {
    double r = g_r[i];
    // positions are wrapped but may be centred on zero
    r = fmod(r, box_length);
    if (r < 0.0)
      r += box_length;
    int ib = (int)(r * inv_bin_length);
    if (ib >= nb)
      ib = nb - 1; // guard against r == box_length
    if (ib == b) {
      ax += g_jx[i];
      ay += g_jy[i];
      az += g_jz[i];
    }
  }
  s_x[tid] = ax;
  s_y[tid] = ay;
  s_z[tid] = az;
  __syncthreads();

  for (int offset = HF_BLOCK >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_x[tid] += s_x[tid + offset];
      s_y[tid] += s_y[tid + offset];
      s_z[tid] += s_z[tid + offset];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_out[b * 3 + 0] = s_x[0];
    g_out[b * 3 + 1] = s_y[0];
    g_out[b * 3 + 2] = s_z[0];
  }
}

Dump_Heat_Flux::Dump_Heat_Flux(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 5) {
    PRINT_INPUT_ERROR(
      "dump_heat_flux should have 1 parameter, or 4 in binned form:\n"
      "  dump_heat_flux <sample_interval>\n"
      "  dump_heat_flux <sample_interval> bin/1d <x|y|z> <nbin>\n");
  }
  if (!is_valid_int(tokens[1], &sample_interval)) {
    PRINT_INPUT_ERROR("dump_heat_flux: sample_interval should be an integer.\n");
  }
  if (sample_interval <= 0) {
    PRINT_INPUT_ERROR("dump_heat_flux: sample_interval should be positive.\n");
  }
  if (num_param == 5) {
    if (tokens[2] != "bin/1d") {
      PRINT_INPUT_ERROR("dump_heat_flux: the binned form is bin/1d.\n");
    }
    use_bins = true;
    if (tokens[3] == "x") {
      axis = 0;
    } else if (tokens[3] == "y") {
      axis = 1;
    } else if (tokens[3] == "z") {
      axis = 2;
    } else {
      PRINT_INPUT_ERROR("dump_heat_flux: the axis should be x, y or z.\n");
    }
    if (!is_valid_int(tokens[4], &nbin)) {
      PRINT_INPUT_ERROR("dump_heat_flux: nbin should be an integer.\n");
    }
    if (nbin <= 0) {
      PRINT_INPUT_ERROR("dump_heat_flux: nbin should be positive.\n");
    }
    printf(
      "    dump the heat flux of %d bins along %c every %d steps into "
      "heat_flux.out.\n",
      nbin,
      "xyz"[axis],
      sample_interval);
  } else {
    printf("    dump the heat flux every %d steps into heat_flux.out.\n", sample_interval);
  }
  dump = true;
  action_name = "dump_heat_flux";
}

void Dump_Heat_Flux::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!dump)
    return;

  const int number_of_frames_expected = number_of_steps / sample_interval;
  if (number_of_frames_expected < 2) {
    PRINT_INPUT_ERROR(
      "dump_heat_flux: the run is too short for the requested sample interval.");
  }

  volume = box.get_volume();
  if (!(volume > 0.0)) {
    PRINT_INPUT_ERROR("dump_heat_flux: the box volume is not positive.");
  }

  if (use_bins) {
    // F-10 (r21): gpu_bin_heat() bins a RAW Cartesian coordinate with
    // fmod(r, a[axis]) and normalises by V/nbin.  Both are only valid for an
    // orthogonal cell that is periodic along the binning axis.  For a tilted
    // box the bin coordinate must be the fractional coordinate
    // s = h^{-1} r (not r_axis), and for a non-periodic axis the bin volume is
    // not V/nbin.  Rather than emit silently wrong numbers, reject the
    // combination here with an explicit message.
    box.set_is_orthogonal();
    if (!box.is_orthogonal) {
      PRINT_INPUT_ERROR(
        "dump_heat_flux bin=: tilted (non-orthogonal) cells are not supported.\n"
        "        Binning uses a Cartesian coordinate; a triclinic cell needs\n"
        "        fractional coordinates and a Jacobian-corrected bin volume.\n"
        "        Use the global (unbinned) heat flux instead.\n");
    }
    const int pbc[3] = {box.pbc_x, box.pbc_y, box.pbc_z};
    if (!pbc[axis]) {
      PRINT_INPUT_ERROR(
        "dump_heat_flux bin=: the binning axis must be periodic.\n"
        "        The bin volume V/nbin assumes periodic slabs.\n");
    }
    const double a[3] = {
      sqrt(box.cpu_h[0] * box.cpu_h[0] + box.cpu_h[3] * box.cpu_h[3] +
           box.cpu_h[6] * box.cpu_h[6]),
      sqrt(box.cpu_h[1] * box.cpu_h[1] + box.cpu_h[4] * box.cpu_h[4] +
           box.cpu_h[7] * box.cpu_h[7]),
      sqrt(box.cpu_h[2] * box.cpu_h[2] + box.cpu_h[5] * box.cpu_h[5] +
           box.cpu_h[8] * box.cpu_h[8])};
    if (!(a[axis] > 0.0)) {
      PRINT_INPUT_ERROR("dump_heat_flux: the box length along the axis is zero.");
    }
    bin_length = a[axis] / nbin;
  }

  // F-11 (r21): STATUS DECLARATION.  The spin-resolved heat current in
  // compute_heat.cu is NOT a validated quantitative total energy flux for a
  // general NEP (see PHYSICS_DERIVATION.md, section 4).  Its output is usable
  // only as a diagnostic time series; any Green-Kubo integral from it must be
  // reported as BLOCKED, not as a thermal conductivity.
  if (atom.has_spin) {
    printf(
      "WARNING [BLOCKED]: spin-resolved heat flux is NOT a validated total\n"
      "  energy current for a general NEP.  e_i^spin = -1/2 S_i.B_i requires\n"
      "  a homogeneous quadratic (Heisenberg) Hamiltonian; the magnon channel\n"
      "  uses a deterministic dS/dt with no thermostat/source terms and omits\n"
      "  the external ME/Zeeman contribution to the local energy derivative.\n"
      "  heat_flux.out / hac.out / kappa.out from this run are DIAGNOSTIC ONLY;\n"
      "  do not quote them as a quantitative thermal conductivity.\n");
  }

  const int N = atom.number_of_atoms;
  // `compute_heat` (3-argument form) writes five components per atom:
  // jx_in, jx_out, jy_in, jy_out, jz.  This is the SAME convention used by
  // `compute_hac` and `compute_hnemd`, so the numbers written here are
  // directly comparable with hac.out and kappa.out.
  // The 5-argument form of `compute_heat` additionally adds the convective
  // term sum_i epsilon_i v_i.  It must NOT be used here: `potential_per_atom`
  // carries the raw DFT total-energy scale (about -1.3e3 eV/atom for this
  // CrN potential), so the convective term amplifies the round-off level of
  // sum_i v_i by that offset and produces a heat current ~10^3 times too
  // large.  See REPORT_validation.md, fix F9.
  const size_t n5 = static_cast<size_t>(N) * 5;
  if (heat_per_atom_.size() != n5) {
    heat_per_atom_.resize(n5);
    heat_cpu_.resize(n5);
  }
  const size_t n3 = static_cast<size_t>(N) * 3;
  if (heat_xyz_.size() != n3) {
    heat_xyz_.resize(n3);
  }
  // Round 9 (task 48): split the flux into its phonon and magnon channels.
  split_channels = atom.has_spin && !use_bins;
  if (split_channels) {
    if (heat_phonon_.size() != n5) {
      heat_phonon_.resize(n5);
      heat_phonon_cpu_.resize(n5);
      heat_magnon_.resize(n5);
      heat_magnon_cpu_.resize(n5);
    }
  }
  if (use_bins) {
    bin_out_.resize(static_cast<size_t>(nbin) * 3);
    bin_cpu_.resize(static_cast<size_t>(nbin) * 3);
  }

  number_of_frames = 0;

  // Truncate, do not append: heat_flux.out belongs to this run only.
  fid = fopen("heat_flux.out", "w");
  if (fid == nullptr) {
    PRINT_INPUT_ERROR("dump_heat_flux: cannot open heat_flux.out for writing.");
  }
  fprintf(fid, "# heat flux time series written by `dump_heat_flux`\n");
  fprintf(fid, "# volume = %.15e A^3, %s\n", volume, use_bins ? "binned" : "global");
  if (use_bins) {
    fprintf(
      fid, "# %d bins along %c, bin length %.15e A\n", nbin, "xyz"[axis], bin_length);
    fprintf(fid, "# columns: time(ps) then Jx Jy Jz of every bin   [eV/(A^2 ps)]\n");
    fprintf(fid, "# (each bin is normalised by its own volume V/nbin)\n");
  } else if (split_channels) {
    fprintf(
      fid, "# columns: time(ps)  Jx Jy Jz |J|  Jpx Jpy Jpz |Jp|  Jmx Jmy Jmz |Jm|"
      "   [eV/(A^2 ps)]\n");
    fprintf(fid, "# J = J_p + J_m;  J_p = sum_i (W_i + e_i^spin I).v_i (phonon),\n");
    fprintf(fid, "# J_m = sum_j A_j : dS_j/dt (magnon, conductive).\n");
    fprintf(fid, "# Green-Kubo: kappa_pp ~ <J_p J_p>, kappa_mm ~ <J_m J_m>,\n");
    fprintf(fid, "# kappa_pm = kappa_mp ~ <J_p J_m> (Onsager: both are T-odd).\n");
  } else {
    fprintf(fid, "# columns: time(ps)  Jx  Jy  Jz  |J|   [eV/(A^2 ps)]\n");
  }
  fprintf(fid, "# 1 eV/(A^2 ps) = %.8e W/m^2\n", 1.602176634e13);
  fflush(fid);
}

void Dump_Heat_Flux::end_of_step(
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
  if (!dump || fid == nullptr)
    return;
  if ((step + 1) % sample_interval != 0)
    return;

  const int N = atom.number_of_atoms;
  if (atom.has_spin) {
    // Round 9 (task 48): the full J = J_p + J_m, with the two channels kept
    // apart for the Green-Kubo decomposition.
    compute_heat(
      atom.virial_per_atom,
      force.spin_flux_A,
      atom.velocity_per_atom,
      atom.spin,
      atom.torque,
      atom.spin_gamma,
      atom.spin_alpha,
      atom.spin_b_ext,
      heat_per_atom_,
      split_channels ? &heat_phonon_ : nullptr,
      split_channels ? &heat_magnon_ : nullptr);
  } else {
    compute_heat(atom.virial_per_atom, atom.velocity_per_atom, heat_per_atom_);
  }

  gpu_pack_heat<<<(N - 1) / HF_BLOCK + 1, HF_BLOCK>>>(
    N,
    heat_per_atom_.data(),
    heat_per_atom_.data() + N,
    heat_per_atom_.data() + N * 2,
    heat_per_atom_.data() + N * 3,
    heat_per_atom_.data() + N * 4,
    heat_xyz_.data(),
    heat_xyz_.data() + N,
    heat_xyz_.data() + N * 2);
  GPU_CHECK_KERNEL

  // natural units (eV * A / natural-time) -> eV/(A^2 ps):
  //   1 natural time unit = TIME_UNIT_CONVERSION fs, so multiply the velocity
  //   by 1000/TIME_UNIT_CONVERSION to get A/ps, then divide by the volume.
  const double t_ps = global_time * TIME_UNIT_CONVERSION / 1000.0;

  if (use_bins) {
    gpu_bin_heat<<<nbin, HF_BLOCK>>>(
      N,
      axis,
      1.0 / bin_length,
      bin_length * nbin,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + N,
      atom.position_per_atom.data() + N * 2,
      heat_xyz_.data(),
      heat_xyz_.data() + N,
      heat_xyz_.data() + N * 2,
      bin_out_.data());
    GPU_CHECK_KERNEL
    bin_out_.copy_to_host(bin_cpu_.data());
    const double factor = 1000.0 / TIME_UNIT_CONVERSION / (volume / nbin);
    fprintf(fid, "%25.15e", t_ps);
    for (int b = 0; b < nbin; ++b) {
      fprintf(
        fid, " %25.15e %25.15e %25.15e",
        bin_cpu_[b * 3 + 0] * factor,
        bin_cpu_[b * 3 + 1] * factor,
        bin_cpu_[b * 3 + 2] * factor);
    }
    fprintf(fid, "\n");
  } else {
    heat_per_atom_.copy_to_host(heat_cpu_.data());
    double jx = 0.0, jy = 0.0, jz = 0.0;
    for (int n = 0; n < N; ++n) {
      // jx = jx_in + jx_out, jy = jy_in + jy_out, jz = jz
      jx += heat_cpu_[n] + heat_cpu_[n + N];
      jy += heat_cpu_[n + N * 2] + heat_cpu_[n + N * 3];
      jz += heat_cpu_[n + N * 4];
    }
    const double factor = 1000.0 / TIME_UNIT_CONVERSION / volume;
    jx *= factor;
    jy *= factor;
    jz *= factor;
    const double jn = sqrt(jx * jx + jy * jy + jz * jz);
    if (!split_channels) {
      fprintf(
        fid, "%25.15e %25.15e %25.15e %25.15e %25.15e\n", t_ps, jx, jy, jz, jn);
    } else {
      heat_phonon_.copy_to_host(heat_phonon_cpu_.data());
      heat_magnon_.copy_to_host(heat_magnon_cpu_.data());
      double px = 0.0, py = 0.0, pz = 0.0;
      double mx = 0.0, my = 0.0, mz = 0.0;
      for (int n = 0; n < N; ++n) {
        px += heat_phonon_cpu_[n] + heat_phonon_cpu_[n + N];
        py += heat_phonon_cpu_[n + N * 2] + heat_phonon_cpu_[n + N * 3];
        pz += heat_phonon_cpu_[n + N * 4];
        mx += heat_magnon_cpu_[n] + heat_magnon_cpu_[n + N];
        my += heat_magnon_cpu_[n + N * 2] + heat_magnon_cpu_[n + N * 3];
        mz += heat_magnon_cpu_[n + N * 4];
      }
      px *= factor;
      py *= factor;
      pz *= factor;
      mx *= factor;
      my *= factor;
      mz *= factor;
      fprintf(
        fid,
        "%25.15e %25.15e %25.15e %25.15e %25.15e"
        " %25.15e %25.15e %25.15e %25.15e"
        " %25.15e %25.15e %25.15e %25.15e\n",
        t_ps,
        jx, jy, jz, jn,
        px, py, pz, sqrt(px * px + py * py + pz * pz),
        mx, my, mz, sqrt(mx * mx + my * my + mz * mz));
    }
  }

  // Flush every 512 frames so that a long run can be inspected while it runs
  // without paying for a flush on every sample.
  if (++number_of_frames % 512 == 0) {
    fflush(fid);
  }
}

void Dump_Heat_Flux::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (!dump)
    return;
  if (fid != nullptr) {
    fflush(fid);
    fclose(fid);
    fid = nullptr;
    printf("    wrote %lld heat flux frames to heat_flux.out.\n", number_of_frames);
  }
  dump = false;
}
