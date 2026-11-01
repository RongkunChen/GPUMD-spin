/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, or (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with GPUMD.  If not, see <https://www.gnu.org/licenses/>.
*/

/*-----------------------------------------------------------------------------100
Dump the magnetisation of the spin system at a given interval.

The reduction runs on the host: it happens once per dump interval, not once per
MD step, and it keeps the per-type breakdown trivial to implement.
-------------------------------------------------------------------------------*/

#include "dump_magnetization.cuh"
#include "model/box.cuh"
#include "parse_utilities.cuh"
#include "utilities/common.cuh"
#include <cmath>
#include "utilities/error.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <cmath>
#include <cstring>
#include <vector>

Dump_Magnetization::Dump_Magnetization(const std::vector<std::string>& tokens)
{
  parse(tokens);
  action_name = "dump_magnetization";
}

void Dump_Magnetization::parse(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  dump_ = true;
  printf("Dump magnetization\n");

  if (num_param < 2) {
    PRINT_INPUT_ERROR("dump_magnetization should have at least 1 parameter.");
  }
  if (!is_valid_int(tokens[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("dump interval should be positive.");
  }
  bool has_filename = false;
  for (size_t k = 2; k < tokens.size(); ++k) {
    if (tokens[k] == "warmup") {
      if (k + 1 >= tokens.size() || !is_valid_int(tokens[k + 1], &warmup_samples_)) {
        PRINT_INPUT_ERROR("dump_magnetization warmup needs a sample count.");
      }
      if (warmup_samples_ < 0) {
        PRINT_INPUT_ERROR("dump_magnetization warmup should not be negative.");
      }
      ++k;
    } else if (!has_filename) {
      filename_ = tokens[k];
      has_filename = true;
    } else {
      PRINT_INPUT_ERROR("Unknown dump_magnetization parameter.");
    }
  }
  printf("    every %d steps.\n", dump_interval_);
  printf("    into %s.\n", filename_.c_str());
  if (warmup_samples_ > 0) {
    printf("    fluctuation accumulates after %d warm-up samples.\n", warmup_samples_);
  }
}

void Dump_Magnetization::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (!dump_)
    return;
  if (!atom.has_spin) {
    PRINT_INPUT_ERROR("dump_magnetization requires a model.xyz with a spin column.");
  }
  num_types_ = static_cast<int>(atom.cpu_type_size.size());
  if (num_types_ < 1) {
    num_types_ = 1;
  }
  atoms_of_type_.assign(size_t(num_types_), 0);
  for (size_t n = 0; n < atom.cpu_type.size(); ++n) {
    const int t = atom.cpu_type[n];
    if (t >= 0 && t < num_types_) {
      atoms_of_type_[t]++;
    }
  }
  // The fluctuation is a property of *this* run, so a fresh run starts from
  // scratch even though the file itself is opened in append mode.
  number_of_atoms_ = atom.number_of_atoms;
  n_samples_ = 0;
  n_written_ = 0;
  sum_total_[0] = sum_total_[1] = sum_total_[2] = 0.0;
  sum2_total_ = 0.0;
  sum_per_type_.assign(size_t(num_types_) * 3, 0.0);
  sum2_per_type_.assign(size_t(num_types_), 0.0);
  sat_total_ = 0.0;
  sat_per_type_.assign(size_t(num_types_), 0.0);

  file_ = my_fopen(filename_.c_str(), "a");
  fprintf(file_, "# dump_magnetization %d\n", dump_interval_);
  // P99 / R32: version 3 appends Msat/Mnorm columns (total and per type).
  fprintf(file_, "# format_version 3\n");
  fprintf(file_, "# chi = (<M.M> - <M>.<M>) / (N kB T) in mu_B^2 / eV, kB = %.10e eV/K\n",
          K_B);
  fprintf(file_, "# num_atoms %d\n", atom.number_of_atoms);
  fprintf(
    file_,
    "# cell %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e %.10e\n",
    box.cpu_h[0], box.cpu_h[3], box.cpu_h[6],
    box.cpu_h[1], box.cpu_h[4], box.cpu_h[7],
    box.cpu_h[2], box.cpu_h[5], box.cpu_h[8]);
  fprintf(file_, "# dt_output %.10e fs\n", time_step * dump_interval_ * TIME_UNIT_CONVERSION);
  fprintf(file_, "# columns step T Mx My Mz |M|");
  for (int t = 0; t < num_types_; ++t) {
    fprintf(file_, " Mx%d My%d Mz%d |M|%d", t, t, t, t);
  }
  // P28: running fluctuation and susceptibility, computed over the samples of
  // this run.  format_version is bumped to 2 because these columns are new.
  fprintf(file_, " nSample varM chi");
  for (int t = 0; t < num_types_; ++t) {
    fprintf(file_, " varM%d chi%d", t, t);
  }
  // P99 / R32: saturation baseline and the reduced order parameter.  Appended
  // at the end so that parsers written for format_version 2 keep working.
  fprintf(file_, " Msat Mnorm");
  for (int t = 0; t < num_types_; ++t) {
    fprintf(file_, " Msat%d Mnorm%d", t, t);
  }
  fprintf(
    file_,
    "\n# Msat = sum_i |S_i| (mu_B);  Mnorm = |sum_i S_i| / Msat  "
    "(1 = collinear, 1/sqrt(N) = random)\n");
  fflush(file_);
}

void Dump_Magnetization::end_of_step(
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
  if (!dump_)
    return;
  if ((step + 1) % dump_interval_ != 0)
    return;

  const int N = atom.number_of_atoms;
  std::vector<float> s(static_cast<size_t>(N) * 3, 0.0f);
  atom.spin.copy_to_host(s.data());

  double total[3] = {0.0, 0.0, 0.0};
  std::vector<double> per_type(static_cast<size_t>(num_types_) * 3, 0.0);
  // P99 / R32: accumulate sum_i |S_i| in the same sweep.
  sat_total_ = 0.0;
  std::vector<double> sat_type(static_cast<size_t>(num_types_), 0.0);
  for (int n = 0; n < N; ++n) {
    const int t = (n < static_cast<int>(atom.cpu_type.size())) ? atom.cpu_type[n] : 0;
    const double sx = static_cast<double>(s[n]);
    const double sy = static_cast<double>(s[n + static_cast<size_t>(N)]);
    const double sz = static_cast<double>(s[n + static_cast<size_t>(2) * N]);
    const double sn = std::sqrt(sx * sx + sy * sy + sz * sz);
    sat_total_ += sn;
    for (int d = 0; d < 3; ++d) {
      const double v = static_cast<double>(s[n + static_cast<size_t>(d) * N]);
      total[d] += v;
      if (t >= 0 && t < num_types_) {
        per_type[t * 3 + d] += v;
      }
    }
    if (t >= 0 && t < num_types_) {
      sat_type[t] += sn;
    }
  }
  sat_per_type_ = sat_type;
  cpu_total_[0] = total[0];
  cpu_total_[1] = total[1];
  cpu_total_[2] = total[2];
  cpu_per_type_ = per_type;

  // `temperature` is the *target* final temperature of the run, which is only
  // meaningful for a thermostat.  Under `ensemble nve` (and any other ensemble
  // that never sets it) it is left uninitialised, and the T column of the
  // dump -- and chi = var(M)/(N kB T) with it -- came out as garbage (a
  // denormal such as 9.88e-324, which is finite and positive and therefore
  // slips past an `isfinite` test).  Any temperature below 1e-6 K is not a
  // physical target, so treat it as "unset" and fall back to the instantaneous
  // kinetic temperature, which thermo[0] already carries.
  double t_reported = temperature;
  if (!std::isfinite(t_reported) || !(t_reported > 1.0e-6)) {
    double cpu_thermo[8];
    thermo.copy_to_host(cpu_thermo, 8);
    if (std::isfinite(cpu_thermo[0]) && cpu_thermo[0] > 1.0e-6) {
      t_reported = cpu_thermo[0];
    }
  }
  write_magnetization(step, t_reported);
}

void Dump_Magnetization::write_magnetization(const int step, const double temperature)
{
  // --- running fluctuation accumulators (P28) -------------------------
  // Samples before the warm-up threshold are written but not accumulated.
  const bool counted = (n_written_ >= warmup_samples_);
  ++n_written_;
  if (!counted) {
    fprintf(file_, "%d%20.10e%20.10e%20.10e%20.10e%20.10e",
            step, temperature, cpu_total_[0], cpu_total_[1], cpu_total_[2],
            std::sqrt(cpu_total_[0] * cpu_total_[0] + cpu_total_[1] * cpu_total_[1] +
                      cpu_total_[2] * cpu_total_[2]));
    for (int t = 0; t < num_types_; ++t) {
      const double inv = (atoms_of_type_[t] > 0) ? (1.0 / atoms_of_type_[t]) : 0.0;
      const double mx = cpu_per_type_[t * 3 + 0] * inv;
      const double my = cpu_per_type_[t * 3 + 1] * inv;
      const double mz = cpu_per_type_[t * 3 + 2] * inv;
      fprintf(file_, "%20.10e%20.10e%20.10e%20.10e", mx, my, mz,
              std::sqrt(mx * mx + my * my + mz * mz));
    }
    fprintf(file_, "%10lld%20.10e%20.10e", 0LL, 0.0, 0.0);
    for (int t = 0; t < num_types_; ++t) {
      fprintf(file_, "%20.10e%20.10e", 0.0, 0.0);
    }
    // P99 / R32: Msat / Mnorm are instantaneous, so they are valid even for
    // the warm-up rows that are excluded from the fluctuation accumulators.
    fprintf(file_, "%20.10e%20.10e", sat_total_, norm_total() / sat_total_norm());
    for (int t = 0; t < num_types_; ++t) {
      const double mx = sat_per_type_[t] > 0.0
                          ? (cpu_per_type_[t * 3 + 0] / sat_per_type_[t])
                          : 0.0;
      const double my = sat_per_type_[t] > 0.0
                          ? (cpu_per_type_[t * 3 + 1] / sat_per_type_[t])
                          : 0.0;
      const double mz = sat_per_type_[t] > 0.0
                          ? (cpu_per_type_[t * 3 + 2] / sat_per_type_[t])
                          : 0.0;
      fprintf(file_, "%20.10e%20.10e", sat_per_type_[t], std::sqrt(mx * mx + my * my + mz * mz));
    }
    fprintf(file_, "\n");
    fflush(file_);
    return;
  }
  n_samples_++;
  for (int d = 0; d < 3; ++d) {
    sum_total_[d] += cpu_total_[d];
  }
  sum2_total_ += cpu_total_[0] * cpu_total_[0] + cpu_total_[1] * cpu_total_[1] +
                 cpu_total_[2] * cpu_total_[2];
  for (int t = 0; t < num_types_; ++t) {
    for (int d = 0; d < 3; ++d) {
      sum_per_type_[t * 3 + d] += cpu_per_type_[t * 3 + d];
    }
    sum2_per_type_[t] += cpu_per_type_[t * 3 + 0] * cpu_per_type_[t * 3 + 0] +
                         cpu_per_type_[t * 3 + 1] * cpu_per_type_[t * 3 + 1] +
                         cpu_per_type_[t * 3 + 2] * cpu_per_type_[t * 3 + 2];
  }
  const double inv_n = 1.0 / static_cast<double>(n_samples_);
  const double mean2 = (sum_total_[0] * sum_total_[0] + sum_total_[1] * sum_total_[1] +
                        sum_total_[2] * sum_total_[2]) *
                       inv_n * inv_n;
  const double var_total = sum2_total_ * inv_n - mean2;
  // chi = var(M) / (N kB T); undefined at T = 0, reported as 0 there.
  const double denom = static_cast<double>(number_of_atoms_) * K_B * temperature;
  const double chi_total = (denom > 0.0 && var_total > 0.0) ? var_total / denom : 0.0;

  const double norm = std::sqrt(
    cpu_total_[0] * cpu_total_[0] + cpu_total_[1] * cpu_total_[1] +
    cpu_total_[2] * cpu_total_[2]);
  fprintf(
    file_, "%d%20.10e%20.10e%20.10e%20.10e%20.10e",
    step, temperature, cpu_total_[0], cpu_total_[1], cpu_total_[2], norm);
  for (int t = 0; t < num_types_; ++t) {
    const double inv = (atoms_of_type_[t] > 0) ? (1.0 / atoms_of_type_[t]) : 0.0;
    const double mx = cpu_per_type_[t * 3 + 0] * inv;
    const double my = cpu_per_type_[t * 3 + 1] * inv;
    const double mz = cpu_per_type_[t * 3 + 2] * inv;
    fprintf(
      file_, "%20.10e%20.10e%20.10e%20.10e", mx, my, mz,
      std::sqrt(mx * mx + my * my + mz * mz));
  }
  fprintf(file_, "%10lld%20.10e%20.10e", n_samples_, var_total, chi_total);
  for (int t = 0; t < num_types_; ++t) {
    const double m2 = (sum_per_type_[t * 3 + 0] * sum_per_type_[t * 3 + 0] +
                       sum_per_type_[t * 3 + 1] * sum_per_type_[t * 3 + 1] +
                       sum_per_type_[t * 3 + 2] * sum_per_type_[t * 3 + 2]) *
                      inv_n * inv_n;
    const double var_t = sum2_per_type_[t] * inv_n - m2;
    const double chi_t = (denom > 0.0 && var_t > 0.0) ? var_t / denom : 0.0;
    fprintf(file_, "%20.10e%20.10e", var_t, chi_t);
  }
  // P99 / R32: Msat and the reduced order parameter Mnorm.
  fprintf(file_, "%20.10e%20.10e", sat_total_, norm / sat_total_norm());
  for (int t = 0; t < num_types_; ++t) {
    const double mx = sat_per_type_[t] > 0.0 ? (cpu_per_type_[t * 3 + 0] / sat_per_type_[t])
                                             : 0.0;
    const double my = sat_per_type_[t] > 0.0 ? (cpu_per_type_[t * 3 + 1] / sat_per_type_[t])
                                             : 0.0;
    const double mz = sat_per_type_[t] > 0.0 ? (cpu_per_type_[t * 3 + 2] / sat_per_type_[t])
                                             : 0.0;
    fprintf(
      file_, "%20.10e%20.10e", sat_per_type_[t],
      std::sqrt(mx * mx + my * my + mz * mz));
  }
  fprintf(file_, "\n");
  fflush(file_);
}

void Dump_Magnetization::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (dump_) {
    fclose(file_);
    file_ = nullptr;
    dump_ = false;
  }
}
