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

#pragma once
#include "action.cuh"
#include "integrate/integrate.cuh"
#include "model/atom.cuh"
#include "model/group.cuh"
#include <string>
#include <vector>
#include <cmath>
class Box;
class Atom;
class Force;
class Integrate;

// Dump the total and the per-type magnetisation of the spin system.
//
//   dump_magnetization <interval> [<filename>]
//
// M = sum_i S_i, written every <interval> steps to <filename>
// (magnetization.out by default), together with the instantaneous
// temperature so that an M(T) curve can be read off directly.  Per type the
// average vector is reported, that is M_type / N_type, so that the numbers
// are comparable between types with different site counts.
//
// The same file also carries the running fluctuation of M,
//
//   var(M) = <M.M> - <M>.<M>,      chi = var(M) / (N kB T),
//
// which is what an M(T) plot cannot show: chi peaks at the Curie temperature,
// so Tc can be located straight from the dump without a separate analysis
// pass.  The running sums accumulate over the dump samples of the current run
// and are reset in pre_run(), i.e. they describe this run only.
//
//   dump_magnetization <interval> [<filename>] [warmup <n>]
//
// `warmup <n>` drops the first n dump samples from the fluctuation (they are
// still written as rows).  Without it the initial relaxation of a spin system
// started from a DFT configuration dominates the variance.
class Dump_Magnetization : public Action
{
public:
  Dump_Magnetization(const std::vector<std::string>& tokens);
  void parse(const std::vector<std::string>& tokens);
  virtual void pre_run(
    const int number_of_steps,
    const double time_step,
    Integrate& integrate,
    std::vector<Group>& group,
    Atom& atom,
    Box& box,
    Force& force);

  virtual void end_of_step(
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
      Force& force);

  virtual void post_run(
    Atom& atom,
    Box& box,
    Integrate& integrate,
    const int number_of_steps,
    const double time_step,
    const double temperature);

private:
  bool dump_ = false;
  int dump_interval_ = 1;
  int num_types_ = 1;
  // Number of dump samples discarded from the fluctuation accumulators.  A
  // spin system started from a DFT configuration relaxes for a while before it
  // equilibrates (in the Fe test cell |M| moves by more than a factor of two),
  // and that transient would dominate <M.M> - <M>.<M> and put a spurious peak
  // at the lowest temperature.  Default 0 = accumulate everything.
  int warmup_samples_ = 0;
  std::string filename_ = "magnetization.out";
  FILE* file_ = nullptr;
  std::vector<int> atoms_of_type_;
  double cpu_total_[3] = {0.0, 0.0, 0.0};
  std::vector<double> cpu_per_type_;
  // P99 / R32: the saturation baseline sum_i |S_i|, total and per type.
  // `|M|` is the *unnormalised* sum |sum_i S_i| and the per-type column is
  // |sum S| / N_type, so neither can be compared between two runs that differ
  // in atom count or in the spin length (|S| is not exactly conserved: the
  // frozen-lattice Fe runs show it drifting 2.2633 -> 2.2666).  Reporting
  // Msat alongside makes the reduced order parameter
  //     Mnorm = |sum_i S_i| / sum_i |S_i|
  // available in the file itself, instead of forcing every reader to recover
  // it from gpumd.log.  Mnorm is dimensionless: 1 = perfectly collinear,
  // 1/sqrt(N) = fully random.
  double sat_total_ = 0.0;
  std::vector<double> sat_per_type_;
  // Running fluctuation accumulators (P28).  sum_/sum2_ are over the dump
  // samples; var = sum2/n - (sum/n).(sum/n), and chi = var / (N kB T).
  long long n_written_ = 0;   // dump samples written so far
  long long n_samples_ = 0;   // dump samples fed into the accumulators
  double sum_total_[3] = {0.0, 0.0, 0.0};
  double sum2_total_ = 0.0;
  std::vector<double> sum_per_type_;
  std::vector<double> sum2_per_type_;
  int number_of_atoms_ = 0;
  void write_magnetization(const int step, const double temperature);
  // P99 / R32: |sum_i S_i| and the guarded saturation baseline.  The guard
  // keeps Mnorm finite (0) if every spin length is zero.
  double norm_total() const
  {
    return std::sqrt(
      cpu_total_[0] * cpu_total_[0] + cpu_total_[1] * cpu_total_[1] +
      cpu_total_[2] * cpu_total_[2]);
  }
  double sat_total_norm() const { return (sat_total_ > 0.0) ? sat_total_ : 1.0; }
};
