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

/*  dump_heat_flux
 *  --------------
 *  Writes the *time series* of the macroscopic heat flux, which is the raw
 *  input of every Green-Kubo (EMD) calculation and the quantity that closes
 *  the energy balance in a NEMD run.  Upstream GPUMD only exposes the heat
 *  current through `compute_hac`, which immediately reduces it to an
 *  autocorrelation function and throws the series away.  That is fine for
 *  the built-in kappa workflow but makes it impossible to
 *    - apply a custom window / filter before integrating the correlation,
 *    - split the flux into in-plane and cross-plane components afterwards,
 *    - compute the flux-flux cross correlation with another observable,
 *    - verify that the NEMD flux has reached a steady state.
 *  This action keeps the series instead.
 *
 *  run.in:
 *      dump_heat_flux <sample_interval>
 *      dump_heat_flux <sample_interval> bin/1d <x|y|z> <nbin>
 *
 *  The first form writes the *global* flux, which is what Green-Kubo needs:
 *
 *      heat_flux.out:  time(ps)  Jx  Jy  Jz  |J|      [eV / (A^2 ps)]
 *
 *  The second form writes one flux vector per bin along an axis:
 *
 *      heat_flux.out:  time(ps)  Jx(b0) Jy(b0) Jz(b0)  Jx(b1) ...  [eV/(A^2 ps)]
 *
 *  The binned form is the one NEMD needs.  In a periodic NEMD setup the heat
 *  source sits next to the sink on *both* sides, so the heat flows away from
 *  the source in two opposite directions and the global sum cancels to zero
 *  (measured: <Jx> = 0.97 +- 1.24, i.e. indistinguishable from zero, while the
 *  temperature profile is a clean 395 K -> 204 K ramp).  Only the flux in the
 *  slab between the source and the sink is the conducted flux.
 */

#pragma once
#include "action.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <string>
#include <vector>

class Dump_Heat_Flux : public Action
{
public:
  Dump_Heat_Flux(const std::vector<std::string>& tokens);

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
  int sample_interval = 1;
  bool dump = false;
  bool use_bins = false;
  int axis = 0;    // 0 -> x, 1 -> y, 2 -> z
  int nbin = 1;
  double bin_length = 1.0; // box length along `axis` divided by nbin
  FILE* fid = nullptr;
  // Private scratch: `compute_hac` owns Atom::heat_per_atom, and writing into
  // it from here would corrupt the HAC accumulation when both keywords are
  // present in the same run.
  GPU_Vector<double> heat_per_atom_;  // 5 components per atom (HAC/HNEMD form)
  GPU_Vector<double> heat_xyz_;       // packed Cartesian jx, jy, jz (N each)
  std::vector<double> heat_cpu_;
  GPU_Vector<double> bin_out_;   // binned form: nbin * 3 on the device
  std::vector<double> bin_cpu_;
  double volume = 1.0;
  long long number_of_frames = 0;
  // Round 9 (task 48): when the model carries spins the file also carries the
  // two channels separately, so that a Green-Kubo post-processing can build
  //   kappa_pp ~ <J_p J_p>,   kappa_mm ~ <J_m J_m>,   kappa_pm ~ <J_p J_m>
  // and hence the full  kappa = kappa_pp + kappa_pm + kappa_mp + kappa_mm.
  bool split_channels = false;
  GPU_Vector<double> heat_phonon_;  // 5 components per atom, J_p
  GPU_Vector<double> heat_magnon_;  // 5 components per atom, J_m
  std::vector<double> heat_phonon_cpu_;
  std::vector<double> heat_magnon_cpu_;
};
