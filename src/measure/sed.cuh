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

/*  compute_sed  --  spectral energy density
 *  ----------------------------------------
 *  Upstream GPUMD has no classical SED.  It offers `compute_gkma` /
 *  `compute_hnema`, but those are modal-decomposition methods and need an
 *  external eigenvector.in produced by a lattice-dynamics code; they cannot
 *  be used when only an MD trajectory is available, and they do not give a
 *  q-resolved spectrum directly.
 *
 *  This action implements the standard SED of Thomas, Turney, McGaughey and
 *  Amon (Phys. Rev. B 81, 081411(R) (2010)):
 *
 *      V_a(q, t) = sum_i sqrt(m_i) v_{i,a}(t) exp( i q . r_i^0 )
 *
 *      Phi(q, nu) = sum_a | Int dt V_a(q, t) exp(-2 pi i nu t) |^2
 *
 *  where r_i^0 is a *reference* (equilibrium / initial, unwrapped) position.
 *  Using unwrapped positions is essential: with wrapped positions the phase
 *  jumps by 2 pi whenever an atom crosses the box, which smears the whole
 *  spectrum.
 *
 *  With `spin` mode the velocity is replaced by the spin vector itself,
 *      V_a(q, t) = sum_i S_{i,a}(t) exp( i q . r_i^0 )
 *  which gives the magnon (spin-wave) spectral density of a spin-aware NEP.
 *
 *  run.in:
 *      compute_sed <sample_interval> <num_q> <direction> [<mode>] [<series>]
 *        <direction> : x | y | z    (the q points are k * 2pi/L, k=0..num_q-1)
 *        <mode>      : velocity (default) | spin
 *        <series>    : 1 (default) also write the raw time series to
 *                      sed_qv.out, 0 to skip it
 *
 *  Output:
 *      sed.out     nu(THz)  Phi(q_0,nu) Phi(q_1,nu) ... Phi(q_{num_q-1},nu)
 *      sed_qv.out  t(ps)  Re/Im of V_a(q_k, t) for every (q_k, a)   [if series=1]
 *
 *  Requirements: the sampling run must be `ensemble nve`.  A thermostat or a
 *  barostat perturbs the dynamics and the spectra are no longer the harmonic
 *  normal modes; equilibrate in a previous `run` block and sample here.
 */

#pragma once
#include "action.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <cstdio>
#include <string>
#include <vector>

class SED : public Action
{
public:
  SED(const std::vector<std::string>& tokens);

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
  bool compute = false;
  int sample_interval = 1;
  int num_q = 1;
  int direction = 0; // 0 -> x, 1 -> y, 2 -> z
  bool spin_mode = false;
  bool dump_series = true;

  int number_of_frames = 0;
  int frame_index = 0;
  double q_step = 0.0; // 2 pi / L_dir   [1/A]
  double q_dir_[3] = {1.0, 0.0, 0.0}; // unit vector along the box vector (= q direction)
  double dt_sample_ps = 0.0;

  // reference positions, snapshotted once in pre_run (unwrapped, 3N doubles)
  GPU_Vector<double> r0_;
  // per-frame projection, 3 * num_q complex numbers = 6 * num_q doubles
  GPU_Vector<double> frame_gpu_;
  std::vector<double> frame_cpu_;
  // full series on the device: [frame][q*3+a][re,im]
  GPU_Vector<double> series_gpu_;
};
