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
Run simulation according to the inputs in the run.in file.
------------------------------------------------------------------------------*/

#include "cohesive.cuh"
#include "force/force.cuh"
#include "integrate/ensemble.cuh"
#include "integrate/integrate.cuh"
#include "measure/measure.cuh"
#include "minimize/minimize.cuh"
#include "model/box.cuh"
#include "model/read_xyz.cuh"
#include "phonon/hessian.cuh"
#include "replicate.cuh"
#include "run.cuh"
#include "spin/spin_field_provider.cuh"
#include <cstdio>
#include "utilities/error.cuh"
#include "utilities/compact_nep.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/main_common.cuh"
#include "utilities/read_file.cuh"
#include "utilities/run_input.cuh"
#include "velocity.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>

// F21: process exit status.  A run cut short by the R19 spin guard used to exit
// with 0, so every wrapper script read a truncated trajectory as a success.
// 0 = ok, 2 = stalled by the spin guard, 3 = non-finite potential energy.
int g_run_status = 0;

// How the spin update interleaves with the lattice step.
//   0 = staggered (historical, now opt-in): one spin step of dt after the force
//       call, using F(r_{n+1}, S_n).  The lattice kick then uses the previous
//       step's spin state and the reported energy belongs to S_{n+1/2}: first
//       order in dt and a mixed output state.
//   1 = strang (default since S05B): S(dt/2) V(dt) S(dt/2), with the force for
//       each VV kick evaluated at the single spin state that exists during V,
//       and a final force/energy evaluation at (r_{n+1}, S_{n+1}).
// Keyword: `spin_split staggered|strang`.
int g_spin_split = 1;

// F23 / risk #13: the spin propagator uses a torque frozen over the whole
// (sub)step, which for alpha = 0 is an exact rotation about a fixed axis.
// Subdividing without refreshing the torque therefore changes nothing; the
// first-order error is the stale torque.  `spin_substeps` only pays off with
// `spin_torque_refresh on`, which re-evaluates the torque between sub-steps
// (as LAMMPS does in fix nve/spin) at one extra force call per sub-step.
// Defaults keep the historical path: one sub-step, no refresh.
int g_spin_substeps = 1;
int g_spin_torque_refresh = 0;

// F25: `velocity` was seen; emit its note at the next `run`, and only if spin
// dynamics is on (non-spin stdout must stay identical to the baseline).
static bool velocity_note_pending = false;

static __global__ void gpu_find_largest_v2(
  int N, int number_of_rounds, double* g_vx, double* g_vy, double* g_vz, double* g_v2_max)
{
  int tid = threadIdx.x;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = round * 1024 + tid;
    if (n < N) {
      double vx = g_vx[n];
      double vy = g_vy[n];
      double vz = g_vz[n];
      double v2 = vx * vx + vy * vy + vz * vz;
      if (s_data[tid] < v2) {
        s_data[tid] = v2;
      }
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_data[tid] < s_data[tid + offset]) {
        s_data[tid] = s_data[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_v2_max[0] = s_data[0];
  }
}

__device__ double device_v2_max[1];

static void calculate_time_step(
  double max_distance_per_step,
  GPU_Vector<double>& velocity_per_atom,
  double initial_time_step,
  double& time_step)
{
  if (max_distance_per_step <= 0.0) {
    return;
  }
  const int N = velocity_per_atom.size() / 3;
  double* gpu_v2_max;
  CHECK(gpuGetSymbolAddress((void**)&gpu_v2_max, device_v2_max));
  gpu_find_largest_v2<<<1, 1024>>>(
    N,
    (N - 1) / 1024 + 1,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + N * 2,
    gpu_v2_max);
  GPU_CHECK_KERNEL
  double cpu_v2_max[1] = {0.0};
  CHECK(gpuMemcpy(cpu_v2_max, gpu_v2_max, sizeof(double), gpuMemcpyDeviceToHost));
  double cpu_v_max = sqrt(cpu_v2_max[0]);
  double time_step_min = max_distance_per_step / cpu_v_max;

  if (time_step_min < initial_time_step) {
    time_step = time_step_min;
  } else {
    time_step = initial_time_step;
  }
}

static bool parse_initial_replicate(
  const RunInput& run_input, int replicate_size[3])
{
  for (const auto& line : run_input.lines()) {
    if (line.tokens.empty()) {
      continue;
    }
    if (line.tokens[0] != "replicate") {
      return false;
    }
    parse_replicate(line.tokens, replicate_size);
    return true;
  }
  return false;
}

Run::Run(const RunInput& run_input)
{
  print_line_1();
  printf("Started initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  has_replicate_ = parse_initial_replicate(run_input, replicate_size_);

  initialize_position(
    run_input,
    has_velocity_in_xyz,
    number_of_types,
    box,
    group,
    atom);
  first_potential_filename_ = get_first_potential_filename(run_input);

  if (has_replicate_) {
    Replicate(replicate_size_, box, atom, group);
  }

  if (atom.number_of_atoms < 2) {
    PRINT_INPUT_ERROR("Number of atoms should >= 2.");
  }

  velocity.initialize_cpu(
    has_velocity_in_xyz, 300, atom, false, 123);

  if (has_velocity_in_xyz) {
    printf("Initialized velocities with data in model.xyz.\n");
  } else {
    printf("Initialized velocities with default T = 300 K.\n");
  }

  allocate_memory_gpu(group, atom, thermo);

  print_line_1();
  printf("Finished initializing positions and related parameters.\n");
  fflush(stdout);
  print_line_2();

  execute_run_in(run_input);
}

void Run::execute_run_in(const RunInput& run_input)
{
  print_line_1();
  printf("Started executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

  bool first_effective_command = true;
  for (const auto& line : run_input.lines()) {
    if (!line.tokens.empty()) {
      std::vector<std::string> tokens = line.tokens;
      if (tokens.size() >= 2 && tokens[0] == "potential") {
        tokens[1] = get_compact_nep_filename(tokens[1]);
      }
      if (has_replicate_ && first_effective_command &&
          tokens[0] == "replicate") {
        first_effective_command = false;
        continue;
      }
      first_effective_command = false;
      parse_one_keyword(tokens, run_input);
    }
  }

  if (integrate.has_ensemble()) {
    PRINT_INPUT_ERROR("The last ensemble is not followed by a run.");
  }

  print_line_1();
  printf("Finished executing the commands in run.in.\n");
  fflush(stdout);
  print_line_2();

}

void Run::compute_force()
{
  if (is_pimd(integrate.get_type())) {
    for (int k = 0; k < integrate.get_number_of_beads(); ++k) {
      force.compute(
        box,
        atom.position_beads[k],
        atom.type,
        group,
        atom.potential_beads[k],
        atom.force_beads[k],
        atom.virial_beads[k],
        atom.velocity_beads[k],
        atom.mass);
    }
  } else {
    force.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom,
      atom.velocity_per_atom,
      atom.mass,
      atom.position_image.size() > 0 ? atom.position_image.data() : nullptr,
      atom.has_spin ? atom.spin.data() : nullptr,
      atom.has_spin ? atom.torque.data() : nullptr);
  }
}

void Run::perform_a_run(const int number_of_steps)
{
  integrate.initialize(time_step, atom, box, group);
  measure.pre_run(number_of_steps, time_step, integrate, group, atom, box, force);

  // F16 / R12: start every run with an empty spin work budget.
  spin.reset_work();
  if (spin.enabled) {
    // suggestion #3: keep the budget in a file as well, so the
    // injection/dissipation ratio can be followed WHILE the run is going
    // (an online runaway criterion instead of a post-mortem).
    FILE* fid = fopen("spin_work.out", "w");
    if (fid) {
      fprintf(fid,
              "# F16 spin work budget, cumulative since the start of this run\n"
              "# step  W_rot(eV)  W_noise(eV)  W_ext(eV)\n");
      fclose(fid);
    }
  }

  // F17 / R12 + R1: a finite SLLG temperature on an NVE lattice is the exact
  // combination that made long CrN runs run away (CUDA 700 at 435k steps,
  // dE = +221 eV).  The thermal field is a net energy SOURCE; with no lattice
  // thermostat to absorb it the total energy grows monotonically.  Say so up
  // front instead of letting the run die much later with a deferred CUDA error.
  if (spin.enabled && spin.temperature > 0.0 &&
      integrate.get_type() == EnsembleType::NVE) {
    printf(
      "WARNING: spin_llg is running with a finite temperature (%g K) while the\n"
      "         lattice is integrated in the NVE ensemble.  The SLLG thermal field\n"
      "         is a net energy source (risk R12): measured on CrN at 1024 atoms it\n"
      "         pumps +6.8e-08 eV/atom/step into the lattice and the run heats up\n"
      "         monotonically (+4.5 eV over 3e5 steps); the same protocol at 128\n"
      "         atoms cools, because the injection only wins when it exceeds the\n"
      "         Gilbert damping sink.  Use a lattice thermostat (e.g.\n"
      "         'ensemble nvt_lan 300 300 100'), or set the spin temperature to 0,\n"
      "         or keep the run below ~3e5 steps.\n",
      spin.temperature);
    fflush(stdout);
  }

  // setup force for the first integrate step
  compute_force();
  atom.update_unwrapped_position(box);
  measure.setup_force(time_step, integrate, group, atom, box, force);

  // F18 / R19: finite-temperature spin dynamics can leave the collinear
  // training manifold and drive the coupled potential into an unphysical deep
  // well before the deferred neighbour-list CUDA error is reported.  Keep an
  // independent online reference for temperature, potential energy and spin
  // work.  The guard below stops the run cleanly; it does not pretend to repair
  // a potential that lacks high-temperature non-collinear training data.
  // F19: the guard must also cover conservative runs.  Round 10 showed that
  // NVE + deterministic LLG (alpha = 0, no stochastic field at all) heats the
  // lattice from 304 K to 1409 K in 6.1e5 steps and then dies with the deferred
  // CUDA 700, while the F16 spin-work budget stays below 0.04 eV -- the
  // spin/lattice splitting leaks energy that -sum_i B_i.dS_i does not see.
  // Requiring spin.temperature > 0 therefore left the worst case unprotected.
  const bool spin_guard_active = spin.enabled;
  const int spin_guard_interval = 1000;
  double spin_guard_initial_pe = 0.0;
  double spin_guard_previous_pe = 0.0;
  double spin_guard_previous_work = 0.0;
  double spin_guard_reference_temperature = 0.0;
  double spin_guard_reference_energy = 0.0;
  int spin_guard_injection_streak = 0;
  bool spin_guard_triggered = false;
  bool spin_guard_primed = false;
  // F20 (round 12): a frozen-lattice SLLG run (`fix` covering every atom) has
  // no lattice kinetic energy, so thermo[0] is 0/0 = NaN and the total energy
  // is not conserved (the SLLG thermostat is a heat bath, not a conservative
  // dynamics).  That is a legitimate configuration for measuring the purely
  // magnetic heat current, so the guard must not stop it.  Tracked here so the
  // references are re-primed at the start of the frozen stage.
  bool spin_guard_frozen_seen = false;
  int completed_steps = 0;

  // F23: scratch force, so that a mid-step torque refresh can call
  // compute_force() without disturbing the force that closes the kick (F15).
  GPU_Vector<double> force_scratch;
  if (spin.enabled && g_spin_torque_refresh) {
    force_scratch.resize(atom.number_of_atoms * 3);
  }
  if (spin.enabled && g_spin_substeps > 1 && !g_spin_torque_refresh) {
    // Resolved here rather than in the parser, because run.in may declare the
    // two keywords in either order.
    printf("WARNING: spin_substeps = %d without `spin_torque_refresh on` changes\n"
           "         nothing: for alpha = 0 the propagator is an exact rotation\n"
           "         about the frozen field, and rotations about one axis compose\n"
           "         exactly.  Add `spin_torque_refresh on` to actually buy accuracy.\n",
           g_spin_substeps);
  }

  // S05A: field provider for the conservative (alpha = 0, T = 0) spin update.
  // Constructed after every setup step of the run, right before the loop, so the
  // pointers it captures (positions, types, group, box) are final.  Unused unless
  // spin.propagate() takes the implicit-midpoint branch.
  NepSpinFieldProvider<Atom, Force> spin_field_provider(
    atom, force, box, group,
    atom.position_image.size() > 0 ? atom.position_image.data() : nullptr);

  double initial_time_step = time_step;

  const auto time_begin = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < number_of_steps; ++step) {

    velocity.correct_velocity(step, group, atom);

    calculate_time_step(
      max_distance_per_step, atom.velocity_per_atom, initial_time_step, time_step);
    global_time += time_step;

    // S05B: symmetric (Strang) spin-lattice splitting
    //     S(dt/2)  V(dt)  S(dt/2)
    // Every force evaluation below belongs to a state that actually exists in
    // the splitting, and the step *ends* with a force/energy evaluation at the
    // final (r_{n+1}, S_{n+1}) so that thermo, trajectory, restart and the
    // magnetisation dumps all read one and the same state:
    //   A  S: S_n -> S_{n+1/2}        field at (r_n,     S_n)
    //   B  F(r_n,     S_{n+1/2})      first VV kick
    //   C  VV: half kick + drift      r_n -> r_{n+1}
    //   D  F(r_{n+1}, S_{n+1/2})      second VV kick
    //   E  VV: second half kick
    //   F  S: S_{n+1/2} -> S_{n+1}    field at (r_{n+1}, S_{n+1/2})
    //   G  F(r_{n+1}, S_{n+1})        the isochronous output state
    // `spin_split staggered` keeps the historical ordering (spin advanced once,
    // after the force call, so the VV kick uses the previous step's spin state
    // and the reported energy belongs to S_{n+1/2}).  It is kept only to
    // reproduce pre-S05B runs and warns when selected.
    const bool symmetric_split = spin.enabled && g_spin_split == 1;

    // A: first spin half step.  The conservative branch re-evaluates the field
    // itself at (r_n, S_n), so no stale torque enters here.
    if (symmetric_split) {
      spin.propagate(
        0.5 * time_step * TIME_UNIT_CONVERSION,
        atom.spin.data(),
        atom.torque.data(),
        atom.number_of_atoms,
        0,
        &spin_field_provider);
    }

    // B: force at (r_n, S_{n+1/2}) -- the state the V sub-step starts from.
    // The torque in atom.torque at this point is the one from the previous
    // step, i.e. evaluated at (r_n, S_n), which is exactly right for step A
    // above but *not* for the VV kick; recomputing here is what makes the
    // lattice see one single spin state during V.
    if (symmetric_split) {
      compute_force();
    }

    integrate.compute1(time_step, step, number_of_steps, group, box, atom, thermo);
    measure.post_integrate1(step, time_step, integrate, group, atom, box, force);

    force.advance_temperature();

    measure.pre_force(step, time_step, integrate, group, atom, box, force);
    // D: force at (r_{n+1}, S_{n+1/2}); this is also the force that closes the
    // velocity-Verlet step below.
    compute_force();

    // Legacy staggered path: one full spin step, torque frozen at (r_{n+1}, S_n).
    if (spin.enabled && !symmetric_split) {
      const int nsub = g_spin_substeps;
      const double sdt = time_step * TIME_UNIT_CONVERSION / nsub;
      for (int ksub = 0; ksub < nsub; ++ksub) {
        spin.propagate(
          sdt,
          atom.spin.data(),
          atom.torque.data(),
          atom.number_of_atoms,
          0,
          &spin_field_provider);

        if (g_spin_torque_refresh && ksub + 1 < nsub) {
          const size_t nbytes = sizeof(double) * 3 * atom.number_of_atoms;
          cudaMemcpy(force_scratch.data(), atom.force_per_atom.data(), nbytes,
                     cudaMemcpyDeviceToDevice);
          compute_force();
          cudaMemcpy(atom.force_per_atom.data(), force_scratch.data(), nbytes,
                     cudaMemcpyDeviceToDevice);
        }
      }
    }

    // The magnon heat current must use the field the spins actually saw,
    // otherwise `add_bfield` runs report a flux foreign to their trajectory.
    if (spin.enabled) {
      spin.get_applied_external_field(atom.spin_b_ext);
    }

    atom.update_unwrapped_position(box);
    measure.post_force(step, time_step, integrate, group, atom, box, force);

    // E: second half kick, with the force evaluated at (r_{n+1}, S_{n+1/2}).
    integrate.compute2(time_step, step, number_of_steps, group, box, atom, thermo, force);
    atom.update_unwrapped_position(box);

    // F + G: second spin half step, then the state the step is reported at.
    if (symmetric_split) {
      spin.propagate(
        0.5 * time_step * TIME_UNIT_CONVERSION,
        atom.spin.data(),
        atom.torque.data(),
        atom.number_of_atoms,
        0,
        &spin_field_provider);

      // G: the spin has moved since D, so potential, forces, virial and torque
      // are stale for the final state.  Re-evaluate them once, and refresh the
      // thermo vector that the integrations computed from the pre-G arrays.
      compute_force();
      integrate.find_thermo(
        box.get_volume(),
        group,
        atom.mass,
        atom.potential_per_atom,
        atom.velocity_per_atom,
        atom.virial_per_atom,
        thermo);
      spin.get_applied_external_field(atom.spin_b_ext);
    }

    measure.end_of_step(
      number_of_steps,
      step,
      integrate.get_fixed_group(),
      integrate.get_move_group(),
      global_time,
      integrate.get_temperature2(),
      integrate,
      box,
      group,
      thermo,
      atom,
      force);

    completed_steps = step + 1;

    // F18 / F19 online guard.  Four independent signatures are checked every
    // 1000 steps, starting one interval late so that the reference state is a
    // relaxed configuration and not the (off-minimum) input snapshot:
    //   (1) sustained stochastic injection (W_noise > 1.05 |W_rot|);
    //   (2) a potential-energy fall > 0.15 eV/atom from the reference or
    //       > 0.05 eV/atom within one guard interval;
    //   (3) a lattice-temperature excursion above max(1.5*Tref, Tref+300K);
    //   (4) F19: in a conservative (NVE) run, a total-energy drift above
    //       0.02 eV/atom -- the spin/lattice splitting can pump the lattice
    //       without showing up in the -sum_i B_i.dS_i work budget at all.
    // R19 shows up as signature (2)/(3); the round-10 NVE failure as (4).
    // F24 / risk #14: a NaN cell used to finish with exit code 0 (CrN, stage E:
    // outside its V/atom training window every step returns PE = NaN, gpumd
    // writes header-only thermo.out files and returns 0, so every downstream
    // script treats the run as a success).  Check PE/KE every guard interval and
    // fail loudly with exit code 3.  T is NOT checked: a frozen lattice
    // legitimately has T = 0/0.
    if (spin_guard_interval > 0 &&
        (completed_steps % 1000 == 0 || completed_steps == number_of_steps)) {
      integrate.find_thermo(
        box.get_volume(),
        group,
        atom.mass,
        atom.potential_per_atom,
        atom.velocity_per_atom,
        atom.virial_per_atom,
        thermo);
      double nan_check[3] = {0.0, 0.0, 0.0};
      thermo.copy_to_host(nan_check, 3);
      if (!std::isfinite(nan_check[2]) || !std::isfinite(nan_check[1])) {
        g_run_status = 3;
        printf("\nERROR: non-finite energy at step %d (PE = %g, KE = %g).\n"
               "       The potential returned NaN/inf, which almost always means the\n"
               "       configuration left the training domain of the model (see the\n"
               "       V/atom range of train.xyz).  Stopping with exit code 3 instead\n"
               "       of finishing with exit code 0 and a thermo.out full of NaN.\n",
               completed_steps,
               nan_check[2],
               nan_check[1]);
        fflush(stdout);
        break;
      }
    }

    if (spin_guard_active && completed_steps % spin_guard_interval == 0) {
      integrate.find_thermo(
        box.get_volume(),
        group,
        atom.mass,
        atom.potential_per_atom,
        atom.velocity_per_atom,
        atom.virial_per_atom,
        thermo);
      // F26: find_thermo writes [T, KE, PE, virial ...]; the guard needs all
      // three scalars, and it needs PE from slot 2.  It used to read slot 1,
      // which is the kinetic energy, so the potential-energy signatures and the
      // energy-conservation signature were all watching KE instead.
      double current_thermo[3] = {0.0, 0.0, 0.0};
      thermo.copy_to_host(current_thermo, 3);
      double w_guard[3];
      spin.read_work(w_guard);

      const double n_inv = 1.0 / std::max(1, atom.number_of_atoms);
      const double w_net = w_guard[0] + w_guard[1] + w_guard[2];
      // Total energy = PE + the kinetic energy reconstructed from the
      // temperature exactly the way Dump_Thermo does it.  thermo[] layout is
      // [0] = T, [1] = PE (eV), [2..7] = virial in Voigt order.
      // F26: total energy is PE + KE, both already in thermo.  The old form
      // added KE twice (thermo[1] + 3/2 N kB T = 2 KE for a classical system).
      const double current_energy = current_thermo[2] + current_thermo[1];

      // F20 (round 12): detect the frozen-lattice case (see the declaration of
      // spin_guard_frozen_seen).  Every atom being fixed makes the lattice
      // temperature undefined, so the temperature and energy-conservation
      // criteria are switched off and the references are re-primed from the
      // frozen configuration.
      bool frozen_lattice = false;
      {
        const int fg = integrate.get_fixed_group();
        const int fm = integrate.get_fixed_grouping_method();
        if (fg >= 0 && fm >= 0 && fm < static_cast<int>(group.size())) {
          frozen_lattice = (group[fm].cpu_size[fg] == atom.number_of_atoms);
        }
      }
      if (frozen_lattice && !spin_guard_frozen_seen) {
        spin_guard_frozen_seen = true;
        spin_guard_primed = false;
        printf(
          "    [F20] frozen-lattice spin dynamics detected: the R19 guard "
          "switches to\n"
          "          spin-only mode (lattice-temperature and "
          "energy-conservation\n"
          "          criteria disabled; references re-primed).\n");
      }

      if (!spin_guard_primed) {
        // F19: prime every reference from a *relaxed* configuration.  Sampling
        // at step 0 reads the input positions, which for a DFT snapshot are far
        // from the model minimum; the first ~1000 steps are then a legitimate
        // relaxation transient that used to be mis-reported as a
        // "potential-energy collapse" (reproduced on Fe, round 10).
        spin_guard_initial_pe = current_thermo[2];
        spin_guard_previous_pe = current_thermo[2];
        spin_guard_previous_work = w_net;
        spin_guard_reference_temperature = current_thermo[0];
        spin_guard_reference_energy = current_energy;
        spin_guard_injection_streak = 0;
        spin_guard_primed = true;
      }

      const double injection_ratio =
        (std::fabs(w_guard[0]) > 1.0e-12) ? w_guard[1] / std::fabs(w_guard[0]) : 0.0;
      // F21: criterion (1) (injection_ratio > 1.05 && w_net/atom > 0.02 eV) is
      // DISABLED.  The w_net it tests is dominated by the damping relaxing
      // toward the noise field (sigma_SLLG ~ 19 rms|B_perp| at gamma = 0.01);
      // its 2048-fold cancellation with W_rot leaves a residual that is an
      // O(dS^2) discretisation artefact, not a magnetic energy change (Fe,
      // 3456 atoms: w_net = -2.34e3 eV while the total energy moved by +5.9 eV).
      // The definition is wrong, not the precision, so the criterion is removed
      // instead of retuned.  Criteria (3) and (4) below read thermo.out directly
      // and are kept.  Measurements: docs/07-fixes-r17.md.
      const bool injection_bad = false;
      (void)injection_ratio;
      spin_guard_injection_streak = injection_bad ? spin_guard_injection_streak + 1 : 0;

      const double pe_drop_total = (spin_guard_initial_pe - current_thermo[2]) * n_inv;
      const double pe_drop_interval = (spin_guard_previous_pe - current_thermo[2]) * n_inv;
      const double work_interval = (w_net - spin_guard_previous_work) * n_inv;
      const double energy_drift = (current_energy - spin_guard_reference_energy) * n_inv;
      const bool conservative_run = (integrate.get_type() == EnsembleType::NVE);
      // F26 (root cause): `Integrate::temperature` is never assigned for NVE and
      // holds an indeterminate value.  In the released binary it happens to be a
      // tiny POSITIVE denormal, so the old `if (!(t > 0))` fallback never fired,
      // and max(1.5*t, t+300) collapsed to exactly 300 K -- every NVE run at
      // T >= 300 K was stopped as a "lattice temperature excursion".  The floor
      // of 1 K below is what makes the fallback actually take effect.
      double target_temperature = integrate.get_temperature();
      if (!(target_temperature > 1.0)) {
        target_temperature = spin_guard_reference_temperature;
      }
      if (!(target_temperature > 1.0)) {
        target_temperature = current_thermo[0];
      }
      if (!(target_temperature > 1.0)) {
        target_temperature = 300.0;
      }
      const double temperature_limit =
        std::max(1.5 * target_temperature, target_temperature + 300.0);

      // F20: in frozen-lattice mode thermo[0] is NaN by construction and the
      // total energy is not a conserved quantity, so neither is a runaway
      // signature.  Only the potential energy and the spin work budget are
      // still meaningful there.
      const bool non_finite =
        (!frozen_lattice && !std::isfinite(current_thermo[0])) ||
        !std::isfinite(current_thermo[1]) || !std::isfinite(current_thermo[2]) ||
        !std::isfinite(w_net);
      const bool temperature_bad = !frozen_lattice && current_thermo[0] > temperature_limit;
      const bool potential_bad = pe_drop_total > 0.15 || pe_drop_interval > 0.05;
      const bool injection_runaway = spin_guard_injection_streak >= 2;
      // F19: a conservative run must conserve energy.  A drift above
      // 0.02 eV/atom means the spin/lattice splitting is pumping the lattice
      // (round 10: +0.21 and +0.77 eV/atom before the deferred CUDA 700).
      const bool energy_bad =
        conservative_run && !frozen_lattice && std::fabs(energy_drift) > 0.02;

      if (non_finite ||
          (spin_guard_primed &&
           (temperature_bad || potential_bad || injection_runaway || energy_bad))) {
        spin_guard_triggered = true;
        const char* reason = non_finite
          ? "non-finite thermo/work value"
          : (energy_bad
               ? "energy-conservation drift in a conservative run"
               : (temperature_bad
                    ? "lattice temperature excursion"
                    : (potential_bad
                         ? "potential-energy collapse outside the training domain"
                         : "sustained stochastic spin-energy injection")));
        printf(
          "SPIN RUNAWAY GUARD STOP (R19) at step %d: %s.\n"
          "    T = %.6g K (limit %.6g), PE drop = %.6g eV/atom total, "
          "%.6g eV/atom/interval, dW = %+.6g eV/atom/interval, "
          "W_noise/|W_rot| = %.6g, dE = %+.6g eV/atom.\n"
          "    The partial trajectory is closed cleanly.  Do not use it as an "
          "equilibrium sample; add non-collinear/high-temperature structures to "
          "the training set.\n",
          completed_steps,
          reason,
          current_thermo[0],
          temperature_limit,
          pe_drop_total,
          pe_drop_interval,
          work_interval,
          injection_ratio,
          energy_drift);
        FILE* fid = fopen("spin_work.out", "a");
        if (fid) {
          fprintf(
            fid,
            "# GUARD_STOP step=%d reason=%s T=%.10g T_limit=%.10g "
            "PE_drop_total_eV_per_atom=%.10g PE_drop_interval_eV_per_atom=%.10g "
            "dW_interval_eV_per_atom=%+.10g injection_ratio=%.10g "
            "dE_conservative_eV_per_atom=%+.10g\n",
            completed_steps,
            reason,
            current_thermo[0],
            temperature_limit,
            pe_drop_total,
            pe_drop_interval,
            work_interval,
            injection_ratio,
            energy_drift);
          fclose(fid);
        }
        // F21: leave a machine-readable marker next to spin_work.out so that a
        // truncated trajectory can never be mistaken for a finished one.
        {
          FILE* mark = fopen("SPIN_GUARD_STOP", "w");
          if (mark) {
            fprintf(mark,
                    "step=%d\nrequested=%d\ndelivered_fraction=%.6f\nreason=%s\n"
                    "T_K=%.10g\nPE_drop_total_eV_per_atom=%.10g\n"
                    "dE_conservative_eV_per_atom=%+.10g\n",
                    completed_steps,
                    number_of_steps,
                    (double)completed_steps / (double)number_of_steps,
                    reason,
                    current_thermo[0],
                    pe_drop_total,
                    energy_drift);
            fclose(mark);
          }
        }
        fflush(stdout);
        break;
      }

      spin_guard_previous_pe = current_thermo[2];
      spin_guard_previous_work = w_net;
    }

    int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
    if (0 == (step + 1) % base) {
      printf("    %d steps completed.\n", step + 1);
      // F16 / R12: report the spin work budget alongside the progress line so
      // the three terms can be seen growing instead of only at the end.
      if (spin.enabled) {
        spin.print_work("progress");
        // P86: magnetic time-scale diagnostics (only |B| and the SLLG field
        // amplitude, both already on the device; ~10 progress lines per run).
        spin.report(atom.spin.data(), atom.torque.data(), atom.number_of_atoms);
        double w[3];
        spin.read_work(w);
        FILE* fid = fopen("spin_work.out", "a");
        if (fid) {
          fprintf(fid, "%d %.10g %.10g %.10g\n", step + 1, w[0], w[1], w[2]);
          fclose(fid);
        }
      }
      fflush(stdout);
    }
  }

  if (spin.enabled) {
    spin.print_work(spin_guard_triggered ? "guard stop" : "end of run");
    double w[3];
    spin.read_work(w);
    FILE* fid = fopen("spin_work.out", "a");
    if (fid) {
      fprintf(fid, "%d %.10g %.10g %.10g\n", completed_steps, w[0], w[1], w[2]);
      fclose(fid);
    }
  }

  print_line_1();
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;

  printf("Time used for this run = %g second.\n", time_used.count());
  double run_speed = atom.number_of_atoms * (completed_steps * 1.0 / time_used.count());
  printf("Speed of this run = %g atom*step/second.\n", run_speed);
  if (spin_guard_triggered) {
    g_run_status = 2;   // F21: make the truncation visible to the shell
    printf("Run stopped safely after %d of %d requested steps by the R19 guard.\n",
           completed_steps, number_of_steps);
  }
  print_line_2();

  measure.post_run(
    atom, box, integrate, completed_steps, time_step, integrate.get_temperature2());

  integrate.finalize(atom, box);
  velocity.finalize();
  force.finalize();
  max_distance_per_step = 0.0;
}

void Run::parse_one_keyword(
  const std::vector<std::string>& tokens, const RunInput& run_input)
{
  if (tokens[0] == "replicate") {
    PRINT_INPUT_ERROR("replicate must be the first effective command.");
  }

  const int num_param = tokens.size();
  const int max_num_param = 32;
  if (num_param > max_num_param)
    PRINT_INPUT_ERROR("The number of parameters should be less than 32.\n");

#ifdef GPUMD_WPE_ENABLED
  force.wpe_process_command(tokens, atom.number_of_atoms);
#endif

  if (tokens[0] == "potential") {
    force.parse_potential(tokens, box, atom.type.size(), run_input);
  } else if (tokens[0] == "minimize") {
    Minimize minimize;
    minimize.parse_minimize(
      tokens,
      integrate.get_fixed_group(),
      integrate.get_fixed_grouping_method(),
      force,
      box,
      atom,
      group);
  } else if (tokens[0] == "compute_phonon") {
    Hessian hessian;
    hessian.parse(tokens);
    if (!has_replicate_) {
      PRINT_INPUT_ERROR("replicate keyword not found in run.in file.");
    }
    hessian.compute(force, box, atom, group, replicate_size_);
  } else if (tokens[0] == "compute_cohesive") {
    Cohesive cohesive;
    cohesive.parse(tokens, 0);
    cohesive.compute(box, atom, group, force);
  } else if (tokens[0] == "compute_elastic") {
    Cohesive cohesive;
    cohesive.parse(tokens, 1);
    cohesive.compute(box, atom, group, force);
  } else if (tokens[0] == "change_box") {
    parse_change_box(tokens);
  } else if (tokens[0] == "velocity") {
    parse_velocity(tokens);
    // F25 / risk #15: `velocity` puts the target temperature into the KINETIC
    // energy only; the positions stay on the input lattice.  Measured on
    // 1024-atom bcc Fe, `velocity 300` + `ensemble nve` ran 300 -> 47 -> 150 K
    // in 100 fs with E conserved, so a "the bath heats the lattice" conclusion
    // drawn from such a run is an equipartition artefact.  The note is emitted
    // at the first `run` and only for spin runs, so that non-spin stdout stays
    // byte-identical to the accepted baseline.
    velocity_note_pending = true;
  } else if (tokens[0] == "ensemble") {
    integrate.parse_ensemble(tokens, atom, box, group);
  } else if (tokens[0] == "time_step") {
    parse_time_step(tokens);
  } else if (tokens[0] == "spin_llg") {
    spin.parse(tokens);
    spin.initialize(atom.number_of_atoms, atom.has_spin);
    // S08B: let `dump_restart` reach the spin object, so the checkpoint can
    // carry the RNG position and the |S| anchor (Atom is what every measure
    // class already sees; see the comment on the field).
    atom.spin_dynamics = &spin;
    // Round 9 (task 48): the heat-current code needs gamma and alpha to turn
    // the torque into the deterministic LLG rate dS/dt.  Atom is the only
    // object every measurement class already sees.
    atom.spin_gamma = spin.gamma;
    atom.spin_alpha = spin.alpha;
    // `time_step` is read before `spin_llg` in run.in, so it is known here.
    // It is stored in natural units (1 natural = TIME_UNIT_CONVERSION fs),
    // the check wants femtoseconds.
    spin.check_discretisation(time_step * TIME_UNIT_CONVERSION);
  } else if (tokens[0] == "spin_split") {
    // F22: choose how the spin update is interleaved with the lattice step.
    if (tokens.size() != 2) {
      PRINT_INPUT_ERROR("spin_split should have 1 parameter: staggered or strang.\n");
    }
    if (tokens[1] == "staggered") {
      g_spin_split = 0;
      printf("WARNING: spin_split staggered is the legacy first-order coupling.\n"
             "         The velocity-Verlet kick uses the previous step's spin\n"
             "         state and the reported energy belongs to S(n+1/2), not to\n"
             "         S(n+1).  Use the default (strang) unless you are\n"
             "         reproducing a pre-S05B run.\n");
    } else if (tokens[1] == "strang") {
      g_spin_split = 1;
    } else {
      PRINT_INPUT_ERROR("spin_split should be staggered or strang.\n");
    }
    printf("Spin/lattice splitting: %s\n",
           g_spin_split == 1 ? "strang, S(dt/2) V(dt) S(dt/2), 2nd order"
                             : "staggered (historical), 1st order in the coupling");
  } else if (tokens[0] == "spin_substeps") {
    // F23: number of spin sub-updates per MD step.
    if (tokens.size() != 2) {
      PRINT_INPUT_ERROR("spin_substeps should have 1 parameter: <n>.\n");
    }
    g_spin_substeps = atoi(tokens[1].c_str());
    if (g_spin_substeps < 1) {
      PRINT_INPUT_ERROR("spin_substeps must be >= 1.\n");
    }
    printf("Spin sub-steps per MD step: %d\n", g_spin_substeps);
  } else if (tokens[0] == "spin_torque_refresh") {
    // F23: re-evaluate -dE/dS between spin sub-steps.  This is the only way the
    // sub-stepping buys accuracy; it costs one force evaluation per sub-step.
    if (tokens.size() != 2) {
      PRINT_INPUT_ERROR("spin_torque_refresh should have 1 parameter: on or off.\n");
    }
    if (tokens[1] == "on") {
      g_spin_torque_refresh = 1;
    } else if (tokens[1] == "off") {
      g_spin_torque_refresh = 0;
    } else {
      PRINT_INPUT_ERROR("spin_torque_refresh should be on or off.\n");
    }
    printf("Spin torque refresh between sub-steps: %s\n",
           g_spin_torque_refresh ? "on (force is restored, only the torque is refreshed)"
                                 : "off");
  } else if (tokens[0] == "add_bfield") {
    // External magnetic field acting on the spins, in eV/mu_B -- the same unit
    // as the torque -dE/dS.  This is the spin-side counterpart of the external
    // driving used by NEMD: a static field tilts the magnetic order, an
    // oscillating one (omega > 0) drives FMR / spin pumping.
    //   add_bfield <Bx> <By> <Bz> [<omega_in_THz>]
    if (num_param != 4 && num_param != 5) {
      PRINT_INPUT_ERROR("add_bfield should have 3 or 4 parameters.");
    }
    if (!spin.enabled) {
      PRINT_INPUT_ERROR("add_bfield requires `spin_llg` to be set first.");
    }
    double bx = 0.0, by = 0.0, bz = 0.0, omega = 0.0;
    if (!is_valid_real(tokens[1], &bx) || !is_valid_real(tokens[2], &by) ||
        !is_valid_real(tokens[3], &bz)) {
      PRINT_INPUT_ERROR("add_bfield: Bx, By and Bz should be real numbers.");
    }
    if (num_param == 5) {
      if (!is_valid_real(tokens[4], &omega)) {
        PRINT_INPUT_ERROR("add_bfield: omega should be a real number.");
      }
      if (omega < 0.0) {
        PRINT_INPUT_ERROR("add_bfield: omega should be >= 0.");
      }
    }
    spin.set_external_field(bx, by, bz, omega);
    printf(
      "External magnetic field: (%g, %g, %g) eV/mu_B%s.\n",
      bx,
      by,
      bz,
      (omega > 0.0) ? " (oscillating)" : " (static)");
  } else if (tokens[0] == "seed") {
    // Pin every random stream of this run.  Must appear before anything that
    // draws random numbers, i.e. before `run`.
    if (num_param != 2) {
      PRINT_INPUT_ERROR("seed should have 1 parameter.");
    }
    int s = 0;
    if (!is_valid_int(tokens[1], &s)) {
      PRINT_INPUT_ERROR("seed should be an integer.");
    }
    set_user_prng_seed(s);
    printf("Global PRNG seed is %d.\n", s);
  } else if (tokens[0] == "correct_velocity") {
    parse_correct_velocity(tokens, group);
  } else if (tokens[0] == "fix") {
    integrate.parse_fix(tokens, group);
  } else if (tokens[0] == "move") {
    integrate.parse_move(tokens, group);
  } else if (tokens[0] == "kspace") {
    if (has_seen_kspace_command) {
      PRINT_INPUT_ERROR("kspace can only appear once.");
    }
    has_seen_kspace_command = true;
  } else if (tokens[0] == "dftd3") {
    if (has_seen_dftd3_command) {
      PRINT_INPUT_ERROR("dftd3 can only appear once.");
    }
    has_seen_dftd3_command = true;
  } else if (tokens[0] == "run") {
    // Re-apply the global seed right before the integration starts.
    // `velocity ... seed <n>` is implemented by calling srand() for every
    // velocity component of every atom (main_gpumd/velocity.cu), which leaves
    // the process-wide rand() state at a deterministic place; `fix`, the
    // thermostats and every curand initialisation draw from that same rand().
    // Without this re-seed, a `velocity` command appearing *after* `seed`
    // would silently undo the pinning.
    if (user_prng_seed() >= 0) {
      srand(static_cast<unsigned int>(user_prng_seed()));
    }
    // F25: the `velocity` note is deferred to here so that it is only printed
    // for spin runs (see the velocity branch) -- non-spin stdout must stay
    // byte-identical to the accepted baseline, which the regression suite
    // compares byte for byte.
    if (velocity_note_pending && spin.enabled) {
      printf("NOTE: `velocity` sets the kinetic temperature only; the positions are\n"
             "      still on the input lattice.  If the next ensemble is NVE the\n"
             "      measured T will settle near HALF this value once the energy is\n"
             "      shared with the potential.  Equilibrate with a thermostat\n"
             "      (nvt_lan / npt_scr) before any NVE measurement.\n");
      velocity_note_pending = false;
    }
    parse_run(tokens);
  } else if (!measure.parse_action(
               tokens,
               number_of_types,
               integrate,
               group,
               atom,
               box,
               force,
               first_potential_filename_)) {
    PRINT_KEYWORD_ERROR(tokens[0].c_str());
  }
}

void Run::parse_velocity(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  double initial_temperature;
  int seed = 0;
  bool use_seed = false;
  if (!(num_param == 2 || num_param == 4)) {
    PRINT_INPUT_ERROR("velocity should have 1 or 3 parameters.\n");
  }

  if (!is_valid_real(tokens[1], &initial_temperature)) {
    PRINT_INPUT_ERROR("initial temperature should be a real number.\n");
  }
  if (initial_temperature <= 0.0) {
    PRINT_INPUT_ERROR("initial temperature should be a positive number.\n");
  }

  if (num_param == 4) {
    if (tokens[2] != "seed") {
      PRINT_INPUT_ERROR("The second parameter for velocity should be 'seed'.\n");
    }
    use_seed = true;
    if (!is_valid_int(tokens[3], &seed) || seed <= 0) {
      PRINT_INPUT_ERROR("seed should be a positive integer.\n");
    }
  }

  velocity.initialize_cpu(
    has_velocity_in_xyz,
    initial_temperature,
    atom,
    use_seed,
    seed);
  atom.velocity_per_atom.copy_from_host(atom.cpu_velocity_per_atom.data());
  if (!has_velocity_in_xyz) {
    printf("Initialized velocities with input T = %g K.\n", initial_temperature);
  }
}

void Run::parse_correct_velocity(
  const std::vector<std::string>& tokens, const std::vector<Group>& group)
{
  const int num_param = tokens.size();
  printf("Correct linear and angular momenta.\n");

  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("correct_velocity should have 1 or 2 parameters.\n");
  }
  if (!is_valid_int(tokens[1], &velocity.velocity_correction_interval)) {
    PRINT_INPUT_ERROR("velocity correction interval should be an integer.\n");
  }
  if (velocity.velocity_correction_interval < 10) {
    PRINT_INPUT_ERROR("velocity correction interval should >= 10.\n");
  }

  printf("    every %d steps.\n", velocity.velocity_correction_interval);

  if (num_param == 3) {
    if (!is_valid_int(tokens[2], &velocity.velocity_correction_group_method)) {
      PRINT_INPUT_ERROR("velocity correction group method should be an integer.\n");
    }
    if (velocity.velocity_correction_group_method < 0) {
      PRINT_INPUT_ERROR("grouping method should >= 0.\n");
    }
    if (velocity.velocity_correction_group_method >= group.size()) {
      PRINT_INPUT_ERROR("grouping method should < maximum number of grouping methods.\n");
    }
  }

  if (velocity.velocity_correction_group_method < 0) {
    printf("    for the whole system.\n");
  } else {
    printf(
      "    for individual groups in group method %d.\n", velocity.velocity_correction_group_method);
  }

  velocity.do_velocity_correction = true;
}

void Run::parse_time_step(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 3) {
    PRINT_INPUT_ERROR("time_step should have 1 or 2 parameters.\n");
  }
  if (!is_valid_real(tokens[1], &time_step)) {
    PRINT_INPUT_ERROR("time_step should be a real number.\n");
  }
  printf("Time step for this run is %g fs.\n", time_step);
  time_step /= TIME_UNIT_CONVERSION;
  if (num_param == 3) {
    if (!is_valid_real(tokens[2], &max_distance_per_step)) {
      PRINT_INPUT_ERROR("max distance per step should be a real number.\n");
    }
    if (max_distance_per_step <= 0.0) {
      PRINT_INPUT_ERROR("max distance per step should > 0.\n");
    }
    printf("    max distance per step = %g A.\n", max_distance_per_step);
  }
}

void Run::parse_run(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  int number_of_steps;
  if (num_param != 2) {
    PRINT_INPUT_ERROR("run should have 1 parameter.\n");
  }
  if (!is_valid_int(tokens[1], &number_of_steps)) {
    PRINT_INPUT_ERROR("number of steps should be an integer.\n");
  }
  if (number_of_steps <= 0) {
    PRINT_INPUT_ERROR("number of steps should be positive.\n");
  }
  if (!integrate.has_ensemble()) {
    PRINT_INPUT_ERROR("An ensemble must be specified before each run.");
  }
  printf("Run %d steps.\n", number_of_steps);

  // set target temperature for temperature-dependent NEP
  force.set_temperature_range(
    integrate.get_temperature1(), integrate.get_temperature2(), number_of_steps);

  perform_a_run(number_of_steps);
}

static __global__ void gpu_deform_atom(
  int N,
  double mu0,
  double mu1,
  double mu2,
  double mu3,
  double mu4,
  double mu5,
  double mu6,
  double mu7,
  double mu8,
  double* g_x,
  double* g_y,
  double* g_z)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    double x_old = g_x[i];
    double y_old = g_y[i];
    double z_old = g_z[i];
    g_x[i] = mu0 * x_old + mu1 * y_old + mu2 * z_old;
    g_y[i] = mu3 * x_old + mu4 * y_old + mu5 * z_old;
    g_z[i] = mu6 * x_old + mu7 * y_old + mu8 * z_old;
  }
}

void Run::parse_change_box(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2 && num_param != 4 && num_param != 7) {
    PRINT_INPUT_ERROR("change_box can only have 1 or 3 or 6 parameters\n.");
  }

  double deformation_matrix[3][3] = {0.0};

  if (!is_valid_real(tokens[1], &deformation_matrix[0][0])) {
    PRINT_INPUT_ERROR("box change parameter in xx should be a number.");
  }
  deformation_matrix[1][1] = deformation_matrix[2][2] = deformation_matrix[0][0];

  if (num_param >= 4) {
    if (!is_valid_real(tokens[2], &deformation_matrix[1][1])) {
      PRINT_INPUT_ERROR("box change parameter in yy should be a number.");
    }
    if (!is_valid_real(tokens[3], &deformation_matrix[2][2])) {
      PRINT_INPUT_ERROR("box change parameter in zz should be a number.");
    }
  }

  if (num_param == 7) {
    if (!is_valid_real(tokens[4], &deformation_matrix[1][2])) {
      PRINT_INPUT_ERROR("box change parameter in yz should be a number.");
    }
    if (!is_valid_real(tokens[5], &deformation_matrix[0][2])) {
      PRINT_INPUT_ERROR("box change parameter in xz should be a number.");
    }
    if (!is_valid_real(tokens[6], &deformation_matrix[0][1])) {
      PRINT_INPUT_ERROR("box change parameter in xy should be a number.");
    }
    deformation_matrix[1][0] = deformation_matrix[0][1];
    deformation_matrix[2][0] = deformation_matrix[0][2];
    deformation_matrix[2][1] = deformation_matrix[1][2];
  }

  printf("Change box:\n");
  printf("    in xx by %g A.\n", deformation_matrix[0][0]);
  printf("    in yy by %g A.\n", deformation_matrix[1][1]);
  printf("    in zz by %g A.\n", deformation_matrix[2][2]);
  printf("    in yz and zy by strain %g.\n", deformation_matrix[1][2]);
  printf("    in xz and zx by strain %g.\n", deformation_matrix[0][2]);
  printf("    in xy and yz by strain %g.\n", deformation_matrix[0][1]);

  for (int d = 0; d < 3; ++d) {
    deformation_matrix[d][d] =
      (box.cpu_h[d * 3 + d] + deformation_matrix[d][d]) / box.cpu_h[d * 3 + d];
  }

  printf("    Deformation matrix =\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", deformation_matrix[d1][d2]);
    }
    printf("\n");
  }

  printf("    Original box h = [a, b, c] is\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", box.cpu_h[d1 * 3 + d2]);
    }
    printf("\n");
  }

  double h_old[9];
  for (int i = 0; i < 9; ++i) {
    h_old[i] = box.cpu_h[i];
  }

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      double tmp = 0.0;
      for (int k = 0; k < 3; ++k) {
        tmp += deformation_matrix[r][k] * h_old[k * 3 + c];
      }
      box.cpu_h[r * 3 + c] = tmp;
    }
  }
  box.get_inverse();

  const int number_of_atoms = atom.position_per_atom.size() / 3;
  gpu_deform_atom<<<(number_of_atoms - 1) / 128 + 1, 128>>>(
    number_of_atoms,
    deformation_matrix[0][0],
    deformation_matrix[0][1],
    deformation_matrix[0][2],
    deformation_matrix[1][0],
    deformation_matrix[1][1],
    deformation_matrix[1][2],
    deformation_matrix[2][0],
    deformation_matrix[2][1],
    deformation_matrix[2][2],
    atom.position_per_atom.data(),
    atom.position_per_atom.data() + number_of_atoms,
    atom.position_per_atom.data() + number_of_atoms * 2);
  GPU_CHECK_KERNEL

  printf("    Changed box h = [a, b, c] is\n");
  for (int d1 = 0; d1 < 3; ++d1) {
    printf("        ");
    for (int d2 = 0; d2 < 3; ++d2) {
      printf("%g ", box.cpu_h[d1 * 3 + d2]);
    }
    printf("\n");
  }
}
