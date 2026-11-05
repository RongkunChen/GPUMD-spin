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
The driver class for minimizers.

V4 adds the magnetic moments to the minimization.  The syntax is backward
compatible: the positional arguments are exactly the pre-V4 ones and the new
options are trailing key=value tokens.

    minimize sd   <force_tol> <steps>                    [spin=<on|off|auto>] ...
    minimize fire <force_tol> <steps> [<box_change> [<hydrostatic>]] [spin=...] ...

    spin=on             relax the magnetic moments together with the positions
    spin=off            pre-V4 behaviour (positions only)
    spin=auto           on when `spin_llg` was read in run.in, off otherwise
                        (the default, i.e. a spin run minimises its spins)
    spin_tol=<x>        torque tolerance in eV/mu_B, default 1e-3
    alpha_damp=<x>      fictitious damping of the spin relaxation, default 1.0
    discrete_factor=<x> adaptive-step factor, default 10.0 (LAMMPS defaults)

`alpha_damp` and `discrete_factor` are the LAMMPS `min_modify` keywords, so a
LAMMPS user can carry the numbers over unchanged.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "model/atom.cuh"
#include "minimize.cuh"
#include "minimizer_fire.cuh"
#include "minimizer_fire_box_change.cuh"
#include "minimizer_sd.cuh"
#include "spin/spin.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <memory>
#include <string>

void Minimize::parse_minimize(
  const std::vector<std::string>& tokens,
  int fixed_group,
  int fixed_grouping_method,
  Force& force,
  Box& box,
  Atom& atom,
  std::vector<Group>& group,
  Spin* spin)
{
  const int num_param = tokens.size();
  int minimizer_type = 0;
  int number_of_steps = 0;
  double force_tolerance = 0.0;
  int box_change = 0;
  int hydrostatic_strain = 0;
  std::unique_ptr<Minimizer> minimizer;

  // ---- V4 spin options (defaults: LAMMPS min_style spin) ----------------
  int spin_mode = 1; // 0 = off, 1 = auto, 2 = on
  double spin_tolerance = 1.0e-3;
  double alpha_damp = 1.0;
  double discrete_factor = 10.0;

  // Split the tail into positional values and key=value options.  Anything
  // containing '=' is an option, so `minimize fire 1e-3 1000 spin=on` works
  // without a placeholder for <box_change>.
  std::vector<std::string> positional;
  for (int i = 1; i < num_param; ++i) {
    const std::string& t = tokens[i];
    const size_t eq = t.find('=');
    if (eq == std::string::npos) {
      positional.push_back(t);
      continue;
    }
    const std::string key = t.substr(0, eq);
    const std::string val = t.substr(eq + 1);
    if (key == "spin") {
      if (val == "on") {
        spin_mode = 2;
      } else if (val == "off") {
        spin_mode = 0;
      } else if (val == "auto") {
        spin_mode = 1;
      } else {
        PRINT_INPUT_ERROR("minimize: spin= must be on, off or auto.");
      }
    } else if (key == "spin_tol") {
      if (!is_valid_real(val, &spin_tolerance) || spin_tolerance <= 0.0) {
        PRINT_INPUT_ERROR("minimize: spin_tol= must be a positive number (eV/mu_B).");
      }
    } else if (key == "alpha_damp") {
      if (!is_valid_real(val, &alpha_damp) || alpha_damp <= 0.0) {
        PRINT_INPUT_ERROR("minimize: alpha_damp= must be a positive number.");
      }
    } else if (key == "discrete_factor") {
      if (!is_valid_real(val, &discrete_factor) || discrete_factor <= 0.0) {
        PRINT_INPUT_ERROR("minimize: discrete_factor= must be a positive number.");
      }
    } else {
      PRINT_INPUT_ERROR(
        "minimize: unknown option.  Use spin=, spin_tol=, alpha_damp=, "
        "discrete_factor=.");
    }
  }

  const int npos = static_cast<int>(positional.size());
  if (npos < 3) {
    PRINT_INPUT_ERROR("minimize needs at least: <method> <force_tol> <steps>.");
  }

  if (positional[0] == "sd") {
    minimizer_type = 0;

    if (npos != 3) {
      PRINT_INPUT_ERROR("minimize sd should have 2 parameters.");
    }

    if (!is_valid_real(positional[1], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(positional[2], &number_of_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }
    if (number_of_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }
  } else if (positional[0] == "fire") {
    minimizer_type = 1;

    if (!((npos >= 3) && (npos <= 5))) {
      PRINT_INPUT_ERROR("minimize fire should have 2 to 4 parameters.");
    }

    if (!is_valid_real(positional[1], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(positional[2], &number_of_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }
    if (number_of_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }

    if (npos >= 4) {
      if (!is_valid_int(positional[3], &box_change)) {
        PRINT_INPUT_ERROR("Box_change should be an integer.");
      }
      if (!(box_change == 0 || box_change == 1)) {
        PRINT_INPUT_ERROR("Box_change should be 1 or 0.");
      }

      if (box_change == 1) {
        minimizer_type = 2;
      }
    }

    if (npos >= 5) {
      if (!is_valid_int(positional[4], &hydrostatic_strain)) {
        PRINT_INPUT_ERROR("Hydrostatic_strain should be an integer.");
      }
      if (!(hydrostatic_strain == 0 || hydrostatic_strain == 1)) {
        PRINT_INPUT_ERROR("Hydrostatic_strain should be 1 or 0.");
      }
    }
  } else {
    PRINT_INPUT_ERROR("Invalid minimizer.");
  }

  // ---- decide whether the magnetic moments take part --------------------
  const bool has_spin_dof =
    atom.has_spin && atom.spin.size() == static_cast<size_t>(atom.number_of_atoms) * 3;
  bool use_spin = false;
  if (spin_mode == 2) {
    if (!has_spin_dof) {
      PRINT_INPUT_ERROR(
        "minimize: spin=on was requested but model.xyz carries no spin column.");
    }
    use_spin = true;
  } else if (spin_mode == 1) {
    use_spin = has_spin_dof && spin != nullptr && spin->enabled;
  }

  switch (minimizer_type) {
    case 0:
      printf("\nStart to do an energy minimization.\n");
      printf("    using the steepest descent method.\n");
      printf("    with fixed box.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", number_of_steps);

      minimizer.reset(new Minimizer_SD(
        fixed_group, fixed_grouping_method, atom.number_of_atoms, number_of_steps,
        force_tolerance));
      break;
    case 1:
      printf("\nStart to do an energy minimization.\n");
      printf("    using the fast inertial relaxation engine (FIRE) method.\n");
      printf("    with fixed box.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", number_of_steps);

      minimizer.reset(
        new Minimizer_FIRE(atom.number_of_atoms, number_of_steps, force_tolerance));
      break;
    case 2:
      printf("\nStart to do an energy minimization.\n");
      printf("    using the fast inertial relaxation engine (FIRE) method.\n");
      printf("    with variable box.\n");
      if (hydrostatic_strain == 1) {
        printf("    with hydrostatic pressure.\n");
      }
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", number_of_steps);

      minimizer.reset(new Minimizer_FIRE_Box_Change(
        atom.number_of_atoms, number_of_steps, force_tolerance, hydrostatic_strain));
      break;
    default:
      PRINT_INPUT_ERROR("Invalid minimizer.");
      break;
  }

  minimizer->set_spin_minimization(use_spin, spin_tolerance, alpha_damp, discrete_factor);
  minimizer->compute(force, box, atom, atom.position_per_atom, group);
}
