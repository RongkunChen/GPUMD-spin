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
Dump a restart file
--------------------------------------------------------------------------------------------------*/

#include "dump_restart.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "spin/spin.cuh"
#include "utilities/common.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#include "utilities/read_file.cuh"
#include <vector>
#include <cstring>

Dump_Restart::Dump_Restart(const std::vector<std::string>& tokens)
{
  parse(tokens);
  action_name = "dump_restart";
}

void Dump_Restart::parse(const std::vector<std::string>& tokens)
{
  const int num_param = tokens.size();
  if (num_param != 2) {
    PRINT_INPUT_ERROR("dump_restart should have 1 parameter.");
  }
  if (!is_valid_int(tokens[1], &dump_interval_)) {
    PRINT_INPUT_ERROR("restart dump interval should be an integer.");
  }
  if (dump_interval_ <= 0) {
    PRINT_INPUT_ERROR("restart dump interval should > 0.");
  }
  dump_ = true;
  printf("Dump restart every %d steps.\n", dump_interval_);
}

void Dump_Restart::pre_run(
  const int number_of_steps,
  const double time_step,
  Integrate& integrate,
  std::vector<Group>& group,
  Atom& atom,
  Box& box,
  Force& force)
{
  if (dump_) {
    // nothing
  }
}

void Dump_Restart::end_of_step(
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

  FILE* fid = my_fopen("restart.xyz", "w");

  const int number_of_atoms = atom.number_of_atoms;

  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());

  fprintf(fid, "%d\n", number_of_atoms);

  fprintf(
    fid, "pbc=\"%c %c %c\" ", box.pbc_x ? 'T' : 'F', box.pbc_y ? 'T' : 'F', box.pbc_z ? 'T' : 'F');

  fprintf(
    fid,
    "Lattice=\"%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\" ",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);

  // A4: the spin vectors are part of the state.  Without this column a restart
  // silently lost the magnetic configuration (read_xyz then sees no spin:R:3
  // and the spin-aware NEP is evaluated in the S = 0 limit).  The torque is
  // deliberately NOT written: it is recomputed from (R, S) on the next step.
  if (atom.has_spin) {
    atom.spin.copy_to_host(atom.cpu_spin.data());
  }
  const char* spin_props = atom.has_spin ? ":spin:R:3" : "";
  if (group.size() == 0) {
    fprintf(
      fid, "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3%s\n", spin_props);
  } else {
    fprintf(
      fid,
      "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:%d%s\n",
      int(group.size()),
      spin_props);
  }

  for (int n = 0; n < number_of_atoms; n++) {
    const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
    fprintf(
      fid,
      "%s %.17g %.17g %.17g %.17g %.17g %.17g %.17g ",
      atom.cpu_atom_symbol[n].c_str(),
      atom.cpu_position_per_atom[n],
      atom.cpu_position_per_atom[n + number_of_atoms],
      atom.cpu_position_per_atom[n + 2 * number_of_atoms],
      atom.cpu_mass[n],
      atom.cpu_velocity_per_atom[n] * natural_to_A_per_fs,
      atom.cpu_velocity_per_atom[n + number_of_atoms] * natural_to_A_per_fs,
      atom.cpu_velocity_per_atom[n + 2 * number_of_atoms] * natural_to_A_per_fs);

    for (int m = 0; m < group.size(); ++m) {
      fprintf(fid, "%d ", group[m].cpu_label[n]);
    }

    if (atom.has_spin) {
      // component-major, the same layout read_xyz expects for `spin:R:3`
      fprintf(
        fid,
        "%.9g %.9g %.9g",
        (double)atom.cpu_spin[n],
        (double)atom.cpu_spin[n + number_of_atoms],
        (double)atom.cpu_spin[n + 2 * number_of_atoms]);
    }

    fprintf(fid, "\n");
  }

  fflush(fid);
  fclose(fid);

  // S08B: the magnetic part of the state that restart.xyz cannot hold -- the
  // per-atom curand position and the |S| anchor.  Written only when spin
  // dynamics is on; the file is atomic and self-describing.
  if (atom.has_spin && atom.spin_dynamics != nullptr &&
      atom.spin_dynamics->enabled) {
    atom.spin_dynamics->save_restart_state("spin_restart.bin", step + 1,
                                           global_time);
  }
}

void Dump_Restart::post_run(
  Atom& atom,
  Box& box,
  Integrate& integrate,
  const int number_of_steps,
  const double time_step,
  const double temperature)
{
  if (dump_) {
    dump_ = false;
  }
}
