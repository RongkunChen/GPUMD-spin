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
The neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

#include "neighbor.cuh"
#include "nep.cuh"
#include "nep_small_box.cuh"
#include "utilities/common.cuh"
#include "utilities/compact_nep.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_parameters.cuh"
#include "utilities/nep_utilities.cuh"
#include "utilities/run_input.cuh"
#include "utilities/spin_descriptor.cuh"
#include "utilities/spin_descriptor_report.cuh"
#include <cstring>
#include <fstream>
#ifdef S3A2_DIAG
#include "s3a2_diag.cuh"
#endif
#include <iostream>
#include <cstddef>
#include <string>
#include <vector>

const std::string ELEMENTS[NUM_ELEMENTS] = {
  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",
  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
  "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
  "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
  "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu"};

void NEP::initialize_dftd3(const RunInput& run_input)
{
  has_dftd3 = false;
  for (const auto& line : run_input.lines()) {
    const std::vector<std::string>& tokens = line.tokens;
    if (!tokens.empty() && tokens[0] == "dftd3") {
      has_dftd3 = true;
      if (tokens.size() != 4) {
        std::cout << "dftd3 must have 3 parameters\n";
        exit(1);
      }
      std::string xc_functional = tokens[1];
      float rc_potential = get_double_from_token(tokens[2], __FILE__, __LINE__);
      float rc_coordination_number = get_double_from_token(tokens[3], __FILE__, __LINE__);
      dftd3.initialize(
        xc_functional,
        rc_potential,
        rc_coordination_number,
        get_first_potential_filename(run_input));
      break;
    }
  }
}

// Stage03B: strict integer parsing for the ANN line of nep.txt.
//
// get_int_from_token() wraps std::stoi(), which happily accepts trailing
// garbage ("30abc" -> 30), leading spaces and a leading "0x".  Used on the
// architecture line that means a malformed nep.txt can be read as a *different*
// network instead of being rejected, so the two architecture fields are parsed
// here instead: the whole token must be an optional sign followed by digits.
static bool parse_strict_int(const std::string& token, int* value)
{
  if (token.empty()) {
    return false;
  }
  size_t i = 0;
  bool negative = false;
  if (token[0] == '-' || token[0] == '+') {
    negative = (token[0] == '-');
    i = 1;
    if (token.size() == 1) {
      return false;
    }
  }
  long long magnitude = 0;
  for (; i < token.size(); ++i) {
    if (token[i] < '0' || token[i] > '9') {
      return false;
    }
    magnitude = magnitude * 10 + (token[i] - '0');
    if (magnitude > 2147483647LL) {
      return false;
    }
  }
  *value = static_cast<int>(negative ? -magnitude : magnitude);
  return true;
}

NEP::NEP(const char* file_potential, const int num_atoms, const RunInput& run_input)
{
  std::ifstream input(file_potential);
  if (!input.is_open()) {
    std::cout << "Failed to open " << file_potential << std::endl;
    exit(1);
  }

  std::vector<std::string> tokens = get_tokens(input);
  if (tokens.size() < 3) {
    std::cout << "The first line of nep.txt should have at least 3 items." << std::endl;
    exit(1);
  }
  if (tokens[0] == "nep4") {
    zbl.enabled = false;
  } else if (tokens[0] == "nep4_zbl") {
    zbl.enabled = true;
  } else if (tokens[0] == "nep4_spin") {
    zbl.enabled = false;
    paramb.enable_spin = true;
  } else if (tokens[0] == "nep4_temperature") {
    paramb.model_type = 3;
  } else if (tokens[0] == "nep4_zbl_temperature") {
    paramb.model_type = 3;
    zbl.enabled = true;
  } else {
    std::cout << tokens[0]
              << " is an unsupported NEP model. We only support NEP4 models now."
              << std::endl;
    exit(1);
  }
  paramb.num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  if (tokens.size() != 2 + paramb.num_types) {
    std::cout << "The first line of nep.txt should have " << paramb.num_types << " atom symbols."
              << std::endl;
    exit(1);
  }

  if (paramb.num_types == 1) {
    printf("Use the NEP4 potential with %d atom type.\n", paramb.num_types);
  } else {
    printf("Use the NEP4 potential with %d atom types.\n", paramb.num_types);
  }

  for (int n = 0; n < paramb.num_types; ++n) {
    int atomic_number = 0;
    for (int m = 0; m < NUM_ELEMENTS; ++m) {
      if (tokens[2 + n] == ELEMENTS[m]) {
        atomic_number = m + 1;
        break;
      }
    }
    zbl.atomic_numbers[n] = atomic_number;
    printf("    type %d (%s with Z = %d).\n", n, tokens[2 + n].c_str(), zbl.atomic_numbers[n]);
  }

  // zbl
  if (zbl.enabled) {
    tokens = get_tokens(input);
    if (tokens.size() != 3 && tokens.size() != 4) {
      std::cout << "This line should be zbl rc_inner rc_outer [zbl_factor]." << std::endl;
      exit(1);
    }
    zbl.rc_inner = get_double_from_token(tokens[1], __FILE__, __LINE__);
    zbl.rc_outer = get_double_from_token(tokens[2], __FILE__, __LINE__);
    if (zbl.rc_inner == 0 && zbl.rc_outer == 0) {
      zbl.flexible = true;
      printf("    has the flexible ZBL potential\n");
    } else {
      if (tokens.size() == 4) {
        paramb.typewise_cutoff_zbl_factor = get_double_from_token(tokens[3], __FILE__, __LINE__);
        paramb.use_typewise_cutoff_zbl = true;
        printf("    has the universal ZBL with typewise cutoff with a factor of %g.\n",
          paramb.typewise_cutoff_zbl_factor);
      } else {
        printf(
          "    has the universal ZBL with inner cutoff %g A and outer cutoff %g A.\n",
          zbl.rc_inner,
          zbl.rc_outer);
      }
    }
  }

  std::vector<std::string> pending;
  bool has_pending = false;
  const auto next_tokens = [&input, &pending, &has_pending]() {
    if (has_pending) {
      has_pending = false;
      return pending;
    }
    return get_tokens(input);
  };
  const auto push_back = [&pending, &has_pending](
                           const std::vector<std::string>& line) {
    pending = line;
    has_pending = true;
  };

  if (paramb.enable_spin) {
    tokens = get_tokens(input);
    if (tokens[0] != "spin" || tokens.size() != size_t(paramb.num_types) + 1) {
      std::cout << "This line should be spin followed by one flag per atom type." << std::endl;
      exit(1);
    }
    paramb.use_spin.assign(size_t(paramb.num_types), 0);
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.use_spin[n] = get_int_from_token(tokens[1 + n], __FILE__, __LINE__);
    }
    printf("    spin-aware model, spin types:");
    for (int n = 0; n < paramb.num_types; ++n) {
      printf(" %d", paramb.use_spin[n]);
    }
    printf("\n");

    // Optional `spin_layout` / `spin_mag_scale` / `spin_blocks` lines.  They are
    // absent in a legacy model, in which case the implicit block order of
    // version 1 applies.
    //
    // Stage03A.1: this used to peek with tellg()/seekg().  On MSVC that round
    // trip is not exact for an LF-only file -- the stream resumes a few bytes
    // past the mark and the next header line comes back truncated -- so a
    // nep.txt with LF endings was rejected as malformed while the same file
    // with CRLF endings worked.  The peek is now one line of pushback and the
    // stream is never repositioned.  Same syntax and order as before.
    paramb.spin_layout_version = SPIN_LAYOUT_VERSION_IMPLICIT;
    paramb.spin_use_sia = false;
    paramb.spin_mag_scale = 0.25f;
    tokens = next_tokens();
    for (;;) {
      if (tokens.empty()) {
        break;   // end of file
      }
      if (tokens[0] != "spin_layout" && tokens[0] != "spin_mag_scale" &&
          tokens[0] != "spin_blocks") {
        push_back(tokens);   // first mandatory line: zbl or cutoff
        break;
      }
      if (tokens[0] == "spin_layout") {
      if (tokens.size() != 2 + SPIN_NUM_BLOCKS) {
        std::cout << "This line should be spin_layout version followed by the "
                  << SPIN_NUM_BLOCKS << " block names." << std::endl;
        exit(1);
      }
      paramb.spin_layout_version = get_int_from_token(tokens[1], __FILE__, __LINE__);
      // The kernels below implement exactly these five blocks, concatenated in
      // this order; anything else would silently misread the descriptor.  Two
      // layouts are understood: version 3 ("Sia" in slot 4) and version 2/1
      // ("Senv" in slot 4).  See utilities/spin_descriptor.cuh for why Senv was
      // replaced by Sia.
      const char* prefix_ok[SPIN_BLOCK_SLOT_REPLACED] = {
        SPIN_BLOCK_NAME_SEX, SPIN_BLOCK_NAME_SAN, SPIN_BLOCK_NAME_SDM};
      bool prefix_matches = true;
      for (int d = 0; d < SPIN_BLOCK_SLOT_REPLACED; ++d) {
        if (tokens[2 + d] != prefix_ok[d]) {
          prefix_matches = false;
        }
      }
      const bool tail_ok = (tokens[2 + SPIN_NUM_BLOCKS - 1] == SPIN_BLOCK_NAME_SMAG);
      const bool slot4_is_sia = (tokens[2 + SPIN_BLOCK_SLOT_REPLACED] == SPIN_BLOCK_NAME_SIA);
      const bool slot4_is_senv = (tokens[2 + SPIN_BLOCK_SLOT_REPLACED] == SPIN_BLOCK_NAME_SENV);
      if (prefix_matches && tail_ok && slot4_is_sia &&
          paramb.spin_layout_version == SPIN_LAYOUT_VERSION) {
        paramb.spin_use_sia = true;
      } else if (prefix_matches && tail_ok && slot4_is_senv &&
                 (paramb.spin_layout_version == SPIN_LAYOUT_VERSION_SENV ||
                  paramb.spin_layout_version == SPIN_LAYOUT_VERSION_IMPLICIT)) {
        paramb.spin_use_sia = false;
      } else {
        std::cout << "spin_layout version " << paramb.spin_layout_version
                  << " / block list does not match anything implemented by this build"
                  << " (it understands v" << SPIN_LAYOUT_VERSION_SENV << " = Sex San Sdm Senv Smag"
                  << " and v" << SPIN_LAYOUT_VERSION << " = Sex San Sdm Sia Smag)."
                  << std::endl;
        exit(1);
      }
      printf(
        "    spin descriptor layout (v%d): Sex San Sdm %s Smag\n",
        paramb.spin_layout_version,
        paramb.spin_use_sia ? SPIN_BLOCK_NAME_SIA : SPIN_BLOCK_NAME_SENV);
      tokens = next_tokens();
      continue;
      }
      if (tokens[0] == "spin_mag_scale") {
      if (tokens.size() != 2) {
        std::cout << "This line should be spin_mag_scale value." << std::endl;
        exit(1);
      }
      paramb.spin_mag_scale = get_double_from_token(tokens[1], __FILE__, __LINE__);
      printf("    spin |S| scaling = %g.\n", paramb.spin_mag_scale);
      tokens = next_tokens();
      continue;
      }
    // F26: an ablation model states which blocks are active.  Absent for a full
    // model, which is why this is an optional trailing line rather than a new
    // required field.  Evaluating an ablation with all five blocks on would use
    // five untrained weight sets, so getting this wrong is not a small error.
    if (!tokens.empty() && tokens[0] == "spin_blocks") {
      paramb.spin_block_mask = 0;
      for (size_t t = 1; t < tokens.size(); ++t) {
        const std::string& b = tokens[t];
        if (b == SPIN_BLOCK_NAME_SEX) paramb.spin_block_mask |= SPIN_BLOCK_BIT_SEX;
        else if (b == SPIN_BLOCK_NAME_SAN) paramb.spin_block_mask |= SPIN_BLOCK_BIT_SAN;
        else if (b == SPIN_BLOCK_NAME_SDM) paramb.spin_block_mask |= SPIN_BLOCK_BIT_SDM;
        else if (b == SPIN_BLOCK_NAME_SIA || b == SPIN_BLOCK_NAME_SENV)
          paramb.spin_block_mask |= SPIN_BLOCK_BIT_SLOT4;
        else if (b == SPIN_BLOCK_NAME_SMAG) paramb.spin_block_mask |= SPIN_BLOCK_BIT_SMAG;
        else {
          std::cout << "Unknown block name '" << b << "' on the spin_blocks line." << std::endl;
          exit(1);
        }
      }
      if (paramb.spin_block_mask == 0) {
        std::cout << "The spin_blocks line enables no block." << std::endl;
        exit(1);
      }
      printf("    spin descriptor blocks (ablation model):");
      if (paramb.spin_block_mask & SPIN_BLOCK_BIT_SEX) printf(" Sex");
      if (paramb.spin_block_mask & SPIN_BLOCK_BIT_SAN) printf(" San");
      if (paramb.spin_block_mask & SPIN_BLOCK_BIT_SDM) printf(" Sdm");
      if (paramb.spin_block_mask & SPIN_BLOCK_BIT_SLOT4) printf(" %s",
        paramb.spin_use_sia ? SPIN_BLOCK_NAME_SIA : SPIN_BLOCK_NAME_SENV);
      if (paramb.spin_block_mask & SPIN_BLOCK_BIT_SMAG) printf(" Smag");
      printf("\n");
      tokens = next_tokens();
      continue;
      }
    }
  }

  // S06: state what the enabled blocks actually are, once, after every header
  // line has been read (the layout and the mask can appear in either order).
  // Nothing here changes the evaluation; it is the model's own declaration of
  // its symmetry class and of the two blocks that are identically zero for
  // whole classes of configurations.
  if (paramb.enable_spin) {
    spin_report_declaration(paramb.spin_layout_version, paramb.spin_block_mask);
  }

  // cutoff
  tokens = next_tokens();
  if (tokens.size() != 5 && tokens.size() != paramb.num_types * 2 + 3) {
    std::cout << "cutoff should have 4 or num_types * 2 + 2 parameters.\n";
    exit(1);
  }
  if (tokens.size() == 5) {
    paramb.rc_radial[0] = get_double_from_token(tokens[1], __FILE__, __LINE__);
    paramb.rc_angular[0] = get_double_from_token(tokens[2], __FILE__, __LINE__);
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = paramb.rc_radial[0];
      paramb.rc_angular[n] = paramb.rc_angular[0];
    }
    printf("    radial cutoff = %g A.\n", paramb.rc_radial[0]);
    printf("    angular cutoff = %g A.\n", paramb.rc_angular[0]);
  } else {
    printf("    cutoff = \n");
    for (int n = 0; n < paramb.num_types; ++n) {
      paramb.rc_radial[n] = get_double_from_token(tokens[1 + n * 2], __FILE__, __LINE__);
      paramb.rc_angular[n] = get_double_from_token(tokens[2 + n * 2], __FILE__, __LINE__);
      printf("    (%g A, %g A)\n", paramb.rc_radial[n], paramb.rc_angular[n]);
    }
  }
  for (int n = 0; n < paramb.num_types; ++n) {
    if (paramb.rc_radial[n] > paramb.rc_radial_max) {
      paramb.rc_radial_max = paramb.rc_radial[n];
    }
  }
  paramb.rc_radial_max_inv = 1.0f / paramb.rc_radial_max;

  int MN_radial = get_int_from_token(tokens[tokens.size() - 2], __FILE__, __LINE__);
  int MN_angular = get_int_from_token(tokens[tokens.size() - 1], __FILE__, __LINE__);
  printf("    MN_radial = %d.\n", MN_radial);
  if (MN_radial > 819) {
    std::cout << "The maximum number of neighbors exceeds 819. Please reduce this value."
              << std::endl;
    exit(1);
  }
  paramb.MN_radial = int(ceil(MN_radial * 1.25));
  paramb.MN_angular = int(ceil(MN_angular * 1.25));
  printf("    enlarged MN_radial = %d.\n", paramb.MN_radial);
  printf("    enlarged MN_angular = %d.\n", paramb.MN_angular);

  // n_max 10 8
  tokens = next_tokens();
  if (tokens.size() != 3) {
    std::cout << "This line should be n_max n_max_radial n_max_angular." << std::endl;
    exit(1);
  }
  paramb.n_max_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.n_max_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    n_max_radial = %d.\n", paramb.n_max_radial);
  printf("    n_max_angular = %d.\n", paramb.n_max_angular);

  // basis_size 10 8
  tokens = next_tokens();
  if (tokens.size() != 3) {
    std::cout << "This line should be basis_size basis_size_radial basis_size_angular."
              << std::endl;
    exit(1);
  }
  paramb.basis_size_radial = get_int_from_token(tokens[1], __FILE__, __LINE__);
  paramb.basis_size_angular = get_int_from_token(tokens[2], __FILE__, __LINE__);
  printf("    basis_size_radial = %d.\n", paramb.basis_size_radial);
  printf("    basis_size_angular = %d.\n", paramb.basis_size_angular);

  // l_max
  tokens = next_tokens();
  if (tokens.size() < 4) {
    std::cout << "This line should be l_max l_max_3body has_q_222 has_q_1111 [has_q_112] [has_q_123] [has_q_233] [has_q_134]." << std::endl;
    exit(1);
  }

  paramb.L_max = get_int_from_token(tokens[1], __FILE__, __LINE__);
  printf("    l_max_3body = %d.\n", paramb.L_max);
  paramb.num_L = paramb.L_max;

  paramb.has_q_222 = get_int_from_token(tokens[2], __FILE__, __LINE__);
  paramb.has_q_1111 = get_int_from_token(tokens[3], __FILE__, __LINE__);
  if (tokens.size() >= 5) {
    paramb.has_q_112 = get_int_from_token(tokens[4], __FILE__, __LINE__);
  }
  if (tokens.size() >= 6) {
    paramb.has_q_123 = get_int_from_token(tokens[5], __FILE__, __LINE__);
  }
  if (tokens.size() >= 7) {
    paramb.has_q_233 = get_int_from_token(tokens[6], __FILE__, __LINE__);
  }
  if (tokens.size() >= 8) {
    paramb.has_q_134 = get_int_from_token(tokens[7], __FILE__, __LINE__);
  }
  printf("    has_q_222 = %d.\n", paramb.has_q_222);
  printf("    has_q_1111 = %d.\n", paramb.has_q_1111);
  printf("    has_q_112 = %d.\n", paramb.has_q_112);
  printf("    has_q_123 = %d.\n", paramb.has_q_123);
  printf("    has_q_233 = %d.\n", paramb.has_q_233);
  printf("    has_q_134 = %d.\n", paramb.has_q_134);
  if (paramb.has_q_222) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_1111) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_112) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_123) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_233) {
    paramb.num_L += 1;
  }
  if (paramb.has_q_134) {
    paramb.num_L += 1;
  }

  paramb.dim_angular = (paramb.n_max_angular + 1) * paramb.num_L;

  // ANN
  //
  // Stage03B: this used to accept any three-token line (the keyword itself was
  // never checked) and read only the first neuron count, silently ignoring the
  // second.  A nep.txt written for a two-hidden-layer model therefore produced
  // the wrong parameter count here and the whole parameter block was read at
  // the wrong offsets.  The architecture is now validated completely, before
  // any parameter array is sized or read, and a second hidden layer is refused
  // explicitly because the MD evaluator only implements one hidden layer.
  tokens = next_tokens();
  if (tokens.empty()) {
    std::cout << "The ANN line of " << file_potential << " is missing or empty."
              << std::endl;
    exit(1);
  }
  if (tokens[0] != "ANN") {
    std::cout << "Expected the ANN line of " << file_potential
              << " to start with 'ANN', but found '" << tokens[0] << "'."
              << std::endl;
    exit(1);
  }
  if (tokens.size() != 3) {
    std::cout << "The ANN line of " << file_potential
              << " should be 'ANN <num_neurons1> <num_neurons2>', but it has "
              << tokens.size() << " token(s)." << std::endl;
    exit(1);
  }
  int ann_num_neurons1 = 0;
  int ann_num_neurons2 = 0;
  if (!parse_strict_int(tokens[1], &ann_num_neurons1)) {
    std::cout << "The first field of the ANN line of " << file_potential
              << " is not an integer: '" << tokens[1] << "'." << std::endl;
    exit(1);
  }
  if (!parse_strict_int(tokens[2], &ann_num_neurons2)) {
    std::cout << "The second field of the ANN line of " << file_potential
              << " is not an integer: '" << tokens[2] << "'." << std::endl;
    exit(1);
  }
  if (ann_num_neurons1 <= 0) {
    // Stage03B.1: zero neurons is NOT a usable architecture for this evaluator.
    // With num_neurons1 == 0 the ANN collapses to "energy = -b1" for every atom
    // and the parameter block degenerates (w0/b0/w1 all empty), which is never
    // what a real model means: it is either a truncated file or a zeroed field.
    // The check is <= 0 so a negative value gets the same message.
    std::cout << "The first field of the ANN line of " << file_potential
              << " must be > 0, but it is " << ann_num_neurons1 << "."
              << std::endl;
    exit(1);
  }
  if (ann_num_neurons2 < 0) {
    std::cout << "The second field of the ANN line of " << file_potential
              << " must be >= 0, but it is " << ann_num_neurons2 << "."
              << std::endl;
    exit(1);
  }
  if (ann_num_neurons2 > 0) {
    std::cout << "The ANN line of " << file_potential << " requests "
              << ann_num_neurons1 << "-" << ann_num_neurons2
              << "-1, i.e. two hidden layers, but the MD evaluator in this "
                 "build only implements one hidden layer (ANN n1 0).  "
                 "Re-export the model with a single hidden layer; the trainer "
                 "and the specialized config exporter still accept two."
              << std::endl;
    exit(1);
  }
  annmb.num_neurons1 = ann_num_neurons1;
  // 5 blocks: Sex, San, Sdm, Senv and the on-site |S| channel (Smag)
  paramb.dim_spin = paramb.enable_spin ? 5 * (paramb.n_max_angular + 1) : 0;
  annmb.dim = (paramb.n_max_radial + 1) + paramb.dim_angular + paramb.dim_spin;
  // B-6: q[], Fp[] and q_scaler[] are fixed-size stack arrays of MAX_DIM floats
  // in every kernel.  Nothing used to check this, so an over-large descriptor
  // (e.g. l_max = 12 with all has_q_* enabled, plus the 5 spin blocks) wrote
  // past the end of those arrays.  Fail loudly instead.
  if (annmb.dim > MAX_DIM) {
    std::cout << "The descriptor dimension " << annmb.dim << " (radial + angular + spin) exceeds"
              << " MAX_DIM = " << MAX_DIM << "; rebuild with a larger MAX_DIM in"
              << " utilities/nep_utilities.cuh or reduce n_max/l_max." << std::endl;
    exit(1);
  }
  if (paramb.enable_spin) {
    // Spin Route B: the MD side now reads the spin vectors from a `spin:R:3`
    // column in model.xyz and propagates them with the LLG integrator when the
    // `spin_llg` keyword is present. Without that column every spin descriptor
    // component is zero, i.e. the model is evaluated in the S = 0 limit: the
    // energy and forces stay consistent with the trainer, but the magnetic
    // contribution is switched off.
    printf("    WARNING: this is a spin-aware NEP model. Add a 'spin:R:3' column to\n");
    printf("             model.xyz, otherwise it is evaluated in the S = 0 limit.\n");
  }
  nep_model_type = paramb.model_type;
  if (paramb.model_type == 3) {
    annmb.dim += 1;
  }
  printf("    ANN = %d-%d-1.\n", annmb.dim, annmb.num_neurons1);

  // calculated parameters:
  rc = paramb.rc_radial_max; // largest cutoff
  paramb.num_types_sq = paramb.num_types * paramb.num_types;

  annmb.num_para_ann = (annmb.dim + 2) * annmb.num_neurons1 * paramb.num_types + 1;
  printf("    number of neural network parameters = %d.\n", annmb.num_para_ann);
  int num_para_descriptor =
    paramb.num_types_sq * ((paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1) +
                           (paramb.n_max_angular + 1) * (paramb.basis_size_angular + 1));
  printf("    number of descriptor parameters = %d.\n", num_para_descriptor);
  annmb.num_para = annmb.num_para_ann + num_para_descriptor;
  printf("    total number of parameters = %d.\n", annmb.num_para);

  paramb.num_c_radial =
    paramb.num_types_sq * (paramb.n_max_radial + 1) * (paramb.basis_size_radial + 1);

  // NN and descriptor parameters
  std::vector<float> parameters(annmb.num_para + annmb.dim);
  for (int n = 0; n < annmb.num_para + annmb.dim; ++n) {
    tokens = get_tokens(input);
    parameters[n] = get_double_from_token(tokens[0], __FILE__, __LINE__);
  }
  // refactor descriptor parameters memory
  std::vector<float> descriptor_parameters = get_descriptor_parameters_type_pair(
    parameters,
    annmb.num_para_ann,
    paramb.num_types,
    paramb.n_max_radial,
    paramb.n_max_angular,
    paramb.basis_size_radial,
    paramb.basis_size_angular);
  nep_data.parameters.resize(annmb.num_para + annmb.dim);
  nep_data.parameters.copy_from_host(parameters.data());
  nep_data.descriptor_parameters_type_pair.resize(num_para_descriptor);
  nep_data.descriptor_parameters_type_pair.copy_from_host(descriptor_parameters.data());
  update_potential(nep_data.parameters.data(), annmb);
  annmb.c_type_pair = nep_data.descriptor_parameters_type_pair.data();
  annmb.q_scaler = nep_data.parameters.data() + annmb.num_para;

  // flexible zbl potential parameters
  if (zbl.flexible) {
    int num_type_zbl = (paramb.num_types * (paramb.num_types + 1)) / 2;
    for (int d = 0; d < 10 * num_type_zbl; ++d) {
      tokens = get_tokens(input);
      zbl.para[d] = get_double_from_token(tokens[0], __FILE__, __LINE__);
    }
    zbl.num_types = paramb.num_types;
  }

  nep_data.f12x.resize(num_atoms * paramb.MN_angular);
  nep_data.f12y.resize(num_atoms * paramb.MN_angular);
  nep_data.f12z.resize(num_atoms * paramb.MN_angular);
  neighbor_manager.initialize(rc, num_atoms, paramb.MN_radial);
  nep_data.NN_radial.resize(num_atoms);
  nep_data.NL_radial.resize(static_cast<size_t>(num_atoms) * paramb.MN_radial);
  nep_data.NN_angular.resize(num_atoms);
  nep_data.NL_angular.resize(num_atoms * paramb.MN_angular);
  nep_data.Fp.resize(static_cast<size_t>(num_atoms) * annmb.dim);
  nep_data.sum_fxyz.resize(
    static_cast<size_t>(num_atoms) * (paramb.n_max_angular + 1) * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1));
  nep_data.cpu_NN_radial.resize(num_atoms);
  nep_data.cpu_NN_angular.resize(num_atoms);
  if (paramb.enable_spin) {
    nep_data.q_spin.resize(static_cast<size_t>(num_atoms) * paramb.dim_spin);
    nep_data.use_spin.resize(paramb.num_types);
    nep_data.use_spin.copy_from_host(paramb.use_spin.data());
  }

  initialize_dftd3(run_input);
}

NEP::~NEP(void)
{
  // nothing
}

void NEP::update_potential(float* parameters, ANN& ann)
{
  float* pointer = parameters;
  for (int t = 0; t < paramb.num_types; ++t) {
    ann.w0[t] = pointer;
    pointer += ann.num_neurons1 * ann.dim;
    ann.b0[t] = pointer;
    pointer += ann.num_neurons1;
    ann.w1[t] = pointer;
    pointer += ann.num_neurons1;
  }
  ann.b1 = pointer;
  pointer += 1;

  ann.c = pointer;
}

static __global__ void find_neighbor_list_large_box(
  NEP::ParaMB paramb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* __restrict__ g_NN_global,
  const int* __restrict__ g_NL_global,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int t1 = g_type[n1];
  int count_radial = 0;
  int count_angular = 0;

  for (int i1 = 0; i1 < g_NN_global[n1]; ++i1) {
    int n2 = g_NL_global[static_cast<size_t>(N) * i1 + n1];
    float x12 = g_x[n2] - x1;
    float y12 = g_y[n2] - y1;
    float z12 = g_z[n2] - z1;
    apply_mic(box, x12, y12, z12);
    float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
    int t2 = g_type[n2];
    float rc_radial = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
    float rc_angular = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
    if (d12_square >= rc_radial * rc_radial) {
      continue;
    }
    g_NL_radial[static_cast<size_t>(N) * count_radial++ + n1] = n2;
    if (d12_square < rc_angular * rc_angular) {
      g_NL_angular[count_angular++ * N + n1] = n2;
    }
  }

  g_NN_radial[n1] = count_radial;
  g_NN_angular[n1] = count_angular;
}
#include "../utilities/spin_descriptor_kernels.cuh"
static __global__ void find_descriptor(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz,
  const float* __restrict__ g_q_spin)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};
    // neighbour-order-independent accumulators (see nep_acc_t)
    nep_acc_t q_acc[MAX_NUM_N] = {0.0};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];

      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        q_acc[n] += static_cast<nep_acc_t>(gn12);
      }
    }
    // exactly one narrowing: deterministic, independent of neighbour order
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      q[n] = static_cast<float>(q_acc[n]);
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      nep_acc_t s_acc[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s_acc);
      }
      float s[NUM_OF_ABC];
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        s[abc] = static_cast<float>(s_acc[abc]);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[static_cast<size_t>(N) * (n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) + n1] = s[abc];
      }
    }

    // spin descriptor components (Route B), evaluated in find_descriptor_spin
    if (g_q_spin != nullptr) {
      const int base = (paramb.n_max_radial + 1) + paramb.dim_angular;
      const int nn1 = paramb.n_max_angular + 1;
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        q[base + n] = g_q_spin[n1 + static_cast<size_t>(N) * n];
        q[base + nn1 + n] = g_q_spin[n1 + static_cast<size_t>(N) * (nn1 + n)];
        q[base + 2 * nn1 + n] = g_q_spin[n1 + static_cast<size_t>(N) * (2 * nn1 + n)];
        q[base + 3 * nn1 + n] = g_q_spin[n1 + static_cast<size_t>(N) * (3 * nn1 + n)];
        q[base + 4 * nn1 + n] = g_q_spin[n1 + static_cast<size_t>(N) * (4 * nn1 + n)];
      }
    }

#ifdef S3A2_DIAG
    // raw descriptor of the target atom, taken in place (before the scaler) 
    if (s3a2_is_target(n1)) {
      for (int d = 0; d < annmb.dim; ++d) {
        g_s3a2_q_raw[d] = q[d];
      }
      g_s3a2_type = t1;
    }
#endif
    // normalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

#ifdef S3A2_DIAG
    if (s3a2_is_target(n1)) {
      for (int d = 0; d < annmb.dim; ++d) {
        g_s3a2_q_scaled[d] = q[d];
      }
      s3a2_ann_one_layer_diag(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1],
        annmb.w1[t1], annmb.b1, q, F, Fp);
    } else
#endif
    apply_ann_one_layer(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[t1],
      annmb.b0[t1],
      annmb.w1[t1],
      annmb.b1,
      q,
      F,
      Fp);
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[static_cast<size_t>(N) * d + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}

static __global__ void find_force_radial(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
      int t2 = g_type[n2];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      float fc12, fcp12;
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_radial, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index_12 = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          int c_index_21 = get_c_index(
            t2 * paramb.num_types + t1, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index_12];
          gnp21 += fnp12[k] * annmb.c_type_pair[c_index_21];
        }
        float tmp12 = g_Fp[static_cast<size_t>(N) * n + n1] * gnp12 * d12inv;
        float tmp21 = g_Fp[static_cast<size_t>(N) * n + n2] * gnp21 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
          f21[d] -= tmp21 * r12[d];
        }
      }
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx += r12[0] * f21[0];
      s_syy += r12[1] * f21[1];
      s_szz += r12[2] * f21[2];
      s_sxy += r12[0] * f21[1];
      s_sxz += r12[0] * f21[2];
      s_syx += r12[1] * f21[0];
      s_syz += r12[1] * f21[2];
      s_szx += r12[2] * f21[0];
      s_szy += r12[2] * f21[1];
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
  }
}

static __global__ void find_partial_force_angular(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[static_cast<size_t>(N) * (paramb.n_max_radial + 1 + d) + n1];
    }
    for (int n = 0; n < paramb.n_max_angular + 1; ++n) {
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        sum_fxyz[n * NUM_OF_ABC + abc] =
          g_sum_fxyz[static_cast<size_t>(N) * (n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) + n1];
      }
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc_and_fcp(rc, rcinv, d12, fc12, fcp12);

      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(paramb.basis_size_angular, rcinv, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
          gnp12 += fnp12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}

static __global__ void find_force_ZBL(
  NEP::ParaMB paramb,
  const int N,
  const NEP::ZBL zbl,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    float s_pe = 0.0f;
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(float(zi), 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float r12[3] = {x12, y12, z12};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      int type2 = g_type[n2];
      int zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(float(zj), 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexible) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      s_fx += f12[0] - f21[0];
      s_fy += f12[1] - f21[1];
      s_fz += f12[2] - f21[2];
      s_sxx -= r12[0] * f12[0];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_syz -= r12[1] * f12[2];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_pe += f * 0.5f;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * N] += s_sxx;
    g_virial[n1 + 1 * N] += s_syy;
    g_virial[n1 + 2 * N] += s_szz;
    g_virial[n1 + 3 * N] += s_sxy;
    g_virial[n1 + 4 * N] += s_sxz;
    g_virial[n1 + 5 * N] += s_syz;
    g_virial[n1 + 6 * N] += s_syx;
    g_virial[n1 + 7 * N] += s_szx;
    g_virial[n1 + 8 * N] += s_szy;
    g_pe[n1] += s_pe;
  }
}

// large box fo MD applications
void NEP::compute_large_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  neighbor_manager.update(
    box, 
    type, 
    position_per_atom);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    neighbor_manager.get_candidate_NN().data(),
    neighbor_manager.get_candidate_NL().data(),
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data());
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    nep_data.NN_radial.copy_to_host(nep_data.cpu_NN_radial.data());
    nep_data.NN_angular.copy_to_host(nep_data.cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < nep_data.cpu_NN_radial[n]) {
        radial_actual = nep_data.cpu_NN_radial[n];
      }
      if (angular_actual < nep_data.cpu_NN_angular[n]) {
        angular_actual = nep_data.cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  const bool spin_active = paramb.enable_spin && spin_ != nullptr && torque_ != nullptr;
  if (spin_active) {
    // GPU_Vector::resize() frees and reallocates unconditionally, and a
    // cudaFree in the middle of a stream would stall (or, with work in
    // flight on another stream, fail).  Allocate only when it is needed.
    if (torque_accumulator_.size() != static_cast<size_t>(N) * 3) {
      torque_accumulator_.resize(static_cast<size_t>(N) * 3);
    }
    if (Aspin_out_ != nullptr) {
      zero_Aspin<<<(N - 1) / 64 + 1, 64>>>(N, Aspin_out_);
      GPU_CHECK_KERNEL
    }
    zero_torque_md<<<(N - 1) / 64 + 1, 64>>>(
      N,
      torque_accumulator_.data(),
      torque_accumulator_.data() + N,
      torque_accumulator_.data() + N * 2);
    GPU_CHECK_KERNEL
    find_descriptor_spin<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      spin_,
      spin_ + N,
      spin_ + N * 2,
      nep_data.use_spin.data(),
      nep_data.q_spin.data());
    GPU_CHECK_KERNEL
  }

  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data(),
    spin_active ? nep_data.q_spin.data() : nullptr);
  GPU_CHECK_KERNEL

  if (spin_active && getenv("SPIN_DEBUG") != nullptr) {
    static int dbg_calls = 0;
    if (dbg_calls++ == 0) {
      std::vector<float> qs(nep_data.q_spin.size());
      nep_data.q_spin.copy_to_host(qs.data());
      std::vector<float> par(nep_data.parameters.size());
      nep_data.parameters.copy_to_host(par.data());
      const float* qsc = par.data() + annmb.num_para;
      const int base = (paramb.n_max_radial + 1) + paramb.dim_angular;
      FILE* fp = fopen("spin_debug.out", "w");
      for (int a = 0; a < N; ++a) {
        for (int d = 0; d < paramb.dim_spin; ++d) {
          fprintf(fp, "%.8g ", qs[a + static_cast<size_t>(N) * d] * qsc[base + d]);
        }
        fprintf(fp, "\n");
      }
      fclose(fp);
      printf("[SPIN_DEBUG] wrote spin_debug.out (dim_spin=%d)\n", paramb.dim_spin);
    }
  }


  find_force_radial<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data());
  GPU_CHECK_KERNEL

  find_properties_many_body(
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    position_per_atom,
    force_per_atom,
    virial_per_atom);
  GPU_CHECK_KERNEL

  // spin (Route B): magnetic torque and the spin contribution to the force
  if (spin_active) {
    find_force_spin<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      nep_data.Fp.data(),
      spin_,
      spin_ + N,
      spin_ + N * 2,
      nep_data.use_spin.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      torque_accumulator_.data(),
      torque_accumulator_.data() + N,
      torque_accumulator_.data() + N * 2,
      Aspin_out_);
    GPU_CHECK_KERNEL
    // Round the double accumulator down to the float torque the integrator
    // reads.  Doing it here rather than in the accumulation kernel is what
    // keeps the whole reduction order independent.
    convert_torque_to_float<<<(N - 1) / 64 + 1, 64>>>(
      N,
      torque_accumulator_.data(),
      torque_accumulator_.data() + N,
      torque_accumulator_.data() + N * 2,
      torque_,
      torque_ + N,
      torque_ + N * 2);
    GPU_CHECK_KERNEL
  }


  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

// small box possibly used for active learning:
void NEP::compute_small_box(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = type.size() * big_neighbor_size;

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    ebox,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    std::vector<int> cpu_NN_radial(type.size());
    std::vector<int> cpu_NN_angular(type.size());
    small_box_data.NN_radial.copy_to_host(cpu_NN_radial.data());
    small_box_data.NN_angular.copy_to_host(cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < cpu_NN_radial[n]) {
        radial_actual = cpu_NN_radial[n];
      }
      if (angular_actual < cpu_NN_angular[n]) {
        angular_actual = cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  const bool spin_active = paramb.enable_spin && spin_ != nullptr && torque_ != nullptr;
  if (spin_active) {
    // GPU_Vector::resize() frees and reallocates unconditionally, and a
    // cudaFree in the middle of a stream would stall (or, with work in
    // flight on another stream, fail).  Allocate only when it is needed.
    if (torque_accumulator_.size() != static_cast<size_t>(N) * 3) {
      torque_accumulator_.resize(static_cast<size_t>(N) * 3);
    }
    if (Aspin_out_ != nullptr) {
      zero_Aspin<<<(N - 1) / 64 + 1, 64>>>(N, Aspin_out_);
      GPU_CHECK_KERNEL
    }
    zero_torque_md<<<(N - 1) / 64 + 1, 64>>>(
      N,
      torque_accumulator_.data(),
      torque_accumulator_.data() + N,
      torque_accumulator_.data() + N * 2);
    GPU_CHECK_KERNEL
    find_descriptor_spin_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      type.data(),
      spin_,
      spin_ + N,
      spin_ + N * 2,
      nep_data.use_spin.data(),
      nep_data.q_spin.data());
    GPU_CHECK_KERNEL
  }


  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data(),
    spin_active ? nep_data.q_spin.data() : nullptr);
  GPU_CHECK_KERNEL

  if (spin_active && getenv("SPIN_DEBUG") != nullptr) {
    static int dbg_calls = 0;
    if (dbg_calls++ == 0) {
      std::vector<float> qs(nep_data.q_spin.size());
      nep_data.q_spin.copy_to_host(qs.data());
      std::vector<float> par(nep_data.parameters.size());
      nep_data.parameters.copy_to_host(par.data());
      const float* qsc = par.data() + annmb.num_para;
      const int base = (paramb.n_max_radial + 1) + paramb.dim_angular;
      FILE* fp = fopen("spin_debug.out", "w");
      for (int a = 0; a < N; ++a) {
        for (int d = 0; d < paramb.dim_spin; ++d) {
          fprintf(fp, "%.8g ", qs[a + static_cast<size_t>(N) * d] * qsc[base + d]);
        }
        fprintf(fp, "\n");
      }
      fclose(fp);
      printf("[SPIN_DEBUG] wrote spin_debug.out (dim_spin=%d)\n", paramb.dim_spin);
    }
  }

  find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    nep_data.Fp.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  // spin (Route B): magnetic torque and the spin contribution to the force
  if (spin_active) {
    find_force_spin_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      annmb,
      N,
      N1,
      N2,
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      type.data(),
      nep_data.Fp.data(),
      spin_,
      spin_ + N,
      spin_ + N * 2,
      nep_data.use_spin.data(),
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      torque_accumulator_.data(),
      torque_accumulator_.data() + N,
      torque_accumulator_.data() + N * 2,
      Aspin_out_);
    GPU_CHECK_KERNEL
    // Round the double accumulator down to the float torque the integrator
    // reads.  Doing it here rather than in the accumulation kernel is what
    // keeps the whole reduction order independent.
    convert_torque_to_float<<<(N - 1) / 64 + 1, 64>>>(
      N,
      torque_accumulator_.data(),
      torque_accumulator_.data() + N,
      torque_accumulator_.data() + N * 2,
      torque_,
      torque_ + N,
      torque_ + N * 2);
    GPU_CHECK_KERNEL
  }


  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      type.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

static bool get_expanded_box(const double rc, const Box& box, NEP::ExpandedBox& ebox)
{
  double volume = box.get_volume();
  double thickness_x = volume / box.get_area(0);
  double thickness_y = volume / box.get_area(1);
  double thickness_z = volume / box.get_area(2);
  ebox.num_cells[0] = box.pbc_x ? int(ceil(2.0 * rc / thickness_x)) : 1;
  ebox.num_cells[1] = box.pbc_y ? int(ceil(2.0 * rc / thickness_y)) : 1;
  ebox.num_cells[2] = box.pbc_z ? int(ceil(2.0 * rc / thickness_z)) : 1;

  bool is_small_box = false;
  if (box.pbc_x && thickness_x <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_y && thickness_y <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }
  if (box.pbc_z && thickness_z <= 2.5 * (rc + 1.0)) {
    is_small_box = true;
  }

  if (is_small_box) {
    if (thickness_x > 10 * rc || thickness_y > 10 * rc || thickness_z > 10 * rc) {
      std::cout << "Error:\n"
                << "    The box has\n"
                << "        a thickness < 2.5 radial cutoffs in a periodic direction.\n"
                << "        and a thickness > 10 radial cutoffs in another direction.\n"
                << "    Please increase the periodic direction(s).\n";
      exit(1);
    }

    ebox.h[0] = box.cpu_h[0] * ebox.num_cells[0];
    ebox.h[3] = box.cpu_h[3] * ebox.num_cells[0];
    ebox.h[6] = box.cpu_h[6] * ebox.num_cells[0];
    ebox.h[1] = box.cpu_h[1] * ebox.num_cells[1];
    ebox.h[4] = box.cpu_h[4] * ebox.num_cells[1];
    ebox.h[7] = box.cpu_h[7] * ebox.num_cells[1];
    ebox.h[2] = box.cpu_h[2] * ebox.num_cells[2];
    ebox.h[5] = box.cpu_h[5] * ebox.num_cells[2];
    ebox.h[8] = box.cpu_h[8] * ebox.num_cells[2];

    ebox.h[9] = ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7];
    ebox.h[10] = ebox.h[2] * ebox.h[7] - ebox.h[1] * ebox.h[8];
    ebox.h[11] = ebox.h[1] * ebox.h[5] - ebox.h[2] * ebox.h[4];
    ebox.h[12] = ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8];
    ebox.h[13] = ebox.h[0] * ebox.h[8] - ebox.h[2] * ebox.h[6];
    ebox.h[14] = ebox.h[2] * ebox.h[3] - ebox.h[0] * ebox.h[5];
    ebox.h[15] = ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6];
    ebox.h[16] = ebox.h[1] * ebox.h[6] - ebox.h[0] * ebox.h[7];
    ebox.h[17] = ebox.h[0] * ebox.h[4] - ebox.h[1] * ebox.h[3];
    double det = ebox.h[0] * (ebox.h[4] * ebox.h[8] - ebox.h[5] * ebox.h[7]) +
                 ebox.h[1] * (ebox.h[5] * ebox.h[6] - ebox.h[3] * ebox.h[8]) +
                 ebox.h[2] * (ebox.h[3] * ebox.h[7] - ebox.h[4] * ebox.h[6]);
    for (int n = 9; n < 18; n++) {
      ebox.h[n] /= det;
    }
  }

  return is_small_box;
}

void NEP::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const bool is_small_box = get_expanded_box(paramb.rc_radial_max, box, ebox);
  if (is_small_box) {
    // update small_box_data
    const int current_num_atoms = type.size();
    if (small_box_data.NN_radial.size() != current_num_atoms) {
        const int big_neighbor_size = 2000;
        const int size_x12 = current_num_atoms * big_neighbor_size;

        small_box_data.NN_radial.resize(current_num_atoms);
        small_box_data.NL_radial.resize(size_x12);
        small_box_data.NN_angular.resize(current_num_atoms);
        small_box_data.NL_angular.resize(size_x12);
        small_box_data.r12.resize(size_x12 * 6);
    }

    compute_small_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  } else {
    compute_large_box(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

static __global__ void find_descriptor(
  const float temperature,
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int N2,
  const Box box,
  const int* g_NN,
  const int* g_NL,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N2) {
    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    float q[MAX_DIM] = {0.0f};
    // neighbour-order-independent accumulators (see nep_acc_t)
    nep_acc_t q_acc[MAX_NUM_N] = {0.0};

    // get radial descriptors
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[static_cast<size_t>(N) * i1 + n1];
      float x12 = g_x[n2] - x1;
      float y12 = g_y[n2] - y1;
      float z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      float fc12;
      int t2 = g_type[n2];
      float rc = (paramb.rc_radial[t1] + paramb.rc_radial[t2]) * 0.5f;
      float rcinv = 1.0f / rc;
      find_fc(rc, rcinv, d12, fc12);
      float fn12[MAX_NUM_N];
      find_fn(paramb.basis_size_radial, rcinv, d12, fc12, fn12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2, n, k, paramb.n_max_radial, paramb.basis_size_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        q_acc[n] += static_cast<nep_acc_t>(gn12);
      }
    }
    // exactly one narrowing: deterministic, independent of neighbour order
    for (int n = 0; n <= paramb.n_max_radial; ++n) {
      q[n] = static_cast<float>(q_acc[n]);
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      nep_acc_t s_acc[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        float x12 = g_x[n2] - x1;
        float y12 = g_y[n2] - y1;
        float z12 = g_z[n2] - z1;
        apply_mic(box, x12, y12, z12);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        int t2 = g_type[n2];
        float rc = (paramb.rc_angular[t1] + paramb.rc_angular[t2]) * 0.5f;
        float rcinv = 1.0f / rc;
        find_fc(rc, rcinv, d12, fc12);
        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, rcinv, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = get_c_index(
            t1 * paramb.num_types + t2,
            n,
            k,
            paramb.n_max_angular,
            paramb.basis_size_angular,
            paramb.num_c_radial);
          gn12 += fn12[k] * annmb.c_type_pair[c_index];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s_acc);
      }
      float s[NUM_OF_ABC];
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        s[abc] = static_cast<float>(s_acc[abc]);
      }
      find_q(
        paramb.L_max, paramb.has_q_222, paramb.has_q_1111, paramb.has_q_112, paramb.has_q_123, paramb.has_q_233, paramb.has_q_134,
        paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      for (int abc = 0; abc < (paramb.L_max + 1) * (paramb.L_max + 1) - 1; ++abc) {
        g_sum_fxyz[static_cast<size_t>(N) * (n * ((paramb.L_max + 1) * (paramb.L_max + 1) - 1) + abc) + n1] = s[abc];
      }
    }

    // normalize descriptor
    q[annmb.dim - 1] = temperature;
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * annmb.q_scaler[d];
    }

    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    apply_ann_one_layer(
      annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp);
    g_pe[n1] += F;

    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[static_cast<size_t>(N) * d + n1] = Fp[d] * annmb.q_scaler[d];
    }
  }
}

void NEP::compute_large_box(
  const float temperature,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  neighbor_manager.update(
    box, 
    type, 
    position_per_atom);

  find_neighbor_list_large_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    neighbor_manager.get_candidate_NN().data(),
    neighbor_manager.get_candidate_NL().data(),
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data());
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    nep_data.NN_radial.copy_to_host(nep_data.cpu_NN_radial.data());
    nep_data.NN_angular.copy_to_host(nep_data.cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < nep_data.cpu_NN_radial[n]) {
        radial_actual = nep_data.cpu_NN_radial[n];
      }
      if (angular_actual < nep_data.cpu_NN_angular[n]) {
        angular_actual = nep_data.cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  find_descriptor<<<grid_size, BLOCK_SIZE>>>(
    temperature,
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  find_force_radial<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_radial.data(),
    nep_data.NL_radial.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_partial_force_angular<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data());
  GPU_CHECK_KERNEL

  find_properties_many_body(
    box,
    nep_data.NN_angular.data(),
    nep_data.NL_angular.data(),
    nep_data.f12x.data(),
    nep_data.f12y.data(),
    nep_data.f12z.data(),
    position_per_atom,
    force_per_atom,
    virial_per_atom);
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      box,
      nep_data.NN_angular.data(),
      nep_data.NL_angular.data(),
      type.data(),
      position_per_atom.data(),
      position_per_atom.data() + N,
      position_per_atom.data() + N * 2,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

// small box possibly used for active learning:
void NEP::compute_small_box(
  const float temperature,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = type.size();
  const int grid_size = (N2 - N1 - 1) / BLOCK_SIZE + 1;

  const int big_neighbor_size = 2000;
  const int size_x12 = type.size() * big_neighbor_size;

  find_neighbor_list_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    N,
    N1,
    N2,
    box,
    ebox,
    type.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5);
  GPU_CHECK_KERNEL

  static int num_calls = 0;
  if (num_calls++ % 1000 == 0) {
    std::vector<int> cpu_NN_radial(type.size());
    std::vector<int> cpu_NN_angular(type.size());
    small_box_data.NN_radial.copy_to_host(cpu_NN_radial.data());
    small_box_data.NN_angular.copy_to_host(cpu_NN_angular.data());
    int radial_actual = 0;
    int angular_actual = 0;
    for (int n = 0; n < N; ++n) {
      if (radial_actual < cpu_NN_radial[n]) {
        radial_actual = cpu_NN_radial[n];
      }
      if (angular_actual < cpu_NN_angular[n]) {
        angular_actual = cpu_NN_angular[n];
      }
    }
    std::ofstream output_file("neighbor.out", std::ios_base::app);
    output_file << "Neighbor info at step " << num_calls - 1 << ": "
                << "radial(max=" << paramb.MN_radial << ",actual=" << radial_actual
                << "), angular(max=" << paramb.MN_angular << ",actual=" << angular_actual << ")."
                << std::endl;
    output_file.close();
  }

  find_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
    temperature,
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    potential_per_atom.data(),
    nep_data.Fp.data(),
    virial_per_atom.data(),
    nep_data.sum_fxyz.data());
  GPU_CHECK_KERNEL

  find_force_radial_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_radial.data(),
    small_box_data.NL_radial.data(),
    type.data(),
    small_box_data.r12.data(),
    small_box_data.r12.data() + size_x12,
    small_box_data.r12.data() + size_x12 * 2,
    nep_data.Fp.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  find_force_angular_small_box<<<grid_size, BLOCK_SIZE>>>(
    paramb,
    annmb,
    N,
    N1,
    N2,
    small_box_data.NN_angular.data(),
    small_box_data.NL_angular.data(),
    type.data(),
    small_box_data.r12.data() + size_x12 * 3,
    small_box_data.r12.data() + size_x12 * 4,
    small_box_data.r12.data() + size_x12 * 5,
    nep_data.Fp.data(),
    nep_data.sum_fxyz.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data());
  GPU_CHECK_KERNEL

  if (zbl.enabled) {
    find_force_ZBL_small_box<<<grid_size, BLOCK_SIZE>>>(
      paramb,
      N,
      zbl,
      N1,
      N2,
      small_box_data.NN_angular.data(),
      small_box_data.NL_angular.data(),
      type.data(),
      small_box_data.r12.data() + size_x12 * 3,
      small_box_data.r12.data() + size_x12 * 4,
      small_box_data.r12.data() + size_x12 * 5,
      force_per_atom.data(),
      force_per_atom.data() + N,
      force_per_atom.data() + N * 2,
      virial_per_atom.data(),
      potential_per_atom.data());
    GPU_CHECK_KERNEL
  }
}

void NEP::compute(
  const float temperature,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  const bool is_small_box = get_expanded_box(paramb.rc_radial_max, box, ebox);

  if (is_small_box) {
    // update small_box_data
    const int current_num_atoms = type.size();
    if (small_box_data.NN_radial.size() != current_num_atoms) {
        const int big_neighbor_size = 2000;
        const int size_x12 = current_num_atoms * big_neighbor_size;

        small_box_data.NN_radial.resize(current_num_atoms);
        small_box_data.NL_radial.resize(size_x12);
        small_box_data.NN_angular.resize(current_num_atoms);
        small_box_data.NL_angular.resize(size_x12);
        small_box_data.r12.resize(size_x12 * 6);
    }

    compute_small_box(
      temperature,
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  } else {
    compute_large_box(
      temperature,
      box,
      type,
      position_per_atom,
      potential_per_atom,
      force_per_atom,
      virial_per_atom);
  }

  if (has_dftd3) {
    dftd3.compute(
      box, type, position_per_atom, potential_per_atom, force_per_atom, virial_per_atom);
  }
}

const GPU_Vector<int>& NEP::get_NN_radial_ptr() { return nep_data.NN_radial; }

const GPU_Vector<int>& NEP::get_NL_radial_ptr() { return nep_data.NL_radial; }
