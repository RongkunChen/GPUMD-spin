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

#pragma once
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

// Spin Route B: descriptor layout written to nep.txt.
//   spin_layout <version> <block> <block> ...
// enumerates the spin descriptor blocks in the order in which they are
// concatenated into the ANN input.  Version 1 is implicit (the line is absent
// and the order is understood to be the one below); version 2 and later spell
// it out, so that an MD build whose kernels implement a different set of blocks
// can refuse the model instead of silently misreading it.
#include "../utilities/spin_descriptor.cuh"

// Canonical block list of the current layout version (3): the 4th block is the
// single-ion anisotropy channel "Sia".  Must stay in sync with the kernels in
// main_nep/nep.cu, main_nep/nep_specialized.cu (JIT), force/nep.cu (MD) and
// main_replica/force/nep.cu (replica MD).
inline const std::vector<std::string>& spin_layout_blocks()
{
  static const std::vector<std::string> blocks = {"Sex", "San", "Sdm", "Sia", "Smag"};
  return blocks;
}

// Block list of layout versions 1 and 2, whose 4th block is the legacy "Senv"
// channel (linear in one spin, hence time-reversal odd -- see the long note in
// utilities/spin_descriptor.cuh).  Kept so that existing models still load.
inline const std::vector<std::string>& spin_layout_blocks_legacy()
{
  static const std::vector<std::string> blocks = {"Sex", "San", "Sdm", "Senv", "Smag"};
  return blocks;
}

// Default |S| -> u scaling of the Smag block.  Overridable with the
// `spin_mag_scale` keyword in nep.in, and recorded in nep.txt.
#define SPIN_MAG_SCALE_DEFAULT 0.25f

// Default random seed used when nep.in has no `seed` keyword (para.seed < 0).
// Reproducibility fix (r16).  The device side has always been deterministic:
// SNES::initialize_curand_states() is called with `para.seed >= 0 ? para.seed
// : 1234567`.  The host side, however, seeded std::mt19937 from the system
// clock unless the binary was built with -DDEBUG, so two runs of the *same*
// binary on the *same* nep.in produced different initial mu, and hence
// different parameters, losses and nep.txt.  That made NEP training
// irreproducible by default and made value-level regression testing
// impossible (all 22 training cases in tests_regression omit `seed`).
// The host RNG now falls back to the same constant the device already used,
// so a training run is bit-reproducible with or without an explicit `seed`.
// Users who want different restarts still have the `seed` keyword.
#define NEP_DEFAULT_SEED 1234567

// the model hyperparameters as recorded in the header of a nep.txt file
struct NepTxtHeader {
  std::string model_token;
  int version;
  int model_type;
  int charge_mode;
  int vdw;
  int charge_vdw;
  bool enable_zbl;
  bool flexible_zbl;
  bool use_typewise_cutoff_zbl;
  float zbl_rc_inner;
  float zbl_rc_outer;
  float typewise_cutoff_zbl_factor;
  int num_types;
  std::vector<std::string> elements;
  bool has_multiple_cutoffs;
  std::vector<float> rc_radial;
  std::vector<float> rc_angular;
  int n_max_radial;
  int n_max_angular;
  int basis_size_radial;
  int basis_size_angular;
  int L_max;
  int has_q_222; // as written to nep.txt, that is 2 or 0
  int has_q_1111;
  int has_q_112;
  int has_q_123;
  int has_q_233;
  int has_q_134;
  int num_neurons1;
  int num_neurons2;
  int spin;                   // 1 for a spin-aware (magnetic) model
  std::vector<int> use_spin;  // per atom type: 1 if the type carries a spin vector
  int spin_layout_version = 1;          // 1 = implicit, 2 = explicit (Senv), 3 = explicit (Sia)
  std::vector<std::string> spin_blocks; // descriptor blocks, in concatenation order
  // True when the 4th spin block is the single-ion anisotropy "Sia" (layout 3);
  // false keeps the legacy time-reversal-odd "Senv" block (layout 1/2).
  bool spin_use_sia = false;
  int spin_block_mask = SPIN_BLOCK_MASK_ALL;  // F26 ablation mask
  bool has_spin_blocks = false;               // F26: the line was present
  float spin_mag_scale = SPIN_MAG_SCALE_DEFAULT; // |S| -> u scaling of the Smag block
  bool has_spin_mag_scale = false;  // true only when the file actually carries the line
  int number_of_header_lines; // 6 without a zbl line, 7 with one, plus one for the spin line
};

// parse the header of a nep.txt file; returns false and sets error if it cannot be parsed
bool read_nep_txt_header(const std::string& filename, NepTxtHeader& header, std::string& error);

class Parameters
{
public:
  Parameters();

  // parameters to be read in
  int version;            // NEP version; only NEP4 is supported
  int batch_size;         // number of configurations in one batch
  int num_types;          // number of atom types
  int population_size;    // population size for SNES
  int seed;               // seed of the SNES random draws; negative selects NEP_DEFAULT_SEED
  int maximum_generation; // maximum number of generations for SNES;
  int save_potential;     // number of generations between writing a checkpoint nep.txt file.
  int save_potential_format;   // format of checkpoint nep.txt file name
  int save_potential_restart;  // if restart files should be written or not. 0=no, 1=yes
  int output_interval;    // number of generations between writing loss.out, nep.txt, nep.restart and related output
  int num_neurons1;       // number of neurons in the first hidden layer
  int num_neurons2;       // number of neurons in the second hidden layer; 0 if unused
  int num_hidden_layers;  // number of hidden layers
  int basis_size_radial;
  int basis_size_angular;
  int n_max_radial;       // maximum order of the radial Chebyshev polynomials
  int n_max_angular;      // maximum order of the angular Chebyshev polynomials
  int L_max;              // maximum order of the three-body spherical harmonics
  int has_q_222;          // has q_222
  int has_q_1111;         // has q_1111
  int has_q_112;          // has q_112
  int has_q_123;          // has q_123
  int has_q_233;          // has q_233
  int has_q_134;          // has q_134
  float lambda_1;         // weight parameter for L1 regularization loss
  float lambda_2;         // weight parameter for L2 regularization loss
  float lambda_e;         // weight parameter for energy RMSE loss
  float lambda_f;         // weight parameter for force RMSE loss
  float lambda_v;         // weight parameter for virial RMSE loss
  float lambda_shear;     // extra weight parameter for shear virial
  float lambda_q;         // weight for global charge
  float lambda_z;         // weight for BEC
  float lambda_d;         // weight for the energy-difference loss over the pairs in ediff.in
  float lambda_m;         // weight for the magnetic torque (-dE/dS) RMSE loss
  bool enable_spin;       // true for a spin-aware (magnetic) NEP model
  int dim_spin;           // number of spin descriptor components
  std::vector<int> use_spin; // per atom type: 1 = this type carries a spin vector
  float spin_mag_scale = SPIN_MAG_SCALE_DEFAULT; // |S| -> u scaling of the Smag block
  bool spin_use_sia = false; // 4th spin block is Sia (layout 3) rather than Senv
  int spin_block_mask = SPIN_BLOCK_MASK_ALL;  // F26 ablation mask from nep.txt
  bool has_spin_blocks = false;               // F26: the line was present
  float spin_mag_scale_from_file = 0.0f; // value recorded in an existing nep.txt
  bool has_spin_mag_scale_from_file = false;
  bool spin_mag_scale_line_present = false; // the nep.txt carries an explicit line
  float force_delta;      // a parameter used to modify the force loss
  bool enable_zbl;        // true for including the universal ZBL potential
  bool flexible_zbl;      // true for including the flexible ZBL potential
  float zbl_rc_inner;     // inner cutoff for the universal ZBL potential
  float zbl_rc_outer;     // outer cutoff for the universal ZBL potential
  int model_type; // 0=potential, 1=dipole, 2=polarizability, 3=temperature-dependent free energy
  int prediction; // 0=no, 1=yes
  float initial_para;
  float sigma0;
  int atomic_v;
  bool use_typewise_cutoff_zbl;
  float typewise_cutoff_zbl_factor;
  int output_descriptor;
  int charge_mode; // add dynamic charge to NEP potential model
  int vdw;         // add environment-dependent vdW to ordinary NEP
  int charge_vdw;  // combined charge-vdW model
  bool has_bec = false; // check if there are target BEC values
  // set by EnergyDifference when ediff.in resolves a training combination
  bool has_ediff_combinations = false;
  int flip_charge = 0; // 1 for flipping charges upon restarting
  int fine_tune = 0; // fine_tune option; 0=no, 1=yes
  int fine_tune_descriptor = 1; // fine-tune descriptor; 0=no, 1=yes
  std::string fine_tune_nep_txt = "";
  std::string fine_tune_nep_restart = "";
  bool import_q_scaler = false; // read q_scaler from the local nep.txt instead of recomputing it
  bool nep_compile = true; // runtime-specialize NEP-family training kernels

  // check if a parameter has been set:
  bool is_model_type_set;
  bool is_prediction_set;
  bool is_version_set;
  bool is_type_set;
  bool is_cutoff_set;
  bool is_n_max_set;
  bool is_basis_size_set;
  bool is_l_max_set;
  bool is_neuron_set;
  bool is_lambda_1_set;
  bool is_lambda_2_set;
  bool is_lambda_e_set;
  bool is_lambda_f_set;
  bool is_lambda_v_set;
  bool is_atomic_v_set;
  bool is_lambda_shear_set;
  bool is_lambda_d_set;
  bool is_lambda_m_set;
  bool is_spin_set;
  bool is_spin_mag_scale_set;
  bool is_batch_set;
  bool is_population_set;
  bool is_seed_set;
  bool is_generation_set;
  bool is_save_potential_set;
  bool is_output_interval_set;
  bool is_type_weight_set;
  bool is_force_delta_set;
  bool is_zbl_set;
  bool is_use_typewise_cutoff_zbl_set;
  bool is_charge_mode_set;
  bool is_vdw_set;
  bool is_charge_vdw_set;

  // other parameters
  int dim;                            // dimension of the descriptor vector
  int dim_radial;                     // number of radial descriptor components
  int dim_angular;                    // number of angular descriptor components
  int number_of_variables;            // total number of parameters (NN and descriptor)
  int number_of_variables_ann;        // number of parameters in the ANN only
  int number_of_variables_ann_1;      // number of parameters in the ANN for one element
  int number_of_variables_descriptor; // number of parameters in the descriptor only
  int number_of_nep_txt_header_lines = 0; // header length of the nep.txt file that has been checked

  // some arrays

  std::vector<float> type_weight_cpu; // relative force weight for different atom types (CPU)
  std::vector<float> q_scaler_cpu;    // used to scale some descriptor components (CPU)
  std::vector<std::string> elements;  // atom symbols
  std::vector<int> atomic_numbers;
  std::vector<float> zbl_para;        // parameters of zbl potential
  std::vector<float> rc_radial;       // radial cutoff distance
  std::vector<float> rc_angular;      // angular cutoff distance
  float rc_radial_max = 0.0f;         // maximal radial cutoff
  float rc_angular_max = 0.0f;        // maximal angular cutoff
  bool has_multiple_cutoffs = false;

  GPU_Vector<float> q_scaler_gpu[16]; // used to scale some descriptor components (GPU)
  GPU_Vector<float> q_scaler_max[16]; // used to scale some descriptor components (GPU)
  GPU_Vector<float> q_scaler_min[16]; // used to scale some descriptor components (GPU)

private:
  void set_default_parameters();
  void read_nep_in();
  void read_zbl_in();
  void calculate_parameters();
  void report_inputs();
  void check_existing_model();
  void check_nep_txt(const std::string& filename, const bool fatal, const char* remedy);
  void check_nep_restart();
  void compare_with_nep_txt(const std::string& filename, std::vector<std::string>& mismatches);

  void parse_one_keyword(std::vector<std::string>& tokens);

  void parse_model_type(const char** param, int num_param);
  void parse_prediction(const char** param, int num_param);
  void parse_version(const char** param, int num_param);
  void parse_type(const char** param, int num_param);
  void parse_type_weight(const char** param, int num_param);
  void parse_zbl(const char** param, int num_param);
  void parse_cutoff(const char** param, int num_param);
  void parse_n_max(const char** param, int num_param);
  void parse_basis_size(const char** param, int num_param);
  void parse_l_max(const char** param, int num_param);
  void parse_neuron(const char** param, int num_param);
  void parse_lambda_1(const char** param, int num_param);
  void parse_lambda_2(const char** param, int num_param);
  void parse_lambda_e(const char** param, int num_param);
  void parse_lambda_f(const char** param, int num_param);
  void parse_lambda_v(const char** param, int num_param);
  void parse_lambda_q(const char** param, int num_param);
  void parse_lambda_z(const char** param, int num_param);
  void parse_lambda_d(const char** param, int num_param);
  void parse_lambda_m(const char** param, int num_param);
  void parse_spin(const char** param, int num_param);
  void parse_spin_mag_scale(const char** param, int num_param);
  void parse_spin_blocks(const char** param, int num_param);
  void parse_lambda_shear(const char** param, int num_param);
  void parse_force_delta(const char** param, int num_param);
  void parse_batch(const char** param, int num_param);
  void parse_population(const char** param, int num_param);
  void parse_seed(const char** param, int num_param);
  void parse_generation(const char** param, int num_param);
  void parse_initial_para(const char** param, int num_param);
  void parse_sigma0(const char** param, int num_param);
  void parse_atomic_v(const char** param, int num_param);
  void parse_use_typewise_cutoff_zbl(const char** param, int num_param);
  void parse_output_descriptor(const char** param, int num_param);
  void parse_charge_mode(const char** param, int num_param);
  void parse_vdw(const char** param, int num_param);
  void parse_charge_vdw(const char** param, int num_param);
  void parse_fine_tune(const char** param, int num_param);
  void parse_save_potential(const char** param, int num_param);
  void parse_output_interval(const char** param, int num_param);
  void parse_import_q_scaler(const char** param, int num_param);
  void parse_nep_compile(const char** param, int num_param);
};
