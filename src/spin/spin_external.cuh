/*
    Spin Route B, V4: external magnetic field and magneto-electric coupling.

    Two terms are provided, both modelled on the LAMMPS SPIN package:

    1) Zeeman field  --  LAMMPS `fix precession/spin zeeman H x y z`

           E_Z = -sum_i S_i . B_ext,   S_i in mu_B, B_ext in eV/mu_B

       LAMMPS carries a unit spin vector and hard-codes g = 2, so its
       B[eV] = g mu_B B[T] = 1.157676e-4 * B[T].  GPUMD carries the MAGNETIC
       MOMENT (Fe = 2.0 mu_B) in the spin column, so the same physical Zeeman
       energy is obtained with B[eV/mu_B] = mu_B[eV/T] B[T]: the g and the
       moment length are already in S.  g = 1 is the default in run.cu.
       The time law is B_ext(t) = B * env(t) with env = 1, sin(2 pi f t) or
       cos(2 pi f t); f in THz, t in fs.  That is how an FMR / spin-pumping
       drive is set up.

    2) Magneto-electric coupling  --  LAMMPS `pair_style spin/magelec`

           H_ME = sum_{i<j} kappa_ij (E x r_ij) . (S_i x S_j)

       the electric-polarisation-induced Dzyaloshinskii-Moriya term of
       Katsura-Nagaosa-Balatsky (PRL 95, 057205 (2005)) in the form implemented
       by Tranchida et al. (J. Comput. Phys. 372, 406 (2018)).  With
       D_ij = kappa (E x r_ij) the associated torque and force are

           omega_i = -sum_j S_j x D_ij          (eV/mu_B)
           F_i     = -sum_j (S_i x S_j) x E     (eV/A)

       i.e. exactly the LAMMPS expressions.  Only the constant differs, because
       GPUMD carries S in mu_B where LAMMPS carries a unit vector:

           kappa_GPUMD [eV/(A mu_B^2)] = kappa_LAMMPS / |S|^2

       Units here (r21, self-consistent -- see PHYSICS_DERIVATION.md):
         energy eV, length A, spin in mu_B, kappa in eV/(A mu_B^2),
         E a DIMENSIONLESS screened-polarisation vector (the code applies no
         unit conversion to E).  Then
             [kappa] [E x r] [S_i x S_j] = eV/(A mu_B^2) * A * mu_B^2 = eV.
         Declaring E in "eV" as well gave eV^2 and was a documentation error;
         E is not an energy.  If E is instead supplied as a physical electric
         field in V/A = eV/(e A), kappa must be rescaled accordingly; the code
         has no separate E unit flag and never converts E.
         Zeeman: the structure file's spin column is the MAGNETIC MOMENT in
         mu_B (not angular momentum in hbar).  Therefore
             B[eV/mu_B] = mu_B[eV/T] * B[T] = 5.7883818066e-5 * B[T],
         with NO extra Lande factor.  g is already contained in gamma
         (g mu_B/hbar = 3.0385 mu_B/(eV fs), g = 2); multiplying by g again
         would double the field.  `magfield ... g=` exists only for inputs that
         carry angular momentum instead of a moment.  (r20's header claimed
         B = 1.157676e-4 * B[T] with g = 2; the code has always used g = 1.)

    Both terms are applied inside Force::compute(), i.e. at the same place the
    potential is evaluated, so every driver (MD, both minimizers, the replica
    build) sees them without extra plumbing.  They only ever *add* to
    atom.torque / force_per_atom / potential_per_atom / virial_per_atom; the
    NEP arrays are never overwritten.
*/

#pragma once
#include "utilities/gpu_vector.cuh"
#include <string>
#include <vector>

class Box;

class SpinExternal
{
public:
  bool zeeman_enabled = false;
  double b_peak[3] = {0.0, 0.0, 0.0}; // eV/mu_B, amplitude before the envelope
  int zeeman_wave = 0;                // 0 = static, 1 = sin, 2 = cos
  double zeeman_freq_thz = 0.0;
  double zeeman_phase_rad = 0.0;
  double time_fs = 0.0; // accumulated simulation time, drives the envelope

  // ---- magneto-electric ----
  bool magelec_enabled = false;
  double me_kappa = 0.0;            // eV/(A mu_B^2), default for every pair
  double me_E[3] = {0.0, 0.0, 0.0}; // eV
  double me_cutoff = 0.0;           // A
  int num_types = 0;

  // ---- R22-NonHeat: versioned C2 smooth cutoff (opt-in) -------------------
  // R22-COMPAT-01.  `me_cutoff_mode == 0` reproduces the r21 Hamiltonian
  // EXACTLY (hard step at r_c).  It is DISCONTINUOUS in energy, force, spin
  // gradient and virial at r = r_c and is therefore DEPRECATED and NOT a
  // reliable default for long conservative dynamics.  It is the default ONLY
  // so that every r21 input keeps its r21 numbers unchanged; no existing
  // input or model silently changes its predictions.
  // `me_cutoff_mode == 1` is the R22 fix, selected explicitly with
  //     magelec <rc> <kappa> Ex Ey Ez cutoff=smooth [smooth_width=<frac>]
  // and multiplies H_ME by the C2 quintic switch
  //     f_c(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5, x = (r-r_on)/(r_c-r_on).
  // f_c(0)=1, f_c(1)=f'_c(0)=f'_c(1)=f''_c(0)=f''_c(1)=0, so the energy
  // AND its first two derivatives are continuous at both ends.  df_c/dr is
  // included in the force and hence in all nine virial components; the spin
  // gradient carries f_c.  The smoothness order is C2 (not C3: f'''_c is
  // discontinuous at the two switching endpoints).
  int me_cutoff_mode = 0;          // 0 = r21 legacy hard cutoff, 1 = R22 C2
  double me_smooth_fraction = 0.2; // (r_c - r_on)/r_c, in (0, 1]
  double me_r_on = 0.0;            // A, = r_c * (1 - me_smooth_fraction)
  void set_magelec_switch(const int mode, const double r_on);

  // F-06 (r21): separate "was this pair explicitly set?" mask.  A negative
  // kappa is a legal coupling; using kv >= 0 as the set/unset sentinel made
  // every negative value silently fall back to the global me_kappa.
  std::vector<float> me_kappa_pair;      // num_types^2, value used iff set
  std::vector<float> me_kappa_pair_set;  // num_types^2, 1 = explicitly set

  // ---- diagnostics ------------------------------------------------------
  // `applied_b` is host side and free; the two energies live in a device
  // accumulator and are only pulled back by report(), so an MD or minimisation
  // step never stalls on them.
  double applied_b[3] = {0.0, 0.0, 0.0}; // field actually used in the last call
  double energy_zeeman = 0.0;            // eV, sum_i -S_i.B   (after report())
  double energy_magelec = 0.0;           // eV, sum_{i<j} H_ME  (after report())

  // ---- R22-NonHeat: versioned energy/work ledger ---------------------------
  // W_drive is the work of the EXPLICIT time dependence of the external
  // field.  The integrator samples B at the midpoint t_{n+1/2} and holds it
  // constant for the whole step, so B jumps between steps.  At a jump the
  // spins are continuous, hence
  //     delta H_Z = -sum_i S_i(t_n) . [ B(t_{n+1/2}) - B(t_{n-1/2}) ]
  // is EXACT for the discrete scheme.  Accumulated on the device (no per-step
  // host synchronisation), read only when a ledger row is written.
  void compute_applied_b();               // evaluate the envelope at time_fs
  void reset_ledger();                    // once per run block
  void record_drive_work(
    const float* sx, const float* sy, const float* sz, const int N);
  double read_drive_work();
  void read_energies();                   // pull Zeeman/ME totals to the host

  bool any() const { return zeeman_enabled || magelec_enabled; }

  void set_zeeman(
    const double b[3], const int wave, const double freq_thz, const double phase_rad);
  void set_magelec(const double cutoff, const double kappa, const double E[3]);
  void set_magelec_pair(const int type_i, const int type_j, const double kappa);
  void allocate_types(const int number_of_types);
  // F-09 (r21): set (rather than increment) the envelope clock, so a restart
  // can restore the phase of a time-dependent field exactly.
  void set_time_fs(const double t_fs) { time_fs = t_fs; }
  void advance_time(const double dt_fs) { time_fs += dt_fs; }
  // F-06 (r21): upload a dirty CPU parameter table before the next kernel.
  void sync_kappa_pair();
  bool kappa_pair_dirty() const { return kappa_pair_dirty_; }
  const GPU_Vector<float>& kappa_pair_device() const { return kappa_pair_; }
  const GPU_Vector<float>& kappa_pair_mask_device() const { return kappa_pair_mask_; }
  // Reads the device energy accumulators and prints one line per active term.
  void report(const char* tag = "external terms");

  // Adds the external terms to the arrays the potential has just written.
  // spin/torque must be non-null (3N, component-major).  Any of
  // virial/force/potential may be null when only some of them are wanted.
  void apply(
    Box& box,
    const int number_of_atoms,
    const int* type,
    const double* x,
    const double* y,
    const double* z,
    const float* sx,
    const float* sy,
    const float* sz,
    float* tx,
    float* ty,
    float* tz,
    double* potential,
    double* fx,
    double* fy,
    double* fz,
    double* virial);

private:
  void apply_zeeman(
    const int number_of_atoms,
    const float* sx, const float* sy, const float* sz,
    float* tx, float* ty, float* tz,
    double* potential);
  void apply_magelec(
    Box& box,
    const int number_of_atoms,
    const int* type,
    const double* x, const double* y, const double* z,
    const float* sx, const float* sy, const float* sz,
    float* tx, float* ty, float* tz,
    double* potential,
    double* fx, double* fy, double* fz,
    double* virial);

  GPU_Vector<int> cell_count_;
  GPU_Vector<int> cell_count_sum_;
  GPU_Vector<int> cell_contents_;
  GPU_Vector<float> kappa_pair_;      // device copy of me_kappa_pair
  GPU_Vector<float> kappa_pair_mask_; // device copy of me_kappa_pair_set
  bool kappa_pair_dirty_ = true;      // upload before the next apply_magelec()
  GPU_Vector<double> energy_acc_; // [0] Zeeman, [1] magneto-electric, device side
  GPU_Vector<double> drive_acc_;  // [0] cumulative W_drive, device side (R22)
  bool drive_ready_ = false;      // first field sample only primes prev_b_
  double prev_b_[3] = {0.0, 0.0, 0.0};
};
