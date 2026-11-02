/*
    Spin Route B: LLG (Landau-Lifshitz-Gilbert) spin dynamics for GPUMD.

    The magnetic torque  w_i = -dE/dS_i  produced by a spin-aware NEP model is
    the effective field acting on the spin (in eV / mu_B).  The spin is
    propagated with the LLG form used in most atomistic spin-dynamics codes:

        dS/dt = -gamma  S x B  -  gamma*alpha/|S|  S x (s x B)
        B     = w + b(t)          (effective field = -dE/dS + thermal field)

    The first term is the precession (energy conserving), the second is the
    Gilbert damping (energy decreasing).  |S| is conserved because both terms
    are perpendicular to S.

    NOTE on the gamma convention (Stage04): this is the Landau-Lifshitz (LL)
    form, i.e. the damping coefficient is gamma*alpha.  The Gilbert form

        dS/dt = -gamma_G S x B + (alpha/|S|) S x dS/dt

    maps onto it with gamma_G = gamma (1 + alpha^2), so `gamma` here is
    gamma_LL = gamma_G / (1 + alpha^2).  Everything below uses gamma = gamma_LL.

    Integrator (F13): the equation is NOT integrated with explicit Euler.  The
    precession is applied as an exact Rodrigues rotation about B-hat by
    gamma|B|dt and the damping as the analytic relaxation
    tan(theta/2) -> tan(theta/2) exp(-gamma alpha |B| dt), combined in a Strang
    split.  Both maps are isometries, so |S| is conserved to round-off and the
    scheme is unconditionally stable in gamma|B|dt.  The explicit-Euler +
    renormalise form it replaced lagged by ~theta^3/3 per step with a fixed
    sign, pumping +8e-9 eV/atom/step into the lattice even at alpha = 0; that
    error grows as |B|^3, which is what makes long NVE + SLLG runs diverge.

    run.in keyword:
        spin_llg  <gamma>  <alpha>  [<temperature>  [<seed>]]
        spin_llg  gamma=<g>  alpha=<a>  [T=<T>]  [seed=<n>]        (F21)
    gamma       : gyromagnetic ratio, natural GPUMD units (1/fs per eV/mu_B).
                  The physical value is g*mu_B/hbar = 3.0385; anything much
                  smaller makes the SLLG noise dominate the exchange field and
                  invalidates every dynamical quantity (equilibrium is fine).
    alpha       : dimensionless Gilbert damping
    temperature : (optional, default 0) temperature of the stochastic
                  Landau-Lifshitz (SLLG) thermal field, in kelvin.
                  A value of 0 reproduces the deterministic LLG exactly.
    seed        : (optional, default 12345) RNG seed for the thermal field.

    Thermal field (fluctuation-dissipation for the form actually integrated):

        <b_i(t) b_j(t')> = 2 alpha kB T / (gamma (1 + alpha^2) |S_i|)
                           delta_ij delta(t-t')

    The 1/(1 + alpha^2) belongs to the LL form above: the FDT is derived for
    the Gilbert form with gamma_G = gamma (1 + alpha^2), and the amplitude
    carries 1/gamma_G.  Dropping it makes the stationary state Boltzmann at
    T (1 + alpha^2) instead of T -- a factor 2 in effective temperature at
    alpha = 1, 1.25x at alpha = 0.5, 1.01x at alpha = 0.1.

    discretised with the time step dt as (one draw per atom per Cartesian
    component, so Var(b_k) = prefactor/|S_i| independently of k)

        b = sqrt(2 alpha kB T / (gamma (1 + alpha^2) |S_i| dt)) * xi,
        xi ~ N(0,1)

    which has the right dimension:  [kB T] = eV, [gamma] = mu_B/(eV fs),
    [|S|] = mu_B, [dt] = fs  =>  [b] = eV/mu_B, the same unit as the torque.
    Only the component of b perpendicular to S matters (S x b kills the
    longitudinal part), so the magnitude |S| stays constant.

    Known edge case (R11): the amplitude diverges as |S| -> 0, so an atom with
    a numerically tiny (rather than exactly zero) |S| receives a thermal field
    orders of magnitude larger than a magnetic one.  The LLG increment scales
    as sqrt(|S| * <b^2>), so it still vanishes for |S| -> 0 and the physics is
    unaffected; initialize() prints a one-off diagnostic when it happens.

    Equilibrium and alpha (round 15, P110/P111) -- supersedes R29/R31:

    The stationary distribution of this map does NOT depend on alpha.  Both
    sub-maps are isometries and each step is the exact flow of the frozen-field
    SLLG, so the only approximation is that the noise is piecewise constant
    over dt (Wong-Zakai), which converges to the Stratonovich equation that the
    FDT amplitude above was derived for.  Verified against Metropolis Monte
    Carlo on an interacting simple-cubic Heisenberg ferromagnet
    (runs/spectra/p110_sllg_interacting.py): M/M_MC = 1.007 at alpha = 0.01
    and 0.997 at alpha = 0.03, with the exact Hamiltonian energy drift 58x
    smaller than the bias of the first-order work diagnostic.

    What alpha *does* control is the relaxation rate, tau_Gilbert =
    1/(gamma alpha |B|).  Because the equilibrium is alpha independent but the
    rate is not, a small alpha needs a proportionally longer run: rounds 13/14
    read the resulting kinetic arrest as an alpha-dependent equilibrium
    (Fe "M collapses at alpha = 0.003", CrN "T_N moves from 281 to 368 K").

    Corollary: W_noise/|W_rot| is a BIASED diagnostic, not an equilibrium
    test.  |W_rot| carries a geometric part gamma alpha dt |B|^2 that grows
    with dt, so the ratio looks like 1 - c/alpha with c ~ dt even though the
    dynamics is exact.  Judge equilibrium by a stationary potential energy
    (frozen lattice) or by comparison with Monte Carlo.

    Units: the time step is the MD time step in fs (GPUMD natural units are
    converted with TIME_UNIT_CONVERSION inside run.cu, so the value handed to
    propagate() is already in fs).
*/

#pragma once
// NOTE: this header must NOT include "model/atom.cuh".  src/spin/spin.cu is
// compiled once and its object file is linked into BOTH gpumd and
// gpumd_replica, but the two builds have different `Atom` layouts
// (src/model/atom.cuh has position_image / unwrapped_position / heat_per_atom
// and the PIMD vectors, src/main_replica/model/atom.cuh does not).  A single
// translation unit that reads Atom::spin would therefore use the wrong member
// offsets in one of the two executables -- that is exactly the bug that made
// cudaMemcpyAsync fail with "invalid argument" in the replica path.  Spin
// therefore only ever sees raw device pointers, supplied by the caller, which
// is compiled against the matching Atom definition.
#include "utilities/gpu_macro.cuh"
#include "utilities/gpu_vector.cuh"
#ifdef USE_HIP
  #include <hiprand/hiprand_kernel.h>
#else
  #include <curand_kernel.h>
#endif

// Field provider for the conservative (alpha = 0, T = 0) spin update.
// The implicit midpoint rule needs the deterministic effective field
// w = -dE/dS evaluated at the *midpoint* spin configuration, i.e. after every
// iteration has updated every spin.  Re-using the torque handed to propagate()
// (which is the field at the beginning of the step) is exactly the defect this
// interface removes.
// This class deliberately knows nothing about Atom or Force: spin.cu is a
// single translation unit linked into both gpumd and gpumd_replica, whose Atom
// layouts differ (see the note at the top of this file).  The concrete
// implementation is src/spin/spin_field_provider.cuh, a template instantiated
// by each caller against its own Atom.
class SpinFieldProvider
{
public:
  virtual ~SpinFieldProvider(void) {}
  // g_spin / g_field are 3N device buffers, component-major (x block, y block,
  // z block), the same layout Spin::propagate uses.  Must not modify g_spin.
  virtual void compute_field(
    const float* g_spin, float* g_field, const int number_of_atoms) = 0;
};

class Spin
{
public:
  bool enabled = false;   // spin_llg was requested in run.in
  bool has_spin = false;  // model.xyz actually carries a spin column
  double gamma = 1.0;     // gyromagnetic ratio
  double alpha = 0.1;     // Gilbert damping
  double temperature = 0.0; // SLLG thermostat temperature in K (0 = no noise)
  // Round 11 (P86): kept so that the diagnostic `report()` can evaluate the
  // SLLG thermal-field amplitude without an extra host round trip.
  double time_step_fs = 1.0;
  int seed = 12345;       // RNG seed for the thermal field

  // ---- external magnetic field (the spin-side "driving field" of NEMD) ----
  // `add_bfield <Bx> <By> <Bz> [<omega>]` in run.in.  Units eV/mu_B, i.e. the
  // same unit as the torque w = -dE/dS.  omega is in THz and 0 means a static
  // field; a non-zero omega makes it B(t) = B sin(2 pi omega t), which is how
  // an FMR / spin-pumping experiment drives the system.
  double b_ext_x = 0.0, b_ext_y = 0.0, b_ext_z = 0.0;
  double b_ext_omega = 0.0;
  double elapsed_time_fs = 0.0;
  double b_ext_applied[3] = {0.0, 0.0, 0.0}; // last values handed to the kernel

  void set_external_field(double bx, double by, double bz, double omega_thz);
  bool has_external_field() const
  {
    return b_ext_x != 0.0 || b_ext_y != 0.0 || b_ext_z != 0.0;
  }

  // The `add_bfield` values (already multiplied by the sin(2 pi omega t)
  // envelope) that were used in the most recent propagate() call.  The magnon
  // heat current must use the field the spins actually saw, otherwise the
  // Green-Kubo flux is inconsistent with the trajectory whenever `add_bfield`
  // is active.  Zero when no field or before the first step.
  void get_applied_external_field(double b[3]) const
  {
    b[0] = b_ext_applied[0];
    b[1] = b_ext_applied[1];
    b[2] = b_ext_applied[2];
  }

  // tokens = {"spin_llg", "<gamma>", "<alpha>"[, "<temperature>"[, "<seed>"]]}
  void parse(const std::vector<std::string>& tokens);

  // Copy only the run.in settings.  Never copy the device buffers (noise_,
  // spin_new_): GPU_Vector has raw-pointer semantics, so a bitwise copy would
  // make two Spin objects free the same allocation.
  void copy_settings_from(const Spin& other)
  {
    enabled = other.enabled;
    gamma = other.gamma;
    alpha = other.alpha;
    temperature = other.temperature;
    seed = other.seed;
    has_spin = false; // set by initialize() for this replica
  }

  // Print a warning when the explicit LLG update is not resolved by the time
  // step.  Measured on this CrN model at dt = 1 fs:
  //   gamma = 0.01  stable (T stays at the thermostat set point)
  //   gamma = 0.1   lattice is drained, T -> ~3 K
  //   gamma = 1.0   lattice is pumped,  T -> 10^4 K within 5 ps
  //   gamma = 3.04  same, NVE runs die with a CUDA 700 once the atoms fly apart
  // Noise needs alpha >= 1e-3 to dissipate what the thermal field injects;
  // alpha = 2e-4 was seen to diverge.
  void check_discretisation(double time_step_fs);

  // Called once after model.xyz has been read.  Takes plain values (not an
  // Atom&) for the reason given at the top of this file.
  void initialize(const int number_of_atoms, const bool has_spin);

  // One LLG step with the torque computed by the force routine.
  // `g_spin`   : device array of 3N floats, the spins S_i (read and written).
  // `g_torque` : device array of 3N floats, the effective field w_i = -dE/dS_i.
  // `stream` selects the CUDA stream the kernel and the ping-pong copy are
  // enqueued on.  The MD path passes the default stream (0); the replica path
  // must pass the slot stream, otherwise the LLG kernel would race with the
  // force kernel that writes the torque on the slot stream.
  // S05A: alpha = 0 and T = 0 take the conservative implicit-midpoint branch,
  // which needs `provider` to re-evaluate the field at the midpoint spins.  A
  // null provider in that branch is a hard error (no silent fallback to the
  // frozen g_torque).  The finite-temperature / finite-alpha branch ignores it.
  void propagate(
    const double time_step,
    float* g_spin,
    const float* g_torque,
    const int number_of_atoms,
    gpuStream_t stream = 0,
    SpinFieldProvider* provider = nullptr);

  // ---- S05A: conservative implicit midpoint (fixed lattice, alpha = T = 0) --
  // The iteration stops when max_i |S_new^(k+1) - S_new^(k)| <= midpoint_tolerance
  // (mu_B, per component); more than midpoint_max_iterations is a hard failure.
  // 1e-6 is ~8 float32 ulp at |S| = 1.  That is the floor that matters: the
  // iterate itself is stored as float32, so a residual below ~1.2e-7 (2^-23)
  // can never be reached no matter how many iterations are allowed.  Measured
  // with the NEP case: a 1e-7 tolerance stalls at exactly 1.19e-7.  Eight ulp
  // is still ~1e-6 of the spin length, i.e. far below any physical accuracy
  // this scheme is asked for.
  double midpoint_tolerance = 1.0e-6;
  int midpoint_max_iterations = 50;
  // Diagnostics of what the solver actually did (read after the run).
  long long midpoint_steps = 0;              // conservative steps taken
  long long midpoint_field_evaluations = 0;  // field re-evaluations in total
  int midpoint_max_iterations_used = 0;      // worst step
  double midpoint_max_residual = 0.0;        // worst converged residual

  // ---- S08A: stochastic integrator and its diagnostics ------------------
  // Two schemes are implemented for the finite-temperature branch:
  //   sllg_scheme = 0  "geometric": the pre-S08A map.  Precession as an exact
  //                    Rodrigues rotation about the total field, Gilbert
  //                    relaxation as the analytic tan(theta/2) rescaling,
  //                    combined in a Strang split about the field frozen at
  //                    the beginning of the step.  Exact for a constant field,
  //                    so its only discretisation error is the piecewise
  //                    constant noise (Wong-Zakai -> Stratonovich).
  //   sllg_scheme = 1  "midpoint": stochastic implicit midpoint.  The same
  //                    noise draw is held fixed for the whole step, the
  //                    deterministic field is re-evaluated at the midpoint
  //                    spin on every iteration, and the update
  //                        S_new = S_old - gdt m x H - gdt alpha/|m| m x (m x H)
  //                    with m = (S_old + S_new)/2 is norm preserving in exact
  //                    arithmetic, because (S_new - S_old) is perpendicular to
  //                    m and (S_new + S_old) = 2m.
  // Set with `scheme=geometric|midpoint` on the spin_llg line.  The default is
  // the scheme that passed the Stage08A battery (see that stage's summary);
  // both were measured with the same driver and the same criteria.
  int sllg_scheme = 0;
  bool sllg_scheme_set = false;

  // Diagnostics of the stochastic branch.  The two norm numbers are the ones
  // the acceptance test reads: pre_projection is the relative |S| error of the
  // raw map (the integrator itself), norm_deviation is the largest relative
  // distance from the per-atom anchor after the constraint projection (the
  // stored trajectory).  They are accumulated as maxima over the whole run,
  // not read at the last step.
  long long sllg_steps = 0;
  long long sllg_field_evaluations = 0;
  int sllg_max_iterations_used = 0;
  double sllg_max_residual = 0.0;
  double sllg_max_pre_projection_norm_error = 0.0;
  long long sllg_nonconverged_steps = 0;

  // ---- F16: spin work / energy accounting (risk R12) --------------------
  // U(r, S) is one function, so rotating the spins by dS changes the potential
  // energy by dU = -B_det.dS with B_det = w = -dU/dS (the deterministic
  // torque).  The LLG kernel rotates about the *total* field, so the same
  // quantity is split exactly into three bookkeeping terms:
  //     dU = W_rot + W_noise + W_ext
  //     W_rot   = -b_total.dS   (Rodrigues rotation gives b_total.dS = 0, so
  //                              this is entirely the Gilbert relaxation;
  //                              <= 0, a one-way energy sink)
  //     W_noise = +b_noise.dS   (work of the SLLG thermal field; this is the
  //                              term that makes the thermal field a net
  //                              energy SOURCE)
  //     W_ext   = +b_ext.dS     (work of `add_bfield`)
  // All three are accumulated on the device (one block-reduced atomicAdd per
  // block per step, so the cost is negligible) and can be read at any time.
  // Before F16, "the thermal field injects energy" could only be inferred by
  // differencing two 300k-step runs; it is now a directly measured number.
  // S08A: largest relative |S| error the raw stochastic map produced, measured
  // before the constraint projection, accumulated over the whole run.
  double sllg_pre_projection_norm_error(gpuStream_t stream = 0);

  void reset_work(gpuStream_t stream = 0);
  void read_work(double w[3], gpuStream_t stream = 0);
  void print_work(const char* tag, gpuStream_t stream = 0);

  // Diagnostics: mean |S| and the total "magnetic energy" drift monitor.
  void report(const float* g_spin, const float* g_torque, const int number_of_atoms);

  // Total magnetisation sum_i S_i (host, 3 components), for thermostats tests.
  void magnetization(const float* g_spin, const int number_of_atoms, double m[3]) const;

  // ---- S08B: restart state ------------------------------------------------
  // restart.xyz carries (R, v, S) and nothing else.  Two pieces of the magnetic
  // state are not in it and cannot be reconstructed from it:
  //   * the per-atom curand state, i.e. where the thermal noise stream stands.
  //     Re-seeding from `seed` restarts the stream at step 0, which gives a
  //     different trajectory even though the distribution is the same;
  //   * the per-atom |S| anchor used by the constraint projection.
  // `spin_restart.bin` stores those, plus the identity fields (N, model hash,
  // gamma/alpha/T, dt, scheme, step counters) needed to refuse a mismatch.
  // Written by Dump_Restart next to restart.xyz, read back only when run.in
  // says `spin_llg ... restart=spin_restart.bin` -- never silently.
  void set_restart_file(const std::string& path) { restart_file_ = path; }
  bool save_restart_state(const char* path, const long long physics_step,
                          const double global_time);
  bool load_restart_state(const char* path);
  long long restart_physics_step() const { return restart_physics_step_; }
  bool has_restart_digest() const { return !restart_digest_.empty(); }
  const std::string& restart_digest() const { return restart_digest_; }

private:
  int number_of_atoms = 0;
  // P22: the thermal field is drawn on the device with one curand state per
  // atom, so a step no longer has to read |S| back to the host and upload a
  // freshly drawn noise vector.  Besides removing two transfers and a stream
  // synchronize per step, it also keeps the draw inside the slot stream, which
  // is what the replica path needs.
  // S05A: implicit-midpoint workspace (allocated on first use).
  void propagate_conservative(
    const double time_step,
    float* g_spin,
    const int number_of_atoms,
    const float bex,
    const float bey,
    const float bez,
    SpinFieldProvider* provider,
    gpuStream_t stream);
  GPU_Vector<float> spin_mid_;      // (S_old + S_new)/2
  GPU_Vector<float> spin_prev_;     // previous iterate, for the residual
  GPU_Vector<float> field_mid_;     // deterministic field at spin_mid_
  GPU_Vector<double> mid_residual_; // one element: max |S_new - S_prev|

  // S08A: stochastic implicit midpoint (finite T).  Needs the provider for the
  // same reason the conservative branch does: the deterministic field must be
  // re-evaluated at the midpoint iterate, never taken from g_torque.
  void propagate_stochastic_midpoint(
    const double time_step,
    float* g_spin,
    const int number_of_atoms,
    const float bex, const float bey, const float bez,
    const float* bx, const float* by, const float* bz,
    SpinFieldProvider* provider,
    gpuStream_t stream);

  // S08A: project every spin onto its own magnitude anchor and record the
  // largest pre-projection relative error.  |S_i| is a conserved quantity of
  // the equation, so the anchor is a constraint, not a correction of an
  // integration error: the pre-projection number is what says whether the map
  // preserves the norm.  The anchor is refreshed only when |S| changes by more
  // than 1e-4 (a restart or a replicate with different moments).
  void project_onto_norm_anchor(float* g_spin, const int N, gpuStream_t stream);
  GPU_Vector<float> norm_anchor_;   // per-atom |S| anchor
  GPU_Vector<double> sllg_scalar_;  // one element: max pre-projection error

  GPU_Vector<gpurandState> states_;
  GPU_Vector<float> noise_;
  GPU_Vector<float> spin_new_; // ping-pong buffer for one LLG step
  GPU_Vector<double> work_;    // F16: [0] W_rot, [1] W_noise, [2] W_ext (eV)
  long long debug_step_ = 0;
  bool warned_small_spin_ = false; // F14: R11 diagnostic, printed at most once

  // S08B: restart bookkeeping (see the public block above).
  std::string restart_file_;         // from `spin_llg ... restart=<file>`
  std::string restart_digest_;       // hex digest of the loaded payload
  long long restart_physics_step_ = 0; // step index recorded in the checkpoint
};
