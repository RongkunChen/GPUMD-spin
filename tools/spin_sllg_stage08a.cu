/*
    S08A: driver for the finite-temperature SLLG integrator.

    Every trajectory here is produced by the *production* Spin::propagate():
    this file only supplies the effective field of the test Hamiltonian and
    records what comes out.  No private time stepper is used anywhere.

        single   H = -B.S          one spin, constant field
        dimer    H = -J S1.S2 - B.(S1+S2)
        cons     H = -B.S          alpha = 0, T = 0: the S05A conservative
                                   solver, kept here as a regression check

    usage:
      single <gamma> <alpha> <T> <B> <dt> <steps> <seed> <scheme> <stride> <out>
      dimer  <gamma> <alpha> <T> <J> <B> <dt> <steps> <seed> <scheme> <stride> <out>
      cons   <gamma> <B> <dt> <steps> <out>

    <scheme> is geometric | midpoint and is passed straight to spin_llg as
    scheme=..., so the run states which integrator it used.

    The field provider runs entirely on the device (a fill kernel for the
    constant case, a two-body kernel for the dimer).  A host round trip inside
    compute_field() would put a stream synchronisation inside every midpoint
    iteration and make a 1e5-step production-style run take hours; the provider
    is the test physics, not the integrator, so keeping it on the device does
    not change what is being verified.
*/
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "spin/spin.cuh"
#include "spin/spin_field_provider.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Same block size the production kernels use (STAGE08A_BLOCK is private to
// spin.cu, so it is repeated here for the two provider kernels only).
#define STAGE08A_BLOCK 128

// Intended dimer initial state: S1 = (1,0,0), S2 = (0,0.6,0.8), both unit
// vectors and perpendicular, so <S1.S2> is a non-trivial observable and the
// spin length the references assume (|S| = 1) is the spin length the run uses.
// The python checker holds the same two vectors (DIMER_S1 / DIMER_S2) and
// re-derives them from the step-0 row of the trajectory, so the two sides
// cannot drift apart silently.
static const float DIMER_S1[3] = {1.0f, 0.0f, 0.0f};
static const float DIMER_S2[3] = {0.0f, 0.6f, 0.8f};

// |S| of spin n in an SoA buffer laid out as [Sx(0..N-1)][Sy(..)][Sz(..)].
static double soa_norm(const std::vector<float>& s, const int N, const int n)
{
  double q = 0.0;
  for (int c = 0; c < 3; ++c) {
    const double v = (double)s[c * N + n];
    q += v * v;
  }
  return std::sqrt(q);
}

static double soa_dot(const std::vector<float>& s, const int N,
                      const int i, const int j)
{
  double d = 0.0;
  for (int c = 0; c < 3; ++c) {
    d += (double)s[c * N + i] * (double)s[c * N + j];
  }
  return d;
}

static __global__ void fill_constant(
  const int N, float* f, const double bx, const double by, const double bz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  f[n] = (float)bx;
  f[N + n] = (float)by;
  f[2 * N + n] = (float)bz;
}

// B_i = J S_j + B_ext with j the other spin (N = 2), i.e. -dH/dS_i for
// H = -J S1.S2 - B.(S1+S2).
static __global__ void fill_dimer(
  const int N, const float* s, float* f, const double j,
  const double bx, const double by, const double bz)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const int o = (N == 2) ? (1 - n) : n;
  f[n] = (float)(j * (double)s[o] + bx);
  f[N + n] = (float)(j * (double)s[N + o] + by);
  f[2 * N + n] = (float)(j * (double)s[2 * N + o] + bz);
}

class AnalyticField : public SpinFieldProvider
{
public:
  enum Kind { CONSTANT, DIMER };
  Kind kind = CONSTANT;
  double bx = 0.0, by = 0.0, bz = 0.0;
  double j = 0.0;
  long long calls = 0;

  void compute_field(const float* g_spin, float* g_field, const int N) override
  {
    const int grid = (N - 1) / STAGE08A_BLOCK + 1;
    if (kind == CONSTANT) {
      fill_constant<<<grid, STAGE08A_BLOCK>>>(N, g_field, bx, by, bz);
    } else {
      fill_dimer<<<grid, STAGE08A_BLOCK>>>(N, g_spin, g_field, j, bx, by, bz);
    }
    GPU_CHECK_KERNEL
    ++calls;
  }
};

static void dump_state(FILE* out, const int step, const std::vector<float>& s,
                       const int N)
{
  std::fprintf(out, "%d", step);
  for (int n = 0; n < N; ++n) {
    std::fprintf(out, " %.9g %.9g %.9g", (double)s[n], (double)s[N + n],
                 (double)s[2 * N + n]);
  }
  std::fprintf(out, "\n");
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s single <gamma> <alpha> <T> <B> <dt> <steps> <seed> "
                 "<scheme> <stride> <out>\n"
                 "       %s dimer  <gamma> <alpha> <T> <J> <B> <dt> <steps> "
                 "<seed> <scheme> <stride> <out>\n"
                 "       %s cons   <gamma> <B> <dt> <steps> <out>\n",
                 argv[0], argv[0], argv[0]);
    return 2;
  }
  const std::string mode = argv[1];
  double gamma = 3.04, alpha = 0.0, T = 0.0, B = 0.0, J = 0.0;
  double dt = 1.0;
  int steps = 100, seed = 20261, stride = 1;
  std::string scheme = "geometric";
  std::string out_path;

  int N = 1;
  AnalyticField analytic;
  std::vector<float> spin0;

  if (mode == "single" || mode == "dimer") {
    gamma = atof(argv[2]);
    alpha = atof(argv[3]);
    T = atof(argv[4]);
    if (mode == "single") {
      // single: gamma alpha T B dt steps seed scheme stride out -> 10 args
      if (argc < 12) return 2;
      B = atof(argv[5]);
      dt = atof(argv[6]);
      steps = atoi(argv[7]);
      seed = atoi(argv[8]);
      scheme = argv[9];
      stride = atoi(argv[10]);
      out_path = argv[11];
      N = 1;
      analytic.kind = AnalyticField::CONSTANT;
      analytic.bz = B;
      spin0 = {1.0f, 0.0f, 0.0f};
    } else {
      // dimer: gamma alpha T J B dt steps seed scheme stride out -> 11 args
      if (argc < 13) return 2;
      J = atof(argv[5]);
      B = atof(argv[6]);
      dt = atof(argv[7]);
      steps = atoi(argv[8]);
      seed = atoi(argv[9]);
      scheme = argv[10];
      stride = atoi(argv[11]);
      out_path = argv[12];
      N = 2;
      analytic.kind = AnalyticField::DIMER;
      analytic.j = J;
      analytic.bz = B;
      // SoA, built from the named vectors instead of a hand-written literal:
      //   spin0 = [S1x S2x][S1y S2y][S1z S2z]
      // Writing the same six numbers in AoS order ([S1x S1y S1z][S2x S2y S2z])
      // yields |S1| = 1.16619, |S2| = 0.8 -- the exact defect S08A-R1 fixes.
      spin0.assign(6, 0.0f);
      for (int c = 0; c < 3; ++c) {
        spin0[c * 2 + 0] = DIMER_S1[c];
        spin0[c * 2 + 1] = DIMER_S2[c];
      }
    }
  } else if (mode == "cons") {
    if (argc < 7) return 2;
    gamma = atof(argv[2]);
    B = atof(argv[3]);
    dt = atof(argv[4]);
    steps = atoi(argv[5]);
    out_path = argv[6];
    N = 1;
    analytic.kind = AnalyticField::CONSTANT;
    analytic.bz = B;
    spin0 = {1.0f, 0.0f, 0.0f};
    alpha = 0.0;
    T = 0.0;
  } else {
    std::fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 2;
  }

  // ---- initial-state contract -------------------------------------------
  // The state buffer is SoA.  A dimer written in AoS order silently produces
  // |S1| = 1.16619 and |S2| = 0.8 while every reference (Metropolis MC and the
  // four-dimensional Gauss-Legendre quadrature) assumes unit spins, so the
  // comparison measures the input mistake instead of the integrator.  Assert
  // the intended state before the first step rather than trusting the literal.
  const double norm_tol = 1e-6;
  double initial_norms[3] = {0.0, 0.0, 0.0};
  for (int n = 0; n < N; ++n) {
    initial_norms[n] = soa_norm(spin0, N, n);
    if (std::fabs(initial_norms[n] - 1.0) > norm_tol) {
      std::fprintf(stderr,
                   "[stage08a] FATAL: initial |S%d| = %.9f, expected 1 within "
                   "%.1e.  spin0 must be SoA: [Sx(0..N-1)][Sy(0..N-1)]"
                   "[Sz(0..N-1)].\n",
                   n + 1, initial_norms[n], norm_tol);
      return 4;
    }
  }
  const double initial_dot = (N == 2) ? soa_dot(spin0, N, 0, 1)
                                      : std::nan("");
  if (N == 2) {
    const double dot_ref = (double)DIMER_S1[0] * DIMER_S2[0]
                           + (double)DIMER_S1[1] * DIMER_S2[1]
                           + (double)DIMER_S1[2] * DIMER_S2[2];
    if (std::fabs(initial_dot - dot_ref) > norm_tol) {
      std::fprintf(stderr,
                   "[stage08a] FATAL: initial S1.S2 = %.9f, expected %.9f.\n",
                   initial_dot, dot_ref);
      return 4;
    }
  }
  std::printf("[stage08a] initial |S| = %.9f %.9f", initial_norms[0],
              initial_norms[1]);
  if (N == 2) std::printf("  S1.S2 = %.9f", initial_dot);
  std::printf("\n");

  // ---- box (fixed lattice; the spins never see the atoms) ----
  Box box;
  box.pbc_x = box.pbc_y = box.pbc_z = 1;
  const double L = 10.0;
  for (int k = 0; k < 9; ++k) box.cpu_h[k] = 0.0;
  box.cpu_h[0] = box.cpu_h[4] = box.cpu_h[8] = L;
  box.get_inverse();
  box.set_is_orthogonal();
  for (int k = 0; k < 18; ++k) box.float_h[k] = (float)box.cpu_h[k];

  // ---- production spin object ----
  Spin spin;
  {
    char g[64], a[64], t[64], s[64];
    std::snprintf(g, sizeof g, "%.17g", gamma);
    std::snprintf(a, sizeof a, "%.17g", alpha);
    std::snprintf(t, sizeof t, "%.17g", T);
    std::snprintf(s, sizeof s, "%d", seed);
    std::vector<std::string> tokens = {"spin_llg", std::string("gamma=") + g,
                                       std::string("alpha=") + a,
                                       std::string("T=") + t,
                                       std::string("seed=") + s,
                                       std::string("scheme=") + scheme};
    spin.parse(tokens);
  }
  spin.check_discretisation(dt);
  spin.initialize(N, true);

  GPU_Vector<float> d_spin(static_cast<size_t>(N) * 3);
  GPU_Vector<float> d_torque(static_cast<size_t>(N) * 3);
  d_spin.copy_from_host(spin0.data());
  std::vector<float> zero(static_cast<size_t>(N) * 3, 0.0f);
  d_torque.copy_from_host(zero.data());

  FILE* out = std::fopen(out_path.c_str(), "w");
  if (out == nullptr) {
    std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
    return 3;
  }
  std::fprintf(out, "# stage08a case %s N %d gamma %.17g alpha %.17g T %.17g "
                    "B %.17g J %.17g dt %.17g steps %d seed %d scheme %s "
                    "stride %d\n",
               mode.c_str(), N, gamma, alpha, T, B, J, dt, steps, seed,
               scheme.c_str(), stride);
  std::fprintf(out, "# midpoint_tolerance %.17g max_iterations %d\n",
               spin.midpoint_tolerance, spin.midpoint_max_iterations);
  // What the integrator was actually handed, measured from the buffer it was
  // handed (not from the literal above).  The checker compares this with the
  // |S| its references assume; a mismatch is a harness defect, not physics.
  std::fprintf(out, "# initial_norms");
  for (int n = 0; n < N; ++n) std::fprintf(out, " %.12g", initial_norms[n]);
  std::fprintf(out, "\n");
  if (N == 2) std::fprintf(out, "# initial_dot %.12g\n", initial_dot);
  dump_state(out, 0, spin0, N);

  std::vector<float> host(static_cast<size_t>(N) * 3, 0.0f);
  for (int step = 0; step < steps; ++step) {
    // The production contract: the caller hands propagate() the deterministic
    // field evaluated at the *current* spins (run.cu gets it from
    // Force::compute).  The geometric branch uses this buffer directly; the
    // midpoint branch ignores it and re-evaluates at the midpoint iterates.
    // Skipping this call would feed the geometric branch a zero field, which
    // silently turns the run into pure noise.
    analytic.compute_field(d_spin.data(), d_torque.data(), N);
    spin.propagate(dt, d_spin.data(), d_torque.data(), N, 0, &analytic);
    if ((step + 1) % stride == 0 || step + 1 == steps) {
      d_spin.copy_to_host(host.data());
      dump_state(out, step + 1, host, N);
    }
  }
  CHECK(gpuStreamSynchronize(0));

  // Final diagnostics, all computed here so Python never has to guess.
  d_spin.copy_to_host(host.data());
  double nerr = 0.0;
  for (int n = 0; n < N; ++n) {
    double s0 = 0.0, s1 = 0.0;
    for (int c = 0; c < 3; ++c) {
      s0 += (double)spin0[c * N + n] * spin0[c * N + n];
      s1 += (double)host[c * N + n] * host[c * N + n];
    }
    nerr = std::fmax(nerr, std::fabs(std::sqrt(s1) - std::sqrt(s0)));
  }
  std::fprintf(out, "# final_norm_error %.9e\n", nerr);
  std::fprintf(out, "# pre_projection_norm_error %.9e\n",
               spin.sllg_pre_projection_norm_error());
  std::fprintf(out, "# sllg_steps %lld\n", spin.sllg_steps);
  std::fprintf(out, "# sllg_field_evaluations %lld\n", spin.sllg_field_evaluations);
  std::fprintf(out, "# sllg_max_iterations_used %d\n", spin.sllg_max_iterations_used);
  std::fprintf(out, "# sllg_max_residual %.9e\n", spin.sllg_max_residual);
  std::fprintf(out, "# sllg_nonconverged_steps %lld\n", spin.sllg_nonconverged_steps);
  std::fprintf(out, "# midpoint_steps %lld\n", spin.midpoint_steps);
  std::fprintf(out, "# provider_calls %lld\n", analytic.calls);
  std::fprintf(out, "# end\n");
  std::fclose(out);

  std::printf("[stage08a] %s alpha=%g T=%g B=%g J=%g dt=%g steps=%d scheme=%s "
              "pre_proj_norm=%.2e final_norm=%.2e nonconv=%lld\n",
              mode.c_str(), alpha, T, B, J, dt, steps, scheme.c_str(),
              spin.sllg_pre_projection_norm_error(), nerr,
              spin.sllg_nonconverged_steps);
  return 0;
}
