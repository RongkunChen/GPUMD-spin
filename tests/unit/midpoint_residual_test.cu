/*
 * r21 GPU unit tests for the implicit-midpoint residual and the |S| anchor.
 *
 * This translation unit TEXTUALLY INCLUDES the shipped src/spin/spin.cu, so it
 * exercises the real gpu_midpoint_residual / gpu_project_anchor kernels rather
 * than a re-implementation.  Build (from the package root):
 *
 *   nvcc -O2 -arch=sm_89 -I src -I src/spin \
 *        tests/unit/midpoint_residual_test.cu -o tests/unit/midpoint_residual_test.exe
 *
 * It prints one JSON object per case to stdout and returns 0 only if every
 * case passes.  The "r20" kernel below is an explicit REPRODUCTION of the
 * defect (grid sized for N, n = 3N), included solely to demonstrate the bug;
 * it is never used as a correctness reference.
 *
 * Independent reference: the analytic constant-field implicit-midpoint map
 * for S = (0,1,0), B = (B,0,0), theta = gamma*B*dt is solved in float64 in
 * test_iterations() below, without calling any GPUMD kernel.
 */
#include "../../src/spin/spin.cu"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

static int g_failures = 0;

static void emit(const char* name, bool ok, const std::string& detail)
{
  printf("{\"test\":\"%s\",\"status\":\"%s\",\"detail\":%s}\n",
         name, ok ? "PASS" : "FAIL", detail.c_str());
  if (!ok) ++g_failures;
}

// ---- explicit reproduction of the r20 defect --------------------------------
static __global__ void gpu_max_abs_diff_r20(
  const int n, const float* a, const float* b, double* out)
{
  __shared__ double sh[SPIN_BLOCK_SIZE];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  double v = 0.0;
  if (i < n) v = fabs((double)a[i] - (double)b[i]);
  sh[tid] = v;
  __syncthreads();
  for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
    if (tid < off && sh[tid + off] > sh[tid]) sh[tid] = sh[tid + off];
    __syncthreads();
  }
  if (tid == 0 && sh[0] > 0.0)
    atomicMax(reinterpret_cast<unsigned long long*>(out),
              static_cast<unsigned long long>(__double_as_longlong(sh[0])));
}

static bool residual_case(const int N)
{
  const size_t n3 = (size_t)N * 3;
  GPU_Vector<float> a(n3), b(n3);
  GPU_Vector<double> res(2);
  GPU_Vector<double> res20(1);
  GPU_Vector<int> bad(1);
  std::vector<float> ha(n3, 0.0f), hb(n3, 0.0f);
  const float delta = 0.0125f;
  // x is IDENTICAL; only y and z change -> the r20 x-only check sees zero.
  for (int i = 0; i < N; ++i) {
    ha[i] = 0.25f;            hb[i] = 0.25f;                 // x: unchanged
    ha[i + N] = 0.5f;         hb[i + N] = 0.5f + delta;      // y: changing
    ha[i + 2 * N] = -0.75f;   hb[i + 2 * N] = -0.75f + delta;// z: changing
  }
  a.copy_from_host(ha.data());
  b.copy_from_host(hb.data());
  res.fill(0.0);
  res20.fill(0.0);
  bad.fill(0);
  const int grid = (N - 1) / SPIN_BLOCK_SIZE + 1;
  gpu_zero_midpoint_state<<<1, 1>>>(res.data(), bad.data());
  gpu_midpoint_residual<<<grid, SPIN_BLOCK_SIZE>>>(
    N, a.data(), b.data(), res.data(), bad.data());
  gpu_max_abs_diff_r20<<<grid, SPIN_BLOCK_SIZE>>>((int)n3, a.data(), b.data(), res20.data());
  CHECK(gpuDeviceSynchronize());
  double r[2] = {0, 0}, r20 = 0;
  int nb = 0;
  res.copy_to_host(r);
  res20.copy_to_host(&r20);
  bad.copy_to_host(&nb);
  const double rel = r[0] / fmax(r[1], 1e-30);
  // Independent analytic reference for the r20 DEFECT REPRODUCTION, which is
  // NOT a correctness reference: r20 launches ceil(N/128)*128 threads but
  // hands them n = 3N, so it only ever scans the first
  //   covered = min(3N, ceil(N/128)*128)
  // entries of the component-major array.  x is identical (indices [0,N)),
  // y and z differ by delta (indices [N,3N)).  Therefore r20 sees a nonzero
  // difference iff covered > N.  For N that is an exact multiple of 128 the
  // covered range is exactly the x block and r20 falsely reports ZERO -- that
  // is the r20 bug.  For small or non-aligned N, some y (or z) entries leak
  // into the covered range and r20 happens to see delta.  Assuming r20 == 0
  // for every N was an error in the FIRST version of this test; the analytic
  // coverage model below is computed independently on the host.
  const int covered = (int)fmin((double)n3, (double)grid * SPIN_BLOCK_SIZE);
  const double ref_delta = (double)(hb[N] - ha[N]); // float32-rounded delta
  const double r20_expected = (covered > N) ? ref_delta : 0.0;
  const bool ok = fabs(r[0] - ref_delta) < 1e-6 &&
                  fabs(r20 - r20_expected) < 1e-12 && r[1] >= 0.75 &&
                  nb == 0 && rel > 0;
  char buf[640];
  snprintf(buf, sizeof buf,
           "{\"N\":%d,\"r21_abs\":%.9g,\"r21_rel\":%.9g,\"r21_scale\":%.9g,"
           "\"r21_bad\":%d,\"r20_abs\":%.9g,\"r20_expected\":%.9g,"
           "\"r20_covered\":%d,\"r20_total\":%zu,\"expected\":%.9g,"
           "\"r20_false_convergence\":%s}",
           N, r[0], rel, r[1], nb, r20, r20_expected, covered, n3,
           ref_delta, (covered <= N) ? "true" : "false");
  char name[128];
  snprintf(name, sizeof name, "residual_all_components_N%d", N);
  emit(name, ok, std::string(buf));
  return ok;
}

static void nonfinite_case(const char* tag, const char* which)
{
  const int N = 128;
  const size_t n3 = (size_t)N * 3;
  GPU_Vector<float> a(n3), b(n3);
  GPU_Vector<double> res(2), res20(1);
  GPU_Vector<int> bad(1);
  std::vector<float> ha(n3, 0.5f), hb(n3, 0.5f);
  // Put the anomaly in the z block -- a component r20 NEVER inspected.
  const size_t z0 = 2 * (size_t)N;
  double bad_value = 0.0;
  if (std::strcmp(which, "nan") == 0) bad_value = std::nan("");
  else if (std::strcmp(which, "inf") == 0)
    bad_value = std::numeric_limits<float>::infinity();
  else bad_value = 1.0e30; // finite in float, but an unphysical overflow
  hb[z0 + 7] = (float)bad_value;
  a.copy_from_host(ha.data());
  b.copy_from_host(hb.data());
  res.fill(0.0);
  res20.fill(0.0);
  bad.fill(0);
  const int grid = (N - 1) / SPIN_BLOCK_SIZE + 1;
  gpu_zero_midpoint_state<<<1, 1>>>(res.data(), bad.data());
  gpu_midpoint_residual<<<grid, SPIN_BLOCK_SIZE>>>(
    N, a.data(), b.data(), res.data(), bad.data());
  gpu_max_abs_diff_r20<<<grid, SPIN_BLOCK_SIZE>>>((int)n3, a.data(), b.data(), res20.data());
  CHECK(gpuDeviceSynchronize());
  double r[2] = {0, 0}, r20 = 0;
  int nb = 0;
  res.copy_to_host(r);
  res20.copy_to_host(&r20);
  bad.copy_to_host(&nb);
  // r21 must FLAG the anomaly; the r20 max-reduction must (mis)report zero.
  const bool ok = nb > 0 && r20 == 0.0;
  char buf[400];
  snprintf(buf, sizeof buf,
           "{\"kind\":\"%s\",\"r21_bad\":%d,\"r21_abs\":%.9g,\"r20_abs\":%.9g}",
           which, nb, r[0], r20);
  emit(tag, ok, std::string(buf));
}

// ---- analytic x-unchanged / y,z-changing counterexample -------------------
// Exact implicit midpoint for one spin in a constant field B=(B,0,0),
// S_old=(0,s,0), theta = gamma*B*dt.  The fixed point is
//   S_new = S_old - theta * (S_mid x B_hat),  S_mid = (S_old+S_new)/2,
// solved in closed form below (float64, independent of the CUDA kernels).
static void analytic_iterations()
{
  const int N = 128;
  const double s = 1.0;
  const double theta = 0.1; // gamma*B*dt = 0.1
  // With B along x, S_mid x B_hat = (0, m_z, -m_y).
  // S_new_y = s - theta*m_z ;  S_new_z = 0 + theta*m_y
  // m_y = (s + S_new_y)/2   ;  m_z = S_new_z/2
  // => S_new_y = s - theta*S_new_z/2 ; S_new_z = theta*(s+S_new_y)/2
  // Solve 2x2 exactly:
  const double yn = s * (1.0 - theta * theta / 4.0) / (1.0 + theta * theta / 4.0);
  const double zn = s * theta / (1.0 + theta * theta / 4.0);
  const double xres = 0.0;
  const double fullres = fmax(fabs(yn - s), fabs(zn - 0.0));
  // fixed-point iteration from S_new^0 = S_old, with float32 storage
  float y = (float)s, z = 0.0f;
  int iterations = 0;
  double residual = 1e9;
  std::vector<double> history;
  for (int it = 1; it <= 50; ++it) {
    iterations = it;
    const float py = y, pz = z;
    const double my = (s + py) / 2.0, mz = pz / 2.0;
    y = (float)(s - theta * mz);
    z = (float)(0.0 + theta * my);
    residual = fmax(fabs((double)y - py), fabs((double)z - pz));
    history.push_back(residual);
    if (residual <= 1e-6) break;
  }
  const double norm = sqrt((double)y * y + (double)z * z);
  // The x component is identically zero at every iteration; a residual that
  // only inspects x would declare convergence after ONE iteration.
  const bool ok = xres == 0.0 && fullres > 0.09 && iterations >= 2 &&
                  residual <= 1e-6 && fabs(norm - s) < 1e-6;
  char buf[600];
  snprintf(buf, sizeof buf,
           "{\"N\":%d,\"theta\":%.3g,\"x_residual\":%.3g,"
           "\"full_residual_analytic\":%.9g,\"iterations\":%d,"
           "\"final_residual\":%.3g,\"spin_norm\":%.9g,"
           "\"norm_error\":%.3g,\"history\":[",
           N, theta, xres, fullres, iterations, residual, norm, fabs(norm - s));
  std::string s_buf(buf);
  for (size_t i = 0; i < history.size(); ++i) {
    char t[64];
    snprintf(t, sizeof t, "%s%.3g", i ? "," : "", history[i]);
    s_buf += t;
  }
  s_buf += "]}";
  emit("midpoint_x_unchanged_yz_changing_N128", ok, s_buf);
}

// ---- anchor: error recorded BEFORE projection, anchor never refreshed -----
static void anchor_case()
{
  const int N = 43;
  GPU_Vector<float> sx(N), sy(N), sz(N), anchor(N);
  GPU_Vector<double> max_err(1);
  GPU_Vector<int> bad(1);
  std::vector<float> hx(N, 0.0f), hy(N, 1.0f), hz(N, 0.0f), ha(N, 1.0f);
  // One atom's raw map moved |S| to 1.5 (a 50% error).
  hy[17] = 1.5f;
  sx.copy_from_host(hx.data());
  sy.copy_from_host(hy.data());
  sz.copy_from_host(hz.data());
  anchor.copy_from_host(ha.data());
  max_err.fill(0.0);
  bad.fill(0);
  const int grid = (N - 1) / SPIN_BLOCK_SIZE + 1;
  gpu_project_anchor<<<grid, SPIN_BLOCK_SIZE>>>(
    N, sx.data(), sy.data(), sz.data(), anchor.data(),
    1.0e-4, max_err.data(), bad.data());
  CHECK(gpuDeviceSynchronize());
  double err = 0.0;
  int nb = 0;
  max_err.copy_to_host(&err);
  bad.copy_to_host(&nb);
  std::vector<float> ox(N), oy(N), oz(N), oa(N);
  sx.copy_to_host(ox.data());
  sy.copy_to_host(oy.data());
  sz.copy_to_host(oz.data());
  anchor.copy_to_host(oa.data());
  const double projected_norm = sqrt((double)ox[17] * ox[17] +
                                     (double)oy[17] * oy[17] +
                                     (double)oz[17] * oz[17]);
  const bool ok = fabs(err - 0.5) < 1e-6 && nb > 0 &&
                  fabs(oa[17] - 1.0f) < 1e-6 &&
                  fabs(projected_norm - 1.0) < 1e-6;
  char buf[400];
  snprintf(buf, sizeof buf,
           "{\"N\":%d,\"pre_projection_error\":%.9g,\"violations\":%d,"
           "\"anchor_after\":%.9g,\"projected_norm\":%.9g}",
           N, err, nb, (double)oa[17], projected_norm);
  emit("norm_anchor_error_before_projection", ok, std::string(buf));
}

int main()
{
  printf("# r21 midpoint/anchor GPU unit test (actual shipped kernels)\n");
  for (int N : {16, 43, 128, 129, 432, 1024}) residual_case(N);
  nonfinite_case("nonfinite_nan_flagged", "nan");
  nonfinite_case("nonfinite_inf_flagged", "inf");
  nonfinite_case("overflow_flagged", "overflow");
  analytic_iterations();
  anchor_case();
  printf("# failures = %d\n", g_failures);
  return g_failures == 0 ? 0 : 1;
}
