// CPU tests for the generated metric headers (native/generated/*.cuh) and the
// shared geodesic code (src/rm/geodesic.h) that the NVRTC tracer runs. They
// need no GPU and no CUDA toolkit, so they run in CI.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "kerr.cuh"
#include "minkowski.cuh"
#include "reissner-nordstrom.cuh"
#include "schwarzschild.cuh"
#include "rm/analytic.h"
#include "rm/geodesic.h"
#include "rm/json.h"
#include "rm/tangent.h"
#include "rm/emission.h"

#ifndef RM_GENERATED_DIR
#error "RM_GENERATED_DIR must point at native/generated"
#endif

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond, ...)                                         \
  do {                                                           \
    ++g_checks;                                                  \
    if (!(cond)) {                                               \
      std::printf("  [FAIL] %s:%d: ", __FILE__, __LINE__);       \
      std::printf(__VA_ARGS__);                                  \
      std::printf("\n");                                         \
      ++g_failures;                                              \
    }                                                            \
  } while (0)

static std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::printf("  [FAIL] cannot open %s\n", path.c_str());
    ++g_failures;
    return "{}";
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// |a - b| <= tol * max(1, |b|)
static bool agrees(double a, double b, double tol) { return std::fabs(a - b) <= tol * std::fmax(1.0, std::fabs(b)); }

// --- Generated C++ vs the JS backend -----------------------------------------

template <class M>
static double max_metric_error(const rm::json::Value& cases, double tol) {
  double worst = 0;
  for (const rm::json::Value& c : cases.a) {
    std::vector<double> params = c["params"].numbers();
    std::vector<double> x = c["x"].numbers();
    double g[16], dg[64];
    M::eval(x.data(), params.data(), g, dg);
    const std::vector<double> eg = c["g"].numbers(), edg = c["dg"].numbers(), disk = c["disk"].numbers();
    auto check = [&](double got, double want, const char* what, int k) {
      double err = std::fabs(got - want) / std::fmax(1.0, std::fabs(want));
      if (err > worst) worst = err;
      CHECK(agrees(got, want, tol), "%s[%d] = %.17g, JS backend %.17g", what, k, got, want);
    };
    for (int k = 0; k < 16; k++) check(g[k], eg[k], "g", k);
    for (int k = 0; k < 64; k++) check(dg[k], edg[k], "dg", k);
    check(M::horizon_value(x.data(), params.data()), c["horizon"].num(), "horizon_value", 0);
    check(M::disk_radius(x.data(), params.data()), disk[0], "disk_radius", 0);
    check(M::disk_omega(x.data(), params.data()), disk[1], "disk_omega", 0);
    check(M::disk_inner(x.data(), params.data()), disk[2], "disk_inner", 0);
    check(M::disk_outer(x.data(), params.data()), disk[3], "disk_outer", 0);
    check(M::length_scale(x.data(), params.data()), c["scale"].num(), "length_scale", 0);
  }
  return worst;
}

static void test_generated_matches_js() {
  const rm::json::Value fixture = rm::json::parse(read_file(std::string(RM_GENERATED_DIR) + "/fixtures/metric-agreement.json"));
  const double tol = 1e-12;
  int metrics = 0;
  for (const rm::json::Value& m : fixture["metrics"].a) {
    const std::string& id = m["id"].str();
    const rm::json::Value& cases = m["cases"];
    double worst = -1;
    if (id == "schwarzschild") worst = max_metric_error<rm::schwarzschild::Metric>(cases, tol);
    else if (id == "kerr") worst = max_metric_error<rm::kerr::Metric>(cases, tol);
    else if (id == "minkowski") worst = max_metric_error<rm::minkowski::Metric>(cases, tol);
    else if (id == "reissner-nordstrom") worst = max_metric_error<rm::reissner_nordstrom::Metric>(cases, tol);
    CHECK(worst >= 0, "fixture has metric '%s' with no compiled-in header; add it to test_main.cpp", id.c_str());
    if (worst >= 0) std::printf("    %-18s %zu points, max relative difference %.2e (tolerance %.0e)\n", id.c_str(), cases.size(), worst, tol);
    metrics++;
  }
  CHECK(metrics == 4, "expected 4 metrics in the fixture, found %d", metrics);
}

// --- Generated C++ JVPs vs the JS backend (Phase 4, step 1) --------------------

/** Worst relative difference of metric_jvp and disk_jvp (P = 2) against the JS values. */
template <class M>
static double max_jvp_error(const rm::json::Value& cases, double tol) {
  constexpr int P = 2;
  double worst = 0;
  for (const rm::json::Value& c : cases.a) {
    const std::vector<double> params = c["params"].numbers(), x = c["x"].numbers();
    const std::vector<double> dx = c["dx"].numbers(), dp = c["dp"].numbers();
    const std::vector<double> etg = c["tg"].numbers(), etdg = c["tdg"].numbers();
    const std::vector<double> edisk = c["disk"].numbers(), etdisk = c["tdisk"].numbers();
    double g[16], dg[64], tg[16 * P], tdg[64 * P], out[4], tout[4 * P];
    // A metric with no parameters still gets a valid pointer.
    const double* p = params.empty() ? nullptr : params.data();
    const double* pdp = dp.empty() ? nullptr : dp.data();
    M::template jvp<double, P>(x.data(), p, dx.data(), pdp, g, dg, tg, tdg);
    M::template disk_jvp<double, P>(x.data(), p, dx.data(), pdp, out, tout);
    auto check = [&](const double* got, const std::vector<double>& want, const char* what) {
      double scale = 1;
      for (double w : want) scale = std::fmax(scale, std::fabs(w));
      for (size_t k = 0; k < want.size(); k++) {
        const double err = std::fabs(got[k] - want[k]) / scale;
        if (err > worst) worst = err;
        CHECK(err <= tol, "%s[%zu] = %.17g, JS backend %.17g", what, k, got[k], want[k]);
      }
    };
    check(tg, etg, "tg");
    check(tdg, etdg, "tdg");
    check(out, edisk, "disk");
    check(tout, etdisk, "tdisk");
  }
  return worst;
}

static void test_generated_jvp_matches_js() {
  const rm::json::Value fixture = rm::json::parse(read_file(std::string(RM_GENERATED_DIR) + "/fixtures/jvp-agreement.json"));
  const double tol = 1e-12;
  int metrics = 0;
  for (const rm::json::Value& m : fixture["metrics"].a) {
    const std::string& id = m["id"].str();
    const rm::json::Value& cases = m["cases"];
    double worst = -1;
    if (id == "schwarzschild") worst = max_jvp_error<rm::schwarzschild::Metric>(cases, tol);
    else if (id == "kerr") worst = max_jvp_error<rm::kerr::Metric>(cases, tol);
    else if (id == "minkowski") worst = max_jvp_error<rm::minkowski::Metric>(cases, tol);
    else if (id == "reissner-nordstrom") worst = max_jvp_error<rm::reissner_nordstrom::Metric>(cases, tol);
    CHECK(worst >= 0, "JVP fixture has metric '%s' with no compiled-in header", id.c_str());
    if (worst >= 0) std::printf("    %-18s %zu points x 2 directions, max relative difference %.2e (tolerance %.0e)\n", id.c_str(), cases.size(), worst, tol);
    metrics++;
  }
  CHECK(metrics == 4, "expected 4 metrics in the JVP fixture, found %d", metrics);
}

// --- Tangent rays (Phase 4, step 3) -------------------------------------------

/** Identity directions over theta: direction d moves theta[d]. */
template <int P>
static void identity_dirs(double dirs[P][rm::MAX_THETA]) {
  for (int d = 0; d < P; d++)
    for (int k = 0; k < rm::MAX_THETA; k++) dirs[d][k] = d == k ? 1 : 0;
}

/** max over directions of |a - b| / max(|b| over that direction's 8 components, 1e-300). */
template <int P>
static double tangent_difference(const double* a, const double* b) {
  double worst = 0;
  for (int d = 0; d < P; d++) {
    double num = 0, den = 1e-300;
    for (int i = 0; i < 8; i++) {
      num = std::fmax(num, std::fabs(a[8 * d + i] - b[8 * d + i]));
      den = std::fmax(den, std::fabs(b[8 * d + i]));
    }
    worst = std::fmax(worst, num / den);
  }
  return worst;
}

static rm::TraceOptions<double> fixture_options(const rm::json::Value& o) {
  rm::TraceOptions<double> opt{};
  opt.max_steps = int(o["maxSteps"].num());
  opt.step_fraction = o["stepFraction"].num();
  opt.min_step = o["minStep"].num();
  opt.escape_radius = o["escapeRadius"].num();
  opt.disk = o["disk"].b;
  opt.integrator = rm::INTEGRATOR_RK4;
  return opt;
}

static const char* fate_name(int f) {
  return f == rm::FATE_ESCAPED ? "escaped" : f == rm::FATE_CAPTURED ? "captured" : f == rm::FATE_DISK ? "disk" : "unfinished";
}

struct TangentStats {
  double js = 0, dual = 0;
  int rays = 0;
};

/** The JS fixture rays through the C++ symbolic tangent integrator and through duals. */
template <class M, int P>
static TangentStats tangent_rays_vs_js(const rm::json::Value& m) {
  TangentStats st;
  const rm::TraceOptions<double> opt = fixture_options(m["options"]);
  double dirs[P][rm::MAX_THETA];
  identity_dirs<P>(dirs);
  for (const rm::json::Value& ray : m["rays"].a) {
    const std::vector<double> theta = ray["theta"].numbers();
    double y0[8], S0[8 * P];
    rm::TangentResult<double, P> sym;
    bool ok = false;
    rm::tangent_ray_symbolic<M, double, P>(theta.data(), dirs, opt, y0, S0, sym, ok);
    CHECK(ok, "no static observer");
    const std::string want_fate = ray["fate"].str();
    CHECK(want_fate == fate_name(sym.fate), "fate %s, JS %s", fate_name(sym.fate), want_fate.c_str());
    CHECK(sym.steps == int(ray["steps"].num()), "steps %d, JS %d", sym.steps, int(ray["steps"].num()));
    if (want_fate == "captured") continue;  // the final state of a captured ray is not an observable
    const std::vector<double> jy = ray["y"].numbers(), jS = ray["S"].numbers(), jS0 = ray["S0"].numbers();
    double ey = 0;
    for (int i = 0; i < 8; i++) ey = std::fmax(ey, std::fabs(sym.y[i] - jy[i]) / std::fmax(1.0, std::fabs(jy[i])));
    st.js = std::fmax(st.js, std::fmax(ey, std::fmax(tangent_difference<P>(S0, jS0.data()), tangent_difference<P>(sym.S, jS.data()))));

    const rm::TraceResult<rm::Dual<double, P>> dual = rm::tangent_ray_dual<M, double, P>(theta.data(), dirs, opt, ok);
    CHECK(dual.fate == sym.fate && dual.steps == sym.steps, "dual fate/steps %d/%d vs symbolic %d/%d", dual.fate, dual.steps,
          sym.fate, sym.steps);
    double dS[8 * P];
    for (int i = 0; i < 8; i++)
      for (int d = 0; d < P; d++) dS[8 * d + i] = dual.y[i].d[d];
    st.dual = std::fmax(st.dual, tangent_difference<P>(dS, sym.S));
    st.rays++;
  }
  return st;
}

static void test_tangent_rays() {
  const rm::json::Value fixture = rm::json::parse(read_file(std::string(RM_GENERATED_DIR) + "/fixtures/tangent-rays.json"));
  for (const rm::json::Value& m : fixture["metrics"].a) {
    const std::string& id = m["id"].str();
    TangentStats st;
    if (id == "schwarzschild") st = tangent_rays_vs_js<rm::schwarzschild::Metric, 6>(m);
    else if (id == "kerr") st = tangent_rays_vs_js<rm::kerr::Metric, 7>(m);
    else if (id == "minkowski") st = tangent_rays_vs_js<rm::minkowski::Metric, 6>(m);
    else if (id == "reissner-nordstrom") st = tangent_rays_vs_js<rm::reissner_nordstrom::Metric, 7>(m);
    else CHECK(false, "tangent fixture has metric '%s' with no compiled-in header", id.c_str());
    std::printf("    %-18s %d rays x all inputs: C++ symbolic vs JS %.2e (tolerance 1e-9), dual vs symbolic %.2e (tolerance 1e-10)\n",
                id.c_str(), st.rays, st.js, st.dual);
    CHECK(st.rays >= 4, "%s: only %d comparable rays", id.c_str(), st.rays);
    CHECK(st.js < 1e-9, "%s: C++ symbolic vs JS %g", id.c_str(), st.js);
    CHECK(st.dual < 1e-10, "%s: dual vs symbolic %g", id.c_str(), st.dual);
  }
}

/** Duals through the adaptive RK45 integrator against central differences of the same integrator. */
static void test_dual_rk45_matches_fd() {
  using M = rm::kerr::Metric;
  constexpr int P = 3;  // a, inclination, u
  const double theta0[7] = {1, 0.9, 50, 1.2, 0.3, 0.21, 0.07};
  double dirs[P][rm::MAX_THETA] = {};
  dirs[0][1] = 1;
  dirs[1][2 + rm::THETA_INCLINATION] = 1;
  dirs[2][2 + rm::THETA_U] = 1;
  rm::TraceOptions<double> opt{};
  opt.max_steps = 100000;
  opt.step_fraction = 0.02;
  opt.min_step = 1e-6;
  opt.escape_radius = 50.5;
  opt.disk = true;
  opt.integrator = rm::INTEGRATOR_RK45;
  opt.rtol = 1e-9;
  opt.length_scale = 1;
  bool ok = false;
  const rm::TraceResult<rm::Dual<double, P>> dual = rm::tangent_ray_dual<M, double, P>(theta0, dirs, opt, ok);
  double worst = 0;
  int compared = 0;
  for (int d = 0; d < P; d++) {
    int k = 0;
    while (dirs[d][k] == 0) k++;
    const double h = 1e-6;
    double t[7];
    rm::TraceResult<rm::Dual<double, 1>> r[4];
    const double offs[4] = {2, 1, -1, -2};
    double zero_dir[1][rm::MAX_THETA] = {};
    bool same = true;
    for (int s = 0; s < 4; s++) {
      for (int j = 0; j < 7; j++) t[j] = theta0[j];
      t[k] += offs[s] * h;
      r[s] = rm::tangent_ray_dual<M, double, 1>(t, zero_dir, opt, ok);
      same = same && r[s].fate == dual.fate && r[s].steps == dual.steps && r[s].rejected == dual.rejected;
    }
    if (!same) continue;
    double num = 0, den = 1e-300;
    for (int i = 0; i < 8; i++) {
      const double fd = (-r[0].y[i].v + 8 * r[1].y[i].v - 8 * r[2].y[i].v + r[3].y[i].v) / (12 * h);
      num = std::fmax(num, std::fabs(dual.y[i].d[d] - fd));
      den = std::fmax(den, std::fabs(fd));
    }
    worst = std::fmax(worst, num / den);
    compared++;
  }
  std::printf("    Kerr a = 0.9 RK45 rtol 1e-9, fate %s, %d steps (%d rejected): duals vs 4-point FD over %d inputs, worst %.2e (tolerance 1e-6)\n",
              fate_name(dual.fate), dual.steps, dual.rejected, compared, worst);
  CHECK(compared == P, "accept/reject sequence changed under the FD step for %d inputs", P - compared);
  CHECK(worst < 1e-6, "RK45 duals vs FD %g", worst);
}

/** Disk intensities and d/d(a, i) at model-sky samples: C++ duals (rm/emission.h) vs the JS image model. */
static void test_image_samples() {
  using M = rm::kerr::Metric;
  constexpr int P = 2;
  const rm::json::Value fx = rm::json::parse(read_file(std::string(RM_GENERATED_DIR) + "/fixtures/image-samples.json"));
  const rm::json::Value& o = fx["options"];
  rm::TraceOptions<double> opt{};
  opt.max_steps = int(o["maxSteps"].num());
  opt.step_fraction = o["stepFraction"].num();
  opt.min_step = o["minStep"].num();
  opt.escape_radius = o["escapeRadius"].num();
  opt.disk = true;
  opt.integrator = rm::INTEGRATOR_RK4;
  const double taper = fx["taperStart"].num();
  double worst_I = 0, worst_dI = 0;
  int disk = 0, mismatched = 0;
  for (const rm::json::Value& c : fx["cases"].a) {
    opt.crossing = c["crossing"].str() == "hermite" ? rm::CROSSING_HERMITE : rm::CROSSING_LINEAR;
    std::vector<double> theta = c["theta"].numbers();
    const std::vector<double> xi = c["xi"].numbers(), I = c["I"].numbers(), dI = c["dI"].numbers();
    double dirs[P][rm::MAX_THETA] = {};
    for (int d = 0; d < P; d++) {
      const std::vector<double> row = c["dirs"][d].numbers();
      for (size_t k = 0; k < row.size(); k++) dirs[d][k] = row[k];
    }
    const size_t n = I.size();
    double scale_I = 0, scale_dI[P] = {};
    for (size_t s = 0; s < n; s++) {
      scale_I = std::fmax(scale_I, std::fabs(I[s]));
      for (int d = 0; d < P; d++) scale_dI[d] = std::fmax(scale_dI[d], std::fabs(dI[P * s + d]));
    }
    for (size_t s = 0; s < n; s++) {
      theta[2 + rm::THETA_U] = xi[2 * s] / theta[2 + rm::THETA_RADIUS];
      theta[2 + rm::THETA_V] = xi[2 * s + 1] / theta[2 + rm::THETA_RADIUS];
      typedef rm::Dual<double, P> D;
      D dp[2], dy0[8];
      rm::dual_camera_ray<M, double, P>(theta.data(), dirs, dp, dy0);
      const rm::TraceResult<D> r = rm::trace_ray<M>(dp, dy0, rm::dual_options<double, P>(opt));
      const std::string want = c["fate"][s].str();
      if (want != fate_name(r.fate) || int(c["steps"][s].num()) != r.steps || int(c["crossings"][s].num()) != r.crossings) mismatched++;
      if (r.fate != rm::FATE_DISK) continue;
      disk++;
      const D val = rm::thin_disk_intensity<M, D>(dp, r.y, D(taper));
      worst_I = std::fmax(worst_I, std::fabs(val.v - I[s]) / scale_I);
      for (int d = 0; d < P; d++) worst_dI = std::fmax(worst_dI, std::fabs(val.d[d] - dI[P * s + d]) / scale_dI[d]);
    }
  }
  std::printf("    Kerr a = 0.6, 0.94 (linear), 0.94 (Hermite): %d disk samples; C++ dual vs JS: I %.2e, dI/d(a, i) %.2e (relative to the largest per case; tolerance 1e-10); %d fate/step mismatches\n",
              disk, worst_I, worst_dI, mismatched);
  CHECK(mismatched == 0, "%d samples with a different fate or step count", mismatched);
  CHECK(disk >= 50, "only %d disk samples", disk);
  CHECK(worst_I < 1e-10 && worst_dI < 1e-10, "I %g, dI %g", worst_I, worst_dI);
}

// --- Geodesics with the generated metrics -------------------------------------

using rm::coordinate_ray;

template <class M>
static void final_direction(const double* params, const double y[8], double d[3]) {
  double dy[8];
  rm::geodesic_rhs<M>(params, y, dy);
  const double len = std::sqrt(dy[1] * dy[1] + dy[2] * dy[2] + dy[3] * dy[3]);
  for (int i = 0; i < 3; i++) d[i] = dy[1 + i] / len;
}

static rm::TraceOptions<double> options(double step_fraction, double escape_radius) {
  return {1000000, step_fraction, 1e-4, escape_radius, false};
}

static void test_flat_space_is_straight() {
  using M = rm::minkowski::Metric;
  const double params[1] = {1};
  const double start[3] = {100, 3, -2}, dir[3] = {-1, 0, 0};
  double y[8];
  coordinate_ray<M>(params, start, dir, y);
  auto res = rm::trace_ray<M>(params, y, options(0.01, 300));
  double d[3];
  final_direction<M>(params, res.y, d);
  CHECK(res.fate == rm::FATE_ESCAPED, "fate %d", res.fate);
  CHECK(std::fabs(d[0] + 1) < 1e-12 && std::fabs(res.y[2] - 3) < 1e-9 && std::fabs(res.y[3] + 2) < 1e-9,
        "ray bent in flat space: d = (%g, %g, %g)", d[0], d[1], d[2]);
}

static void test_capture_threshold() {
  using M = rm::schwarzschild::Metric;
  const double params[1] = {1};
  const double b_crit = std::sqrt(27.0);
  const double dir[3] = {-1, 0, 0};
  double y[8];
  const double inside[3] = {1000, 0.99 * b_crit, 0}, outside[3] = {1000, 1.01 * b_crit, 0};
  coordinate_ray<M>(params, inside, dir, y);
  auto in = rm::trace_ray<M>(params, y, options(0.002, 2000));
  coordinate_ray<M>(params, outside, dir, y);
  auto out = rm::trace_ray<M>(params, y, options(0.002, 2000));
  CHECK(in.fate == rm::FATE_CAPTURED, "b = 0.99 sqrt(27) M: fate %d, expected captured", in.fate);
  CHECK(out.fate == rm::FATE_ESCAPED, "b = 1.01 sqrt(27) M: fate %d, expected escaped", out.fate);
}

static void test_weak_field_deflection() {
  using M = rm::schwarzschild::Metric;
  const double params[1] = {1};
  const double b = 1000;
  const double start[3] = {1e7, b, 0}, dir[3] = {-1, 0, 0};
  double y[8];
  coordinate_ray<M>(params, start, dir, y);
  auto res = rm::trace_ray<M>(params, y, options(0.002, 2e7));
  double d[3];
  final_direction<M>(params, res.y, d);
  const double deflection = std::atan2(-d[1], -d[0]);
  const double pi = 3.14159265358979323846;
  const double expected = 4 / b + 15 * pi / (4 * b * b);
  const double rel = std::fabs(deflection - expected) / expected;
  std::printf("    deflection at b = 1000 M: %.9e rad, expected %.9e (relative error %.1e)\n", deflection, expected, rel);
  CHECK(res.fate == rm::FATE_ESCAPED, "fate %d", res.fate);
  CHECK(rel < 1e-3, "relative error %g", rel);
}

static void test_null_constraint() {
  using M = rm::kerr::Metric;
  const double params[2] = {1, 0.9};
  // An escaping ray that passes within ~6 M of the hole. (A captured ray drifts
  // more as it piles up at the horizon: 1.7e-8 for b = 6, identical to the JS
  // integrator to 10 digits.)
  const double start[3] = {1000, 8, 2}, dir[3] = {-1, 0, 0};
  double y[8], dy[8];
  coordinate_ray<M>(params, start, dir, y);
  const double pt0 = std::fabs(y[4]);
  auto res = rm::trace_ray<M>(params, y, options(0.002, 2000));
  const double H = rm::geodesic_rhs<M>(params, res.y, dy);
  CHECK(res.fate == rm::FATE_ESCAPED, "fate %d", res.fate);
  CHECK(std::fabs(H) / (pt0 * pt0) < 1e-8, "Kerr a = 0.9: |H| / p_t^2 = %g after %d steps", std::fabs(H) / (pt0 * pt0), res.steps);
}

/** The camera frame is orthonormal: g^{-1}(f_a, f_b) = diag(-1, 1, 1, 1) for the covariant frame. */
template <class M>
static void check_camera_frame(const char* name, const double* params) {
  rm::CameraFrame cam = rm::static_camera<M>(params, 26, 0.7, 0.1);
  CHECK(cam.ok, "%s: no static observer at r = 26", name);
  double g[16], dg[64], gi[16];
  M::eval(cam.pos, params, g, dg);
  rm::invert4(g, gi);
  double worst = 0;
  for (int a = 0; a < 4; a++)
    for (int b = 0; b < 4; b++) {
      double want = a == b ? (a == 0 ? -1 : 1) : 0;
      worst = std::fmax(worst, std::fabs(rm::dot4(gi, cam.frame[a], cam.frame[b]) - want));
    }
  CHECK(worst < 1e-12, "%s: camera frame not orthonormal (max error %g)", name, worst);
  // Forward points at the origin: its spatial contravariant part is along -pos.
  double fwd[4] = {0, 0, 0, 0};
  for (int mu = 0; mu < 4; mu++)
    for (int nu = 0; nu < 4; nu++) fwd[mu] += gi[4 * mu + nu] * cam.frame[3][nu];
  const double r = std::sqrt(cam.pos[1] * cam.pos[1] + cam.pos[2] * cam.pos[2] + cam.pos[3] * cam.pos[3]);
  const double fl = std::sqrt(fwd[1] * fwd[1] + fwd[2] * fwd[2] + fwd[3] * fwd[3]);
  double cosang = -(fwd[1] * cam.pos[1] + fwd[2] * cam.pos[2] + fwd[3] * cam.pos[3]) / (r * fl);
  CHECK(cosang > 1 - 1e-12, "%s: camera does not look at the origin (cos = %.15f)", name, cosang);
}

/** The generated fused right-hand side (symmetry, no zero derivatives) agrees with the generic one. */
template <class M>
static double fused_rhs_difference(const rm::json::Value& cases) {
  double worst = 0;
  for (const rm::json::Value& c : cases.a) {
    std::vector<double> params = c["params"].numbers(), x = c["x"].numbers();
    // A null momentum at x: p = g k with k = (1, unit spatial direction) adjusted to H = 0.
    const double dir[3] = {0.48, -0.6, 0.64};
    double y[8], fused[8], generic[8];
    coordinate_ray<M>(params.data(), &x[1], dir, y);
    y[0] = x[0];
    const double h_fused = M::rhs(y, params.data(), fused);
    const double h_generic = rm::geodesic_rhs_generic<M>(params.data(), y, generic);
    double scale = 0;
    for (int i = 0; i < 8; i++) scale = std::fmax(scale, std::fabs(generic[i]));
    for (int i = 0; i < 8; i++) worst = std::fmax(worst, std::fabs(fused[i] - generic[i]) / scale);
    worst = std::fmax(worst, std::fabs(h_fused - h_generic) / scale);
  }
  return worst;
}

static void test_fused_rhs_matches_generic() {
  const rm::json::Value fixture = rm::json::parse(read_file(std::string(RM_GENERATED_DIR) + "/fixtures/metric-agreement.json"));
  for (const rm::json::Value& m : fixture["metrics"].a) {
    const std::string& id = m["id"].str();
    double worst = -1;
    if (id == "schwarzschild") worst = fused_rhs_difference<rm::schwarzschild::Metric>(m["cases"]);
    else if (id == "kerr") worst = fused_rhs_difference<rm::kerr::Metric>(m["cases"]);
    else if (id == "minkowski") worst = fused_rhs_difference<rm::minkowski::Metric>(m["cases"]);
    else if (id == "reissner-nordstrom") worst = fused_rhs_difference<rm::reissner_nordstrom::Metric>(m["cases"]);
    std::printf("    %-18s fused vs generic right-hand side: max difference %.2e (relative to the largest component)\n", id.c_str(), worst);
    CHECK(worst >= 0 && worst < 1e-13, "%s: %g", id.c_str(), worst);
  }
}

// --- Integrators against the exact Schwarzschild orbit ------------------------

static void test_exact_deflection_matches_series() {
  // Weak-field series to third order: 4M/b + 15 pi M^2 / 4b^2 + 128 M^3 / 3b^3.
  const double b = 1000, pi = rm::analytic::PI;
  const double series = 4 / b + 15 * pi / (4 * b * b) + 128 / (3 * b * b * b);
  const double exact = rm::analytic::schwarzschild_deflection(b, 1);
  CHECK(std::fabs(exact - series) / series < 1e-7, "quadrature %.15e vs series %.15e", exact, series);
}

struct Sweep {
  double traced, exact;
  int steps;
  rm::RayFate fate;
};

/** Angle swept by a Schwarzschild ray entering at x = 1000 M with offset y = b0, against the exact orbit integral. */
static Sweep schwarzschild_sweep(double b0, const rm::TraceOptions<double>& opt) {
  using M = rm::schwarzschild::Metric;
  const double params[1] = {1};
  const double start[3] = {1000, b0, 0}, dir[3] = {-1, 0, 0};
  double y0[8];
  coordinate_ray<M>(params, start, dir, y0);
  const auto res = rm::trace_ray<M>(params, y0, opt);
  const double b = std::fabs(y0[1] * y0[6] - y0[2] * y0[5]) / std::fabs(y0[4]);  // L / E
  const double r0 = std::hypot(y0[1], y0[2]), r1 = std::sqrt(res.y[1] * res.y[1] + res.y[2] * res.y[2] + res.y[3] * res.y[3]);
  const double exact = rm::analytic::schwarzschild_sweep(b, 1, r0, r1);
  // Only the final position is known, so pick the 2 pi branch nearest the exact sweep.
  const double raw = std::atan2(res.y[2], res.y[1]) - std::atan2(y0[2], y0[1]);
  double traced = std::fabs(raw);
  for (int k = -3; k <= 3; k++) {
    const double c = std::fabs(raw + 2 * rm::analytic::PI * k);
    if (std::fabs(c - exact) < std::fabs(traced - exact)) traced = c;
  }
  return {traced, exact, res.steps, res.fate};
}

static void test_rk4_converges_at_fourth_order() {
  const double fractions[] = {0.04, 0.02, 0.01, 0.005};
  double err[4];
  for (int i = 0; i < 4; i++) {
    rm::TraceOptions<double> opt = {10000000, fractions[i], 1e-4, 2000, false};
    const Sweep s = schwarzschild_sweep(10, opt);
    err[i] = std::fabs(s.traced - s.exact);
    CHECK(s.fate == rm::FATE_ESCAPED, "fate %d", s.fate);
  }
  for (int i = 0; i < 3; i++) {
    const double order = std::log2(err[i] / err[i + 1]);
    std::printf("    RK4 b = 10 M: step fraction %.3f -> %.3f, error %.2e -> %.2e, observed order %.2f\n", fractions[i],
                fractions[i + 1], err[i], err[i + 1], order);
    CHECK(order > 3.6 && order < 4.4, "observed order %.2f", order);
  }
}

static void test_rk45_meets_tolerance() {
  for (double rtol : {1e-6, 1e-8, 1e-10, 1e-12}) {
    rm::TraceOptions<double> opt = {10000000, 0.01, 1e-6, 2000, false, rm::INTEGRATOR_RK45, rtol, 1.0};
    for (double b0 : {5.4, 10.0, 100.0}) {
      const Sweep s = schwarzschild_sweep(b0, opt);
      const double rel = std::fabs(s.traced - s.exact) / s.exact;
      std::printf("    RK45 rtol %.0e, b ~ %5.1f M: sweep %.12f rad, relative error %.2e, %d steps\n", rtol, b0, s.exact, rel, s.steps);
      // Local error control does not bound the global error by rtol exactly; allow 10x.
      CHECK(rel < 10 * rtol, "rtol %g, b %g: relative error %g", rtol, b0, rel);
    }
  }
}

int main() {
  struct Test {
    const char* name;
    void (*fn)();
  };
  const Test tests[] = {
      {"generated_cpp_matches_js_backend", test_generated_matches_js},
      {"generated_jvp_matches_js_backend", test_generated_jvp_matches_js},
      {"tangent_rays_symbolic_vs_js_vs_dual", test_tangent_rays},
      {"dual_rk45_matches_finite_differences", test_dual_rk45_matches_fd},
      {"image_samples_dual_vs_js", test_image_samples},
      {"flat_space_is_straight", test_flat_space_is_straight},
      {"capture_threshold", test_capture_threshold},
      {"weak_field_deflection", test_weak_field_deflection},
      {"null_constraint_kerr", test_null_constraint},
      {"fused_rhs_matches_generic", test_fused_rhs_matches_generic},
      {"exact_deflection_matches_series", test_exact_deflection_matches_series},
      {"rk4_converges_at_fourth_order", test_rk4_converges_at_fourth_order},
      {"rk45_meets_tolerance", test_rk45_meets_tolerance},
      {"camera_frame_orthonormal", [] {
         const double s[1] = {1}, k[2] = {1, 0.9}, m[1] = {1};
         check_camera_frame<rm::schwarzschild::Metric>("schwarzschild", s);
         check_camera_frame<rm::kerr::Metric>("kerr", k);
         check_camera_frame<rm::minkowski::Metric>("minkowski", m);
       }},
  };
  for (const Test& t : tests) {
    const int before = g_failures;
    t.fn();
    std::printf("[%s] %s\n", g_failures == before ? "PASS" : "FAIL", t.name);
  }
  std::printf("%d checks, %d failed\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
