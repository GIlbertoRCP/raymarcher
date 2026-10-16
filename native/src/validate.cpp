// fp64 validation suite. Every check traces rays on the GPU with the generated
// metric (double-precision NVRTC modules) and compares against a result from
// the literature or an exact quadrature, reporting a number, a tolerance and a
// pass/fail. See validate.h.

#include "validate.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "rm/analytic.h"
#include "rm/frame.h"

namespace {

namespace an = rm::analytic;
using Vec3 = std::array<double, 3>;
constexpr double PI = an::PI;
constexpr double DEG = PI / 180;

#define V_CHECK(call)                                                                                      \
  do {                                                                                                     \
    cudaError_t e_ = (call);                                                                               \
    if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e_));      \
  } while (0)

// --- Small JSON writer -------------------------------------------------------------------

std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  return buf;
}

std::string str(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    if (c == '\n') {
      out += "\\n";
      continue;
    }
    out += c;
  }
  return out + "\"";
}

std::string arr(const std::vector<double>& v) {
  std::string out = "[";
  for (size_t i = 0; i < v.size(); i++) out += (i ? ", " : "") + num(v[i]);
  return out + "]";
}

struct Metric {
  std::string name, unit;
  double value, tolerance;
  bool pass;
  std::string note;
};

std::string metric_json(const Metric& m) {
  return "{\"name\": " + str(m.name) + ", \"value\": " + num(m.value) + ", \"tolerance\": " + num(m.tolerance) +
         ", \"unit\": " + str(m.unit) + ", \"pass\": " + (m.pass ? "true" : "false") +
         (m.note.empty() ? "" : ", \"note\": " + str(m.note)) + "}";
}

/** "value <= tolerance" metric. */
Metric at_most(const std::string& name, double value, double tolerance, const std::string& unit, const std::string& note = "") {
  return {name, unit, value, tolerance, std::isfinite(value) && value <= tolerance, note};
}

struct CheckResult {
  std::string id, title, description;
  std::vector<Metric> metrics;
  std::string data;  // JSON object
  bool pass() const {
    return std::all_of(metrics.begin(), metrics.end(), [](const Metric& m) { return m.pass; });
  }
};

std::string shell(const char* cmd) {
#ifdef _WIN32
  FILE* p = _popen(cmd, "r");
#else
  FILE* p = popen(cmd, "r");
#endif
  if (!p) return "";
  std::string out;
  char buf[256];
  while (std::fgets(buf, sizeof buf, p)) out += buf;
#ifdef _WIN32
  _pclose(p);
#else
  pclose(p);
#endif
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
  return out;
}

// --- GPU batches -------------------------------------------------------------------------

class Gpu {
 public:
  Gpu(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog) : tracer_(tracer), catalog_(catalog) {}

  const CatalogEntry& use(const std::string& id) {
    for (const CatalogEntry& e : catalog_) {
      if (e.id != id) continue;
      ModuleOptions options;
      options.batch = true;
      if (!tracer_.select(e, Precision::Double, options)) throw std::runtime_error(tracer_.error());
      return e;
    }
    throw std::runtime_error("metric " + id + " is not in the catalog");
  }

  std::vector<rm::RayOut> coordinate_rays(const rm::BatchOptions& o, const std::vector<Vec3>& starts, const std::vector<Vec3>& dirs) {
    const int n = int(starts.size());
    double *d_s = upload(starts), *d_d = upload(dirs);
    rm::RayOut* d_out = nullptr;
    V_CHECK(cudaMalloc(&d_out, sizeof(rm::RayOut) * n));
    rm::BatchOptions copy = o;
    int count = n;
    void* args[] = {&copy, &d_s, &d_d, &count, &d_out};
    tracer_.launch("trace_coordinate_rays_kernel", unsigned((n + 63) / 64), 64, args);
    return download(d_out, n, {d_s, d_d});
  }

  std::vector<rm::RayOut> camera_rays(const rm::BatchOptions& o, const rm::CameraFrame& cam, const std::vector<Vec3>& dirs) {
    const int n = int(dirs.size());
    double* d_d = upload(dirs);
    rm::RayOut* d_out = nullptr;
    V_CHECK(cudaMalloc(&d_out, sizeof(rm::RayOut) * n));
    rm::BatchOptions copy = o;
    rm::CameraFrame c = cam;
    int count = n;
    void* args[] = {&copy, &c, &d_d, &count, &d_out};
    tracer_.launch("trace_camera_rays_kernel", unsigned((n + 63) / 64), 64, args);
    return download(d_out, n, {d_d});
  }

  /** g and dg at 4-points x (x.size() = 4 n). */
  void metric_eval(const double* params, const std::vector<double>& x, std::vector<double>& g, std::vector<double>& dg) {
    const int n = int(x.size() / 4);
    double *d_p = nullptr, *d_x = nullptr, *d_g = nullptr, *d_dg = nullptr;
    V_CHECK(cudaMalloc(&d_p, sizeof(double) * rm::MAX_PARAMS));
    V_CHECK(cudaMalloc(&d_x, sizeof(double) * x.size()));
    V_CHECK(cudaMalloc(&d_g, sizeof(double) * 16 * n));
    V_CHECK(cudaMalloc(&d_dg, sizeof(double) * 64 * n));
    V_CHECK(cudaMemcpy(d_p, params, sizeof(double) * rm::MAX_PARAMS, cudaMemcpyHostToDevice));
    V_CHECK(cudaMemcpy(d_x, x.data(), sizeof(double) * x.size(), cudaMemcpyHostToDevice));
    int count = n;
    void* args[] = {&d_p, &d_x, &count, &d_g, &d_dg};
    tracer_.launch("metric_eval_kernel", unsigned((n + 63) / 64), 64, args);
    g.resize(16 * n);
    dg.resize(64 * n);
    V_CHECK(cudaMemcpy(g.data(), d_g, sizeof(double) * g.size(), cudaMemcpyDeviceToHost));
    V_CHECK(cudaMemcpy(dg.data(), d_dg, sizeof(double) * dg.size(), cudaMemcpyDeviceToHost));
    for (double* p : {d_p, d_x, d_g, d_dg}) cudaFree(p);
  }

  rm::CameraFrame camera(const double* params, double radius, double yaw, double pitch) {
    return tracer_.camera(params, radius, yaw, pitch);
  }

  NvrtcTracer& tracer() { return tracer_; }

 private:
  double* upload(const std::vector<Vec3>& v) {
    double* d = nullptr;
    V_CHECK(cudaMalloc(&d, sizeof(double) * 3 * v.size()));
    V_CHECK(cudaMemcpy(d, v.data(), sizeof(double) * 3 * v.size(), cudaMemcpyHostToDevice));
    return d;
  }
  std::vector<rm::RayOut> download(rm::RayOut* d_out, int n, std::initializer_list<double*> extra) {
    std::vector<rm::RayOut> out(n);
    V_CHECK(cudaMemcpy(out.data(), d_out, sizeof(rm::RayOut) * n, cudaMemcpyDeviceToHost));
    cudaFree(d_out);
    for (double* p : extra) cudaFree(p);
    return out;
  }
  NvrtcTracer& tracer_;
  const std::vector<CatalogEntry>& catalog_;
};

rm::BatchOptions rk45(const std::vector<double>& params, double rtol, double escape_radius, double length_scale = 1) {
  rm::BatchOptions o{};
  std::copy(params.begin(), params.end(), o.params);
  o.step_fraction = 0.01;
  o.min_step = 1e-9;
  o.escape_radius = escape_radius;
  o.rtol = rtol;
  o.length_scale = length_scale;
  o.max_steps = 2000000;
  o.integrator = rm::INTEGRATOR_RK45;
  o.disk = 0;
  return o;
}

rm::BatchOptions rk4(const std::vector<double>& params, double step_fraction, double escape_radius) {
  rm::BatchOptions o{};
  std::copy(params.begin(), params.end(), o.params);
  o.step_fraction = step_fraction;
  o.min_step = 1e-4;
  o.escape_radius = escape_radius;
  o.max_steps = 20000000;
  o.integrator = rm::INTEGRATOR_RK4;
  return o;
}

const char* fate_name(int f) {
  static const char* names[] = {"escaped", "captured", "disk", "unfinished"};
  return names[std::min(std::max(f, 0), 3)];
}

double radius(const double* y) { return std::sqrt(y[1] * y[1] + y[2] * y[2] + y[3] * y[3]); }

// --- Schwarzschild sweep angle (shared by deflection and convergence) ---------------------

struct SweepRay {
  double b, traced, exact;
  int steps, fate;
};

/**
 * Rays entering at x = start_x with offsets b0 in the equatorial plane, moving
 * in -x; each one's swept angle against the exact orbit integral for its own
 * L/E and its actual start and end radius.
 */
std::vector<SweepRay> schwarzschild_sweeps(Gpu& gpu, const rm::BatchOptions& o, const std::vector<double>& b0, double start_x) {
  std::vector<Vec3> starts, dirs;
  for (double b : b0) {
    starts.push_back({start_x, b, 0});
    dirs.push_back({-1, 0, 0});
  }
  std::vector<SweepRay> out;
  for (const rm::RayOut& r : gpu.coordinate_rays(o, starts, dirs)) {
    const double b = std::fabs(r.y0[1] * r.y0[6] - r.y0[2] * r.y0[5]) / std::fabs(r.y0[4]);
    const double exact = an::schwarzschild_sweep(b, 1, radius(r.y0), radius(r.y));
    const double raw = std::atan2(r.y[2], r.y[1]) - std::atan2(r.y0[2], r.y0[1]);
    double traced = std::fabs(raw);
    for (int k = -4; k <= 4; k++) {  // the final position fixes the angle only modulo 2 pi
      const double c = std::fabs(raw + 2 * PI * k);
      if (std::fabs(c - exact) < std::fabs(traced - exact)) traced = c;
    }
    out.push_back({b, traced, exact, r.steps, r.fate});
  }
  return out;
}

// --- Checks ------------------------------------------------------------------------------

CheckResult check_deflection(Gpu& gpu) {
  gpu.use("schwarzschild");
  const std::vector<double> params = {1};
  const std::vector<double> b0 = {5.21, 5.25, 5.3, 5.5, 6, 7, 8, 10, 15, 20, 30, 50, 100, 300, 1000};
  const double start = 1e5, rtol = 1e-12;
  const auto ref = schwarzschild_sweeps(gpu, rk45(params, rtol, 2 * start), b0, start);
  const auto interactive = schwarzschild_sweeps(gpu, rk4(params, 0.03, 2 * start), b0, start);

  double worst = 0, worst_rk4 = 0;
  bool all_escaped = true;
  std::vector<double> b, err, err_rk4, deflection, series, steps;
  for (size_t i = 0; i < ref.size(); i++) {
    const double e = std::fabs(ref[i].traced - ref[i].exact);
    const double e4 = std::fabs(interactive[i].traced - interactive[i].exact);
    worst = std::max(worst, e);
    worst_rk4 = std::max(worst_rk4, e4);
    all_escaped = all_escaped && ref[i].fate == rm::FATE_ESCAPED;
    b.push_back(ref[i].b);
    err.push_back(e);
    err_rk4.push_back(e4);
    deflection.push_back(an::schwarzschild_deflection(ref[i].b, 1));
    const double bb = ref[i].b;
    series.push_back(4 / bb + 15 * PI / (4 * bb * bb));
    steps.push_back(ref[i].steps);
  }
  CheckResult c;
  c.id = "deflection";
  c.title = "Light deflection vs impact parameter (Schwarzschild)";
  c.description =
      "Rays enter at x = 1e5 M and leave past 2e5 M. The angle each sweeps around the hole is compared with the exact "
      "orbit integral phi = int du / sqrt(1/b^2 - u^2 + 2 M u^3) for its own b = L/E and its actual start and end "
      "radius (adaptive Simpson quadrature). The deflection column is the same integral between infinities minus pi.";
  c.metrics.push_back(at_most("max |swept angle - exact|, RK45 fp64 rtol 1e-12", worst, 1e-9, "rad"));
  c.metrics.push_back({"every reference ray escaped", "", all_escaped ? 1.0 : 0.0, 1, all_escaped, ""});
  c.metrics.push_back({"max |swept angle - exact|, interactive RK4 (step fraction 0.03)", "rad", worst_rk4, NAN, true,
                       "informational: the error of the interactive integrator"});
  c.data = "{\"b_over_M\": " + arr(b) + ", \"exact_deflection_rad\": " + arr(deflection) +
           ", \"weak_field_series_rad\": " + arr(series) + ", \"rk45_abs_error_rad\": " + arr(err) +
           ", \"rk4_interactive_abs_error_rad\": " + arr(err_rk4) + ", \"rk45_steps\": " + arr(steps) + "}";
  return c;
}

CheckResult check_schwarzschild_shadow(Gpu& gpu) {
  gpu.use("schwarzschild");
  const std::vector<double> params = {1};
  double p8[rm::MAX_PARAMS] = {1};
  const std::vector<double> distances = {4, 5, 6, 8, 10, 15, 26, 50, 100, 1000};
  std::vector<double> traced, predicted, err;
  double worst = 0;
  for (double R : distances) {
    const rm::CameraFrame cam = gpu.camera(p8, R, 0, 0);
    const rm::BatchOptions o = rk45(params, 1e-12, std::max(2 * R, 100.0));
    double lo = 0, hi = PI / 2;  // straight at the hole is captured; sideways escapes for R > 3M
    for (int it = 0; it < 60 && hi - lo > 1e-14; it++) {
      const double mid = 0.5 * (lo + hi);
      const int fate = gpu.camera_rays(o, cam, {Vec3{std::sin(mid), 0, std::cos(mid)}})[0].fate;
      if (fate == rm::FATE_CAPTURED) lo = mid;
      else if (fate == rm::FATE_ESCAPED) hi = mid;
      else throw std::runtime_error("shadow bisection: unfinished ray");
    }
    const double a = 0.5 * (lo + hi), p = an::schwarzschild_shadow_angle(R, 1);
    traced.push_back(a);
    predicted.push_back(p);
    err.push_back(std::fabs(a - p));
    worst = std::max(worst, std::fabs(a - p));
  }
  CheckResult c;
  c.id = "shadow-schwarzschild";
  c.title = "Schwarzschild shadow radius vs camera distance";
  c.description =
      "For a static observer at r, the angle between the direction to the hole and the shadow edge is found by "
      "bisection on ray fate (captured / escaped) and compared with sin(alpha) = sqrt(27) M sqrt(1 - 2M/r) / r.";
  c.metrics.push_back(at_most("max |traced - predicted| shadow angle, RK45 fp64 rtol 1e-12", worst, 1e-9, "rad"));
  c.data = "{\"camera_r_over_M\": " + arr(distances) + ", \"traced_rad\": " + arr(traced) + ", \"predicted_rad\": " +
           arr(predicted) + ", \"abs_error_rad\": " + arr(err) + "}";
  return c;
}

CheckResult check_kerr_shadow(Gpu& gpu) {
  gpu.use("kerr");
  const double D = 1e6;
  const std::vector<double> spins = {0.3, 0.6, 0.9, 0.99};
  const std::vector<double> inclinations = {17, 45, 90};
  double worst = 0;
  std::string configs = "[";
  int config_count = 0;
  for (double a : spins) {
    const std::vector<double> params = {1, a};
    const rm::BatchOptions o = rk45(params, 1e-12, 2 * D);
    const double r_pro = an::kerr_photon_orbit_prograde(a, 1), r_retro = an::kerr_photon_orbit_retrograde(a, 1);
    for (double inc : inclinations) {
      const double th = inc * DEG;
      const Vec3 n = {std::sin(th), 0, std::cos(th)}, ea = {0, 1, 0}, eb = {-std::cos(th), 0, std::sin(th)};
      // Bardeen's outline, sampled at spherical photon orbit radii.
      std::vector<std::array<double, 2>> pts;
      const int samples = 40;
      for (int i = 0; i < samples; i++) {
        const double r = r_pro + (r_retro - r_pro) * (i + 0.5) / samples;
        double alpha, beta2;
        an::kerr_shadow_point(r, a, 1, th, alpha, beta2);
        if (beta2 <= 0) continue;
        pts.push_back({alpha, std::sqrt(beta2)});
        pts.push_back({alpha, -std::sqrt(beta2)});
      }
      double amin = 1e300, amax = -1e300;
      for (auto& p : pts) {
        amin = std::min(amin, p[0]);
        amax = std::max(amax, p[0]);
      }
      const double ca = 0.5 * (amin + amax);
      auto ray_start = [&](double al, double be) {
        return Vec3{D * n[0] + al * ea[0] + be * eb[0], D * n[1] + al * ea[1] + be * eb[1], D * n[2] + al * ea[2] + be * eb[2]};
      };
      const Vec3 dir = {-n[0], -n[1], -n[2]};
      if (gpu.coordinate_rays(o, {ray_start(ca, 0)}, {dir})[0].fate != rm::FATE_CAPTURED)
        throw std::runtime_error("Kerr shadow: the outline centre is not inside the shadow");
      // Bisection along the line from the centre through each outline point: s = 1 is Bardeen's edge.
      std::vector<double> lo(pts.size(), 0.5), hi(pts.size(), 1.5);
      for (int it = 0; it < 48; it++) {
        std::vector<Vec3> starts, dirs;
        for (size_t k = 0; k < pts.size(); k++) {
          const double s = 0.5 * (lo[k] + hi[k]);
          starts.push_back(ray_start(ca + s * (pts[k][0] - ca), s * pts[k][1]));
          dirs.push_back(dir);
        }
        const auto res = gpu.coordinate_rays(o, starts, dirs);
        for (size_t k = 0; k < pts.size(); k++) {
          const double s = 0.5 * (lo[k] + hi[k]);
          if (res[k].fate == rm::FATE_CAPTURED) lo[k] = s;
          else hi[k] = s;
        }
      }
      double cfg_worst = 0, cfg_signed_mean = 0;
      std::vector<double> dev;
      for (size_t k = 0; k < pts.size(); k++) {
        const double s = 0.5 * (lo[k] + hi[k]);
        const double len = std::hypot(pts[k][0] - ca, pts[k][1]);
        dev.push_back((s - 1) * len);
        cfg_worst = std::max(cfg_worst, std::fabs((s - 1) * len));
        cfg_signed_mean += (s - 1) * len / pts.size();
      }
      worst = std::max(worst, cfg_worst);
      configs += std::string(config_count++ ? ", " : "") + "{\"a\": " + num(a) + ", \"inclination_deg\": " + num(inc) +
                 ", \"outline_points\": " + std::to_string(pts.size()) + ", \"max_abs_deviation_M\": " + num(cfg_worst) +
                 ", \"mean_signed_deviation_M\": " + num(cfg_signed_mean) + ", \"deviation_M\": " + arr(dev) + "}";
      std::printf("    Kerr shadow a = %.2f, i = %2.0f deg: %zu points, max deviation %.2e M\n", a, inc, pts.size(), cfg_worst);
    }
  }
  CheckResult c;
  c.id = "shadow-kerr";
  c.title = "Kerr shadow outline vs Bardeen (1973)";
  c.description =
      "Parallel rays from a plane 1e6 M away, at inclination i from the spin axis. Along the line from the outline's "
      "centre through each point (alpha, beta) of Bardeen's analytic edge, the capture boundary is found by bisection "
      "on ray fate; the deviation is its distance from Bardeen's point, in M (positive = traced edge outside). Starting "
      "at finite distance costs little here: light reaching a distant observer travels along the outgoing principal "
      "null direction l of the Kerr-Schild form g = eta + f l l, so the f l l correction to the initial momentum is "
      "of order (b / D)^2 ~ 1e-11.";
  c.metrics.push_back(at_most("max |traced - Bardeen| edge distance over a = 0.3, 0.6, 0.9, 0.99 and i = 17, 45, 90 deg", worst, 1e-4, "M"));
  c.data = "{\"observer_distance_M\": " + num(D) + ", \"configurations\": " + configs + "]}";
  return c;
}

/**
 * Circular equatorial geodesics at points (x, 0, 0) from g and its derivatives
 * evaluated on the GPU: Omega solves d_r g_tt + 2 Omega d_r g_tphi + Omega^2 d_r g_phiphi = 0
 * (prograde = larger root), and E = -(g_tt + Omega g_tphi) u^t. At (x, 0, 0), d_phi = x d_y.
 */
void circular_orbits(Gpu& gpu, const double* p8, const std::vector<double>& xs, bool pro, std::vector<double>& energy,
                     std::vector<double>& omega) {
  std::vector<double> pts, g, dg;
  for (double x : xs) pts.insert(pts.end(), {0, x, 0, 0});
  gpu.metric_eval(p8, pts, g, dg);
  energy.clear();
  omega.clear();
  for (size_t i = 0; i < xs.size(); i++) {
    const double x = xs[i];
    const double* G = &g[16 * i];
    const double* D = &dg[64 * i + 16];  // d/dx
    const double gtt = G[0], gtp = x * G[2], gpp = x * x * G[10];
    const double A = D[0], B = G[2] + x * D[2], C = 2 * x * G[10] + x * x * D[10];
    const double disc = std::sqrt(B * B - A * C);
    const double w1 = (-B + disc) / C, w2 = (-B - disc) / C;
    const double w = pro ? std::max(w1, w2) : std::min(w1, w2);
    const double ut = 1 / std::sqrt(-(gtt + 2 * w * gtp + w * w * gpp));
    energy.push_back(-(gtt + w * gtp) * ut);
    omega.push_back(w);
  }
}

/** Minimum of the circular-orbit energy E(x) on [lo, hi] by golden-section search: the ISCO. */
double isco_from_metric(Gpu& gpu, const double* p8, double lo, double hi, bool pro) {
  const double gr = (std::sqrt(5.0) - 1) / 2;
  std::vector<double> e, w;
  for (int it = 0; it < 120 && hi - lo > 1e-13; it++) {
    const double x1 = hi - gr * (hi - lo), x2 = lo + gr * (hi - lo);
    circular_orbits(gpu, p8, {x1, x2}, pro, e, w);
    if (e[0] < e[1]) hi = x2;
    else lo = x1;
  }
  return 0.5 * (lo + hi);
}

/** Launch radius on the x axis that separates tangential rays falling in from escaping: the circular photon orbit. */
double photon_orbit_traced(Gpu& gpu, const std::vector<double>& params, double lo, double hi, bool prograde) {
  const rm::BatchOptions o = rk45(params, 1e-12, 100);
  const Vec3 dir = {0, prograde ? -1.0 : 1.0, 0};  // dir -y at +x follows light moving +y
  for (int it = 0; it < 60 && hi - lo > 1e-14; it++) {
    const double mid = 0.5 * (lo + hi);
    if (gpu.coordinate_rays(o, {Vec3{mid, 0, 0}}, {dir})[0].fate == rm::FATE_CAPTURED) lo = mid;
    else hi = mid;
  }
  return 0.5 * (lo + hi);
}

/** Shadow edge angle seen by a static camera at distance R (bisection on ray fate). */
double shadow_angle_traced(Gpu& gpu, const std::vector<double>& params, double R) {
  double p8[rm::MAX_PARAMS] = {};
  std::copy(params.begin(), params.end(), p8);
  const rm::CameraFrame cam = gpu.camera(p8, R, 0, 0);
  const rm::BatchOptions o = rk45(params, 1e-12, std::max(2 * R, 100.0));
  double lo = 0, hi = PI / 2;
  for (int it = 0; it < 60 && hi - lo > 1e-14; it++) {
    const double mid = 0.5 * (lo + hi);
    const int fate = gpu.camera_rays(o, cam, {Vec3{std::sin(mid), 0, std::cos(mid)}})[0].fate;
    if (fate == rm::FATE_CAPTURED) lo = mid;
    else if (fate == rm::FATE_ESCAPED) hi = mid;
    else throw std::runtime_error("shadow bisection: unfinished ray");
  }
  return 0.5 * (lo + hi);
}

CheckResult check_reissner_nordstrom(Gpu& gpu) {
  gpu.use("reissner-nordstrom");
  const std::vector<double> charges = {0, 0.3, 0.6, 0.8, 0.95};
  const std::vector<double> distances = {10, 26};
  double worst_photon = 0, worst_isco = 0, worst_omega = 0, worst_shadow = 0;
  std::string rows = "[";
  for (size_t k = 0; k < charges.size(); k++) {
    const double q = charges[k];
    const std::vector<double> params = {1, q};
    double p8[rm::MAX_PARAMS] = {1, q};
    const double r_plus = 1 + std::sqrt(1 - q * q);
    const double ph = photon_orbit_traced(gpu, params, r_plus * 1.02, 5, true);
    const double ph_expect = an::rn_photon_orbit(q, 1);
    const double isco = isco_from_metric(gpu, p8, ph_expect * 1.05, 12, true);
    const double isco_expect = an::rn_isco(q, 1);
    std::vector<double> xs, e, w;
    for (double r = 4; r <= 20; r += 2) xs.push_back(r);
    circular_orbits(gpu, p8, xs, true, e, w);
    for (size_t i = 0; i < xs.size(); i++) {
      const double expect = std::sqrt(xs[i] - q * q) / (xs[i] * xs[i]);  // sqrt(M r - Q^2) / r^2, M = 1
      worst_omega = std::max(worst_omega, std::fabs(w[i] - expect) / expect);
    }
    std::vector<double> shadow, shadow_expect;
    for (double R : distances) {
      shadow.push_back(shadow_angle_traced(gpu, params, R));
      shadow_expect.push_back(an::rn_shadow_angle(R, q, 1));
      worst_shadow = std::max(worst_shadow, std::fabs(shadow.back() - shadow_expect.back()));
    }
    worst_photon = std::max(worst_photon, std::fabs(ph - ph_expect));
    worst_isco = std::max(worst_isco, std::fabs(isco - isco_expect));
    rows += std::string(k ? ", " : "") + "{\"q\": " + num(q) + ", \"photon_orbit\": {\"traced\": " + num(ph) +
            ", \"analytic\": " + num(ph_expect) + "}, \"isco\": {\"from_metric\": " + num(isco) + ", \"cubic_root\": " +
            num(isco_expect) + "}, \"shadow_camera_r_over_M\": " + arr(distances) + ", \"shadow_traced_rad\": " + arr(shadow) +
            ", \"shadow_predicted_rad\": " + arr(shadow_expect) + "}";
    std::printf("    RN q = %.2f: photon %.12f (analytic %.12f), ISCO %.9f (cubic %.9f), shadow at 26 M %.12f (predicted %.12f)\n", q,
                ph, ph_expect, isco, isco_expect, shadow.back(), shadow_expect.back());
  }
  CheckResult c;
  c.id = "reissner-nordstrom";
  c.title = "Reissner–Nordström: photon orbit, ISCO and shadow";
  c.description =
      "Charged black hole, Q = q M with q = 0, 0.3, 0.6, 0.8, 0.95. Photon orbit: bisection on the radius of "
      "tangential rays (falling in vs escaping) against r = (3M + sqrt(9M^2 - 8Q^2)) / 2. ISCO: minimum of the "
      "circular-orbit energy from the metric's derivatives on the GPU, against the root of rho^3 - 6 rho^2 + 9 q^2 rho - "
      "4 q^4 = 0 found numerically (not the metric file's closed form). Omega: against sqrt(M r - Q^2) / r^2, the "
      "disk model's formula. Shadow: static cameras at 10 M and 26 M, against sin(alpha) = b_c sqrt(f(r)) / r with "
      "f = 1 - 2M/r + Q^2/r^2 and b_c = r_ph / sqrt(f(r_ph)).";
  c.metrics.push_back(at_most("max |traced - analytic| photon orbit radius", worst_photon, 1e-8, "M"));
  c.metrics.push_back(at_most("max |metric - cubic root| ISCO radius", worst_isco, 1e-6, "M",
                              "golden-section search on E(r); limited by the flatness of E at its minimum"));
  c.metrics.push_back(at_most("max relative |Omega - sqrt(M r - Q^2)/r^2|, r = 4..20 M", worst_omega, 1e-12, ""));
  c.metrics.push_back(at_most("max |traced - predicted| shadow angle", worst_shadow, 1e-9, "rad"));
  c.data = "{\"rows\": " + rows + "]}";
  return c;
}

CheckResult check_orbits(Gpu& gpu) {
  CheckResult c;
  c.id = "orbits";
  c.title = "Circular photon orbits and the ISCO";
  c.description =
      "Photon orbits: rays launched tangentially in the equatorial plane; bisection on the launch radius between "
      "falling in and escaping gives the circular photon orbit, compared with r = 2M (1 + cos(2/3 acos(-+a))). "
      "ISCO: from g and its derivatives evaluated on the GPU, circular geodesics have angular velocity Omega solving "
      "d_r g_tt + 2 Omega d_r g_tphi + Omega^2 d_r g_phiphi = 0; the ISCO is the minimum of their energy, compared with "
      "Bardeen, Press & Teukolsky (1972). Omega is also compared with the Boyer-Lindquist formula "
      "sqrt(M) / (r^1.5 + a sqrt(M)) that the disk model uses. Radii are Boyer-Lindquist (r^2 = x^2 - a^2 in the plane).";
  double worst_photon = 0, worst_isco = 0, worst_omega = 0;
  std::string rows = "[";
  const std::vector<double> spins = {0, 0.3, 0.6, 0.9, 0.99};
  int row_count = 0;
  for (const std::string id : {std::string("schwarzschild"), std::string("kerr")}) {
    gpu.use(id);
    for (double a : (id == "kerr" ? spins : std::vector<double>{0})) {
      const std::vector<double> params = id == "kerr" ? std::vector<double>{1, a} : std::vector<double>{1};
      double p8[rm::MAX_PARAMS] = {};
      std::copy(params.begin(), params.end(), p8);
      const double r_plus = 1 + std::sqrt(1 - a * a);
      auto x_of_r = [&](double r) { return std::sqrt(r * r + a * a); };
      auto r_of_x = [&](double x) { return std::sqrt(x * x - a * a); };

      // Photon orbits by tracing. dir -y at +x follows light moving +y (prograde).
      double photon[2];
      for (int pro = 0; pro < 2; pro++) {
        const rm::BatchOptions o = rk45(params, 1e-12, 100);
        double lo = x_of_r(r_plus * 1.02), hi = x_of_r(5);
        const Vec3 dir = {0, pro ? -1.0 : 1.0, 0};
        for (int it = 0; it < 60 && hi - lo > 1e-14; it++) {
          const double mid = 0.5 * (lo + hi);
          const int fate = gpu.coordinate_rays(o, {Vec3{mid, 0, 0}}, {dir})[0].fate;
          if (fate == rm::FATE_CAPTURED) lo = mid;
          else hi = mid;
        }
        photon[pro] = r_of_x(0.5 * (lo + hi));
      }
      const double ph_pro = an::kerr_photon_orbit_prograde(a, 1), ph_retro = an::kerr_photon_orbit_retrograde(a, 1);
      worst_photon = std::max({worst_photon, std::fabs(photon[1] - ph_pro), std::fabs(photon[0] - ph_retro)});

      // Circular timelike orbits from the metric.
      auto circular = [&](const std::vector<double>& xs, bool pro, std::vector<double>& energy, std::vector<double>& omega) {
        circular_orbits(gpu, p8, xs, pro, energy, omega);
      };
      double isco[2];
      for (int pro = 0; pro < 2; pro++) {
        // Golden-section search for the minimum of E(x).
        const double phr = pro ? ph_pro : ph_retro;
        double lo = x_of_r(phr * 1.05), hi = x_of_r(12);
        const double gr = (std::sqrt(5.0) - 1) / 2;
        std::vector<double> e, w;
        for (int it = 0; it < 120 && hi - lo > 1e-13; it++) {
          const double x1 = hi - gr * (hi - lo), x2 = lo + gr * (hi - lo);
          circular({x1, x2}, pro, e, w);
          if (e[0] < e[1]) hi = x2;
          else lo = x1;
        }
        isco[pro] = r_of_x(0.5 * (lo + hi));
      }
      const double isco_pro = an::kerr_isco(a, 1, true), isco_retro = an::kerr_isco(a, 1, false);
      worst_isco = std::max({worst_isco, std::fabs(isco[1] - isco_pro), std::fabs(isco[0] - isco_retro)});

      // Prograde Omega against the Boyer-Lindquist formula on 4 M .. 20 M.
      std::vector<double> xs, e, w;
      for (double r = 4; r <= 20; r += 2) xs.push_back(x_of_r(r));
      circular(xs, true, e, w);
      for (size_t i = 0; i < xs.size(); i++) {
        const double r = r_of_x(xs[i]);
        const double expect = 1 / (std::pow(r, 1.5) + a);
        worst_omega = std::max(worst_omega, std::fabs(w[i] - expect) / expect);
      }

      rows += std::string(row_count++ ? ", " : "") + "{\"metric\": " + str(id) + ", \"a\": " + num(a) +
              ", \"photon_prograde\": {\"traced\": " + num(photon[1]) + ", \"analytic\": " + num(ph_pro) + "}" +
              ", \"photon_retrograde\": {\"traced\": " + num(photon[0]) + ", \"analytic\": " + num(ph_retro) + "}" +
              ", \"isco_prograde\": {\"from_metric\": " + num(isco[1]) + ", \"analytic\": " + num(isco_pro) + "}" +
              ", \"isco_retrograde\": {\"from_metric\": " + num(isco[0]) + ", \"analytic\": " + num(isco_retro) + "}}";
      std::printf("    %-13s a = %.2f: photon %.10f / %.10f (analytic %.10f / %.10f), ISCO %.8f / %.8f (analytic %.8f / %.8f)\n",
                  id.c_str(), a, photon[1], photon[0], ph_pro, ph_retro, isco[1], isco[0], isco_pro, isco_retro);
    }
  }
  c.metrics.push_back(at_most("max |traced - analytic| circular photon orbit radius", worst_photon, 1e-8, "M"));
  c.metrics.push_back(at_most("max |metric - analytic| ISCO radius", worst_isco, 1e-6, "M",
                              "golden-section search on E(r); limited by the flatness of E at its minimum"));
  c.metrics.push_back(at_most("max relative |Omega - sqrt(M)/(r^1.5 + a sqrt(M))|, r = 4..20 M", worst_omega, 1e-12, ""));
  c.data = "{\"rows\": " + rows + "]}";
  return c;
}

/** Full-frame constraint drift for one scene with RK4 and RK45 (fp64 frame kernels). */
CheckResult check_constraint_drift(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog) {
  const CatalogEntry* kerr = nullptr;
  for (const CatalogEntry& e : catalog)
    if (e.id == "kerr") kerr = &e;
  // Same view as scenes/kerr-a095-i45-nodisk.json.
  const int w = 960, h = 540;
  const double params[rm::MAX_PARAMS] = {1, 0.95};
  const size_t n = size_t(w) * h;
  float *d_hdr = nullptr, *d_drift = nullptr;
  unsigned* d_steps = nullptr;
  unsigned char* d_fate = nullptr;
  V_CHECK(cudaMalloc(&d_hdr, n * 16));
  V_CHECK(cudaMalloc(&d_drift, n * 4));
  V_CHECK(cudaMalloc(&d_steps, n * 4));
  V_CHECK(cudaMalloc(&d_fate, n));
  struct Run {
    std::string name;
    std::vector<double> histogram;  // counts per decade of max |H| / p_t^2
    double median, p99, max, mean_steps, ms, captured;
  };
  std::vector<Run> runs;
  const int lo_decade = -18, hi_decade = 0;
  for (int rk : {0, 1}) {
    ModuleOptions options;
    options.rk45 = rk == 1;
    if (!tracer.select(*kerr, Precision::Double, options)) throw std::runtime_error(tracer.error());
    rm::FrameUniforms u{};
    u.cam = tracer.camera(params, 26, 0, PI / 4);
    std::copy(params, params + rm::MAX_PARAMS, u.params);
    u.fov_scale = std::tan(45 * PI / 360) / std::min(1.0, double(w) / h);
    u.step_fraction = 0.03;
    u.escape_radius = 60 * u.cam.length_scale;
    u.exposure = 0.55;
    u.disk_temp = 4500;
    u.width = w;
    u.height = h;
    u.max_steps = rk ? 200000 : 300;
    u.flags = rm::FLAG_STARS;
    u.rtol = 1e-10;
    cudaEvent_t a, b;
    V_CHECK(cudaEventCreate(&a));
    V_CHECK(cudaEventCreate(&b));
    V_CHECK(cudaEventRecord(a));
    tracer.trace(u, d_hdr, d_steps, d_fate, d_drift);
    V_CHECK(cudaEventRecord(b));
    V_CHECK(cudaEventSynchronize(b));
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    std::vector<float> drift(n);
    std::vector<unsigned> steps(n);
    std::vector<unsigned char> fate(n);
    V_CHECK(cudaMemcpy(drift.data(), d_drift, n * 4, cudaMemcpyDeviceToHost));
    V_CHECK(cudaMemcpy(steps.data(), d_steps, n * 4, cudaMemcpyDeviceToHost));
    V_CHECK(cudaMemcpy(fate.data(), d_fate, n, cudaMemcpyDeviceToHost));
    Run r;
    r.name = rk ? "RK45 rtol 1e-10" : "RK4 step fraction 0.03, max 300 steps (interactive settings)";
    r.histogram.assign(hi_decade - lo_decade + 1, 0);
    std::vector<double> v;
    double step_sum = 0;
    size_t captured = 0;
    for (size_t i = 0; i < n; i++) {
      step_sum += steps[i];
      if (fate[i] == rm::FATE_CAPTURED) {
        captured++;
        continue;
      }
      const double d = drift[i];
      v.push_back(d);
      const int dec = d > 0 ? std::clamp(int(std::floor(std::log10(d))), lo_decade, hi_decade) : lo_decade;
      r.histogram[dec - lo_decade] += 1;
    }
    std::sort(v.begin(), v.end());
    r.median = v[v.size() / 2];
    r.p99 = v[size_t(0.99 * (v.size() - 1))];
    r.max = v.back();
    r.mean_steps = step_sum / n;
    r.ms = ms;
    r.captured = double(captured) / n;
    runs.push_back(r);
    std::printf("    drift %-60s median %.2e, p99 %.2e, max %.2e, %.1f steps/ray, %.0f ms\n", r.name.c_str(), r.median, r.p99,
                r.max, r.mean_steps, ms);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
  }
  cudaFree(d_hdr);
  cudaFree(d_drift);
  cudaFree(d_steps);
  cudaFree(d_fate);

  CheckResult c;
  c.id = "constraint-drift";
  c.title = "Null-constraint drift over a full frame";
  c.description =
      "Kerr a = 0.95, camera at 26 M and 45 degrees inclination (scenes/kerr-a095-i45-nodisk.json), 960x540, fp64. For every "
      "ray that is not captured, the largest |H| / p_t^2 along it (H = 1/2 g^{mu nu} p_mu p_nu is 0 for light). "
      "Histogram by decade from 1e-18 (and below) to 1e0.";
  c.metrics.push_back(at_most("p99 drift, RK45 rtol 1e-10", runs[1].p99, 1e-8, ""));
  c.metrics.push_back({"p99 drift, interactive RK4", "", runs[0].p99, NAN, true, "informational"});
  std::string data = "{\"decades\": [";
  for (int d = lo_decade; d <= hi_decade; d++) data += (d > lo_decade ? ", " : "") + std::to_string(d);
  data += "], \"runs\": [";
  for (size_t i = 0; i < runs.size(); i++) {
    const Run& r = runs[i];
    data += std::string(i ? ", " : "") + "{\"integrator\": " + str(r.name) + ", \"histogram\": " + arr(r.histogram) +
            ", \"median\": " + num(r.median) + ", \"p99\": " + num(r.p99) + ", \"max\": " + num(r.max) +
            ", \"mean_steps\": " + num(r.mean_steps) + ", \"frame_ms\": " + num(r.ms) + ", \"captured_fraction\": " + num(r.captured) + "}";
  }
  c.data = data + "]}";
  return c;
}

/** Least-squares slope of log(err) against log(h). */
double loglog_slope(const std::vector<double>& h, const std::vector<double>& e) {
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = double(h.size());
  for (size_t i = 0; i < h.size(); i++) {
    const double x = std::log(h[i]), y = std::log(e[i]);
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  return (n * sxy - sx * sy) / (n * sxx - sx * sx);
}

CheckResult check_convergence(Gpu& gpu) {
  gpu.use("schwarzschild");
  const std::vector<double> params = {1};
  const std::vector<double> b0 = {6, 10, 30};
  const double start = 1000;
  const std::vector<double> fractions = {0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025};
  std::vector<std::vector<double>> rk4_err(b0.size());
  for (double f : fractions) {
    const auto rays = schwarzschild_sweeps(gpu, rk4(params, f, 2 * start), b0, start);
    for (size_t i = 0; i < rays.size(); i++) rk4_err[i].push_back(std::fabs(rays[i].traced - rays[i].exact) / rays[i].exact);
  }
  const std::vector<double> rtols = {1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13};
  std::vector<std::vector<double>> rk45_err(b0.size()), rk45_steps(b0.size());
  double worst_ratio = 0;
  for (double t : rtols) {
    const auto rays = schwarzschild_sweeps(gpu, rk45(params, t, 2 * start), b0, start);
    for (size_t i = 0; i < rays.size(); i++) {
      const double e = std::fabs(rays[i].traced - rays[i].exact) / rays[i].exact;
      rk45_err[i].push_back(e);
      rk45_steps[i].push_back(rays[i].steps);
      if (t >= 1e-12) worst_ratio = std::max(worst_ratio, e / t);
    }
  }
  // Fit the order where RK4 is in its asymptotic range and above roundoff: the four finest steps.
  double min_order = 1e9, max_order = -1e9;
  std::string rk4_json = "[";
  for (size_t i = 0; i < b0.size(); i++) {
    const std::vector<double> h(fractions.end() - 4, fractions.end()), e(rk4_err[i].end() - 4, rk4_err[i].end());
    const double order = loglog_slope(h, e);
    min_order = std::min(min_order, order);
    max_order = std::max(max_order, order);
    rk4_json += std::string(i ? ", " : "") + "{\"b0_over_M\": " + num(b0[i]) + ", \"relative_error\": " + arr(rk4_err[i]) +
                ", \"fitted_order_finest_4\": " + num(order) + "}";
    std::printf("    RK4 b0 = %4.1f M: fitted order %.3f; relative errors %s\n", b0[i], order, arr(rk4_err[i]).c_str());
  }
  std::string rk45_json = "[";
  for (size_t i = 0; i < b0.size(); i++) {
    rk45_json += std::string(i ? ", " : "") + "{\"b0_over_M\": " + num(b0[i]) + ", \"relative_error\": " + arr(rk45_err[i]) +
                 ", \"accepted_steps\": " + arr(rk45_steps[i]) + "}";
  }
  CheckResult c;
  c.id = "convergence";
  c.title = "Convergence: RK4 order and RK45 tolerance";
  c.description =
      "Schwarzschild rays entering at x = 1000 M with offsets b0 = 6, 10, 30 M. Error = relative error of the swept "
      "angle against the exact orbit integral. RK4: fixed step = fraction * r; the order is the least-squares slope "
      "of log(error) against log(step fraction) over the four finest steps. RK45: error against requested rtol.";
  c.metrics.push_back({"RK4 fitted order (min over b0)", "", min_order, 3.8, min_order >= 3.8 && max_order <= 4.2,
                       "pass if every fitted order is within [3.8, 4.2]; max " + num(max_order)});
  c.metrics.push_back(at_most("max (RK45 relative error / rtol), rtol 1e-4 .. 1e-12", worst_ratio, 10, "",
                              "global error is not strictly bounded by the local tolerance; 10x allowed"));
  c.data = "{\"rk4_step_fractions\": " + arr(fractions) + ", \"rk4\": " + rk4_json + "], \"rk45_rtol\": " + arr(rtols) +
           ", \"rk45\": " + rk45_json + "]}";
  return c;
}

}  // namespace

int run_validation(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys,
                   const std::string& out_dir) {
  std::filesystem::create_directories(out_dir);
  const std::string commit = shell("git rev-parse --short HEAD");
  // Result files this suite and validation/compare.ts write do not count as changes to the code under test.
  const bool dirty = !shell("git status --porcelain -- . \":(exclude)validation/results\" \":(exclude)docs/benchmarks\"").empty();
  char date[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  const std::string header = "\"gpu\": " + str(sys.gpu) + ", \"arch\": " + str(sys.arch) + ", \"driver\": " + str(sys.driver) +
                             ", \"cuda_driver_api\": " + str(sys.cuda_driver_api) + ", \"nvrtc\": " + str(sys.nvrtc) +
                             ", \"precision\": \"fp64\", \"git\": {\"commit\": " + str(commit) +
                             ", \"uncommitted_changes\": " + (dirty ? "true" : "false") + "}, \"date\": " + str(date);

  Gpu gpu(tracer, catalog);
  using Check = std::function<CheckResult()>;
  const std::vector<std::pair<const char*, Check>> checks = {
      {"deflection", [&] { return check_deflection(gpu); }},
      {"shadow-schwarzschild", [&] { return check_schwarzschild_shadow(gpu); }},
      {"shadow-kerr", [&] { return check_kerr_shadow(gpu); }},
      {"orbits", [&] { return check_orbits(gpu); }},
      {"reissner-nordstrom", [&] { return check_reissner_nordstrom(gpu); }},
      {"constraint-drift", [&] { return check_constraint_drift(tracer, catalog); }},
      {"convergence", [&] { return check_convergence(gpu); }},
  };
  std::string summary = "[";
  bool all = true;
  // RM_VALIDATE_ONLY=<check> runs one check (for development; use with --out to a scratch folder).
  const char* only = std::getenv("RM_VALIDATE_ONLY");
  for (size_t i = 0; i < checks.size(); i++) {
    if (only && std::string(only) != checks[i].first) continue;
    std::printf("[%s]\n", checks[i].first);
    const auto t0 = std::chrono::steady_clock::now();
    const CheckResult r = checks[i].second();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string metrics = "[";
    for (size_t k = 0; k < r.metrics.size(); k++) {
      metrics += (k ? ", " : "") + metric_json(r.metrics[k]);
      const Metric& m = r.metrics[k];
      std::printf("  %s %-95s %.3e %s (tolerance %.1e)\n", m.pass ? "PASS" : "FAIL", m.name.c_str(), m.value,
                  m.unit.c_str(), m.tolerance);
    }
    metrics += "]";
    std::ofstream(std::filesystem::path(out_dir) / (r.id + ".json"))
        << "{\n  \"check\": " << str(r.id) << ",\n  \"title\": " << str(r.title) << ",\n  \"description\": "
        << str(r.description) << ",\n  \"pass\": " << (r.pass() ? "true" : "false") << ",\n  \"metrics\": " << metrics
        << ",\n  " << header << ",\n  \"runtime_s\": " << num(seconds) << ",\n  \"data\": " << r.data << "\n}\n";
    summary += std::string(i ? ", " : "") + "{\"check\": " + str(r.id) + ", \"title\": " + str(r.title) +
               ", \"pass\": " + (r.pass() ? "true" : "false") + ", \"metrics\": " + metrics + "}";
    all = all && r.pass();
  }
  std::ofstream(std::filesystem::path(out_dir) / "fp64-suite.json")
      << "{\n  \"suite\": \"fp64 validation (docs/phase-2-plan.md step 5), CUDA NVRTC double-precision modules\",\n  \"pass\": "
      << (all ? "true" : "false") << ",\n  " << header << ",\n  \"checks\": " << summary << "]\n}\n";
  std::printf("%s\n", all ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");
  return all ? 0 : 1;
}
