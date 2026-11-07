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
#include "rm/json.h"
#include "rm/tangent.h"

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

std::string read_file_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

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

  /** Rays from theta vectors with tangents along dirs (P of them); mode 0 symbolic, 1 duals (trace_tangent_rays_kernel). */
  struct Tangents {
    int P = 0;
    std::vector<double> rec;
    std::vector<int> meta;
    const double* y0(int i) const { return rec.data() + size_t(16 + 16 * P) * i; }
    const double* S0(int i) const { return y0(i) + 8; }
    const double* y(int i) const { return y0(i) + 8 + 8 * P; }
    const double* S(int i) const { return y0(i) + 16 + 8 * P; }
    int fate(int i) const { return meta[2 * i]; }
    int steps(int i) const { return meta[2 * i + 1]; }
  };
  using Theta = std::array<double, rm::MAX_THETA>;
  /** kernel_ms, when given, receives the kernel time (synchronised wall clock around the launch). */
  Tangents tangent_rays(const std::string& id, const rm::BatchOptions& o, const std::vector<Theta>& thetas,
                        const std::vector<Theta>& dirs, int mode, Precision precision = Precision::Double,
                        double* kernel_ms = nullptr) {
    for (const CatalogEntry& e : catalog_) {
      if (e.id != id) continue;
      ModuleOptions options;
      options.batch = true;
      options.tangents = int(dirs.size());
      if (!tracer_.select(e, precision, options)) throw std::runtime_error(tracer_.error());
      Tangents t;
      t.P = int(dirs.size());
      const int n = int(thetas.size());
      double *d_theta = nullptr, *d_dirs = nullptr, *d_out = nullptr;
      int* d_meta = nullptr;
      V_CHECK(cudaMalloc(&d_theta, sizeof(Theta) * n));
      V_CHECK(cudaMalloc(&d_dirs, sizeof(Theta) * dirs.size()));
      V_CHECK(cudaMalloc(&d_out, sizeof(double) * (16 + 16 * t.P) * n));
      V_CHECK(cudaMalloc(&d_meta, sizeof(int) * 2 * n));
      V_CHECK(cudaMemcpy(d_theta, thetas.data(), sizeof(Theta) * n, cudaMemcpyHostToDevice));
      V_CHECK(cudaMemcpy(d_dirs, dirs.data(), sizeof(Theta) * dirs.size(), cudaMemcpyHostToDevice));
      rm::BatchOptions copy = o;
      int count = n;
      void* args[] = {&copy, &d_theta, &d_dirs, &count, &d_out, &d_meta};
      V_CHECK(cudaDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      tracer_.launch(mode == 0 ? "trace_tangent_rays_kernel" : "trace_dual_rays_kernel", unsigned((n + 63) / 64), 64, args);
      V_CHECK(cudaDeviceSynchronize());
      if (kernel_ms) *kernel_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      t.rec.resize(size_t(16 + 16 * t.P) * n);
      t.meta.resize(2 * n);
      V_CHECK(cudaMemcpy(t.rec.data(), d_out, sizeof(double) * t.rec.size(), cudaMemcpyDeviceToHost));
      V_CHECK(cudaMemcpy(t.meta.data(), d_meta, sizeof(int) * t.meta.size(), cudaMemcpyDeviceToHost));
      for (void* p : {(void*)d_theta, (void*)d_dirs, (void*)d_out, (void*)d_meta}) cudaFree(p);
      return t;
    }
    throw std::runtime_error("metric " + id + " is not in the catalog");
  }

  /** Model-sky samples (render_samples_*_kernel): I, dI[P s + d] (dual only), meta[3 s ..] = (fate, steps, crossings). */
  struct SampleImage {
    int P = 0;
    std::vector<double> I, dI;
    std::vector<int> meta;
    double ms = 0;
  };
  SampleImage render_samples(const std::string& id, const rm::BatchOptions& o, const Theta& base, const std::vector<Theta>& dirs,
                             const std::vector<double>& xi, double taper, bool dual, Precision precision) {
    for (const CatalogEntry& e : catalog_) {
      if (e.id != id) continue;
      ModuleOptions options;
      options.batch = true;
      options.tangents = std::max(1, int(dirs.size()));
      if (!tracer_.select(e, precision, options)) throw std::runtime_error(tracer_.error());
      SampleImage out;
      out.P = dual ? int(dirs.size()) : 0;
      const int n = int(xi.size() / 2);
      double *d_base = nullptr, *d_dirs = nullptr, *d_xi = nullptr, *d_I = nullptr, *d_dI = nullptr;
      int* d_meta = nullptr;
      V_CHECK(cudaMalloc(&d_base, sizeof(Theta)));
      V_CHECK(cudaMalloc(&d_dirs, sizeof(Theta) * std::max<size_t>(1, dirs.size())));
      V_CHECK(cudaMalloc(&d_xi, sizeof(double) * xi.size()));
      V_CHECK(cudaMalloc(&d_I, sizeof(double) * n));
      V_CHECK(cudaMalloc(&d_dI, sizeof(double) * std::max(1, out.P) * n));
      V_CHECK(cudaMalloc(&d_meta, sizeof(int) * 3 * n));
      V_CHECK(cudaMemcpy(d_base, base.data(), sizeof(Theta), cudaMemcpyHostToDevice));
      if (!dirs.empty()) V_CHECK(cudaMemcpy(d_dirs, dirs.data(), sizeof(Theta) * dirs.size(), cudaMemcpyHostToDevice));
      V_CHECK(cudaMemcpy(d_xi, xi.data(), sizeof(double) * xi.size(), cudaMemcpyHostToDevice));
      rm::BatchOptions copy = o;
      int count = n;
      double tp = taper;
      V_CHECK(cudaDeviceSynchronize());
      const auto t0 = std::chrono::steady_clock::now();
      if (dual) {
        void* args[] = {&copy, &d_base, &d_dirs, &d_xi, &count, &tp, &d_I, &d_dI, &d_meta};
        tracer_.launch("render_samples_dual_kernel", unsigned((n + 63) / 64), 64, args);
      } else {
        void* args[] = {&copy, &d_base, &d_xi, &count, &tp, &d_I, &d_meta};
        tracer_.launch("render_samples_plain_kernel", unsigned((n + 63) / 64), 64, args);
      }
      V_CHECK(cudaDeviceSynchronize());
      out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      out.I.resize(n);
      out.meta.resize(3 * n);
      V_CHECK(cudaMemcpy(out.I.data(), d_I, sizeof(double) * n, cudaMemcpyDeviceToHost));
      V_CHECK(cudaMemcpy(out.meta.data(), d_meta, sizeof(int) * 3 * n, cudaMemcpyDeviceToHost));
      if (dual) {
        out.dI.resize(size_t(out.P) * n);
        V_CHECK(cudaMemcpy(out.dI.data(), d_dI, sizeof(double) * out.dI.size(), cudaMemcpyDeviceToHost));
      }
      for (void* p : {(void*)d_base, (void*)d_dirs, (void*)d_xi, (void*)d_I, (void*)d_dI, (void*)d_meta}) cudaFree(p);
      return out;
    }
    throw std::runtime_error("metric " + id + " is not in the catalog");
  }

  /** edge_bisect_kernel: segs[4 i ..] = (x start, x end, y, expected start branch); returns out[4 i ..] = (x*, I start side, I end side, flag). */
  std::vector<double> edge_bisect(const std::string& id, const rm::BatchOptions& o, const Theta& base, const std::vector<double>& segs,
                                  int iters, double taper) {
    for (const CatalogEntry& e : catalog_) {
      if (e.id != id) continue;
      ModuleOptions options;
      options.batch = true;
      options.tangents = 2;
      if (!tracer_.select(e, Precision::Double, options)) throw std::runtime_error(tracer_.error());
      const int n = int(segs.size() / 4);
      std::vector<double> out(segs.size());
      if (n == 0) return out;
      double *d_base = nullptr, *d_seg = nullptr, *d_out = nullptr;
      V_CHECK(cudaMalloc(&d_base, sizeof(Theta)));
      V_CHECK(cudaMalloc(&d_seg, sizeof(double) * segs.size()));
      V_CHECK(cudaMalloc(&d_out, sizeof(double) * segs.size()));
      V_CHECK(cudaMemcpy(d_base, base.data(), sizeof(Theta), cudaMemcpyHostToDevice));
      V_CHECK(cudaMemcpy(d_seg, segs.data(), sizeof(double) * segs.size(), cudaMemcpyHostToDevice));
      rm::BatchOptions copy = o;
      int count = n, it = iters;
      double tp = taper;
      void* args[] = {&copy, &d_base, &d_seg, &count, &it, &tp, &d_out};
      tracer_.launch("edge_bisect_kernel", unsigned((n + 31) / 32), 32, args);
      V_CHECK(cudaDeviceSynchronize());
      V_CHECK(cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost));
      for (void* p : {(void*)d_base, (void*)d_seg, (void*)d_out}) cudaFree(p);
      return out;
    }
    throw std::runtime_error("metric " + id + " is not in the catalog");
  }

  /** Capture boundary along model-sky rays (ox, oy, ex, ey) between distances t0 and t1: out[2 i ..] = (distance, flag). */
  std::vector<double> curve_bisect(const std::string& id, rm::BatchOptions o, const Theta& base, const std::vector<double>& rays,
                                   double t0, double t1, int iters) {
    for (const CatalogEntry& e : catalog_) {
      if (e.id != id) continue;
      ModuleOptions options;
      options.batch = true;
      options.tangents = 2;
      if (!tracer_.select(e, Precision::Double, options)) throw std::runtime_error(tracer_.error());
      const int n = int(rays.size() / 4);
      std::vector<double> out(2 * size_t(n));
      if (n == 0) return out;
      double *d_base = nullptr, *d_ray = nullptr, *d_out = nullptr;
      V_CHECK(cudaMalloc(&d_base, sizeof(Theta)));
      V_CHECK(cudaMalloc(&d_ray, sizeof(double) * rays.size()));
      V_CHECK(cudaMalloc(&d_out, sizeof(double) * out.size()));
      V_CHECK(cudaMemcpy(d_base, base.data(), sizeof(Theta), cudaMemcpyHostToDevice));
      V_CHECK(cudaMemcpy(d_ray, rays.data(), sizeof(double) * rays.size(), cudaMemcpyHostToDevice));
      o.disk = 0;
      int count = n, it = iters;
      void* args[] = {&o, &d_base, &d_ray, &count, &t0, &t1, &it, &d_out};
      tracer_.launch("curve_bisect_kernel", unsigned((n + 31) / 32), 32, args);
      V_CHECK(cudaDeviceSynchronize());
      V_CHECK(cudaMemcpy(out.data(), d_out, sizeof(double) * out.size(), cudaMemcpyDeviceToHost));
      for (void* p : {(void*)d_base, (void*)d_ray, (void*)d_out}) cudaFree(p);
      return out;
    }
    throw std::runtime_error("metric " + id + " is not in the catalog");
  }

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

// --- Phase 4: ray sensitivities ------------------------------------------------------------

/** E = -p_t, L = x p_y - y p_x and b = L / E with its tangent along direction d. */
void impact_parameter(const double* y, const double* S, int d, double& b, double& db) {
  const double E = -y[4], L = y[1] * y[6] - y[2] * y[5];
  const double dE = -S[8 * d + 4];
  const double dL = S[8 * d + 1] * y[6] + y[1] * S[8 * d + 6] - S[8 * d + 2] * y[5] - y[2] * S[8 * d + 5];
  b = L / E;
  db = (dL * E - L * dE) / (E * E);
}

/** max over directions of |a - b| / max |b| (per direction). */
double tangent_rel(const double* a, const double* b, int P) {
  double worst = 0;
  for (int d = 0; d < P; d++) {
    double num = 0, den = 1e-300;
    for (int i = 0; i < 8; i++) {
      num = std::max(num, std::fabs(a[8 * d + i] - b[8 * d + i]));
      den = std::max(den, std::fabs(b[8 * d + i]));
    }
    worst = std::max(worst, num / den);
  }
  return worst;
}

CheckResult check_sensitivities(Gpu& gpu, const std::string& generated_dir) {
  CheckResult c;
  c.id = "sensitivities";
  c.title = "Ray sensitivities (forward-mode tangents)";
  c.description =
      "Phase 4 step 3: dy/dtheta of traced rays on the GPU in fp64, from the compiler's symbolic JVP (rm/tangent.h) and from "
      "Dual numbers through the ordinary integrator (rm/dual.h), against the JS reference tracer and against analytic "
      "sensitivities: the Schwarzschild shadow angle against camera radius, the Kerr critical impact parameter against spin, and "
      "the Schwarzschild sweep angle against impact parameter.";
  using Theta = Gpu::Theta;
  auto unit = [](int k) {
    Theta t{};
    t[k] = 1;
    return t;
  };
  std::string data = "{";

  // 1. GPU symbolic and dual tangents against the JS fixture (compiler/src/tangent-geodesic.ts).
  {
    const rm::json::Value fixture = rm::json::parse(read_file_text(generated_dir + "/fixtures/tangent-rays.json"));
    double js = 0, dual = 0;
    int rays = 0, mismatched = 0;
    for (const rm::json::Value& m : fixture["metrics"].a) {
      const int P = int(m["P"].num());
      const rm::json::Value& o = m["options"];
      rm::BatchOptions opt = rk4({}, o["stepFraction"].num(), o["escapeRadius"].num());
      opt.min_step = o["minStep"].num();
      opt.max_steps = int(o["maxSteps"].num());
      opt.disk = o["disk"].b ? 1 : 0;
      std::vector<Theta> thetas, dirs;
      for (int d = 0; d < P; d++) dirs.push_back(unit(d));
      for (const rm::json::Value& r : m["rays"].a) {
        Theta t{};
        const std::vector<double> v = r["theta"].numbers();
        std::copy(v.begin(), v.end(), t.begin());
        thetas.push_back(t);
      }
      const Gpu::Tangents sym = gpu.tangent_rays(m["id"].str(), opt, thetas, dirs, 0);
      const Gpu::Tangents dl = gpu.tangent_rays(m["id"].str(), opt, thetas, dirs, 1);
      for (size_t i = 0; i < thetas.size(); i++) {
        const rm::json::Value& r = m["rays"].a[i];
        const bool same = r["fate"].str() == fate_name(sym.fate(int(i))) && int(r["steps"].num()) == sym.steps(int(i)) &&
                          dl.fate(int(i)) == sym.fate(int(i)) && dl.steps(int(i)) == sym.steps(int(i));
        if (!same) mismatched++;
        if (!same || r["fate"].str() == "captured") continue;
        const std::vector<double> jS = r["S"].numbers();
        js = std::max(js, tangent_rel(sym.S(int(i)), jS.data(), P));
        dual = std::max(dual, tangent_rel(dl.S(int(i)), sym.S(int(i)), P));
        rays++;
      }
    }
    c.metrics.push_back(at_most("GPU symbolic tangents vs JS reference, all inputs, " + std::to_string(rays) + " rays (max relative)", js,
                                1e-9, ""));
    c.metrics.push_back(at_most("GPU Dual tangents vs GPU symbolic tangents (max relative)", dual, 1e-10, ""));
    c.metrics.push_back(at_most("rays whose fate or step count differs between JS, symbolic and dual", mismatched, 0, "rays"));
    data += "\"js_fixture\": {\"rays\": " + std::to_string(rays) + ", \"symbolic_vs_js\": " + num(js) + ", \"dual_vs_symbolic\": " +
            num(dual) + "}";
  }

  // 2. Schwarzschild shadow edge: d alpha / dR against sin(alpha) = sqrt(27) M sqrt(1 - 2M/R) / R.
  {
    const double bc = std::sqrt(27.0);
    std::vector<double> Rs = {4, 10, 50, 1000}, measured, analytic;
    double worst = 0;
    const std::vector<Theta> dirs = {unit(1 + rm::THETA_RADIUS), unit(1 + rm::THETA_U)};
    rm::BatchOptions opt = rk4({1}, 0.01, 1e9);
    opt.max_steps = 1;  // only the initial state and its tangent are needed
    for (double R : Rs) {
      Theta t{};
      t[0] = 1;
      t[1 + rm::THETA_RADIUS] = R;
      t[1 + rm::THETA_INCLINATION] = PI / 2;
      double u = 0.5 * bc / R;
      for (int it = 0; it < 30; it++) {
        t[1 + rm::THETA_U] = u;
        const Gpu::Tangents r = gpu.tangent_rays("schwarzschild", opt, {t}, dirs, 0);
        double b, db;
        impact_parameter(r.y0(0), r.S0(0), 1, b, db);
        const double f = std::fabs(b) - bc;
        u -= f / ((b < 0 ? -1 : 1) * db);
        if (std::fabs(f) < 1e-15) break;
      }
      t[1 + rm::THETA_U] = u;
      const Gpu::Tangents r = gpu.tangent_rays("schwarzschild", opt, {t}, dirs, 0);
      double b, bR, bu;
      impact_parameter(r.y0(0), r.S0(0), 0, b, bR);
      impact_parameter(r.y0(0), r.S0(0), 1, b, bu);
      const double m = -bR / bu / (1 + u * u);
      const double s = std::sqrt(1 - 2 / R), sinA = bc * s / R;
      const double dSin = bc * (-s / (R * R) + 1 / (R * R * R * s));
      const double want = (R >= 3 ? 1 : -1) * dSin / std::sqrt(1 - sinA * sinA);
      measured.push_back(m);
      analytic.push_back(want);
      worst = std::max(worst, std::fabs(m - want) / std::fabs(want));
    }
    c.metrics.push_back(at_most("Schwarzschild d(shadow angle)/dR, R = 4, 10, 50, 1000 M (max relative)", worst, 1e-9, ""));
    data += ", \"shadow_angle_vs_radius\": {\"R\": " + arr(Rs) + ", \"measured\": " + arr(measured) + ", \"analytic\": " +
            arr(analytic) + "}";
  }

  // 3. Kerr critical impact parameter: -(dphi/da|_b) / (dphi/db) from escaping equatorial rays at b = b_crit + delta.
  {
    const double R = 100;
    const std::vector<Theta> dirs = {unit(2 + rm::THETA_U), unit(1)};
    rm::BatchOptions init = rk4({1, 0}, 0.002, 1.01 * R);
    init.max_steps = 1;
    double worst = 0, worst_ratio = 1e300;
    std::string rows = "[";
    for (double a : {0.5, 0.9}) {
      for (bool prograde : {true, false}) {
        const double t = prograde ? std::acos(-a) : std::acos(a);
        const double bc = prograde ? -a + 6 * std::cos(t / 3) : -(a + 6 * std::cos(t / 3));
        const double want = prograde ? -1 - 2 * std::sin(t / 3) / std::sqrt(1 - a * a) : -(1 + 2 * std::sin(t / 3) / std::sqrt(1 - a * a));
        std::vector<double> errs;
        for (double delta : {1e-4, 1e-5}) {
          Theta th{};
          th[0] = 1;
          th[1] = a;
          th[2 + rm::THETA_RADIUS] = R;
          th[2 + rm::THETA_INCLINATION] = PI / 2;
          const double target = bc + (bc > 0 ? delta : -delta);
          double u = target / R;
          init.params[0] = 1;
          init.params[1] = a;
          for (int it = 0; it < 40; it++) {
            th[2 + rm::THETA_U] = u;
            const Gpu::Tangents r = gpu.tangent_rays("kerr", init, {th}, dirs, 0);
            double b, db;
            impact_parameter(r.y0(0), r.S0(0), 0, b, db);
            u -= (b - target) / db;
            if (std::fabs(b - target) < 1e-14 * std::fabs(target)) break;
          }
          th[2 + rm::THETA_U] = u;
          rm::BatchOptions opt = rk4({1, a}, 0.002, 1.01 * R);
          const Gpu::Tangents r = gpu.tangent_rays("kerr", opt, {th}, dirs, 0);
          double b, bu, ba;
          impact_parameter(r.y0(0), r.S0(0), 0, b, bu);
          impact_parameter(r.y0(0), r.S0(0), 1, b, ba);
          const double x = r.y(0)[1], y = r.y(0)[2];
          auto dphi = [&](int d) { return (x * r.S(0)[8 * d + 2] - y * r.S(0)[8 * d + 1]) / (x * x + y * y); };
          const double phiB = dphi(0) / bu, phiA = dphi(1) - phiB * ba;
          errs.push_back(r.fate(0) == rm::FATE_ESCAPED ? std::fabs(-phiA / phiB - want) / std::fabs(want) : 1e300);
        }
        worst = std::max(worst, errs[1]);
        worst_ratio = std::min(worst_ratio, errs[0] / errs[1]);
        rows += std::string(rows.size() > 1 ? ", " : "") + "{\"a\": " + num(a) + ", \"prograde\": " + (prograde ? "true" : "false") +
                ", \"db_crit_da\": " + num(want) + ", \"relative_error_at_delta\": {\"1e-4\": " + num(errs[0]) + ", \"1e-5\": " +
                num(errs[1]) + "}}";
      }
    }
    c.metrics.push_back(at_most("Kerr d b_crit/da, a = 0.5 and 0.9, both sides, at b - b_crit = 1e-5 (max relative)", worst, 1e-3, ""));
    c.metrics.push_back({"Kerr d b_crit/da error ratio between b - b_crit = 1e-4 and 1e-5 (linear convergence gives ~10)", "",
                         worst_ratio, 5, worst_ratio >= 5, "minimum over the four cases; must be at least 5"});
    data += ", \"critical_impact_parameter_vs_spin\": " + rows + "]";
  }

  // 4. Schwarzschild d(sweep)/du against the exact orbit integral at the ray's own start and end radii.
  {
    const double R = 1000;
    const std::vector<Theta> dirs = {unit(1 + rm::THETA_U), unit(1 + rm::THETA_V)};
    rm::BatchOptions init = rk4({1}, 0.002, 1.01 * R);
    init.max_steps = 1;
    std::vector<double> bs = {6, 10, 30, 100}, rel;
    double worst = 0;
    for (double bt : bs) {
      Theta th{};
      th[0] = 1;
      th[1 + rm::THETA_RADIUS] = R;
      th[1 + rm::THETA_INCLINATION] = PI / 2;
      double u = bt / R;
      for (int it = 0; it < 30; it++) {
        th[1 + rm::THETA_U] = u;
        const Gpu::Tangents r = gpu.tangent_rays("schwarzschild", init, {th}, dirs, 0);
        double b, db;
        impact_parameter(r.y0(0), r.S0(0), 0, b, db);
        u -= (std::fabs(b) - bt) / ((b < 0 ? -1 : 1) * db);
      }
      th[1 + rm::THETA_U] = u;
      const Gpu::Tangents r = gpu.tangent_rays("schwarzschild", rk4({1}, 0.002, 1.01 * R), {th}, dirs, 0);
      double b, db;
      impact_parameter(r.y0(0), r.S0(0), 0, b, db);
      const double ab = std::fabs(b), dab = (b < 0 ? -1 : 1) * db;
      const double* y = r.y(0);
      const double* S = r.S(0);
      const double x = y[1], yy = y[2], rOut = radius(y), rIn = radius(r.y0(0));
      // Traced backwards, E = -p_t < 0: the curve turns with sign(L) = -sign(b).
      const double measured = -(b < 0 ? -1 : 1) * (x * S[2] - yy * S[1]) / (x * x + yy * yy);
      const double hb = 1e-6 * ab;
      const double dSdb = (an::schwarzschild_sweep(ab + hb, 1, rIn, rOut) - an::schwarzschild_sweep(ab - hb, 1, rIn, rOut)) / (2 * hb);
      const double uo = 1 / rOut;
      const double dSdr = uo * uo / std::sqrt(1 / (ab * ab) - uo * uo + 2 * uo * uo * uo);
      const double drOut = (x * S[1] + yy * S[2] + y[3] * S[3]) / rOut;
      const double want = dSdb * dab + dSdr * drOut;
      rel.push_back(std::fabs(measured - want) / std::fabs(want));
      worst = std::max(worst, rel.back());
    }
    c.metrics.push_back(at_most("Schwarzschild d(sweep)/du vs exact orbit integral, b = 6, 10, 30, 100 M (max relative)", worst, 1e-6, ""));
    data += ", \"sweep_vs_impact_parameter\": {\"b\": " + arr(bs) + ", \"relative_error\": " + arr(rel) + "}";
  }

  // 5. Disk intensity and d/d(a, i) at model-sky samples: GPU duals (render_samples_dual_kernel) vs the JS image model.
  {
    const rm::json::Value fx = rm::json::parse(read_file_text(generated_dir + "/fixtures/image-samples.json"));
    const rm::json::Value& o = fx["options"];
    double worst_I = 0, worst_dI = 0;
    int disk = 0, mismatched = 0;
    for (const rm::json::Value& cs : fx["cases"].a) {
      const std::vector<double> theta = cs["theta"].numbers(), xi = cs["xi"].numbers(), I = cs["I"].numbers(), dI = cs["dI"].numbers();
      Theta base{};
      std::copy(theta.begin(), theta.end(), base.begin());
      std::vector<Theta> dirs;
      for (const rm::json::Value& row : cs["dirs"].a) {
        Theta t{};
        const std::vector<double> v = row.numbers();
        std::copy(v.begin(), v.end(), t.begin());
        dirs.push_back(t);
      }
      rm::BatchOptions opt = rk4({}, o["stepFraction"].num(), o["escapeRadius"].num());
      opt.min_step = o["minStep"].num();
      opt.max_steps = int(o["maxSteps"].num());
      opt.disk = 1;
      opt.crossing = cs["crossing"].str() == "hermite" ? rm::CROSSING_HERMITE : rm::CROSSING_LINEAR;
      const Gpu::SampleImage g = gpu.render_samples("kerr", opt, base, dirs, xi, fx["taperStart"].num(), true, Precision::Double);
      const int P = int(dirs.size());
      double sI = 0, sdI[8] = {};
      for (size_t s = 0; s < I.size(); s++) {
        sI = std::max(sI, std::fabs(I[s]));
        for (int d = 0; d < P; d++) sdI[d] = std::max(sdI[d], std::fabs(dI[P * s + d]));
      }
      for (size_t s = 0; s < I.size(); s++) {
        if (cs["fate"][s].str() != fate_name(g.meta[3 * s]) || int(cs["steps"][s].num()) != g.meta[3 * s + 1] ||
            int(cs["crossings"][s].num()) != g.meta[3 * s + 2])
          mismatched++;
        if (g.meta[3 * s] != rm::FATE_DISK) continue;
        disk++;
        worst_I = std::max(worst_I, std::fabs(g.I[s] - I[s]) / sI);
        for (int d = 0; d < P; d++) worst_dI = std::max(worst_dI, std::fabs(g.dI[P * s + d] - dI[P * s + d]) / sdI[d]);
      }
    }
    c.metrics.push_back(at_most("GPU disk intensity vs JS image model, " + std::to_string(disk) + " Kerr disk samples (relative to the largest)",
                                worst_I, 1e-10, ""));
    c.metrics.push_back(at_most("GPU dual d(intensity)/d(a, i) vs JS image model (relative to the largest per input)", worst_dI, 1e-10, ""));
    c.metrics.push_back(at_most("image samples whose fate or step count differs from JS", mismatched, 0, "samples"));
    data += ", \"image_samples\": {\"disk_samples\": " + std::to_string(disk) + ", \"intensity\": " + num(worst_I) +
            ", \"intensity_derivative\": " + num(worst_dI) + "}";
  }
  c.data = data + "}";
  return c;
}

/** Stratified jittered samples over [-L, L]^2: n x n cells, k x k sub-cells with one jittered point each. */
void stratified(double L, int n, int k, unsigned seed, std::vector<double>& xi, double& w) {
  const int side = n * k;
  xi.resize(size_t(2) * side * side);
  const double sub = 2 * L / side;
  w = sub * sub;
  unsigned s = seed;
  auto rnd = [&] { return (s = s * 1664525u + 1013904223u) / 4294967296.0; };
  for (int j = 0; j < side; j++)
    for (int i = 0; i < side; i++) {
      const size_t m = size_t(j) * side + i;
      xi[2 * m] = -L + (i + rnd()) * sub;
      xi[2 * m + 1] = -L + (j + rnd()) * sub;
    }
}

// --- Phase 4, step 5: the image-loss gradient with the critical-curve edge term ----------------

/** Observed image of n x n pixels (uas), circular Gaussian beam (FWHM, uas), theta_g (uas / M), PA (rad), flux scale A. */
struct ObsSpec {
  int n;
  double pixel, beam, theta_g, pa, A;
};

/**
 * The observed image (compiler/src/image-model.ts, observe) from samples xi with weights w and intensities I,
 * the Gaussian truncated at 5 sigma. With dI (P per sample) also the Jacobian jac[K p + k], K = P + 2: the P traced
 * directions, then theta_g and PA (exact through the kernel).
 */
void assemble(const ObsSpec& sp, const std::vector<double>& xi, const std::vector<double>& w, const std::vector<double>& I,
              const std::vector<double>* dI, int P, std::vector<double>& img, std::vector<double>* jac,
              const std::vector<double>* flow = nullptr) {
  // flow (3 P per sample: V_x, V_y per direction, then div V per direction): the samples move with velocity V
  // along each traced direction, so dI must be the total derivative dI/dtheta + V . grad I, and the kernel and
  // the weight contribute I grad B . V and B I div V (the change of variables of the image integral).
  const int n = sp.n, K = P + 2;
  img.assign(size_t(n) * n, 0.0);
  if (jac) jac->assign(size_t(n) * n * K, 0.0);
  const double sigma = sp.beam / (2 * std::sqrt(2 * std::log(2.0))), s2 = sigma * sigma, norm = 1 / (2 * PI * s2);
  const double c = std::cos(sp.pa), sn = std::sin(sp.pa), scale = sp.A * sp.theta_g * sp.theta_g;
  const int reach = int(std::ceil(5 * sigma / sp.pixel)) + 1;
  const size_t ns = I.size();
  for (size_t s = 0; s < ns; s++) {
    bool any = I[s] != 0;
    for (int d = 0; dI && d < P && !any; d++) any = (*dI)[P * s + d] != 0;
    if (!any) continue;
    const double xx = xi[2 * s], yy = xi[2 * s + 1];
    const double rx = c * xx - sn * yy, ry = sn * xx + c * yy, qx = -sn * xx - c * yy, qy = c * xx - sn * yy;
    const double X = sp.theta_g * rx, Y = sp.theta_g * ry;
    const int ci = int(std::floor(X / sp.pixel + n / 2.0)), cj = int(std::floor(Y / sp.pixel + n / 2.0));
    for (int j = std::max(0, cj - reach); j <= std::min(n - 1, cj + reach); j++) {
      const double dy = (j + 0.5 - n / 2.0) * sp.pixel - Y;
      for (int i = std::max(0, ci - reach); i <= std::min(n - 1, ci + reach); i++) {
        const double dx = (i + 0.5 - n / 2.0) * sp.pixel - X;
        const double r2 = dx * dx + dy * dy;
        if (r2 > 25 * s2) continue;
        const double base = scale * w[s] * norm * std::exp(-r2 / (2 * s2));
        const size_t p = size_t(j) * n + i;
        img[p] += base * I[s];
        if (!jac) continue;
        double* J = jac->data() + K * p;
        for (int d = 0; d < P; d++) J[d] += base * (*dI)[P * s + d];
        if (flow && I[s] != 0) {
          const double* F = flow->data() + size_t(3 * P) * s;
          for (int d = 0; d < P; d++) {
            const double Vx = F[2 * d], Vy = F[2 * d + 1];
            J[d] += base * I[s] * (F[2 * P + d] + sp.theta_g * (dx * (c * Vx - sn * Vy) + dy * (sn * Vx + c * Vy)) / s2);
          }
        }
        J[P] += base * I[s] * (2 / sp.theta_g + (dx * rx + dy * ry) / s2);
        J[P + 1] += base * I[s] * sp.theta_g * (dx * qx + dy * qy) / s2;
      }
    }
  }
}

/**
 * Sample flow near the critical curve, a change of variables of the image integral that makes the gradient
 * estimator converge. Near the curve the n-th subring has width ~ exp(-gamma n), so dI/dtheta at a fixed sky point
 * grows like 1/(distance to the curve): its integral converges but its square does not, and the quadrature of the
 * derivative has unbounded variance. Moving the samples with the curve removes the 1/distance part (what remains
 * grows like log distance, whose square is integrable).
 *   V_d(xi) = v_d(phi) g(rho - rho_c(phi)) e_rho   about the centre O,
 * rho_c(phi) the capture boundary at theta_0, v_d = d rho_c / d theta_d, g(x) = (1 - (x/W)^2)^2 for |x| < W.
 * Any Lipschitz V gives the same gradient in exact arithmetic; the choice only changes the variance.
 */
struct RingFlow {
  double ox = 0, oy = 0, W = 1.5;
  int n = 0, P = 0;
  std::vector<double> rho, v;  // rho_c at phi_k = -pi + 2 pi k / n; v[P k + d]
  /** V (2 per direction) and div V (1 per direction) at (x, y): out[2 d], out[2 d + 1], out[2 P + d]. */
  void at(double x, double y, double* out) const {
    for (int k = 0; k < 3 * P; k++) out[k] = 0;
    if (n == 0) return;
    const double dx = x - ox, dy = y - oy, r = std::hypot(dx, dy);
    if (r < 1e-9) return;
    const double f = (std::atan2(dy, dx) + PI) / (2 * PI) * n;
    const int k = int(std::floor(f));
    const double t = f - k;
    const int k0 = ((k % n) + n) % n, k1 = (k0 + 1) % n;
    const double xx = r - ((1 - t) * rho[k0] + t * rho[k1]);
    if (std::fabs(xx) >= W) return;
    const double q = 1 - (xx / W) * (xx / W), g = q * q, gp = -4 * xx / (W * W) * q;
    for (int d = 0; d < P; d++) {
      const double vd = (1 - t) * v[P * k0 + d] + t * v[P * k1 + d];
      out[2 * d] = vd * g * dx / r;
      out[2 * d + 1] = vd * g * dy / r;
      out[2 * P + d] = vd * (gp + g / r);
    }
  }
};

/** Adds the edge term: at boundary points xi* with coefficients coef[P e + d] (Delta y (I_left - I_right) times the velocity d x_star / d theta_d). */
void add_edge_term(const ObsSpec& sp, const std::vector<double>& pts, const std::vector<double>& coef, int P, std::vector<double>& jac) {
  const int n = sp.n, K = P + 2;
  const double sigma = sp.beam / (2 * std::sqrt(2 * std::log(2.0))), s2 = sigma * sigma, norm = 1 / (2 * PI * s2);
  const double c = std::cos(sp.pa), sn = std::sin(sp.pa), scale = sp.A * sp.theta_g * sp.theta_g;
  const int reach = int(std::ceil(5 * sigma / sp.pixel)) + 1;
  for (size_t e = 0; e < pts.size() / 2; e++) {
    const double X = sp.theta_g * (c * pts[2 * e] - sn * pts[2 * e + 1]), Y = sp.theta_g * (sn * pts[2 * e] + c * pts[2 * e + 1]);
    const int ci = int(std::floor(X / sp.pixel + n / 2.0)), cj = int(std::floor(Y / sp.pixel + n / 2.0));
    for (int j = std::max(0, cj - reach); j <= std::min(n - 1, cj + reach); j++) {
      const double dy = (j + 0.5 - n / 2.0) * sp.pixel - Y;
      for (int i = std::max(0, ci - reach); i <= std::min(n - 1, ci + reach); i++) {
        const double dx = (i + 0.5 - n / 2.0) * sp.pixel - X;
        const double r2 = dx * dx + dy * dy;
        if (r2 > 25 * s2) continue;
        const double B = scale * norm * std::exp(-r2 / (2 * s2));
        for (int d = 0; d < P; d++) jac[K * (size_t(j) * n + i) + d] += B * coef[P * e + d];
      }
    }
  }
}

double loss(const std::vector<double>& img, const std::vector<double>& data, double sigma_n) {
  double L = 0;
  for (size_t p = 0; p < img.size(); p++) L += 0.5 * (img[p] - data[p]) * (img[p] - data[p]) / (sigma_n * sigma_n);
  return L;
}

}  // namespace

int run_gate1(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys, const std::string& out_path) {
  Gpu gpu(tracer, catalog);
  using Theta = Gpu::Theta;
  const double R = 1000, L = 22, taper = 0.75;
  const int base_n = 512, refine = 8, dense_n = 2048, edge_rows = 1024, edge_cols = 2048, bisect_iters = 48;
  const double h_vel = 1e-5;
  struct Scene {
    double a, incl;
  };
  const std::vector<Scene> scenes = {{0.94, 60 * DEG}, {0.5, 30 * DEG}};
  const double beams[2] = {20, 7};
  const char* pnames[4] = {"spin", "inclination", "theta_g", "position angle"};
  std::vector<double> xi_dense, xi_dense2, xi_base;
  double w_dense, w_base;
  stratified(L, dense_n, 1, 101u, xi_dense, w_dense);
  stratified(L, dense_n, 1, 707u, xi_dense2, w_dense);
  stratified(L, base_n, 1, 202u, xi_base, w_base);
  const std::vector<double> wd(xi_dense.size() / 2, w_dense);
  bool all_pass = true;
  std::string metrics_json = "[", scenes_json = "[";
  for (const Scene& sc : scenes) {
    std::printf("scene a = %.2f, i = %.0f deg\n", sc.a, sc.incl / DEG);
    auto options = [&](double a) {
      rm::BatchOptions o = rk4({1, a}, 0.01, 1.01 * R);
      o.disk = 1;
      o.max_steps = 200000;
      o.crossing = rm::CROSSING_HERMITE;
      return o;
    };
    auto theta_at = [&](double a, double incl) {
      Theta t{};
      t[0] = 1;
      t[1] = a;
      t[2 + rm::THETA_RADIUS] = R;
      t[2 + rm::THETA_INCLINATION] = incl;
      return t;
    };
    const Theta t0 = theta_at(sc.a, sc.incl);
    const rm::BatchOptions o0 = options(sc.a);
    // Tangent directions: spin, inclination, and the sample's own sky position (u, v) for the flow term.
    std::vector<Theta> dirs(4, Theta{});
    dirs[0][1] = 1;
    dirs[1][2 + rm::THETA_INCLINATION] = 1;
    dirs[2][2 + rm::THETA_U] = 1;
    dirs[3][2 + rm::THETA_V] = 1;
    const auto dense = [&](double a, double incl) {
      return gpu.render_samples("kerr", options(a), theta_at(a, incl), {}, xi_dense, taper, false, Precision::Double).I;
    };

    // --- The method: refined quadrature with tangents, plus the edge term. ---
    const auto t_m0 = std::chrono::steady_clock::now();
    // Ring flow: the capture boundary rho_c(phi) about its centroid, and d rho_c / d(a, i) by central differences.
    RingFlow flow;
    flow.P = 2;
    flow.n = 1440;
    flow.W = 1.5;  // MODELLING CHOICE of the estimator (changes only its variance)
    auto boundary = [&](double ox, double oy, double a, double incl, size_t& bad) {
      std::vector<double> rays;
      for (int k = 0; k < flow.n; k++) {
        const double phi = -PI + 2 * PI * k / flow.n;
        rays.insert(rays.end(), {ox, oy, std::cos(phi), std::sin(phi)});
      }
      const std::vector<double> out = gpu.curve_bisect("kerr", options(a), theta_at(a, incl), rays, 0.5, 14, 52);
      std::vector<double> rho(flow.n);
      for (int k = 0; k < flow.n; k++) {
        rho[k] = out[2 * k];
        if (out[2 * k + 1] != 0) bad++;
      }
      return rho;
    };
    size_t bad = 0;
    {
      const std::vector<double> r0 = boundary(0, 0, sc.a, sc.incl, bad);
      double cx = 0, cy = 0;
      for (int k = 0; k < flow.n; k++) {
        const double phi = -PI + 2 * PI * k / flow.n;
        cx += r0[k] * std::cos(phi) / flow.n;
        cy += r0[k] * std::sin(phi) / flow.n;
      }
      flow.ox = cx;
      flow.oy = cy;
    }
    flow.rho = boundary(flow.ox, flow.oy, sc.a, sc.incl, bad);
    double flow_check = 0, vmax[2] = {0, 0};
    {
      std::vector<double> v_coarse(2 * flow.n);
      flow.v.assign(2 * flow.n, 0.0);
      for (int d = 0; d < 2; d++)
        for (double h : {1e-4, 1e-3}) {
          const std::vector<double> rp = d == 0 ? boundary(flow.ox, flow.oy, sc.a + h, sc.incl, bad)
                                                : boundary(flow.ox, flow.oy, sc.a, sc.incl + h, bad);
          const std::vector<double> rm_ = d == 0 ? boundary(flow.ox, flow.oy, sc.a - h, sc.incl, bad)
                                                 : boundary(flow.ox, flow.oy, sc.a, sc.incl - h, bad);
          for (int k = 0; k < flow.n; k++) (h < 5e-4 ? flow.v : v_coarse)[2 * k + d] = (rp[k] - rm_[k]) / (2 * h);
        }
      for (int k = 0; k < flow.n; k++)
        for (int d = 0; d < 2; d++) vmax[d] = std::max(vmax[d], std::fabs(flow.v[2 * k + d]));
      for (int k = 0; k < flow.n; k++)
        for (int d = 0; d < 2; d++) flow_check = std::max(flow_check, std::fabs(flow.v[2 * k + d] - v_coarse[2 * k + d]) / vmax[d]);
    }
    const double rho_min = *std::min_element(flow.rho.begin(), flow.rho.end());
    const double rho_max = *std::max_element(flow.rho.begin(), flow.rho.end());
    std::printf("  critical curve: centre (%.4f, %.4f) M, rho_c %.4f .. %.4f M, max |d rho_c/da| %.4f, |d rho_c/di| %.4f M/rad, "
                "velocity at h 1e-4 vs 1e-3 %.1e of max, %zu failed bisections\n",
                flow.ox, flow.oy, rho_min, rho_max, vmax[0], vmax[1], flow_check, bad);
    // Per-sample derivatives: plain (fixed sky point) and with the flow (total derivative along the moving sample).
    auto split = [&](const Gpu::SampleImage& S, const std::vector<double>& xs, size_t s, double* plain, double* moving, double* F) {
      flow.at(xs[2 * s], xs[2 * s + 1], F);
      for (int d = 0; d < 2; d++) {
        plain[d] = S.dI[4 * s + d];
        moving[d] = S.dI[4 * s + d] + (F[2 * d] * S.dI[4 * s + 2] + F[2 * d + 1] * S.dI[4 * s + 3]) / R;
      }
    };
    const Gpu::SampleImage base = gpu.render_samples("kerr", o0, t0, dirs, xi_base, taper, true, Precision::Double);
    // Refine cells whose branch differs from a 4-neighbour's, or whose |dI| (with the flow) is in the top 3%, dilated by one cell.
    auto branch = [&](const std::vector<int>& meta, size_t s) { return meta[3 * s] == rm::FATE_DISK ? 1 + meta[3 * s + 2] : 0; };
    const size_t nb = size_t(base_n) * base_n;
    std::vector<char> flag(nb, 0);
    std::vector<double> mags, dIf_b(2 * nb), F_b(6 * nb);
    for (size_t s = 0; s < nb; s++) {
      double plain[2];
      split(base, xi_base, s, plain, &dIf_b[2 * s], &F_b[6 * s]);
      mags.push_back(std::max(std::fabs(dIf_b[2 * s]), std::fabs(dIf_b[2 * s + 1])));
    }
    std::vector<double> sorted = mags;
    std::sort(sorted.begin(), sorted.end());
    const double cut = sorted[size_t(0.97 * (nb - 1))];
    for (int j = 0; j < base_n; j++)
      for (int i = 0; i < base_n; i++) {
        const size_t s = size_t(j) * base_n + i;
        bool f = mags[s] > cut;
        const int b = branch(base.meta, s);
        if (i > 0) f = f || branch(base.meta, s - 1) != b;
        if (i + 1 < base_n) f = f || branch(base.meta, s + 1) != b;
        if (j > 0) f = f || branch(base.meta, s - base_n) != b;
        if (j + 1 < base_n) f = f || branch(base.meta, s + base_n) != b;
        flag[s] = f;
      }
    std::vector<char> dil = flag;
    for (int j = 0; j < base_n; j++)
      for (int i = 0; i < base_n; i++)
        if (flag[size_t(j) * base_n + i])
          for (int dj = -1; dj <= 1; dj++)
            for (int di = -1; di <= 1; di++) {
              const int jj = j + dj, ii = i + di;
              if (jj >= 0 && jj < base_n && ii >= 0 && ii < base_n) dil[size_t(jj) * base_n + ii] = 1;
            }
    std::vector<double> xi_m, w_m, I_m, dI_m, dIf_m, F_m, xi_ref;
    const double cell = 2 * L / base_n, sub = cell / refine;
    unsigned seed = 303u;
    auto rnd = [&] { return (seed = seed * 1664525u + 1013904223u) / 4294967296.0; };
    size_t refined_cells = 0;
    for (int j = 0; j < base_n; j++)
      for (int i = 0; i < base_n; i++) {
        const size_t s = size_t(j) * base_n + i;
        if (!dil[s]) {
          xi_m.push_back(xi_base[2 * s]);
          xi_m.push_back(xi_base[2 * s + 1]);
          w_m.push_back(w_base);
          I_m.push_back(base.I[s]);
          dI_m.push_back(base.dI[4 * s]);
          dI_m.push_back(base.dI[4 * s + 1]);
          dIf_m.insert(dIf_m.end(), {dIf_b[2 * s], dIf_b[2 * s + 1]});
          F_m.insert(F_m.end(), F_b.begin() + 6 * s, F_b.begin() + 6 * s + 6);
          continue;
        }
        refined_cells++;
        for (int b = 0; b < refine; b++)
          for (int a = 0; a < refine; a++) {
            xi_ref.push_back(-L + i * cell + (a + rnd()) * sub);
            xi_ref.push_back(-L + j * cell + (b + rnd()) * sub);
          }
      }
    const Gpu::SampleImage ref = gpu.render_samples("kerr", o0, t0, dirs, xi_ref, taper, true, Precision::Double);
    for (size_t s = 0; s < ref.I.size(); s++) {
      xi_m.push_back(xi_ref[2 * s]);
      xi_m.push_back(xi_ref[2 * s + 1]);
      w_m.push_back(sub * sub);
      I_m.push_back(ref.I[s]);
      double plain[2], moving[2], F[6];
      split(ref, xi_ref, s, plain, moving, F);
      dI_m.insert(dI_m.end(), {plain[0], plain[1]});
      dIf_m.insert(dIf_m.end(), {moving[0], moving[1]});
      F_m.insert(F_m.end(), F, F + 6);
    }
    // Edge term: branch changes between neighbours along rows of a regular grid, located by bisection;
    // their velocity dx*/dtheta from bisection at theta +- h_vel (central difference).
    std::vector<double> grid_xi;
    const double dxg = 2 * L / edge_cols, dyg = 2 * L / edge_rows;
    for (int j = 0; j < edge_rows; j++)
      for (int i = 0; i < edge_cols; i++) {
        grid_xi.push_back(-L + (i + 0.5) * dxg);
        grid_xi.push_back(-L + (j + 0.5) * dyg);
      }
    const Gpu::SampleImage grid = gpu.render_samples("kerr", o0, t0, {}, grid_xi, taper, false, Precision::Double);
    std::vector<double> segs;
    for (int j = 0; j < edge_rows; j++)
      for (int i = 0; i + 1 < edge_cols; i++) {
        const size_t s = size_t(j) * edge_cols + i;
        const int b0 = branch(grid.meta, s), b1 = branch(grid.meta, s + 1);
        if (b0 == b1) continue;
        segs.insert(segs.end(), {grid_xi[2 * s], grid_xi[2 * s + 2], grid_xi[2 * s + 1], double(b0)});
      }
    const size_t ne = segs.size() / 4;
    const std::vector<double> at0 = gpu.edge_bisect("kerr", o0, t0, segs, bisect_iters, taper);
    std::vector<double> pts(2 * ne), coef(2 * ne, 0.0), coef_half(2 * ne, 0.0), coef_plain(2 * ne, 0.0);
    size_t dropped = 0;
    std::vector<char> ok(ne, 1);
    for (size_t e = 0; e < ne; e++) {
      pts[2 * e] = at0[4 * e];
      pts[2 * e + 1] = segs[4 * e + 2];
      if (at0[4 * e + 3] != 0) ok[e] = 0;
    }
    // With the flow the edge moves relative to the samples: the row crossing of a sample row moves at
    // dx*/dtheta + (dx*/dy) V_y - V_x. dx*/dy from bisection on the rows y +- 1e-5.
    std::vector<double> slope(ne, 0.0);
    {
      const double dyb = 1e-5;
      std::vector<double> sp_ = segs, sm_ = segs;
      for (size_t e = 0; e < ne; e++) {
        sp_[4 * e + 2] += dyb;
        sm_[4 * e + 2] -= dyb;
      }
      const std::vector<double> up = gpu.edge_bisect("kerr", o0, t0, sp_, bisect_iters, taper);
      const std::vector<double> dn = gpu.edge_bisect("kerr", o0, t0, sm_, bisect_iters, taper);
      for (size_t e = 0; e < ne; e++) {
        double F[6];
        flow.at(pts[2 * e], pts[2 * e + 1], F);
        const bool moving = F[0] != 0 || F[1] != 0 || F[2] != 0 || F[3] != 0;
        if (moving && (up[4 * e + 3] != 0 || dn[4 * e + 3] != 0)) ok[e] = 0;
        slope[e] = (up[4 * e] - dn[4 * e]) / (2 * dyb);
      }
    }
    for (int d = 0; d < 2; d++) {
      Theta tp = t0, tm = t0;
      tp[d == 0 ? 1 : 2 + rm::THETA_INCLINATION] += h_vel;
      tm[d == 0 ? 1 : 2 + rm::THETA_INCLINATION] -= h_vel;
      const rm::BatchOptions op = options(d == 0 ? sc.a + h_vel : sc.a), om = options(d == 0 ? sc.a - h_vel : sc.a);
      const std::vector<double> ap = gpu.edge_bisect("kerr", op, tp, segs, bisect_iters, taper);
      const std::vector<double> am = gpu.edge_bisect("kerr", om, tm, segs, bisect_iters, taper);
      for (size_t e = 0; e < ne; e++) {
        if (ap[4 * e + 3] != 0 || am[4 * e + 3] != 0) ok[e] = 0;
        const double v = (ap[4 * e] - am[4 * e]) / (2 * h_vel);
        double F[6];
        flow.at(pts[2 * e], pts[2 * e + 1], F);
        coef_plain[2 * e + d] = dyg * (at0[4 * e + 1] - at0[4 * e + 2]) * v;
        coef[2 * e + d] = dyg * (at0[4 * e + 1] - at0[4 * e + 2]) * (v + slope[e] * F[2 * d + 1] - F[2 * d]);
      }
    }
    for (size_t e = 0; e < ne; e++)
      if (!ok[e]) {
        dropped++;
        coef[2 * e] = coef[2 * e + 1] = coef_plain[2 * e] = coef_plain[2 * e + 1] = 0;
      }
    // Resolution check: the edge term from every other row only (twice the row spacing).
    for (size_t e = 0; e < ne; e++) {
      const int row = int(std::floor((segs[4 * e + 2] + L) / dyg));
      if (row % 2 == 0) {
        coef_half[2 * e] = 2 * coef[2 * e];
        coef_half[2 * e + 1] = 2 * coef[2 * e + 1];
      }
    }
    const double method_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_m0).count();
    std::printf("  method: %zu base + %zu refined cells (%zu samples), %zu edge crossings (%zu dropped), %.0f s\n", nb, refined_cells,
                I_m.size(), ne, dropped, method_s);

    // --- Truth: central differences of the loss computed from 2048^2-sample images. ---
    // Spin and inclination: h = 3.2e-2, 1.6e-2, 8e-3 on two independent jitter sets. At fixed samples the
    // FD noise comes from samples whose branch flips (photon-ring subrings), and its relative size falls like
    // 1/sqrt(h); Richardson extrapolation of (3.2e-2, 1.6e-2) removes the O(h^2) bias, and (1.6e-2, 8e-3)
    // checks it.
    const double a_true = sc.a + 0.04, i_true = sc.incl + 3 * DEG;
    const std::vector<double> I_true = dense(a_true, i_true), I_d0 = dense(sc.a, sc.incl);
    const std::vector<double> I_d0b = gpu.render_samples("kerr", o0, t0, {}, xi_dense2, taper, false, Precision::Double).I;
    const double h_fd[3] = {3.2e-2, 1.6e-2, 8e-3};
    std::vector<double> Ip[2][2][3], Im[2][2][3];  // [seed][param][h]
    for (int sd = 0; sd < 2; sd++)
      for (int d = 0; d < 2; d++)
        for (int hh = 0; hh < 3; hh++) {
          const double h = h_fd[hh];
          const std::vector<double>& xs = sd ? xi_dense2 : xi_dense;
          const double ap = d == 0 ? sc.a + h : sc.a, am = d == 0 ? sc.a - h : sc.a;
          const double ip = d == 1 ? sc.incl + h : sc.incl, im = d == 1 ? sc.incl - h : sc.incl;
          Ip[sd][d][hh] = gpu.render_samples("kerr", options(ap), theta_at(ap, ip), {}, xs, taper, false, Precision::Double).I;
          Im[sd][d][hh] = gpu.render_samples("kerr", options(am), theta_at(am, im), {}, xs, taper, false, Precision::Double).I;
        }
    std::string beams_json = "[";
    for (double beam : beams) {
      const ObsSpec sp0{64, 3, beam, 3.8, 0.4, 1};
      ObsSpec spt = sp0;
      spt.theta_g *= 1.03;
      spt.pa += 0.1;
      std::vector<double> data, img, jac;
      assemble(spt, xi_dense, wd, I_true, nullptr, 0, data, nullptr);
      double dmax = 0;
      for (double v : data) dmax = std::max(dmax, v);
      const double sigma_n = 0.01 * dmax;  // MODELLING CHOICE for the test: uniform noise at 1% of the peak
      // Method gradient (with the ring flow): residuals of the method's own image.
      assemble(sp0, xi_m, w_m, I_m, &dIf_m, 2, img, &jac, &F_m);
      std::vector<double> jac_edge(jac.size(), 0.0), jac_half(jac.size(), 0.0), img_d0;
      add_edge_term(sp0, pts, coef, 2, jac_edge);
      add_edge_term(sp0, pts, coef_half, 2, jac_half);
      // Diagnostics: the plain estimator (fixed sky points) on the same samples, and the flow estimator on the
      // unrefined base samples only.
      std::vector<double> img_p, jac_p, jac_edge_p, img_b, jac_b;
      assemble(sp0, xi_m, w_m, I_m, &dI_m, 2, img_p, &jac_p);
      jac_edge_p.assign(jac_p.size(), 0.0);
      add_edge_term(sp0, pts, coef_plain, 2, jac_edge_p);
      assemble(sp0, xi_base, std::vector<double>(nb, w_base), base.I, &dIf_b, 2, img_b, &jac_b, &F_b);
      assemble(sp0, xi_dense, wd, I_d0, nullptr, 0, img_d0, nullptr);
      // g: the method (its own residuals). gj: the same Jacobian with the dense image's residuals (Jacobian error only).
      double g[4] = {0, 0, 0, 0}, g_edge[2] = {0, 0}, g_half[2] = {0, 0}, gj[4] = {0, 0, 0, 0}, g_plain[2] = {0, 0}, g_base[2] = {0, 0};
      double res_diff = 0, res_norm = 0;
      for (size_t p = 0; p < img.size(); p++) {
        const double r = (img[p] - data[p]) / (sigma_n * sigma_n);
        const double rd = (img_d0[p] - data[p]) / (sigma_n * sigma_n);
        const double rb = (img_b[p] - data[p]) / (sigma_n * sigma_n);
        res_diff += (r - rd) * (r - rd);
        res_norm += rd * rd;
        for (int k = 0; k < 4; k++) {
          g[k] += r * jac[4 * p + k];
          gj[k] += rd * (jac[4 * p + k] + (k < 2 ? jac_edge[4 * p + k] : 0));
        }
        for (int k = 0; k < 2; k++) {
          g_edge[k] += r * jac_edge[4 * p + k];
          g_half[k] += r * jac_half[4 * p + k];
          g_plain[k] += r * (jac_p[4 * p + k] + jac_edge_p[4 * p + k]);
          g_base[k] += rb * (jac_b[4 * p + k] + jac_edge[4 * p + k]);
        }
      }
      const double residual_mismatch = std::sqrt(res_diff / res_norm);
      // Truth.
      auto L_of = [&](const std::vector<double>& I, const ObsSpec& sp, int sd) {
        std::vector<double> im;
        assemble(sp, sd ? xi_dense2 : xi_dense, wd, I, nullptr, 0, im, nullptr);
        return loss(im, data, sigma_n);
      };
      // fd[seed][param][h]: spin and inclination at the three h; theta_g and PA at 1e-4 (relative) and 5e-5 in slots 0, 1.
      double fd[2][4][3] = {};
      for (int sd = 0; sd < 2; sd++) {
        for (int d = 0; d < 2; d++)
          for (int hh = 0; hh < 3; hh++)
            fd[sd][d][hh] = (L_of(Ip[sd][d][hh], sp0, sd) - L_of(Im[sd][d][hh], sp0, sd)) / (2 * h_fd[hh]);
        for (int k = 2; k < 4; k++)
          for (int hh = 0; hh < 2; hh++) {
            const double h = (k == 2 ? 1e-4 * sp0.theta_g : 1e-4) / (hh ? 2 : 1);
            ObsSpec spp = sp0, spm = sp0;
            (k == 2 ? spp.theta_g : spp.pa) += h;
            (k == 2 ? spm.theta_g : spm.pa) -= h;
            const std::vector<double>& I0 = sd ? I_d0b : I_d0;
            fd[sd][k][hh] = (L_of(I0, spp, sd) - L_of(I0, spm, sd)) / (2 * h);
          }
      }
      std::string params_json = "[";
      for (int k = 0; k < 4; k++) {
        const double total = k < 2 ? g[k] + g_edge[k] : g[k];
        // Per seed: spin and inclination, Richardson of (3.2e-2, 1.6e-2), checked against Richardson of (1.6e-2, 8e-3);
        // theta_g and PA, FD at the smaller h. Truth = mean of the two seeds; its standard error |t1 - t2| / 2.
        double ts[2], tc[2];
        for (int sd = 0; sd < 2; sd++) {
          ts[sd] = k < 2 ? (4 * fd[sd][k][1] - fd[sd][k][0]) / 3 : fd[sd][k][1];
          tc[sd] = k < 2 ? (4 * fd[sd][k][2] - fd[sd][k][1]) / 3 : fd[sd][k][1];
        }
        const double truth = 0.5 * (ts[0] + ts[1]);
        const double truth_se = 0.5 * std::fabs(ts[0] - ts[1]) / std::fabs(truth);
        const double truth_bias = std::fabs(0.5 * (tc[0] + tc[1]) - truth) / std::fabs(truth);
        const double rel = std::fabs(total - truth) / std::fabs(truth);
        const double rel_smooth = k < 2 ? std::fabs(g[k] - truth) / std::fabs(truth) : rel;
        const double rel_jac = std::fabs(gj[k] - truth) / std::fabs(truth);
        const double rel_half = k < 2 ? std::fabs(g[k] + g_half[k] - truth) / std::fabs(truth) : rel;
        const double rel_plain = k < 2 ? std::fabs(g_plain[k] - truth) / std::fabs(truth) : rel;
        const double rel_base = k < 2 ? std::fabs(g_base[k] - truth) / std::fabs(truth) : rel;
        const double fd_spread = std::fabs(fd[0][k][0] - fd[0][k][1]) / std::fabs(truth);
        const bool pass = rel <= 0.01, truth_ok = truth_se <= 0.003;
        all_pass = all_pass && pass && truth_ok;
        std::printf("  beam %4.0f uas, d L/d(%s): method %.6e (smooth %.6e + edge %.6e), truth %.6e (seeds %.6e, %.6e; "
                    "standard error %.1e %s, Richardson (1.6e-2, 8e-3) differs by %.1e); error %.3e %s | without edge %.3e, edge on "
                    "half the rows %.3e, Jacobian with dense residuals %.3e, residual mismatch %.2e | without the ring flow %.3e, "
                    "flow on the 512^2 base only %.3e\n",
                    beam, pnames[k], total, g[k], k < 2 ? g_edge[k] : 0.0, truth, ts[0], ts[1], truth_se, truth_ok ? "ok" : "TOO LARGE",
                    truth_bias, rel, pass ? "PASS" : "FAIL", rel_smooth, rel_half, rel_jac, residual_mismatch, rel_plain, rel_base);
        char name[200];
        std::snprintf(name, sizeof name, "a = %.2f, i = %.0f deg, beam %.0f uas: d(loss)/d(%s) vs FD of 2048^2-sample images", sc.a,
                      sc.incl / DEG, beam, pnames[k]);
        metrics_json += std::string(metrics_json.size() > 1 ? ", " : "") +
                        metric_json(at_most(name, rel, 0.01, "relative",
                                            "gate 1; truth = mean over two jitter sets of Richardson of FD at 3.2e-2 and 1.6e-2 (spin, "
                                            "inclination), standard error " + num(truth_se) + ", Richardson (1.6e-2, 8e-3) differs by " +
                                            num(truth_bias)));
        std::snprintf(name, sizeof name, "a = %.2f, i = %.0f deg, beam %.0f uas: d(loss)/d(%s) truth standard error (two jitter sets)", sc.a,
                      sc.incl / DEG, beam, pnames[k]);
        metrics_json += ", " + metric_json(at_most(name, truth_se, 0.003, "relative", "the gate is meaningful only if the truth is known to well under 1%"));
        params_json += std::string(k ? ", " : "") + "{\"parameter\": " + str(pnames[k]) + ", \"method\": " + num(total) +
                       ", \"smooth\": " + num(g[k]) + ", \"edge\": " + num(k < 2 ? g_edge[k] : 0.0) + ", \"fd\": [[" +
                       num(fd[0][k][0]) + ", " + num(fd[0][k][1]) + ", " + num(fd[0][k][2]) + "], [" + num(fd[1][k][0]) + ", " +
                       num(fd[1][k][1]) + ", " + num(fd[1][k][2]) + "]], \"truth_per_seed\": [" + num(ts[0]) + ", " + num(ts[1]) +
                       "], \"truth\": " + num(truth) + ", \"truth_standard_error\": " + num(truth_se) +
                       ", \"truth_richardson_check\": " + num(truth_bias) + ", \"relative_error\": " + num(rel) +
                       ", \"relative_error_without_edge\": " + num(rel_smooth) + ", \"relative_error_edge_half_rows\": " +
                       num(rel_half) + ", \"relative_error_jacobian_with_dense_residuals\": " + num(rel_jac) +
                       ", \"residual_mismatch\": " + num(residual_mismatch) + ", \"fd_spread\": " + num(fd_spread) +
                       ", \"relative_error_without_ring_flow\": " + num(rel_plain) + ", \"relative_error_flow_base_only\": " +
                       num(rel_base) + "}";
      }
      beams_json += std::string(beams_json.size() > 1 ? ", " : "") + "{\"beam_fwhm_uas\": " + num(beam) + ", \"parameters\": " +
                    params_json + "]}";
    }
    scenes_json += std::string(scenes_json.size() > 1 ? ", " : "") + "{\"spin\": " + num(sc.a) + ", \"inclination_deg\": " +
                   num(sc.incl / DEG) + ", \"refined_cells\": " + std::to_string(refined_cells) + ", \"method_samples\": " +
                   std::to_string(I_m.size()) + ", \"edge_crossings\": " + std::to_string(ne) + ", \"edge_crossings_dropped\": " +
                   std::to_string(dropped) + ", \"method_seconds\": " + num(method_s) + ", \"ring_flow\": {\"centre\": [" + num(flow.ox) +
                   ", " + num(flow.oy) + "], \"rho_c_min\": " + num(rho_min) + ", \"rho_c_max\": " + num(rho_max) +
                   ", \"max_drho_da\": " + num(vmax[0]) + ", \"max_drho_di\": " + num(vmax[1]) + ", \"velocity_h_check\": " +
                   num(flow_check) + ", \"failed_bisections\": " + std::to_string(bad) + "}, \"beams\": " + beams_json + "]}";
  }
  const std::string commit = shell("git rev-parse --short HEAD");
  const bool dirty = !shell("git status --porcelain -- . \":(exclude)validation/results\" \":(exclude)docs/benchmarks\"").empty();
  char date[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  std::ofstream(out_path)
      << "{\n  \"check\": \"image-gradient-gate\",\n  \"title\": \"Gate 1: image-loss gradient vs finite differences of supersampled "
         "images\",\n  \"description\": \"Kerr, camera at 1000 M, model sky +-22 M, thin disk g^4 F tau, RK4 step fraction 0.01 with "
         "the cubic Hermite disk crossing, fp64. Observed 64 x 64 image of 3 uas pixels, theta_g = 3.8 uas/M, PA = 0.4, Gaussian beam "
         "20 and 7 uas FWHM (truncated at 5 sigma). Data: the 2048^2-sample image at (a + 0.04, i + 3 deg, 1.03 theta_g, PA + 0.1); "
         "loss 1/2 sum (I - D)^2 / sigma^2 with sigma = 1% of the data peak (a test choice). Method: 512^2 stratified samples with "
         "forward-mode (dual) tangents, cells near branch changes or in the top 3% of |dI| (dilated by one) refined 8 x 8. Samples "
         "move with the ring flow V = (d rho_c/d theta) g(rho - rho_c) e_rho about the critical curve's centroid (rho_c by 52 "
         "bisections on capture along 1440 directions, velocity by central differences at 1e-4, g = (1 - (x/1.5 M)^2)^2): per "
         "sample the total derivative dI/dtheta + V . grad I from tangents along (a, i, u, v), plus I grad B . V and B I div V "
         "(a change of variables of the image integral; it removes the 1/distance growth of dI/dtheta at the photon-ring "
         "subrings, whose square is not integrable). Plus the edge term sum over row crossings of Delta y (I_left - I_right) "
         "(dx*/dtheta + (dx*/dy) V_y - V_x), crossings found between neighbours of a "
         "2048 x 1024 regular grid by 48 bisections, dx*/dtheta by bisection at theta +- 1e-5. Truth: central differences of the "
         "loss of 2048^2-sample images on two independent jitter sets, averaged (spin and inclination: Richardson extrapolation "
         "of h = 3.2e-2 and 1.6e-2, checked against 1.6e-2 and 8e-3; theta_g and PA: 5e-5 relative); the gate also requires the "
         "truth's standard error from the two sets to be at most 0.3%. Diagnostics per parameter: the error without the edge term, with the edge term from every other "
         "row, of the method's Jacobian contracted with the dense image's residuals (Jacobian error alone), without the ring "
         "flow (fixed sky points, the estimator of the first two runs), and with the flow on the unrefined 512^2 base only.\",\n  \"pass\": "
      << (all_pass ? "true" : "false") << ",\n  \"metrics\": " << metrics_json << "],\n  \"gpu\": " << str(sys.gpu) << ", \"driver\": "
      << str(sys.driver) << ", \"nvrtc\": " << str(sys.nvrtc) << ", \"precision\": \"fp64\",\n  \"git\": {\"commit\": " << str(commit)
      << ", \"uncommitted_changes\": " << (dirty ? "true" : "false") << "}, \"date\": " << str(date) << ",\n  \"scenes\": " << scenes_json
      << "]\n}\n";
  std::printf("%s\nwrote %s\n", all_pass ? "GATE 1 PASSED" : "GATE 1 FAILED", out_path.c_str());
  return all_pass ? 0 : 1;
}

/** Per-sample adjoint: R_s = sum_p rd_p dImg_p / d(w_s I_s), so d L / d theta = sum_s w_s R_s dI_s/dtheta. */
static std::vector<double> backproject(const ObsSpec& sp, const std::vector<double>& xi, const std::vector<double>& rd) {
  const int n = sp.n;
  const double sigma = sp.beam / (2 * std::sqrt(2 * std::log(2.0))), s2 = sigma * sigma, norm = 1 / (2 * PI * s2);
  const double c = std::cos(sp.pa), sn = std::sin(sp.pa), scale = sp.A * sp.theta_g * sp.theta_g;
  const int reach = int(std::ceil(5 * sigma / sp.pixel)) + 1;
  std::vector<double> R(xi.size() / 2, 0.0);
  for (size_t s = 0; s < R.size(); s++) {
    const double X = sp.theta_g * (c * xi[2 * s] - sn * xi[2 * s + 1]), Y = sp.theta_g * (sn * xi[2 * s] + c * xi[2 * s + 1]);
    const int ci = int(std::floor(X / sp.pixel + n / 2.0)), cj = int(std::floor(Y / sp.pixel + n / 2.0));
    double acc = 0;
    for (int j = std::max(0, cj - reach); j <= std::min(n - 1, cj + reach); j++) {
      const double dy = (j + 0.5 - n / 2.0) * sp.pixel - Y;
      for (int i = std::max(0, ci - reach); i <= std::min(n - 1, ci + reach); i++) {
        const double dx = (i + 0.5 - n / 2.0) * sp.pixel - X;
        const double r2 = dx * dx + dy * dy;
        if (r2 > 25 * s2) continue;
        acc += rd[size_t(j) * n + i] * scale * norm * std::exp(-r2 / (2 * s2));
      }
    }
    R[s] = acc;
  }
  return R;
}

int run_gate1_diag(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog) {
  // Where does the gate-1 Jacobian error live? Dual tangents on the dense 2048^2 samples themselves, against
  // per-sample central differences at several h, split into samples that keep their branch (smooth) and
  // samples that change it (jumps), and binned by model-sky radius.
  Gpu gpu(tracer, catalog);
  using Theta = Gpu::Theta;
  const double R = 1000, L = 22, taper = 0.75;
  const int dense_n = 2048;
  std::vector<double> xi;
  double w;
  stratified(L, dense_n, 1, 101u, xi, w);
  const size_t ns = xi.size() / 2;
  const std::vector<double> wd(ns, w);
  struct Scene {
    double a, incl;
  };
  const std::vector<Scene> scenes = {{0.5, 30 * DEG}, {0.94, 60 * DEG}};
  const double hs[4] = {8e-3, 4e-3, 1e-3, 1e-5};
  const char* pnames[2] = {"spin", "inclination"};
  auto branch = [](const std::vector<int>& meta, size_t s) { return meta[3 * s] == rm::FATE_DISK ? 1 + meta[3 * s + 2] : 0; };
  for (const Scene& sc : scenes) {
    std::printf("scene a = %.2f, i = %.0f deg\n", sc.a, sc.incl / DEG);
    auto options = [&](double a) {
      rm::BatchOptions o = rk4({1, a}, 0.01, 1.01 * R);
      o.disk = 1;
      o.max_steps = 200000;
      o.crossing = rm::CROSSING_HERMITE;
      return o;
    };
    auto theta_at = [&](double a, double incl) {
      Theta t{};
      t[0] = 1;
      t[1] = a;
      t[2 + rm::THETA_RADIUS] = R;
      t[2 + rm::THETA_INCLINATION] = incl;
      return t;
    };
    std::vector<Theta> dirs(2);
    dirs[0] = Theta{};
    dirs[0][1] = 1;
    dirs[1] = Theta{};
    dirs[1][2 + rm::THETA_INCLINATION] = 1;
    const auto t0c = std::chrono::steady_clock::now();
    const Gpu::SampleImage D = gpu.render_samples("kerr", options(sc.a), theta_at(sc.a, sc.incl), dirs, xi, taper, true, Precision::Double);
    std::printf("  dense dual render: %.0f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0c).count());
    const std::vector<double> I_true =
        gpu.render_samples("kerr", options(sc.a + 0.04), theta_at(sc.a + 0.04, sc.incl + 3 * DEG), {}, xi, taper, false, Precision::Double).I;
    Gpu::SampleImage Sp[2][4], Sm[2][4];
    for (int d = 0; d < 2; d++)
      for (int k = 0; k < 4; k++) {
        const double h = hs[k];
        const double ap = d == 0 ? sc.a + h : sc.a, am = d == 0 ? sc.a - h : sc.a;
        const double ip = d == 1 ? sc.incl + h : sc.incl, im = d == 1 ? sc.incl - h : sc.incl;
        Sp[d][k] = gpu.render_samples("kerr", options(ap), theta_at(ap, ip), {}, xi, taper, false, Precision::Double);
        Sm[d][k] = gpu.render_samples("kerr", options(am), theta_at(am, im), {}, xi, taper, false, Precision::Double);
      }
    for (double beam : {20.0, 7.0}) {
      const ObsSpec sp0{64, 3, beam, 3.8, 0.4, 1};
      ObsSpec spt = sp0;
      spt.theta_g *= 1.03;
      spt.pa += 0.1;
      std::vector<double> data, img;
      assemble(spt, xi, wd, I_true, nullptr, 0, data, nullptr);
      double dmax = 0;
      for (double v : data) dmax = std::max(dmax, v);
      const double sigma_n = 0.01 * dmax;
      assemble(sp0, xi, wd, D.I, nullptr, 0, img, nullptr);
      std::vector<double> rd(img.size());
      for (size_t p = 0; p < img.size(); p++) rd[p] = (img[p] - data[p]) / (sigma_n * sigma_n);
      const std::vector<double> Rs = backproject(sp0, xi, rd);
      for (int d = 0; d < 2; d++) {
        double G_dual = 0, G_fd[4] = {0, 0, 0, 0}, G_smooth[4] = {0, 0, 0, 0}, G_jump[4] = {0, 0, 0, 0}, G_dual_smooth[4] = {0, 0, 0, 0};
        size_t njump[4] = {0, 0, 0, 0};
        const int nbins = 88;
        std::vector<double> bin_dual(nbins, 0), bin_fd8(nbins, 0), bin_fd1(nbins, 0), bin_jump1(nbins, 0);
        // Per-sample derivative error on smooth samples at h = 1e-5 (absolute, weighted by |w R|).
        double err_abs = 0, mag_abs = 0;
        for (size_t s = 0; s < ns; s++) {
          const double wr = w * Rs[s];
          const double dual = D.dI[2 * s + d];
          G_dual += wr * dual;
          const double rho = std::hypot(xi[2 * s], xi[2 * s + 1]);
          const int bi = std::min(nbins - 1, int(rho / 0.25));
          bin_dual[bi] += wr * dual;
          const int b0 = branch(D.meta, s);
          for (int k = 0; k < 4; k++) {
            const double fd = (Sp[d][k].I[s] - Sm[d][k].I[s]) / (2 * hs[k]);
            G_fd[k] += wr * fd;
            const bool smooth = branch(Sp[d][k].meta, s) == b0 && branch(Sm[d][k].meta, s) == b0;
            if (smooth) {
              G_smooth[k] += wr * fd;
              G_dual_smooth[k] += wr * dual;
            } else {
              G_jump[k] += wr * fd;
              njump[k]++;
            }
            if (k == 0) bin_fd8[bi] += wr * fd;
            if (k == 2) {
              if (smooth) bin_fd1[bi] += wr * fd;
              else bin_jump1[bi] += wr * fd;
            }
            if (k == 3 && smooth) {
              err_abs += std::fabs(wr * (dual - fd));
              mag_abs += std::fabs(wr * dual);
            }
          }
        }
        const double truth = (4 * G_fd[1] - G_fd[0]) / 3;
        std::printf("  beam %2.0f, d L/d(%s): truth (Richardson 8e-3/4e-3) %.6e, dense dual %.6e (error %.3e)\n", beam, pnames[d], truth,
                    G_dual, std::fabs(G_dual - truth) / std::fabs(truth));
        for (int k = 0; k < 4; k++)
          std::printf("    h %.0e: FD %.6e = smooth %.6e (dual on the same samples %.6e) + jumps %.6e over %zu samples\n", hs[k], G_fd[k],
                      G_smooth[k], G_dual_smooth[k], G_jump[k], njump[k]);
        std::printf("    smooth samples, h 1e-5: sum |wR (dual - FD)| / sum |wR dual| = %.3e\n", err_abs / mag_abs);
        std::printf("    radial bins (0.25 M) with |dual - FD(8e-3)| > 0.2%% of |truth|: rho, dual, FD 8e-3, FD 1e-3 smooth, FD 1e-3 jumps\n");
        for (int b = 0; b < nbins; b++)
          if (std::fabs(bin_dual[b] - bin_fd8[b]) > 2e-3 * std::fabs(truth))
            std::printf("      %5.2f  %+.4e  %+.4e  %+.4e  %+.4e\n", 0.25 * b, bin_dual[b], bin_fd8[b], bin_fd1[b], bin_jump1[b]);
      }
    }
  }
  return 0;
}

int run_image_gradients(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys,
                        const std::string& out_path) {
  // Phase 4 step 4 measurement: the total-flux gradient of the forward model from the fixed-sample
  // quadrature (tangents only; no edge term yet) against central differences of 8 x 8-supersampled
  // flux, with a map of where the difference comes from, and fp32 vs fp64 tangents with clipping.
  Gpu gpu(tracer, catalog);
  using Theta = Gpu::Theta;
  const double R = 1000, L = 22, taper = 0.75;
  const int base_n = 128, super = 8, bins = 64;
  struct Scene {
    double a, incl;
  };
  const std::vector<Scene> scenes = {{0.94, 60 * DEG}, {0.5, 30 * DEG}};
  const char* names[2] = {"spin", "inclination"};
  const int theta_k[2] = {1, 2 + rm::THETA_INCLINATION};
  const double h0[2] = {4e-3, 4e-3};
  std::vector<double> xi_base, xi_dense;
  double w_base, w_dense;
  stratified(L, base_n, 1, 7u, xi_base, w_base);
  stratified(L, base_n, super, 11u, xi_dense, w_dense);
  const size_t n_dense = xi_dense.size() / 2, n_base = xi_base.size() / 2;
  std::string scenes_json = "[";
  for (const Scene& sc : scenes) {
    rm::BatchOptions opt = rk4({1, sc.a}, 0.01, 1.01 * R);
    opt.disk = 1;
    opt.max_steps = 200000;
    Theta base{};
    base[0] = 1;
    base[1] = sc.a;
    base[2 + rm::THETA_RADIUS] = R;
    base[2 + rm::THETA_INCLINATION] = sc.incl;
    std::vector<Theta> dirs;
    for (int k : theta_k) {
      Theta t{};
      t[k] = 1;
      dirs.push_back(t);
    }
    std::printf("scene a = %.2f, i = %.0f deg\n", sc.a, sc.incl / DEG);
    const Gpu::SampleImage b64 = gpu.render_samples("kerr", opt, base, dirs, xi_base, taper, true, Precision::Double);
    const Gpu::SampleImage b32 = gpu.render_samples("kerr", opt, base, dirs, xi_base, taper, true, Precision::Float);
    const Gpu::SampleImage d64 = gpu.render_samples("kerr", opt, base, dirs, xi_dense, taper, true, Precision::Double);
    std::printf("  base %zu samples fp64 dual %.0f ms, fp32 dual %.0f ms; dense %zu samples fp64 dual %.0f ms\n", n_base, b64.ms, b32.ms,
                n_dense, d64.ms);
    std::string params_json = "[";
    for (int d = 0; d < 2; d++) {
      // Quadrature gradients (no edge term).
      double gq = 0, gq32 = 0, gqd = 0, F = 0;
      for (size_t s = 0; s < n_base; s++) {
        gq += w_base * b64.dI[2 * s + d];
        gq32 += w_base * b32.dI[2 * s + d];
        F += w_base * b64.I[s];
      }
      for (size_t s = 0; s < n_dense; s++) gqd += w_dense * d64.dI[2 * s + d];
      // Central differences of the dense flux at h0 and h0 / 2.
      double fd[2];
      std::vector<double> Ip, Im;
      std::vector<int> mp, mm;
      for (int hh = 0; hh < 2; hh++) {
        const double h = h0[d] / (hh ? 2 : 1);
        Theta tp = base, tm = base;
        tp[theta_k[d]] += h;
        tm[theta_k[d]] -= h;
        rm::BatchOptions op = opt, om = opt;
        if (d == 0) {
          op.params[1] = sc.a + h;
          om.params[1] = sc.a - h;
        }
        const Gpu::SampleImage P1 = gpu.render_samples("kerr", op, tp, {}, xi_dense, taper, false, Precision::Double);
        const Gpu::SampleImage M1 = gpu.render_samples("kerr", om, tm, {}, xi_dense, taper, false, Precision::Double);
        double fp = 0, fm = 0;
        for (size_t s = 0; s < n_dense; s++) {
          fp += w_dense * P1.I[s];
          fm += w_dense * M1.I[s];
        }
        fd[hh] = (fp - fm) / (2 * h);
        if (hh == 0) {
          Ip = P1.I;
          Im = M1.I;
          mp = P1.meta;
          mm = M1.meta;
        }
      }
      // Where the difference comes from (h0): per-sample FD minus tangent, by class and on a map.
      double by_class[3] = {0, 0, 0};  // fate flipped, same fate but step count changed, smooth
      int count_class[3] = {0, 0, 0};
      std::vector<double> map(size_t(bins) * bins, 0.0);
      for (size_t s = 0; s < n_dense; s++) {
        const double diff = w_dense * ((Ip[s] - Im[s]) / (2 * h0[d]) - d64.dI[2 * s + d]);
        if (diff == 0) continue;
        const int cls = mp[3 * s] != mm[3 * s] ? 0 : (mp[3 * s + 1] != mm[3 * s + 1] ? 1 : 2);
        by_class[cls] += diff;
        count_class[cls]++;
        const int bi = std::min(bins - 1, int((xi_dense[2 * s] + L) / (2 * L) * bins));
        const int bj = std::min(bins - 1, int((xi_dense[2 * s + 1] + L) / (2 * L) * bins));
        map[size_t(bj) * bins + bi] += diff;
      }
      const double truth = fd[1];
      for (double& v : map) v /= std::fabs(truth);
      // fp32 vs fp64 per sample, and clipping of large per-sample tangents.
      std::vector<double> rel32, mags;
      std::vector<int> steps;
      for (size_t s = 0; s < n_base; s++) {
        if (b64.meta[3 * s] != rm::FATE_DISK) continue;
        const double a64 = b64.dI[2 * s + d], a32 = b32.dI[2 * s + d];
        mags.push_back(std::fabs(a64));
        steps.push_back(b64.meta[3 * s + 1]);
        rel32.push_back(std::fabs(a32 - a64));
      }
      double scale = 0;
      for (double m : mags) scale = std::max(scale, m);
      std::vector<double> sorted_mags = mags;
      std::sort(sorted_mags.begin(), sorted_mags.end());
      auto pct = [&](std::vector<double> v, double p) {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[std::min(v.size() - 1, size_t(p * (v.size() - 1)))];
      };
      std::vector<int> sorted_steps = steps;
      std::sort(sorted_steps.begin(), sorted_steps.end());
      const int step95 = sorted_steps.empty() ? 0 : sorted_steps[size_t(0.95 * (sorted_steps.size() - 1))];
      std::vector<double> err_hi, err_lo;
      for (size_t k = 0; k < rel32.size(); k++) (steps[k] > step95 ? err_hi : err_lo).push_back(rel32[k] / scale);
      std::string clip_json = "[";
      for (double q : {0.99, 0.999}) {
        const double cap = pct(mags, q);
        double clipped = 0;
        for (size_t s = 0; s < n_base; s++) {
          const double v = b64.dI[2 * s + d];
          clipped += w_base * (std::fabs(v) > cap ? (v > 0 ? cap : -cap) : v);
        }
        clip_json += std::string(clip_json.size() > 1 ? ", " : "") + "{\"percentile\": " + num(q) +
                     ", \"gradient_change_relative\": " + num((clipped - gq) / std::fabs(gq)) + "}";
      }
      clip_json += "]";
      std::printf("  d(flux)/d(%s): quadrature %.6e (128^2) %.6e (1024^2), FD %.6e (h0) %.6e (h0/2); "
                  "relative difference (1024^2 vs FD h0/2) %.3e; fp32 total %.3e relative\n",
                  names[d], gq, gqd, fd[0], fd[1], (gqd - truth) / truth, (gq32 - gq) / gq);
      std::printf("    difference by class (fraction of the FD gradient): fate flip %.3e (%d samples), step change %.3e (%d), smooth %.3e (%d)\n",
                  by_class[0] / truth, count_class[0], by_class[1] / truth, count_class[1], by_class[2] / truth, count_class[2]);
      params_json += std::string(d ? ", " : "") + "{\"parameter\": " + str(names[d]) + ", \"fd_step\": " + num(h0[d]) +
                     ", \"flux\": " + num(F) + ", \"quadrature_gradient_128\": " + num(gq) + ", \"quadrature_gradient_1024\": " +
                     num(gqd) + ", \"fd_gradient_h0\": " + num(fd[0]) + ", \"fd_gradient_h0_over_2\": " + num(fd[1]) +
                     ", \"relative_difference_1024_vs_fd\": " + num((gqd - truth) / truth) +
                     ", \"relative_difference_128_vs_fd\": " + num((gq - truth) / truth) +
                     ", \"difference_by_class\": {\"fate_flip\": " + num(by_class[0] / truth) + ", \"fate_flip_samples\": " +
                     std::to_string(count_class[0]) + ", \"step_change\": " + num(by_class[1] / truth) + ", \"step_change_samples\": " +
                     std::to_string(count_class[1]) + ", \"smooth\": " + num(by_class[2] / truth) + ", \"smooth_samples\": " +
                     std::to_string(count_class[2]) + "}" + ", \"fp32\": {\"total_gradient_relative_error\": " + num((gq32 - gq) / gq) +
                     ", \"per_sample_error_over_max_median\": " + num(pct(err_lo, 0.5)) + ", \"per_sample_error_over_max_p99\": " +
                     num(pct(err_lo, 0.99)) + ", \"top5pct_steps_error_over_max_median\": " + num(pct(err_hi, 0.5)) +
                     ", \"top5pct_steps_error_over_max_max\": " + num(pct(err_hi, 1.0)) + ", \"max_over_median_tangent\": " +
                     num(scale / std::max(1e-300, pct(mags, 0.5))) + "}, \"clipping\": " + clip_json + ", \"map_bins\": " +
                     std::to_string(bins) + ", \"map\": " + arr(map) + "}";
    }
    scenes_json += std::string(scenes_json.size() > 1 ? ", " : "") + "{\"spin\": " + num(sc.a) + ", \"inclination_deg\": " +
                   num(sc.incl / DEG) + ", \"parameters\": " + params_json + "]}";
  }
  const std::string commit = shell("git rev-parse --short HEAD");
  const bool dirty = !shell("git status --porcelain -- . \":(exclude)validation/results\" \":(exclude)docs/benchmarks\"").empty();
  char date[32];
  const std::time_t now = std::time(nullptr);
  std::strftime(date, sizeof date, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
  std::ofstream(out_path)
      << "{\n  \"check\": \"image-gradients\",\n  \"description\": \"Phase 4 step 4 measurement (not a gate). Kerr, camera at 1000 M, "
         "model sky +-22 M, thin disk g^4 F(r) tau(r) (taper from 0.75 r_out), RK4 step fraction 0.01, fp64 unless stated. Total-flux "
         "gradient from the fixed-sample quadrature with forward-mode tangents (no critical-curve edge term), on 128^2 and 1024^2 "
         "stratified samples, against central differences of the 1024^2 (8 x 8 per cell) flux at steps h0 and h0/2. The map is "
         "sum over samples per bin of w (FD_s - tangent_s) at h0, divided by |FD gradient at h0/2|, on a 64 x 64 grid over the model "
         "sky (row = y). fp32: the same tangents from the fp32 module; clipping caps |dI| at a percentile of the fp64 per-sample "
         "magnitudes.\",\n  \"gpu\": "
      << str(sys.gpu) << ", \"driver\": " << str(sys.driver) << ", \"nvrtc\": " << str(sys.nvrtc) << ",\n  \"git\": {\"commit\": "
      << str(commit) << ", \"uncommitted_changes\": " << (dirty ? "true" : "false") << "}, \"date\": " << str(date)
      << ",\n  \"samples\": {\"base\": " << n_base << ", \"dense\": " << n_dense << ", \"half_width_M\": " << L << "},\n  \"scenes\": "
      << scenes_json << "]\n}\n";
  std::printf("wrote %s\n", out_path.c_str());
  return 0;
}

int run_tangent_benchmark(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys,
                          const std::string& out_path) {
  // Kerr a = 0.94 seen at 60 degrees from 1000 M, a 128 x 128 grid over +-15 M on the sky, disk on,
  // RK4 step fraction 0.01: the configuration the inverse problem will render.
  Gpu gpu(tracer, catalog);
  const int side = 128, n = side * side;
  const double R = 1000, half = 15.0 / R;
  using Theta = Gpu::Theta;
  std::vector<Theta> thetas(n);
  for (int j = 0; j < side; j++)
    for (int i = 0; i < side; i++) {
      Theta& t = thetas[j * side + i];
      t = Theta{};
      t[0] = 1;
      t[1] = 0.94;
      t[2 + rm::THETA_RADIUS] = R;
      t[2 + rm::THETA_INCLINATION] = 60 * DEG;
      t[2 + rm::THETA_U] = -half + 2 * half * (i + 0.5) / side;
      t[2 + rm::THETA_V] = -half + 2 * half * (j + 0.5) / side;
    }
  rm::BatchOptions opt = rk4({1, 0.94}, 0.01, 1.01 * R);
  opt.disk = 1;
  // Directions: spin, inclination, then camera radius and yaw.
  const int order[4] = {1, 2 + rm::THETA_INCLINATION, 2 + rm::THETA_RADIUS, 2 + rm::THETA_YAW};
  std::string rows = "[";
  std::printf("%-6s %-9s %2s %10s %12s %8s\n", "prec", "mode", "P", "ms", "rays/s", "x plain");
  for (Precision prec : {Precision::Float, Precision::Double}) {
    const char* pname = prec == Precision::Double ? "fp64" : "fp32";
    // Baseline: trace_camera_rays_kernel, the ordinary batch kernel, on the same rays.
    double plain_ms = 0;
    {
      for (const CatalogEntry& e : catalog) {
        if (e.id != "kerr") continue;
        ModuleOptions mo;
        mo.batch = true;
        if (!tracer.select(e, prec, mo)) throw std::runtime_error(tracer.error());
      }
      const double p[8] = {1, 0.94};
      const rm::CameraFrame cam = tracer.camera(p, R, 0, PI / 2 - 60 * DEG);
      std::vector<Vec3> dirs(n);
      for (int k = 0; k < n; k++) dirs[k] = {thetas[k][2 + rm::THETA_U], thetas[k][2 + rm::THETA_V], 1};
      double* d_d = nullptr;
      rm::RayOut* d_out = nullptr;
      V_CHECK(cudaMalloc(&d_d, sizeof(double) * 3 * n));
      V_CHECK(cudaMalloc(&d_out, sizeof(rm::RayOut) * n));
      V_CHECK(cudaMemcpy(d_d, dirs.data(), sizeof(double) * 3 * n, cudaMemcpyHostToDevice));
      rm::BatchOptions copy = opt;
      rm::CameraFrame c = cam;
      int count = n;
      void* args[] = {&copy, &c, &d_d, &count, &d_out};
      std::vector<double> times;
      for (int rep = 0; rep < 4; rep++) {
        V_CHECK(cudaDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        tracer.launch("trace_camera_rays_kernel", unsigned((n + 63) / 64), 64, args);
        V_CHECK(cudaDeviceSynchronize());
        if (rep > 0) times.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      cudaFree(d_d);
      cudaFree(d_out);
      std::sort(times.begin(), times.end());
      plain_ms = times[0];
      std::printf("%-6s %-9s %2d %10.1f %12.3e %8.2f\n", pname, "plain", 0, plain_ms, n / (plain_ms * 1e-3), 1.0);
      rows += std::string(rows.size() > 1 ? ", " : "") + "{\"precision\": \"" + pname + "\", \"mode\": \"plain\", \"P\": 0, \"ms\": " +
              num(plain_ms) + ", \"rays_per_s\": " + num(n / (plain_ms * 1e-3)) + ", \"over_plain\": 1}";
    }
    for (int P : {1, 2, 4}) {
      std::vector<Theta> dirs;
      for (int d = 0; d < P; d++) {
        Theta t{};
        t[order[d]] = 1;
        dirs.push_back(t);
      }
      for (int mode : {0, 1}) {
        std::vector<double> times;
        double compile_ms = 0;
        for (int rep = 0; rep < 3; rep++) {
          double ms = 0;
          gpu.tangent_rays("kerr", opt, thetas, dirs, mode, prec, &ms);
          if (rep == 0) compile_ms = tracer.last_compile_ms();
          if (rep > 0) times.push_back(ms);
        }
        std::sort(times.begin(), times.end());
        const double ms = times[0];
        const char* mname = mode == 0 ? "symbolic" : "dual";
        std::printf("%-6s %-9s %2d %10.1f %12.3e %8.2f   (NVRTC %.0f ms)\n", pname, mname, P, ms, n / (ms * 1e-3), ms / plain_ms,
                    compile_ms);
        rows += std::string(", ") + "{\"precision\": \"" + pname + "\", \"mode\": \"" + mname + "\", \"P\": " + std::to_string(P) +
                ", \"ms\": " + num(ms) + ", \"rays_per_s\": " + num(n / (ms * 1e-3)) + ", \"over_plain\": " + num(ms / plain_ms) +
                ", \"nvrtc_compile_ms\": " + num(compile_ms) + "}";
      }
    }
  }
  const std::string commit = shell("git rev-parse --short HEAD");
  std::ofstream(out_path)
      << "{\n  \"benchmark\": \"phase4-step3-tangents\",\n  \"description\": \"Kerr a = 0.94, inclination 60 deg, camera at 1000 M, "
         "128 x 128 rays over +-15 M, disk on, RK4 step fraction 0.01. plain = trace_camera_rays_kernel; symbolic = "
         "trace_tangent_rays_kernel (compiler JVP); dual = trace_dual_rays_kernel (Dual<Real, P> through rm::trace_ray). fp32 modules use "
         "fast math and a 128-register cap, fp64 neither. Times are the fastest of the timed runs after a warm-up (synchronised "
         "wall clock around the launch); the GPU may carry background load.\",\n  \"gpu\": "
      << str(sys.gpu) << ", \"driver\": " << str(sys.driver) << ", \"nvrtc\": " << str(sys.nvrtc) << ", \"git\": " << str(commit)
      << ",\n  \"rays\": " << n << ",\n  \"results\": " << rows << "]\n}\n";
  std::printf("wrote %s\n", out_path.c_str());
  return 0;
}

int run_validation(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, const SystemStrings& sys,
                   const std::string& out_dir, const std::string& generated_dir) {
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
      {"sensitivities", [&] { return check_sensitivities(gpu, generated_dir); }},
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
