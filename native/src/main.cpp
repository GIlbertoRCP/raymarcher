// Raymarcher native viewer: the generic NVRTC tracer (kernels/tracer.cu) on any
// metric in native/generated/catalog.json, with a GLFW + Dear ImGui window and
// CUDA/OpenGL PBO interop. Scene defaults match the web app (web/src/settings.ts).
//
//   raymarcher                         interactive viewer
//   raymarcher --benchmark [--out f]   time every catalog metric at 1080p, 1440p, 4K
//   raymarcher --render <scene.json> --out <prefix>
//                                      render a scene file: image, fate, steps and drift maps
//   raymarcher --validate [--out d]    fp64 validation suite on the GPU, results to validation/results/
//   raymarcher --snapshot <id> <png>   render one frame headless (--size W H, --fp64,
//                                      --sky stars|grid|none, --no-disk, --radius/--yaw/--pitch)
//   raymarcher --capture <s> <png>     viewer: save the window (render + panel) after s seconds and exit

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <nvml.h>
#include <nvrtc.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "nvrtc_tracer.h"
#include "rm/json.h"
#include "validate.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define CUDA_CHECK(call)                                                                                 \
  do {                                                                                                   \
    cudaError_t err_ = (call);                                                                           \
    if (err_ != cudaSuccess) {                                                                           \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
      std::exit(1);                                                                                      \
    }                                                                                                    \
  } while (0)

namespace {

constexpr double PI = 3.14159265358979323846;

// --- Scene settings (web/src/settings.ts defaults) ---------------------------------

enum class Sky { Stars, Grid, None };

struct Settings {
  std::string spacetime = "kerr";
  std::map<std::string, std::vector<double>> params;  // per metric, in declaration order
  double render_scale = 1.0;
  int max_steps = 300;
  double step_fraction = 0.03;
  bool disk = true;
  Sky sky = Sky::Stars;
  bool redshift = false;
  double exposure = 0.55;
  double disk_temp = 4500;
  double cam_radius = 26;  // in units of the metric's length scale
  double cam_yaw = 0.0;
  double cam_pitch = 0.1;
  double fov_degrees = 45;
  Precision precision = Precision::Float;
  int block_x = 16, block_y = 16;
};

const char* sky_name(Sky s) { return s == Sky::Stars ? "stars" : s == Sky::Grid ? "grid" : "none"; }

const CatalogEntry* find_metric(const std::vector<CatalogEntry>& catalog, const std::string& id) {
  for (const CatalogEntry& e : catalog)
    if (e.id == id) return &e;
  return nullptr;
}

std::vector<double>& params_for(Settings& s, const CatalogEntry& m) {
  auto it = s.params.find(m.id);
  if (it == s.params.end()) {
    std::vector<double> defaults;
    for (const ParamInfo& p : m.params) defaults.push_back(p.value);
    it = s.params.emplace(m.id, defaults).first;
  }
  return it->second;
}

/** tan of half the vertical field of view; the setting applies to the narrower side (uniforms.ts). */
double vertical_tan(double fov_degrees, double aspect) {
  return std::tan(fov_degrees * PI / 360.0) / std::min(1.0, aspect);
}

/** Builds the frame uniforms; returns false (and leaves u unchanged) when no static camera exists. */
bool build_uniforms(NvrtcTracer& tracer, Settings& s, const CatalogEntry& m, int w, int h, double time,
                    rm::FrameUniforms& u) {
  std::vector<double>& p = params_for(s, m);
  double params[rm::MAX_PARAMS] = {};
  std::copy(p.begin(), p.end(), params);
  rm::CameraFrame cam = tracer.camera(params, s.cam_radius, s.cam_yaw, s.cam_pitch);
  if (!cam.ok) return false;
  u.cam = cam;
  std::copy(params, params + rm::MAX_PARAMS, u.params);
  u.time = time;
  u.fov_scale = vertical_tan(s.fov_degrees, double(w) / h);
  u.step_fraction = s.step_fraction;
  u.escape_radius = std::max(2 * s.cam_radius, 60.0) * cam.length_scale;
  u.exposure = s.exposure;
  u.disk_temp = s.disk_temp;
  u.width = w;
  u.height = h;
  u.max_steps = s.max_steps;
  u.flags = (s.disk ? rm::FLAG_DISK : 0u) | (s.sky == Sky::Grid ? rm::FLAG_GRID : 0u) |
            (s.sky == Sky::Stars ? rm::FLAG_STARS : 0u) | (s.redshift ? rm::FLAG_REDSHIFT : 0u);
  return true;
}

/** One line with every scene parameter, comparable with the web app's settings. */
std::string scene_readout(Settings& s, const CatalogEntry& m, int w, int h) {
  std::string out = m.id;
  const std::vector<double>& p = params_for(s, m);
  char buf[512];
  for (size_t i = 0; i < m.params.size(); i++) {
    std::snprintf(buf, sizeof buf, " %s=%.3f", m.params[i].name.c_str(), p[i]);
    out += buf;
  }
  std::snprintf(buf, sizeof buf,
                "\ncamera r=%.2f M  yaw=%.2f deg  pitch=%.2f deg  fov=%.0f deg"
                "\nRK4 max steps=%d  step fraction=%.3f"
                "\ndisk=%s  sky=%s  redshift=%s  exposure=%.2f  disk temp=%.0f K"
                "\nimage %dx%d  %s",
                s.cam_radius, s.cam_yaw * 180 / PI, s.cam_pitch * 180 / PI, s.fov_degrees, s.max_steps,
                s.step_fraction, s.disk ? "on" : "off", sky_name(s.sky), s.redshift ? "on" : "off", s.exposure,
                s.disk_temp, w, h, s.precision == Precision::Double ? "fp64" : "fp32");
  return out + buf;
}

// --- Frame statistics ----------------------------------------------------------------

struct RayStats {
  double mean_steps = 0, p95_steps = 0;
  double fate_fraction[4] = {0, 0, 0, 0};  // escaped, captured, disk, unfinished
};

RayStats ray_stats(const unsigned* d_steps, const unsigned char* d_fate, size_t n) {
  std::vector<unsigned> steps(n);
  std::vector<unsigned char> fate(n);
  CUDA_CHECK(cudaMemcpy(steps.data(), d_steps, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(fate.data(), d_fate, n, cudaMemcpyDeviceToHost));
  RayStats r;
  r.mean_steps = std::accumulate(steps.begin(), steps.end(), 0.0) / double(n);
  std::nth_element(steps.begin(), steps.begin() + n * 95 / 100, steps.end());
  r.p95_steps = steps[n * 95 / 100];
  for (unsigned char f : fate) r.fate_fraction[std::min<int>(f, 3)] += 1.0 / double(n);
  return r;
}

/** Device buffers for one resolution. */
struct FrameBuffers {
  int w = 0, h = 0;
  float* hdr = nullptr;
  unsigned* steps = nullptr;
  unsigned char* fate = nullptr;
  unsigned* rgba = nullptr;  // only for headless output

  void resize(int nw, int nh, bool with_rgba) {
    if (nw == w && nh == h && (rgba != nullptr) == with_rgba) return;
    release();
    w = nw;
    h = nh;
    size_t n = size_t(w) * h;
    CUDA_CHECK(cudaMalloc(&hdr, n * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&steps, n * sizeof(unsigned)));
    CUDA_CHECK(cudaMalloc(&fate, n));
    if (with_rgba) CUDA_CHECK(cudaMalloc(&rgba, n * sizeof(unsigned)));
  }
  void release() {
    cudaFree(hdr);
    cudaFree(steps);
    cudaFree(fate);
    cudaFree(rgba);
    hdr = nullptr;
    steps = nullptr;
    fate = nullptr;
    rgba = nullptr;
  }
};

bool write_png(const std::string& path, const unsigned* d_rgba, int w, int h) {
  std::vector<unsigned> px(size_t(w) * h);
  CUDA_CHECK(cudaMemcpy(px.data(), d_rgba, px.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
  return stbi_write_png(path.c_str(), w, h, 4, px.data(), w * 4) != 0;
}

// --- System information for reports ------------------------------------------------

struct SystemInfo {
  std::string gpu, arch, driver;
  int cuda_driver = 0, cuda_runtime = 0, nvrtc_major = 0, nvrtc_minor = 0;
  size_t vram_mb = 0;
};

SystemInfo system_info(const NvrtcTracer& tracer) {
  SystemInfo s;
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  s.gpu = prop.name;
  s.vram_mb = prop.totalGlobalMem / (1024 * 1024);
  s.arch = tracer.device_arch();
  cudaDriverGetVersion(&s.cuda_driver);
  cudaRuntimeGetVersion(&s.cuda_runtime);
  nvrtcVersion(&s.nvrtc_major, &s.nvrtc_minor);
  char version[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = "unknown";
  if (nvmlInit() == NVML_SUCCESS) {
    nvmlSystemGetDriverVersion(version, sizeof version);
    nvmlShutdown();
  }
  s.driver = version;
  return s;
}

std::string cuda_version(int v) { return std::to_string(v / 1000) + "." + std::to_string((v % 1000) / 10); }

// --- Headless modes ---------------------------------------------------------------------

struct Timing {
  double median_ms, min_ms, max_ms;
};

Timing time_frames(NvrtcTracer& tracer, const rm::FrameUniforms& u, FrameBuffers& fb, int frames, int bx, int by) {
  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));
  tracer.trace(u, fb.hdr, fb.steps, fb.fate, nullptr, bx, by);  // warm-up
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> ms;
  for (int i = 0; i < frames; i++) {
    CUDA_CHECK(cudaEventRecord(a));
    tracer.trace(u, fb.hdr, fb.steps, fb.fate, nullptr, bx, by);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float t = 0;
    CUDA_CHECK(cudaEventElapsedTime(&t, a, b));
    ms.push_back(t);
  }
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  std::sort(ms.begin(), ms.end());
  return {ms[ms.size() / 2], ms.front(), ms.back()};
}

/**
 * SIMT lane efficiency of the trace kernel for one frame: useful RK4 steps
 * divided by 32 x (the longest ray in each warp), summed over warps. A warp
 * runs until its longest ray finishes, so rays that end early leave lanes idle.
 */
double lane_efficiency(const unsigned* d_steps, int w, int h, int bx, int by) {
  std::vector<unsigned> steps(size_t(w) * h);
  CUDA_CHECK(cudaMemcpy(steps.data(), d_steps, steps.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
  double useful = 0, issued = 0;
  const int threads = bx * by;
  for (int y0 = 0; y0 < h; y0 += by) {
    for (int x0 = 0; x0 < w; x0 += bx) {
      for (int warp = 0; warp * 32 < threads; warp++) {
        unsigned longest = 0;
        for (int lane = 0; lane < 32; lane++) {
          const int t = warp * 32 + lane;
          const int x = x0 + t % bx, y = y0 + t / bx;
          if (t >= threads || x >= w || y >= h) continue;
          const unsigned s = steps[size_t(y) * w + x];
          useful += s;
          longest = std::max(longest, s);
        }
        issued += 32.0 * longest;
      }
    }
  }
  return issued > 0 ? useful / issued : 1;
}

int run_benchmark(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, Settings base, const std::string& out_path) {
  const SystemInfo sys = system_info(tracer);
  struct Res {
    const char* name;
    int w, h;
  };
  const Res resolutions[] = {{"1080p", 1920, 1080}, {"1440p", 2560, 1440}, {"4K", 3840, 2160}};
  const int blocks[][2] = {{16, 16}, {32, 8}, {8, 8}};
  const char* extra_flags = std::getenv("RM_NVRTC_FLAGS");

  std::printf("Benchmark on %s (%s), driver %s, CUDA driver %s, NVRTC %d.%d%s%s\n", sys.gpu.c_str(), sys.arch.c_str(),
              sys.driver.c_str(), cuda_version(sys.cuda_driver).c_str(), sys.nvrtc_major, sys.nvrtc_minor,
              extra_flags ? ", extra NVRTC flags: " : "", extra_flags ? extra_flags : "");
  std::printf("Scene: web defaults (camera r = %.0f M, yaw %.2f, pitch %.2f, fov %.0f deg, max %d steps, step fraction %.3f, disk on, stars)\n",
              base.cam_radius, base.cam_yaw, base.cam_pitch, base.fov_degrees, base.max_steps, base.step_fraction);
  std::printf(" Metric        | Prec Integ | Res   | Block | Trace ms (min-max)        | Mrays/s | Steps mean/p95 | Lanes | Regs Occ\n");

  std::FILE* json = std::fopen(out_path.c_str(), "w");
  if (!json) {
    std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
    return 1;
  }
  std::fprintf(json,
               "{\n  \"description\": \"NVRTC tracer (native/kernels/tracer.cu) on generated metrics, run via `raymarcher --benchmark`. "
               "Trace-kernel time only, measured with CUDA events: median of the timed frames after one warm-up frame. The kernel also writes per-pixel steps and fate buffers. "
               "lane_efficiency = useful steps / (32 x longest ray per warp), from the per-pixel step counts. occupancy = theoretical, from the register count (cuOccupancyMaxActiveBlocksPerMultiprocessor).\",\n");
  std::fprintf(json, "  \"hardware\": {\"gpu\": \"%s\", \"arch\": \"%s\", \"vram_mb\": %zu},\n", sys.gpu.c_str(), sys.arch.c_str(), sys.vram_mb);
  std::fprintf(json, "  \"software\": {\"driver\": \"%s\", \"cuda_driver_api\": \"%s\", \"cuda_runtime\": \"%s\", \"nvrtc\": \"%d.%d\", \"nvrtc_extra_flags\": \"%s\"},\n",
               sys.driver.c_str(), cuda_version(sys.cuda_driver).c_str(), cuda_version(sys.cuda_runtime).c_str(), sys.nvrtc_major,
               sys.nvrtc_minor, extra_flags ? extra_flags : "");
  std::fprintf(json,
               "  \"scene\": {\"source\": \"web/src/settings.ts defaults\", \"camera_radius_M\": %.3f, \"camera_yaw_rad\": %.4f, \"camera_pitch_rad\": %.4f, "
               "\"fov_degrees\": %.1f, \"max_steps\": %d, \"step_fraction\": %.4f, "
               "\"disk\": true, \"sky\": \"stars\", \"redshift\": false, \"time_s\": 0, \"params\": \"catalog defaults\"},\n",
               base.cam_radius, base.cam_yaw, base.cam_pitch, base.fov_degrees, base.max_steps, base.step_fraction);
  std::fprintf(json, "  \"runs\": [\n");

  FrameBuffers fb;
  bool first = true;
  auto run = [&](const CatalogEntry& m, Precision prec, bool rk45, const Res& res, int bx, int by, int frames) {
    Settings s = base;
    s.precision = prec;
    ModuleOptions options;
    options.rk45 = rk45;
    if (!tracer.select(m, prec, options)) {
      std::fprintf(stderr, "%s\n", tracer.error().c_str());
      std::exit(1);
    }
    const double compile_ms = tracer.last_compile_ms();
    fb.resize(res.w, res.h, false);
    rm::FrameUniforms u{};
    if (!build_uniforms(tracer, s, m, res.w, res.h, 0.0, u)) {
      std::fprintf(stderr, "%s: no static camera at r = %.1f\n", m.id.c_str(), s.cam_radius);
      std::exit(1);
    }
    if (rk45) {
      u.rtol = 1e-10;
      u.max_steps = 200000;
    }
    const Timing t = time_frames(tracer, u, fb, frames, bx, by);
    const RayStats st = ray_stats(fb.steps, fb.fate, size_t(res.w) * res.h);
    const double lanes = lane_efficiency(fb.steps, res.w, res.h, bx, by);
    const KernelInfo k = tracer.trace_kernel_info(bx * by);
    const double mrays = res.w * double(res.h) / (t.median_ms * 1e3);
    const char* pname = prec == Precision::Double ? "fp64" : "fp32";
    const char* iname = rk45 ? "RK45" : "RK4 ";
    std::printf(" %-13s | %s %s | %-5s | %2dx%-2d | %9.3f (%.3f-%.3f) | %7.1f | %5.1f / %4.0f   | %4.0f%% | %3d %3.0f%%\n",
                m.id.c_str(), pname, iname, res.name, bx, by, t.median_ms, t.min_ms, t.max_ms, mrays, st.mean_steps,
                st.p95_steps, 100 * lanes, k.registers, 100 * k.occupancy);
    std::fprintf(json,
                 "%s    {\"metric\": \"%s\", \"precision\": \"%s\", \"integrator\": \"%s\", \"resolution\": \"%s\", \"width\": %d, \"height\": %d, \"block\": [%d, %d], "
                 "\"frames\": %d, \"trace_ms_median\": %.4f, \"trace_ms_min\": %.4f, \"trace_ms_max\": %.4f, \"mrays_per_s\": %.2f, "
                 "\"steps_mean\": %.2f, \"steps_p95\": %.0f, \"lane_efficiency\": %.4f, \"registers\": %d, \"local_bytes\": %d, \"occupancy\": %.4f, "
                 "\"fate\": {\"escaped\": %.5f, \"captured\": %.5f, \"disk\": %.5f, \"unfinished\": %.5f}%s%s}",
                 first ? "" : ",\n", m.id.c_str(), pname, rk45 ? "RK45 rtol 1e-10" : "RK4", res.name, res.w, res.h, bx, by, frames,
                 t.median_ms, t.min_ms, t.max_ms, mrays, st.mean_steps, st.p95_steps, lanes, k.registers, k.local_bytes, k.occupancy,
                 st.fate_fraction[0], st.fate_fraction[1], st.fate_fraction[2], st.fate_fraction[3],
                 rk45 ? ", \"max_steps\": 200000" : "",
                 compile_ms > 0 ? (std::string(", \"nvrtc_compile_ms\": ") + std::to_string(compile_ms)).c_str() : "");
    first = false;
  };

  for (const CatalogEntry& m : catalog)
    for (const Res& res : resolutions)
      for (const auto& b : blocks) run(m, Precision::Float, false, res, b[0], b[1], 20);
  // Reference paths in fp64 (the 3070 runs double at 1/64 of the float rate), 1080p only.
  for (const CatalogEntry& m : catalog) run(m, Precision::Double, false, resolutions[0], 16, 16, 3);
  for (const CatalogEntry& m : catalog) run(m, Precision::Double, true, resolutions[0], 16, 16, 3);

  std::fprintf(json, "\n  ]\n}\n");
  std::fclose(json);
  fb.release();
  std::printf("Wrote %s\n", out_path.c_str());
  return 0;
}

int run_snapshot(NvrtcTracer& tracer, const CatalogEntry& m, Settings s, int w, int h, const std::string& path) {
  if (!tracer.select(m, s.precision)) {
    std::fprintf(stderr, "%s\n", tracer.error().c_str());
    return 1;
  }
  std::printf("NVRTC compile: %.0f ms\n", tracer.last_compile_ms());
  FrameBuffers fb;
  fb.resize(w, h, true);
  rm::FrameUniforms u{};
  if (!build_uniforms(tracer, s, m, w, h, 0.0, u)) {
    std::fprintf(stderr, "No static observer at the camera position.\n");
    return 1;
  }
  Timing t = time_frames(tracer, u, fb, 5, s.block_x, s.block_y);
  tracer.composite(fb.hdr, w, h, float(s.exposure), fb.rgba, false);
  RayStats st = ray_stats(fb.steps, fb.fate, size_t(w) * h);
  const bool ok = write_png(path, fb.rgba, w, h);
  std::printf("%s\n", scene_readout(s, m, w, h).c_str());
  std::printf("trace %.3f ms (median of 5) | steps mean %.1f, p95 %.0f | captured %.2f%%, disk %.2f%%, escaped %.2f%%, unfinished %.2f%%\n",
              t.median_ms, st.mean_steps, st.p95_steps, 100 * st.fate_fraction[1], 100 * st.fate_fraction[2],
              100 * st.fate_fraction[0], 100 * st.fate_fraction[3]);
  std::printf("%s %s\n", ok ? "Wrote" : "FAILED to write", path.c_str());
  fb.release();
  return ok ? 0 : 1;
}

/** Settings, size and time from a scene file (scenes/*.json, web/src/scene-file.ts). */
struct LoadedScene {
  std::string name;
  const CatalogEntry* metric;
  int width, height;
  double time;
};

LoadedScene load_scene(const std::string& path, const std::vector<CatalogEntry>& catalog, Settings& s) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  const rm::json::Value j = rm::json::parse(ss.str());
  LoadedScene scene;
  scene.name = j["name"].str();
  scene.metric = find_metric(catalog, j["spacetime"].str());
  if (!scene.metric) throw std::runtime_error("unknown spacetime " + j["spacetime"].str());
  std::vector<double> params;
  for (const ParamInfo& p : scene.metric->params) params.push_back(j["params"].has(p.name) ? j["params"][p.name].num() : p.value);
  for (const auto& [key, value] : j["params"].o) {
    (void)value;
    bool known = false;
    for (const ParamInfo& p : scene.metric->params) known = known || p.name == key;
    if (!known) throw std::runtime_error(scene.metric->id + " has no parameter " + key);
  }
  s.spacetime = scene.metric->id;
  s.params[scene.metric->id] = params;
  s.cam_radius = j["camera"]["radius"].num();
  s.cam_yaw = j["camera"]["yaw"].num();
  s.cam_pitch = j["camera"]["pitch"].num();
  s.fov_degrees = j["camera"]["fovDegrees"].num();
  scene.width = int(j["resolution"][0].num());
  scene.height = int(j["resolution"][1].num());
  const std::string sky = j["sky"].str();
  s.sky = sky == "grid" ? Sky::Grid : sky == "none" ? Sky::None : Sky::Stars;
  s.disk = j["disk"].b;
  s.redshift = j["redshift"].b;
  s.exposure = j["exposure"].num();
  s.disk_temp = j["diskTemp"].num();
  scene.time = j.has("time") ? j["time"].num() : 0.0;
  if (j["integrator"]["method"].str() != "rk4") throw std::runtime_error("only the rk4 integrator is implemented");
  s.max_steps = int(j["integrator"]["maxSteps"].num());
  s.step_fraction = j["integrator"]["stepFraction"].num();
  return scene;
}

template <typename T>
bool write_raw(const std::string& path, const T* d_src, size_t n) {
  std::vector<T> host(n);
  CUDA_CHECK(cudaMemcpy(host.data(), d_src, n * sizeof(T), cudaMemcpyDeviceToHost));
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char*>(host.data()), std::streamsize(n * sizeof(T)));
  return bool(f);
}

/**
 * Headless render of a scene file. Writes, for the prefix <out>:
 *   <out>.hdr.f32    linear RGB, float32, rows top to bottom
 *   <out>.fate.u8    rm::RayFate per pixel (0 escaped, 1 captured, 2 disk, 3 unfinished)
 *   <out>.steps.u16  RK4 steps per pixel
 *   <out>.drift.f32  |H| / p_t^2 at the end of each ray
 *   <out>.png        the displayed image (bloom, tone mapping, grading)
 *   <out>.json       scene, hardware, timing and summary statistics
 */
int run_render(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, Settings s, const std::string& scene_path,
               const std::string& out, int size_w, int size_h, int frames) {
  const LoadedScene scene = load_scene(scene_path, catalog, s);
  const int w = size_w > 0 ? size_w : scene.width, h = size_h > 0 ? size_h : scene.height;
  const size_t n = size_t(w) * h;
  if (!tracer.select(*scene.metric, s.precision)) {
    std::fprintf(stderr, "%s\n", tracer.error().c_str());
    return 1;
  }
  FrameBuffers fb;
  fb.resize(w, h, true);
  float* d_drift = nullptr;
  CUDA_CHECK(cudaMalloc(&d_drift, n * sizeof(float)));
  rm::FrameUniforms u{};
  if (!build_uniforms(tracer, s, *scene.metric, w, h, scene.time, u)) {
    std::fprintf(stderr, "%s: no static observer at the camera position\n", scene.name.c_str());
    return 1;
  }
  Timing t = time_frames(tracer, u, fb, frames, s.block_x, s.block_y);
  tracer.trace(u, fb.hdr, fb.steps, fb.fate, d_drift, s.block_x, s.block_y);
  tracer.composite(fb.hdr, w, h, float(s.exposure), fb.rgba, false);
  CUDA_CHECK(cudaDeviceSynchronize());
  const RayStats st = ray_stats(fb.steps, fb.fate, n);

  std::filesystem::path prefix(out);
  if (prefix.has_parent_path()) std::filesystem::create_directories(prefix.parent_path());
  std::vector<float> rgba(n * 4);
  CUDA_CHECK(cudaMemcpy(rgba.data(), fb.hdr, rgba.size() * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<float> rgb(n * 3);
  for (size_t i = 0; i < n; i++)
    for (int c = 0; c < 3; c++) rgb[3 * i + c] = rgba[4 * i + c];
  std::ofstream(out + ".hdr.f32", std::ios::binary).write(reinterpret_cast<const char*>(rgb.data()), std::streamsize(rgb.size() * sizeof(float)));
  std::vector<unsigned> steps32(n);
  CUDA_CHECK(cudaMemcpy(steps32.data(), fb.steps, n * sizeof(unsigned), cudaMemcpyDeviceToHost));
  std::vector<unsigned short> steps16(n);
  for (size_t i = 0; i < n; i++) steps16[i] = (unsigned short)std::min(steps32[i], 65535u);
  std::ofstream(out + ".steps.u16", std::ios::binary).write(reinterpret_cast<const char*>(steps16.data()), std::streamsize(n * 2));
  bool ok = write_raw(out + ".fate.u8", fb.fate, n) && write_raw(out + ".drift.f32", d_drift, n) &&
            write_png(out + ".png", fb.rgba, w, h);

  // Constraint drift over rays that did not fall in (captured rays stop where dt/dlambda diverges).
  std::vector<float> drift(n);
  CUDA_CHECK(cudaMemcpy(drift.data(), d_drift, n * sizeof(float), cudaMemcpyDeviceToHost));
  std::vector<unsigned char> fate(n);
  CUDA_CHECK(cudaMemcpy(fate.data(), fb.fate, n, cudaMemcpyDeviceToHost));
  std::vector<float> finite;
  for (size_t i = 0; i < n; i++)
    if (fate[i] != rm::FATE_CAPTURED && std::isfinite(drift[i])) finite.push_back(drift[i]);
  auto quantile = [&](double q) {
    if (finite.empty()) return 0.0;
    size_t k = std::min(finite.size() - 1, size_t(q * (finite.size() - 1)));
    std::nth_element(finite.begin(), finite.begin() + k, finite.end());
    return double(finite[k]);
  };
  const double drift_median = quantile(0.5), drift_p99 = quantile(0.99), drift_max = quantile(1.0);

  const SystemInfo sys = system_info(tracer);
  std::FILE* meta = std::fopen((out + ".json").c_str(), "w");
  if (meta) {
    std::fprintf(meta,
                 "{\n  \"backend\": \"cuda-nvrtc\",\n  \"scene\": \"%s\",\n  \"scene_file\": \"%s\",\n  \"precision\": \"%s\",\n"
                 "  \"width\": %d,\n  \"height\": %d,\n  \"gpu\": \"%s\",\n  \"driver\": \"%s\",\n  \"cuda_driver_api\": \"%s\",\n"
                 "  \"nvrtc\": \"%d.%d\",\n  \"timed_frames\": %d,\n  \"trace_ms_median\": %.4f,\n  \"steps_mean\": %.3f,\n  \"steps_p95\": %.0f,\n"
                 "  \"fate\": {\"escaped\": %.6f, \"captured\": %.6f, \"disk\": %.6f, \"unfinished\": %.6f},\n"
                 "  \"constraint_drift_uncaptured\": {\"median\": %.3e, \"p99\": %.3e, \"max\": %.3e},\n"
                 "  \"files\": [\"hdr.f32\", \"fate.u8\", \"steps.u16\", \"drift.f32\", \"png\"]\n}\n",
                 scene.name.c_str(), std::filesystem::path(scene_path).filename().string().c_str(),
                 s.precision == Precision::Double ? "fp64" : "fp32", w, h, sys.gpu.c_str(), sys.driver.c_str(),
                 cuda_version(sys.cuda_driver).c_str(), sys.nvrtc_major, sys.nvrtc_minor, frames, t.median_ms, st.mean_steps,
                 st.p95_steps, st.fate_fraction[0], st.fate_fraction[1], st.fate_fraction[2], st.fate_fraction[3],
                 drift_median, drift_p99, drift_max);
    std::fclose(meta);
  } else {
    ok = false;
  }
  std::printf("%-28s %s %dx%d  trace %.2f ms  captured %.2f%% disk %.2f%% escaped %.2f%% unfinished %.3f%%  drift p99 %.1e\n",
              scene.name.c_str(), s.precision == Precision::Double ? "fp64" : "fp32", w, h, t.median_ms,
              100 * st.fate_fraction[1], 100 * st.fate_fraction[2], 100 * st.fate_fraction[0], 100 * st.fate_fraction[3],
              drift_p99);
  cudaFree(d_drift);
  fb.release();
  return ok ? 0 : 1;
}

// --- Interactive viewer ---------------------------------------------------------------------

// Atlas palette from DESIGN.md: ink ground, plate text, rule lines, one red accent.
void setup_imgui_style() {
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 0;
  style.FrameRounding = 0;
  style.GrabRounding = 0;
  style.ScrollbarRounding = 0;
  style.WindowBorderSize = 1;
  style.FrameBorderSize = 1;
  style.WindowPadding = ImVec2(12, 12);
  style.ItemSpacing = ImVec2(8, 6);
  auto hex = [](unsigned rgb, float a = 1.0f) {
    return ImVec4(((rgb >> 16) & 255) / 255.0f, ((rgb >> 8) & 255) / 255.0f, (rgb & 255) / 255.0f, a);
  };
  const ImVec4 ink = hex(0x0a0f1c), plate = hex(0xdbe3ec), dim = hex(0x93a1b6), rule = hex(0x34465f),
               scale = hex(0x5b7090), red = hex(0xe4503f);
  ImVec4* c = style.Colors;
  c[ImGuiCol_WindowBg] = hex(0x0a0f1c, 0.92f);
  c[ImGuiCol_Border] = rule;
  c[ImGuiCol_Text] = plate;
  c[ImGuiCol_TextDisabled] = dim;
  c[ImGuiCol_FrameBg] = ink;
  c[ImGuiCol_FrameBgHovered] = hex(0x141c2e);
  c[ImGuiCol_FrameBgActive] = hex(0x1b2640);
  c[ImGuiCol_TitleBg] = ink;
  c[ImGuiCol_TitleBgActive] = ink;
  c[ImGuiCol_Header] = hex(0x141c2e);
  c[ImGuiCol_HeaderHovered] = hex(0x1b2640);
  c[ImGuiCol_HeaderActive] = hex(0x22304f);
  c[ImGuiCol_Button] = ink;
  c[ImGuiCol_ButtonHovered] = hex(0x1b2640);
  c[ImGuiCol_ButtonActive] = hex(0x22304f);
  c[ImGuiCol_SliderGrab] = scale;
  c[ImGuiCol_SliderGrabActive] = red;
  c[ImGuiCol_CheckMark] = red;
  c[ImGuiCol_Separator] = rule;
  c[ImGuiCol_PlotLines] = scale;
  c[ImGuiCol_PopupBg] = ink;
}

struct ViewerInput {
  Settings* s = nullptr;
  bool dragging = false;
  double last_x = 0, last_y = 0;
  bool snapshot = false;
};
ViewerInput g_input;

void mouse_button_callback(GLFWwindow* window, int button, int action, int) {
  if (ImGui::GetIO().WantCaptureMouse && action == GLFW_PRESS) return;
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    g_input.dragging = action == GLFW_PRESS;
    glfwGetCursorPos(window, &g_input.last_x, &g_input.last_y);
  }
}

// Orbit controls as in web/src/viewer.ts.
void cursor_callback(GLFWwindow*, double x, double y) {
  if (g_input.dragging) {
    g_input.s->cam_yaw -= (x - g_input.last_x) * 0.005;
    g_input.s->cam_pitch = std::clamp(g_input.s->cam_pitch + (y - g_input.last_y) * 0.005, -1.45, 1.45);
  }
  g_input.last_x = x;
  g_input.last_y = y;
}

void scroll_callback(GLFWwindow*, double, double yoffset) {
  if (ImGui::GetIO().WantCaptureMouse) return;
  // One wheel notch is ~100 px of deltaY in the browser: radius *= exp(0.1) per notch.
  g_input.s->cam_radius = std::clamp(g_input.s->cam_radius * std::exp(-yoffset * 0.1), 4.0, 80.0);
}

void key_callback(GLFWwindow* window, int key, int, int action, int) {
  if (action != GLFW_PRESS || ImGui::GetIO().WantCaptureKeyboard) return;
  if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
  if (key == GLFW_KEY_P) g_input.snapshot = true;
}

// OpenGL 1.5 buffer entry points (no loader library).
#ifndef GL_PIXEL_UNPACK_BUFFER
#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#endif
#ifndef GL_STREAM_DRAW
#define GL_STREAM_DRAW 0x88E0
#endif
typedef void(APIENTRY* GenBuffersFn)(GLsizei, GLuint*);
typedef void(APIENTRY* BindBufferFn)(GLenum, GLuint);
typedef void(APIENTRY* BufferDataFn)(GLenum, ptrdiff_t, const void*, GLenum);
typedef void(APIENTRY* DeleteBuffersFn)(GLsizei, const GLuint*);

int run_viewer(NvrtcTracer& tracer, const std::vector<CatalogEntry>& catalog, Settings& s, double capture_after,
               const std::string& capture_path) {
  const SystemInfo sys = system_info(tracer);
  if (!glfwInit()) {
    std::fprintf(stderr, "Failed to initialise GLFW\n");
    return 1;
  }
  GLFWwindow* window = glfwCreateWindow(1600, 900, "Raymarcher (CUDA)", nullptr, nullptr);
  if (!window) {
    std::fprintf(stderr, "Failed to create a window\n");
    glfwTerminate();
    return 1;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);  // measure the renderer, not the display
  auto glGenBuffers_ = (GenBuffersFn)glfwGetProcAddress("glGenBuffers");
  auto glBindBuffer_ = (BindBufferFn)glfwGetProcAddress("glBindBuffer");
  auto glBufferData_ = (BufferDataFn)glfwGetProcAddress("glBufferData");
  auto glDeleteBuffers_ = (DeleteBuffersFn)glfwGetProcAddress("glDeleteBuffers");
  if (!glGenBuffers_ || !glBindBuffer_ || !glBufferData_ || !glDeleteBuffers_) {
    std::fprintf(stderr, "OpenGL pixel buffer objects are not available\n");
    return 1;
  }

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::GetIO().IniFilename = nullptr;
  setup_imgui_style();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 130");
  g_input.s = &s;
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetCursorPosCallback(window, cursor_callback);
  glfwSetScrollCallback(window, scroll_callback);
  glfwSetKeyCallback(window, key_callback);

  GLuint tex = 0, pbo = 0;
  glGenTextures(1, &tex);
  glGenBuffers_(1, &pbo);
  cudaGraphicsResource* pbo_resource = nullptr;
  int w = 0, h = 0;
  FrameBuffers fb;
  cudaEvent_t ev[3];
  for (cudaEvent_t& e : ev) CUDA_CHECK(cudaEventCreate(&e));

  const CatalogEntry* metric = find_metric(catalog, s.spacetime);
  Precision compiled_precision = s.precision;
  bool need_select = true;
  std::string status;
  double trace_ms = 0, composite_ms = 0, compile_ms = 0;
  RayStats stats;
  double last_stats = -1;
  float trace_history[120] = {};
  int history_index = 0;
  const auto t_start = std::chrono::steady_clock::now();
  rm::FrameUniforms u{};
  bool have_frame = false;

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    if (need_select || compiled_precision != s.precision) {
      if (tracer.select(*metric, s.precision)) {
        compile_ms = tracer.last_compile_ms() > 0 ? tracer.last_compile_ms() : compile_ms;
        status.clear();
      } else {
        status = tracer.error();
      }
      compiled_precision = s.precision;
      need_select = false;
      last_stats = -1;
    }

    int fw = 0, fh = 0;
    glfwGetFramebufferSize(window, &fw, &fh);
    const int rw = std::max(64, int(fw * s.render_scale)), rh = std::max(64, int(fh * s.render_scale));
    if (rw != w || rh != h) {
      w = rw;
      h = rh;
      if (pbo_resource) cudaGraphicsUnregisterResource(pbo_resource);
      glBindTexture(GL_TEXTURE_2D, tex);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
      glBindBuffer_(GL_PIXEL_UNPACK_BUFFER, pbo);
      glBufferData_(GL_PIXEL_UNPACK_BUFFER, ptrdiff_t(w) * h * 4, nullptr, GL_STREAM_DRAW);
      glBindBuffer_(GL_PIXEL_UNPACK_BUFFER, 0);
      CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard));
      fb.resize(w, h, false);
      last_stats = -1;
    }

    if (status.empty()) {
      if (build_uniforms(tracer, s, *metric, w, h, now, u)) {
        have_frame = true;
      } else {
        status = "No static observer at this camera position (inside an ergoregion); showing the last valid camera.";
        u.time = now;
      }
      if (have_frame) {
        unsigned* d_rgba = nullptr;
        size_t bytes = 0;
        CUDA_CHECK(cudaGraphicsMapResources(1, &pbo_resource));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_rgba, &bytes, pbo_resource));
        CUDA_CHECK(cudaEventRecord(ev[0]));
        tracer.trace(u, fb.hdr, fb.steps, fb.fate, nullptr, s.block_x, s.block_y);
        CUDA_CHECK(cudaEventRecord(ev[1]));
        tracer.composite(fb.hdr, w, h, float(s.exposure), d_rgba, true);
        CUDA_CHECK(cudaEventRecord(ev[2]));
        CUDA_CHECK(cudaEventSynchronize(ev[2]));
        float a = 0, b = 0;
        cudaEventElapsedTime(&a, ev[0], ev[1]);
        cudaEventElapsedTime(&b, ev[1], ev[2]);
        trace_ms = a;
        composite_ms = b;
        trace_history[history_index] = a;
        history_index = (history_index + 1) % 120;
        if (now - last_stats > 0.5) {
          stats = ray_stats(fb.steps, fb.fate, size_t(w) * h);
          last_stats = now;
        }
        if (g_input.snapshot) {
          g_input.snapshot = false;
          unsigned* d_png = nullptr;
          CUDA_CHECK(cudaMalloc(&d_png, size_t(w) * h * 4));
          tracer.composite(fb.hdr, w, h, float(s.exposure), d_png, false);
          const std::string name = "raymarcher_" + metric->id + "_" + std::to_string(std::time(nullptr)) + ".png";
          status = write_png(name, d_png, w, h) ? "Saved " + name : "Could not write " + name;
          cudaFree(d_png);
        }
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &pbo_resource));
        glBindBuffer_(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glBindBuffer_(GL_PIXEL_UNPACK_BUFFER, 0);
      }
    }

    glViewport(0, 0, fw, fh);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    if (have_frame) {
      glEnable(GL_TEXTURE_2D);
      glBindTexture(GL_TEXTURE_2D, tex);
      glBegin(GL_QUADS);
      glTexCoord2f(0, 0); glVertex2f(-1, -1);
      glTexCoord2f(1, 0); glVertex2f(1, -1);
      glTexCoord2f(1, 1); glVertex2f(1, 1);
      glTexCoord2f(0, 1); glVertex2f(-1, 1);
      glEnd();
      glDisable(GL_TEXTURE_2D);
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);
    if (ImGui::Begin("Raymarcher (CUDA)", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::TextUnformatted(metric->name.c_str());
      ImGui::PushTextWrapPos(360);
      ImGui::TextDisabled("%s", metric->description.c_str());
      ImGui::PopTextWrapPos();
      ImGui::Separator();

      if (ImGui::BeginCombo("Spacetime", metric->name.c_str())) {
        for (const CatalogEntry& e : catalog) {
          if (ImGui::Selectable(e.name.c_str(), &e == metric) && &e != metric) {
            metric = &e;
            s.spacetime = e.id;
            need_select = true;
          }
        }
        ImGui::EndCombo();
      }
      std::vector<double>& p = params_for(s, *metric);
      for (size_t i = 0; i < metric->params.size(); i++) {
        const ParamInfo& info = metric->params[i];
        float v = float(p[i]);
        if (ImGui::SliderFloat(info.label.c_str(), &v, float(info.min), float(info.max), "%.3f")) p[i] = v;
      }

      if (ImGui::CollapsingHeader("View", ImGuiTreeNodeFlags_DefaultOpen)) {
        float radius = float(s.cam_radius), fov = float(s.fov_degrees);
        if (ImGui::SliderFloat("Distance (M)", &radius, 4, 80, "%.1f")) s.cam_radius = radius;
        if (ImGui::SliderFloat("Field of view", &fov, 15, 90, "%.0f deg")) s.fov_degrees = fov;
        ImGui::Checkbox("Disk", &s.disk);
        ImGui::SameLine();
        ImGui::Checkbox("Redshift colours", &s.redshift);
        int sky = int(s.sky);
        ImGui::RadioButton("Stars", &sky, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Grid", &sky, 1);
        ImGui::SameLine();
        ImGui::RadioButton("No sky", &sky, 2);
        s.sky = Sky(sky);
        float exposure = float(s.exposure), temp = float(s.disk_temp);
        if (ImGui::SliderFloat("Exposure", &exposure, 0.1f, 3.0f, "%.2f")) s.exposure = exposure;
        if (ImGui::SliderFloat("Peak disk temp", &temp, 2500, 12000, "%.0f K")) s.disk_temp = temp;
      }
      if (ImGui::CollapsingHeader("Quality", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderInt("Max steps", &s.max_steps, 50, 2000);
        float frac = float(s.step_fraction), rs = float(s.render_scale);
        if (ImGui::SliderFloat("Step fraction", &frac, 0.005f, 0.1f, "%.3f")) s.step_fraction = frac;
        if (ImGui::SliderFloat("Render scale", &rs, 0.25f, 1.0f, "%.2f")) s.render_scale = rs;
        bool fp64 = s.precision == Precision::Double;
        if (ImGui::Checkbox("fp64 reference (slow on GeForce)", &fp64)) s.precision = fp64 ? Precision::Double : Precision::Float;
        const char* blocks[] = {"16x16", "32x8", "8x8"};
        int bi = s.block_x == 32 ? 1 : s.block_x == 8 ? 2 : 0;
        if (ImGui::Combo("Block", &bi, blocks, 3)) {
          const int shapes[3][2] = {{16, 16}, {32, 8}, {8, 8}};
          s.block_x = shapes[bi][0];
          s.block_y = shapes[bi][1];
        }
      }
      if (ImGui::CollapsingHeader("Scene", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextUnformatted(scene_readout(s, *metric, w, h).c_str());
      }
      if (ImGui::CollapsingHeader("Measurements", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Text("%s (%s), driver %s", sys.gpu.c_str(), sys.arch.c_str(), sys.driver.c_str());
        ImGui::Text("Trace %.2f ms   composite %.2f ms", trace_ms, composite_ms);
        ImGui::PlotLines("##trace", trace_history, 120, history_index, "trace ms", 0.0f, 50.0f, ImVec2(-1, 36));
        ImGui::Text("%.0f Mrays/s   steps/ray mean %.1f, p95 %.0f", trace_ms > 0 ? w * double(h) / (trace_ms * 1e3) : 0.0,
                    stats.mean_steps, stats.p95_steps);
        ImGui::Text("Captured %.1f%%  disk %.1f%%  escaped %.1f%%  unfinished %.2f%%", 100 * stats.fate_fraction[1],
                    100 * stats.fate_fraction[2], 100 * stats.fate_fraction[0], 100 * stats.fate_fraction[3]);
        ImGui::Text("NVRTC compile of this module: %.0f ms", compile_ms);
      }
      ImGui::Separator();
      ImGui::PushTextWrapPos(360);
      ImGui::TextDisabled("Geometry, lensing, redshift and g^4 beaming are computed from the metric. The disk texture, bloom and colour grading are artistic.");
      ImGui::TextDisabled("Drag to orbit, scroll to zoom, P saves a PNG.");
      ImGui::PopTextWrapPos();
      if (!status.empty()) {
        ImGui::PushTextWrapPos(360);
        ImGui::TextColored(ImVec4(1.0f, 0.48f, 0.41f, 1.0f), "%s", status.c_str());
        ImGui::PopTextWrapPos();
      }
    }
    ImGui::End();
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    if (!capture_path.empty() && now >= capture_after) {
      // Read the finished back buffer, so the image holds exactly this window.
      std::vector<unsigned char> px(size_t(fw) * fh * 4);
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glReadBuffer(GL_BACK);
      glReadPixels(0, 0, fw, fh, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
      stbi_flip_vertically_on_write(1);
      const bool ok = stbi_write_png(capture_path.c_str(), fw, fh, 4, px.data(), fw * 4) != 0;
      stbi_flip_vertically_on_write(0);
      std::printf("%s %s\n%s\n", ok ? "Captured" : "FAILED to capture", capture_path.c_str(),
                  scene_readout(s, *metric, w, h).c_str());
      std::printf("trace %.2f ms, composite %.2f ms, steps/ray mean %.1f\n", trace_ms, composite_ms, stats.mean_steps);
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    glfwSwapBuffers(window);

    // A failed camera only lasts until the camera moves somewhere valid.
    if (status.rfind("No static observer", 0) == 0) status.clear();
  }

  for (cudaEvent_t e : ev) cudaEventDestroy(e);
  if (pbo_resource) cudaGraphicsUnregisterResource(pbo_resource);
  fb.release();
  glDeleteBuffers_(1, &pbo);
  glDeleteTextures(1, &tex);
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Settings s;
  bool benchmark = false, validate = false;
  std::string bench_out = "benchmark_report.json";
  std::string render_scene;
  bool out_given = false;
  std::string snapshot_metric, snapshot_path, capture_path;
  double capture_after = 0;
  int snap_w = 1280, snap_h = 720, frames = 3;
  bool size_given = false;
  for (int i = 1; i < argc; i++) {
    const std::string a = argv[i];
    auto next = [&](int n = 1) { return i + n < argc; };
    if (a == "--benchmark" || a == "-b") benchmark = true;
    else if (a == "--validate") validate = true;
    else if (a == "--out" && next()) {
      bench_out = argv[++i];
      out_given = true;
    } else if (a == "--render" && next()) render_scene = argv[++i];
    else if (a == "--capture" && next(2)) {
      capture_after = std::atof(argv[++i]);
      capture_path = argv[++i];
    }
    else if (a == "--snapshot" && next(2)) {
      snapshot_metric = argv[++i];
      snapshot_path = argv[++i];
    } else if (a == "--size" && next(2)) {
      size_given = true;
      snap_w = std::atoi(argv[++i]);
      snap_h = std::atoi(argv[++i]);
    } else if (a == "--metric" && next()) s.spacetime = argv[++i];
    else if (a == "--fp64") s.precision = Precision::Double;
    else if (a == "--frames" && next()) frames = std::max(1, std::atoi(argv[++i]));
    else if (a == "--no-disk") s.disk = false;
    else if (a == "--sky" && next()) {
      const std::string v = argv[++i];
      s.sky = v == "grid" ? Sky::Grid : v == "none" ? Sky::None : Sky::Stars;
    } else if (a == "--radius" && next()) s.cam_radius = std::atof(argv[++i]);
    else if (a == "--yaw" && next()) s.cam_yaw = std::atof(argv[++i]);
    else if (a == "--pitch" && next()) s.cam_pitch = std::atof(argv[++i]);
    else if (a == "--steps" && next()) s.max_steps = std::atoi(argv[++i]);
    else if (a == "--param" && next(2)) {
      // --param <metric>:<index> <value>, e.g. --param kerr:1 0.95
      const std::string key = argv[++i];
      const double v = std::atof(argv[++i]);
      const size_t colon = key.find(':');
      if (colon != std::string::npos) {
        std::vector<double>& p = s.params[key.substr(0, colon)];
        const size_t idx = std::stoul(key.substr(colon + 1));
        if (p.size() <= idx) p.resize(idx + 1, std::nan(""));
        p[idx] = v;
      }
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", a.c_str());
      return 2;
    }
  }

  try {
    const std::string exe_dir = std::filesystem::absolute(std::filesystem::path(argv[0])).parent_path().string();
    const SourcePaths paths = SourcePaths::locate(exe_dir);
    const std::vector<CatalogEntry> catalog = load_catalog(paths);
    // Fill in --param overrides with defaults for the indices not given.
    for (auto& [id, values] : s.params) {
      const CatalogEntry* m = find_metric(catalog, id);
      if (!m) {
        std::fprintf(stderr, "Unknown metric in --param: %s\n", id.c_str());
        return 2;
      }
      values.resize(m->params.size(), std::nan(""));
      for (size_t k = 0; k < values.size(); k++)
        if (std::isnan(values[k])) values[k] = m->params[k].value;
    }
    NvrtcTracer tracer(paths);
    std::printf("Sources: %s\n", paths.kernels.c_str());

    if (benchmark) return run_benchmark(tracer, catalog, s, bench_out);
    if (validate) {
      const SystemInfo info = system_info(tracer);
      const SystemStrings strings{info.gpu, info.arch, info.driver, cuda_version(info.cuda_driver),
                                  std::to_string(info.nvrtc_major) + "." + std::to_string(info.nvrtc_minor)};
      return run_validation(tracer, catalog, strings, out_given ? bench_out : "validation/results");
    }
    if (!render_scene.empty()) {
      const std::string prefix = out_given ? bench_out : std::filesystem::path(render_scene).stem().string();
      return run_render(tracer, catalog, s, render_scene, prefix, size_given ? snap_w : 0, size_given ? snap_h : 0, frames);
    }
    if (!snapshot_metric.empty()) {
      const CatalogEntry* m = find_metric(catalog, snapshot_metric);
      if (!m) {
        std::fprintf(stderr, "Unknown metric: %s\n", snapshot_metric.c_str());
        return 2;
      }
      return run_snapshot(tracer, *m, s, snap_w, snap_h, snapshot_path);
    }
    if (!find_metric(catalog, s.spacetime)) s.spacetime = catalog.front().id;
    return run_viewer(tracer, catalog, s, capture_after, capture_path);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
}
