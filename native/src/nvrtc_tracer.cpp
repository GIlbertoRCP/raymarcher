#include "nvrtc_tracer.h"

#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "rm/json.h"

namespace fs = std::filesystem;

#define CU_CHECK(call)                                                                     \
  do {                                                                                     \
    CUresult r_ = (call);                                                                  \
    if (r_ != CUDA_SUCCESS) {                                                              \
      const char* msg_ = nullptr;                                                          \
      cuGetErrorString(r_, &msg_);                                                         \
      std::fprintf(stderr, "CUDA driver error at %s:%d: %s\n", __FILE__, __LINE__, msg_); \
      std::exit(1);                                                                        \
    }                                                                                      \
  } while (0)

#define RT_CHECK(call)                                                                                   \
  do {                                                                                                   \
    cudaError_t e_ = (call);                                                                             \
    if (e_ != cudaSuccess) {                                                                             \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(1);                                                                                      \
    }                                                                                                    \
  } while (0)

static std::string read_text(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot read " + path.string());
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

SourcePaths SourcePaths::locate(const std::string& exe_dir) {
  // Prefer the source tree, so `npm run export-cuda` or a kernel edit takes
  // effect on the next launch without a rebuild; fall back to the copy next to
  // the executable.
  std::vector<fs::path> roots;
#ifdef RM_NATIVE_DIR
  roots.emplace_back(RM_NATIVE_DIR);
#endif
  roots.emplace_back(fs::path(exe_dir) / "native");
  for (const fs::path& root : roots) {
    if (fs::exists(root / "kernels" / "tracer.cu") && fs::exists(root / "generated" / "catalog.json")) {
      return {(root / "kernels").string(), (root / "src").string(), (root / "generated").string()};
    }
  }
  throw std::runtime_error("cannot find kernels/tracer.cu and generated/catalog.json");
}

std::vector<CatalogEntry> load_catalog(const SourcePaths& paths) {
  const rm::json::Value doc = rm::json::parse(read_text(fs::path(paths.generated) / "catalog.json"));
  std::vector<CatalogEntry> out;
  for (const rm::json::Value& m : doc["metrics"].a) {
    CatalogEntry e;
    e.id = m["id"].str();
    e.name = m["name"].str();
    e.description = m["description"].str();
    e.header = m["header"].str();
    e.has_horizon = m["hasHorizon"].b;
    e.has_disk = m["hasDisk"].b;
    for (const rm::json::Value& p : m["params"].a) {
      e.params.push_back({p["name"].str(), p["label"].str(), p["value"].num(), p["min"].num(), p["max"].num()});
    }
    if ((int)e.params.size() > rm::MAX_PARAMS) throw std::runtime_error(e.id + ": more than 8 parameters");
    out.push_back(e);
  }
  return out;
}

NvrtcTracer::NvrtcTracer(SourcePaths paths) : paths_(std::move(paths)) {
  RT_CHECK(cudaFree(nullptr));  // create the runtime's primary context, which the driver API then uses
  CU_CHECK(cuInit(0));
  int device = 0, major = 0, minor = 0;
  RT_CHECK(cudaGetDevice(&device));
  RT_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RT_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  arch_ = "sm_" + std::to_string(major) + std::to_string(minor);
  RT_CHECK(cudaMalloc(&d_params_, sizeof(double) * rm::MAX_PARAMS));
  RT_CHECK(cudaMalloc(&d_camera_, sizeof(rm::CameraFrame)));
}

NvrtcTracer::~NvrtcTracer() {
  for (auto& [key, m] : modules_) cuModuleUnload(m.module);
  cudaFree(d_params_);
  cudaFree(d_camera_);
}

bool NvrtcTracer::select(const CatalogEntry& metric, Precision precision, ModuleOptions options) {
  const bool fp64 = precision == Precision::Double;
  // RM_NVRTC_FLAGS adds compiler options (space-separated), for performance experiments.
  std::vector<std::string> extra;
  if (const char* flags = std::getenv("RM_NVRTC_FLAGS")) {
    std::istringstream in(flags);
    for (std::string f; in >> f;) extra.push_back(f);
  }
  std::string key = metric.id + (fp64 ? ":f64" : ":f32") + (options.rk45 ? ":rk45" : ":rk4") + (options.batch ? ":batch" : "") +
                    (std::getenv("RM_NVRTC_PRECISE") ? ":precise" : "");
  for (const std::string& f : extra) key += " " + f;
  last_compile_ms_ = 0;
  auto it = modules_.find(key);
  if (it != modules_.end()) {
    current_ = &it->second;
    return true;
  }

  // Sources are re-read for every new module, so edits apply after switching.
  std::string metric_source;
  try {
    tracer_source_ = read_text(fs::path(paths_.kernels) / "tracer.cu");
    geodesic_source_ = read_text(fs::path(paths_.include) / "rm" / "geodesic.h");
    frame_source_ = read_text(fs::path(paths_.include) / "rm" / "frame.h");
    metric_source = read_text(fs::path(paths_.generated) / metric.header);
  } catch (const std::exception& e) {
    error_ = e.what();
    return false;
  }
  const char* headers[] = {metric_source.c_str(), geodesic_source_.c_str(), frame_source_.c_str()};
  const char* header_names[] = {"metric.cuh", "rm/geodesic.h", "rm/frame.h"};

  auto t0 = std::chrono::steady_clock::now();
  nvrtcProgram prog;
  if (nvrtcCreateProgram(&prog, tracer_source_.c_str(), "tracer.cu", 3, headers, header_names) != NVRTC_SUCCESS) {
    error_ = "nvrtcCreateProgram failed";
    return false;
  }
  const std::string arch_opt = "--gpu-architecture=" + arch_;
  const std::string real_opt = std::string("-DRM_REAL=") + (fp64 ? "double" : "float");
  const std::string metric_opt = "-DRM_METRIC=" + metric.id;
  // RM_NVRTC_VERBOSE=1 prints ptxas register and spill counts for each kernel.
  const bool verbose = std::getenv("RM_NVRTC_VERBOSE") != nullptr;
  std::vector<const char*> opts = {"--std=c++17", arch_opt.c_str(), real_opt.c_str(), metric_opt.c_str(),
                                   options.rk45 ? "-DRM_RK45=1" : "-DRM_RK45=0", options.batch ? "-DRM_BATCH=1" : "-DRM_BATCH=0",
                                   "--generate-line-info"};
  // Measured on the RTX 3070 (docs/benchmarks/phase2-3070.json): capping fp32 kernels
  // at 128 registers lets two 256-thread blocks share an SM. Kerr goes from 168
  // registers to 128 with 40 bytes of spills and runs ~14% faster; fp64 kernels
  // are bound by double-precision throughput and only spill, so they keep the default.
  const bool cap_given = std::any_of(extra.begin(), extra.end(), [](const std::string& f) { return f.find("maxrregcount") != std::string::npos; });
  if (!fp64 && !cap_given) opts.push_back("--maxrregcount=128");
  // Fast math (approximate sqrt, division and transcendentals) for the fp32
  // interactive modules only: on the 21-scene cross-backend set it changed no ray
  // fate and no image difference against WebGPU, and Kerr runs ~29% faster
  // (validation/results/fastmath-vs-webgpu.json). The fp64 reference and validation
  // modules stay IEEE. RM_NVRTC_PRECISE=1 turns it off.
  if (!fp64 && !std::getenv("RM_NVRTC_PRECISE")) opts.push_back("--use_fast_math");
  for (const std::string& f : extra) opts.push_back(f.c_str());
  if (verbose) opts.push_back("--ptxas-options=-v");
  nvrtcResult res = nvrtcCompileProgram(prog, int(opts.size()), opts.data());
  size_t log_size = 0;
  nvrtcGetProgramLogSize(prog, &log_size);
  std::string log(log_size, '\0');
  if (log_size > 1) nvrtcGetProgramLog(prog, &log[0]);
  if (verbose) std::fprintf(stderr, "NVRTC %s:\n%s\n", key.c_str(), log.c_str());
  if (res != NVRTC_SUCCESS) {
    error_ = "NVRTC failed for " + key + ":\n" + log;
    nvrtcDestroyProgram(&prog);
    return false;
  }
  // RM_NVRTC_PTX=<dir> writes the PTX of each module there, for inspection.
  if (const char* ptx_dir = std::getenv("RM_NVRTC_PTX")) {
    size_t ptx_size = 0;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::string ptx(ptx_size, '\0');
    nvrtcGetPTX(prog, &ptx[0]);
    std::ofstream(fs::path(ptx_dir) / (metric.id + (fp64 ? "-f64" : "-f32") + (options.rk45 ? "-rk45" : "") + ".ptx")) << ptx.c_str();
  }
  size_t cubin_size = 0;
  nvrtcGetCUBINSize(prog, &cubin_size);
  std::vector<char> cubin(cubin_size);
  nvrtcGetCUBIN(prog, cubin.data());
  nvrtcDestroyProgram(&prog);
  last_compile_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  Module m;
  CU_CHECK(cuModuleLoadData(&m.module, cubin.data()));
  CU_CHECK(cuModuleGetFunction(&m.trace, m.module, "trace_kernel"));
  CU_CHECK(cuModuleGetFunction(&m.composite, m.module, "composite_kernel"));
  CU_CHECK(cuModuleGetFunction(&m.camera, m.module, "camera_kernel"));
  current_ = &(modules_[key] = m);
  error_.clear();
  return true;
}

rm::CameraFrame NvrtcTracer::camera(const double* params, double cam_radius, double yaw, double pitch) {
  RT_CHECK(cudaMemcpy(d_params_, params, sizeof(double) * rm::MAX_PARAMS, cudaMemcpyHostToDevice));
  void* args[] = {&d_params_, &cam_radius, &yaw, &pitch, &d_camera_};
  CU_CHECK(cuLaunchKernel(current_->camera, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr));
  rm::CameraFrame cam;
  RT_CHECK(cudaMemcpy(&cam, d_camera_, sizeof(cam), cudaMemcpyDeviceToHost));
  return cam;
}

void NvrtcTracer::trace(const rm::FrameUniforms& u, float* d_hdr, unsigned* d_steps, unsigned char* d_fate, float* d_drift,
                        int block_x, int block_y) {
  rm::FrameUniforms copy = u;
  void* args[] = {&copy, &d_hdr, &d_steps, &d_fate, &d_drift};
  const unsigned gx = (u.width + block_x - 1) / block_x, gy = (u.height + block_y - 1) / block_y;
  CU_CHECK(cuLaunchKernel(current_->trace, gx, gy, 1, block_x, block_y, 1, 0, nullptr, args, nullptr));
}

KernelInfo NvrtcTracer::trace_kernel_info(int threads) const {
  KernelInfo k;
  CU_CHECK(cuFuncGetAttribute(&k.registers, CU_FUNC_ATTRIBUTE_NUM_REGS, current_->trace));
  CU_CHECK(cuFuncGetAttribute(&k.local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, current_->trace));
  int blocks = 0, max_threads = 0;
  CU_CHECK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, current_->trace, threads, 0));
  RT_CHECK(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, 0));
  k.occupancy = double(blocks * threads) / max_threads;
  return k;
}

void NvrtcTracer::launch(const char* kernel, unsigned grid, unsigned block, void** args) {
  CUfunction f = nullptr;
  CU_CHECK(cuModuleGetFunction(&f, current_->module, kernel));
  CU_CHECK(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, nullptr, args, nullptr));
}

void NvrtcTracer::composite(const float* d_hdr, int w, int h, float exposure, unsigned* d_out, bool flip_y) {
  int flip = flip_y ? 1 : 0;
  void* args[] = {&d_hdr, &w, &h, &exposure, &d_out, &flip};
  CU_CHECK(cuLaunchKernel(current_->composite, (w + 15) / 16, (h + 15) / 16, 1, 16, 16, 1, 0, nullptr, args, nullptr));
}
