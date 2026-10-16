// Runtime compilation of the generic tracer (kernels/tracer.cu) for one
// generated metric with NVRTC, and launches through the CUDA driver API.
// Switching metric or precision compiles a new module; nothing is rebuilt.
#pragma once

#include <cuda.h>

#include <map>
#include <string>
#include <vector>

#include "rm/frame.h"

struct ParamInfo {
  std::string name;
  std::string label;
  double value, min, max;
};

/** One entry of native/generated/catalog.json. */
struct CatalogEntry {
  std::string id;  // file name in metrics/, and the C++ namespace
  std::string name;
  std::string description;
  std::string header;
  bool has_horizon = false;
  bool has_disk = false;
  std::vector<ParamInfo> params;
};

/** Locations of the runtime sources: tracer.cu, rm/*.h and generated/. */
struct SourcePaths {
  std::string kernels;    // directory holding tracer.cu
  std::string include;    // directory holding rm/geodesic.h and rm/frame.h
  std::string generated;  // native/generated
  static SourcePaths locate(const std::string& exe_dir);
};

std::vector<CatalogEntry> load_catalog(const SourcePaths& paths);

enum class Precision { Float, Double };

/** Variants of the runtime-compiled module (each is compiled and cached separately). */
struct ModuleOptions {
  bool rk45 = false;   // trace_kernel uses adaptive RK45 (RM_RK45)
  bool batch = false;  // include the validation batch kernels (RM_BATCH)
};

/** Static resources of the selected trace kernel. */
struct KernelInfo {
  int registers = 0;
  int local_bytes = 0;
  double occupancy = 0;  // theoretical: resident threads / max threads per SM
};

class NvrtcTracer {
 public:
  explicit NvrtcTracer(SourcePaths paths);
  ~NvrtcTracer();

  /** Compiles (or reuses) the module for this metric and precision. Returns false and sets error() on failure. */
  bool select(const CatalogEntry& metric, Precision precision, ModuleOptions options = {});
  const std::string& error() const { return error_; }
  /** Milliseconds NVRTC took for the selected module (0 when it came from the cache). */
  double last_compile_ms() const { return last_compile_ms_; }
  std::string device_arch() const { return arch_; }

  /** Camera position and covariant frame in double, from the generated metric. */
  rm::CameraFrame camera(const double* params, double cam_radius, double yaw, double pitch);

  /** Traces one frame into d_hdr (RGBA float). d_steps, d_fate and d_drift may be null. */
  void trace(const rm::FrameUniforms& u, float* d_hdr, unsigned* d_steps, unsigned char* d_fate, float* d_drift,
             int block_x = 16, int block_y = 16);
  /** Registers, local memory and theoretical occupancy of trace_kernel at this block size. */
  KernelInfo trace_kernel_info(int threads_per_block) const;
  /** Launches a kernel of the selected module by name on a 1-D grid. */
  void launch(const char* kernel, unsigned grid, unsigned block, void** args);
  /** Bloom, tone mapping and grading into packed RGBA8 (d_out). */
  void composite(const float* d_hdr, int w, int h, float exposure, unsigned* d_out, bool flip_y);

 private:
  struct Module {
    CUmodule module = nullptr;
    CUfunction trace = nullptr, composite = nullptr, camera = nullptr;
  };
  SourcePaths paths_;
  std::string arch_;
  std::string tracer_source_, geodesic_source_, frame_source_;
  std::map<std::string, Module> modules_;
  Module* current_ = nullptr;
  std::string error_;
  double last_compile_ms_ = 0;
  double* d_params_ = nullptr;
  rm::CameraFrame* d_camera_ = nullptr;
};
