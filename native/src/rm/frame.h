// Per-frame inputs of the NVRTC tracer, passed to the kernel by value. Shared
// by the host and the runtime-compiled kernel, so it holds only plain types
// (same layout under MSVC/GCC x64 and NVRTC). Mirrors the WGSL Uniforms struct.
#pragma once

#include "rm/geodesic.h"

namespace rm {

constexpr int MAX_PARAMS = 8;

enum FrameFlags : unsigned {
  FLAG_DISK = 1u,
  FLAG_GRID = 2u,
  FLAG_REDSHIFT = 4u,
  FLAG_STARS = 8u,
};

struct FrameUniforms {
  CameraFrame cam;
  double params[MAX_PARAMS];
  double time;           // seconds; only moves the artistic disk texture
  double fov_scale;      // tan of half the vertical field of view
  double step_fraction;  // RK4 step as a fraction of the coordinate radius
  double escape_radius;
  double exposure;
  double disk_temp;  // peak disk temperature in kelvin
  int width;
  int height;
  int max_steps;
  unsigned flags;
  double rtol;  // RK45 relative tolerance (modules compiled with RM_RK45=1)
};

/** Options for the batch kernels the validation suite launches (fp64 modules). */
struct BatchOptions {
  double params[MAX_PARAMS];
  double step_fraction;
  double min_step;
  double escape_radius;
  double rtol;
  double length_scale;
  int max_steps;
  int integrator;
  int disk;
  int pad_;
};

/** Result of one ray in a batch. */
struct RayOut {
  double y0[8];  // initial state
  double y[8];   // final state
  double max_drift;
  int fate;
  int steps;
  int rejected;
  int pad_;
};

}  // namespace rm
