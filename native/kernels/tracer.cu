// Generic CUDA ray tracer, compiled at runtime with NVRTC (src/nvrtc_tracer.cpp).
// It mirrors web/src/shaders/tracer.wgsl.ts; the geodesic integration itself is
// shared with the host tests through rm/geodesic.h.
//
// Compile-time inputs (NVRTC defines):
//   RM_REAL    float for the interactive view, double for reference renders
//   RM_METRIC  namespace of the generated metric; "metric.cuh" is that header
//   RM_RK45    1: trace_kernel integrates with adaptive RK45 instead of fixed-step RK4
//   RM_BATCH   1: also compile the validation batch kernels

#include "metric.cuh"
#include "rm/frame.h"
#include "rm/geodesic.h"

typedef RM_REAL Real;
typedef rm::RM_METRIC::Metric Metric;

#define PI_F 3.14159265359f

__device__ inline float fractf(float x) { return x - floorf(x); }
__device__ inline float mixf(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline float clampf(float x, float lo, float hi) { return fminf(fmaxf(x, lo), hi); }
__device__ inline float smoothstepf(float e0, float e1, float x) {
  float t = clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
  return t * t * (3.0f - 2.0f * t);
}

struct Rgb {
  float r, g, b;
};
__device__ inline Rgb rgb(float r, float g, float b) { return {r, g, b}; }
__device__ inline Rgb scale(Rgb c, float s) { return {c.r * s, c.g * s, c.b * s}; }
__device__ inline Rgb mixc(Rgb a, Rgb b, float t) { return {mixf(a.r, b.r, t), mixf(a.g, b.g, t), mixf(a.b, b.b, t)}; }

// --- Sky (artistic) ------------------------------------------------------------

__device__ float hash3(float px, float py, float pz) {
  px = fractf(px * 443.8975f);
  py = fractf(py * 397.2973f);
  pz = fractf(pz * 491.1871f);
  float d = px * (py + 19.19f) + py * (pz + 19.19f) + pz * (px + 19.19f);
  px += d;
  py += d;
  pz += d;
  return fractf(px * py * pz);
}

// Point-like stars: each cell of a grid around the sky may hold one star.
__device__ Rgb starfield(float dx, float dy, float dz) {
  const float s = 90.0f;
  float cx = floorf(dx * s), cy = floorf(dy * s), cz = floorf(dz * s);
  if (hash3(cx, cy, cz) < 0.93f) return rgb(0, 0, 0);
  float sx = cx + hash3(cx + 1.7f, cy + 1.7f, cz + 1.7f);
  float sy = cy + hash3(cx + 3.1f, cy + 3.1f, cz + 3.1f);
  float sz = cz + hash3(cx + 5.3f, cy + 5.3f, cz + 5.3f);
  float sl = sqrtf(sx * sx + sy * sy + sz * sz);
  sx /= sl;
  sy /= sl;
  sz /= sl;
  float ex = dx - sx, ey = dy - sy, ez = dz - sz;
  float d = sqrtf(ex * ex + ey * ey + ez * ez) * s;
  float intensity = expf(-d * d * 60.0f) * (0.15f + powf(hash3(cx * 4.321f, cy * 4.321f, cz * 4.321f), 5.0f) * 1.6f);
  float tint = hash3(cx * 8.765f, cy * 8.765f, cz * 8.765f);
  Rgb c = rgb(0.95f, 0.97f, 1.0f);
  if (tint < 0.2f) c = rgb(1.0f, 0.75f, 0.6f);
  else if (tint > 0.85f) c = rgb(0.65f, 0.8f, 1.0f);
  return scale(c, intensity);
}

// Latitude-longitude grid on the celestial sphere.
__device__ Rgb sky_grid(float dx, float dy, float dz) {
  float theta = acosf(clampf(dz, -1.0f, 1.0f));
  float phi = atan2f(dy, dx);
  float u = phi / (2.0f * PI_F) * 24.0f;
  float v = theta / PI_F * 12.0f;
  float cu = fabsf(fractf(u) - 0.5f);
  float cv = fabsf(fractf(v) - 0.5f);
  float line = 1.0f - smoothstepf(0.44f, 0.49f, fmaxf(cu, cv));
  int checker = ((int)floorf(u) + (int)floorf(v)) & 1;
  Rgb base = checker == 1 ? rgb(0.06f, 0.065f, 0.08f) : rgb(0.035f, 0.04f, 0.05f);
  return mixc(scale(rgb(0.85f, 0.62f, 0.32f), 0.55f), base, line);
}

__device__ Rgb background(unsigned flags, float dx, float dy, float dz) {
  if (flags & rm::FLAG_GRID) return sky_grid(dx, dy, dz);
  if (flags & rm::FLAG_STARS) return starfield(dx, dy, dz);
  return rgb(0, 0, 0);
}

// --- Disk emission ---------------------------------------------------------------

// Approximate sRGB chromaticity of a blackbody (Tanner Helland's fit), normalised.
__device__ Rgb blackbody(float temp_k) {
  float t = clampf(temp_k, 1000.0f, 40000.0f) / 100.0f;
  Rgb c;
  if (t <= 66.0f) {
    c.r = 1.0f;
    c.g = clampf(0.39008158f * logf(t) - 0.63184144f, 0.0f, 1.0f);
    c.b = t <= 19.0f ? 0.0f : clampf(0.54320679f * logf(t - 10.0f) - 1.19625409f, 0.0f, 1.0f);
  } else {
    c.r = clampf(1.29293619f * powf(t - 60.0f, -0.1332047592f), 0.0f, 1.0f);
    c.g = clampf(1.12989086f * powf(t - 60.0f, -0.0755148492f), 0.0f, 1.0f);
    c.b = 1.0f;
  }
  return scale(c, 1.0f / fmaxf(fmaxf(c.r, c.g), c.b));
}

__device__ float value_noise(float px, float py) {
  float ix = floorf(px), iy = floorf(py);
  float fx = px - ix, fy = py - iy;
  float ux = fx * fx * (3.0f - 2.0f * fx), uy = fy * fy * (3.0f - 2.0f * fy);
  float a = hash3(ix, iy, 1.0f);
  float b = hash3(ix + 1.0f, iy, 1.0f);
  float c = hash3(ix, iy + 1.0f, 1.0f);
  float d = hash3(ix + 1.0f, iy + 1.0f, 1.0f);
  return mixf(mixf(a, b, ux), mixf(c, d, ux), uy);
}

// Colour of the optically thick thin disk at the crossing y = (x, p), for gas
// with 4-velocity u. The observed/emitted energy ratio is g = 1 / (p . u)
// because p . u = 1 at the camera; bolometric brightness scales as g^4.
__device__ Rgb disk_color(const rm::FrameUniforms& U, const Real* params, const Real y[8], const Real u[4]) {
  const float r = (float)Metric::disk_radius(y, params);
  const float r_in = (float)Metric::disk_inner(y, params);
  const float omega = (float)Metric::disk_omega(y, params);
  const float shift = (float)(Real(1) / (y[4] * u[0] + y[5] * u[1] + y[6] * u[2] + y[7] * u[3]));
  const float disk_temp = (float)U.disk_temp;

  // Thin-disk flux profile F ~ (1 - sqrt(r_in / r)) / r^3 (Shakura-Sunyaev),
  // peak-normalised; local temperature T ~ F^(1/4).
  float flux = (1.0f - sqrtf(r_in / r)) * powf(r_in / r, 3.0f) / 0.05666f;
  float temp = disk_temp * powf(fmaxf(flux, 0.0f), 0.25f);
  float temp_obs = temp * shift;

  // Artistic turbulence that co-rotates with the gas.
  float phi = atan2f((float)y[2], (float)y[1]) - omega * (float)U.time;
  float swirl = value_noise(logf(r) * 9.0f, phi * 6.0f) * 0.6f + value_noise(logf(r) * 23.0f, phi * 15.0f) * 0.4f;
  float texture = 0.55f + 0.45f * swirl;

  if (U.flags & rm::FLAG_REDSHIFT) {
    float s = clampf((shift - 1.0f) * 1.6f, -1.0f, 1.0f);
    Rgb neutral = rgb(0.9f, 0.9f, 0.85f);
    Rgb c = s > 0.0f ? mixc(neutral, rgb(0.2f, 0.5f, 1.0f), s) : mixc(neutral, rgb(1.0f, 0.25f, 0.1f), -s);
    return scale(c, texture * 0.8f);
  }
  float brightness = powf(temp_obs / disk_temp, 4.0f) * texture;
  return scale(blackbody(temp_obs), brightness);
}

// --- Kernels ---------------------------------------------------------------------

// Camera placement and covariant frame, always in double (one thread).
extern "C" __global__ void camera_kernel(const double* params, double cam_radius, double yaw, double pitch,
                                         rm::CameraFrame* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) *out = rm::static_camera<Metric>(params, cam_radius, yaw, pitch);
}

// One ray per pixel. Writes linear HDR RGBA (row 0 = top of the image), and
// optionally the integration steps, fate (rm::RayFate) and the largest
// constraint drift |H| / p_t^2 along the ray per pixel. The integrator is fixed
// at compile time (RM_RK45), so the interactive RK4 kernel carries no RK45 code.
extern "C" __global__ void trace_kernel(rm::FrameUniforms U, float* hdr, unsigned* steps_out,
                                        unsigned char* fate_out, float* drift_out) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= U.width || py >= U.height) return;
  const int idx = py * U.width + px;

  Real params[rm::MAX_PARAMS];
  for (int i = 0; i < rm::MAX_PARAMS; i++) params[i] = Real(U.params[i]);

  // Same pixel-to-direction mapping as the WGSL tracer (uv.y flipped so +up is up).
  const Real aspect = Real(U.width) / Real(U.height);
  const Real ndc_x = (Real(px) + Real(0.5)) / Real(U.width) * Real(2) - Real(1);
  const Real ndc_y = (Real(1) - (Real(py) + Real(0.5)) / Real(U.height)) * Real(2) - Real(1);
  Real y0[8];
  rm::camera_ray<Real>(U.cam, ndc_x * aspect * Real(U.fov_scale), ndc_y * Real(U.fov_scale), Real(1), y0);

  rm::TraceOptions<Real> opt;
  opt.max_steps = U.max_steps;
  opt.step_fraction = Real(U.step_fraction);
  opt.min_step = Real(0.002) * Real(U.cam.length_scale);
  opt.escape_radius = Real(U.escape_radius);
  opt.disk = (U.flags & rm::FLAG_DISK) != 0;
  opt.rtol = Real(U.rtol);
  opt.length_scale = Real(U.cam.length_scale);
#if RM_RK45
  opt.integrator = rm::INTEGRATOR_RK45;
  const rm::TraceResult<Real> res = rm::trace_ray_rk45<Metric, Real>(params, y0, opt);
#else
  opt.integrator = rm::INTEGRATOR_RK4;
  const rm::TraceResult<Real> res = rm::trace_ray_rk4<Metric, Real>(params, y0, opt);
#endif

  Rgb c = rgb(0, 0, 0);
  if (res.fate == rm::FATE_DISK) {
    c = disk_color(U, params, res.y, res.u);
  } else if (res.fate != rm::FATE_CAPTURED) {
    // Escaped (or out of steps): the light arrived from the part of the sky
    // this backwards-traced ray is heading toward.
    Real dy[8];
    rm::geodesic_rhs<Metric>(params, res.y, dy);
    float dx = (float)dy[1], dyy = (float)dy[2], dz = (float)dy[3];
    float l = sqrtf(dx * dx + dyy * dyy + dz * dz);
    c = background(U.flags, dx / l, dyy / l, dz / l);
  }
  hdr[4 * idx + 0] = c.r;
  hdr[4 * idx + 1] = c.g;
  hdr[4 * idx + 2] = c.b;
  hdr[4 * idx + 3] = 1.0f;
  if (steps_out) steps_out[idx] = (unsigned)res.steps;
  if (fate_out) fate_out[idx] = (unsigned char)res.fate;
  if (drift_out) drift_out[idx] = (float)res.max_drift;
}

// --- Batch kernels for the validation suite (raymarcher --validate) ---------------
// Compiled only into modules built with RM_BATCH=1.
#if RM_BATCH

__device__ rm::TraceOptions<Real> batch_options(const rm::BatchOptions& o) {
  rm::TraceOptions<Real> opt;
  opt.max_steps = o.max_steps;
  opt.step_fraction = Real(o.step_fraction);
  opt.min_step = Real(o.min_step);
  opt.escape_radius = Real(o.escape_radius);
  opt.disk = o.disk != 0;
  opt.integrator = o.integrator;
  opt.rtol = Real(o.rtol);
  opt.length_scale = Real(o.length_scale);
  return opt;
}

__device__ void store_ray(const Real y0[8], const rm::TraceResult<Real>& res, rm::RayOut* out) {
  for (int i = 0; i < 8; i++) {
    out->y0[i] = double(y0[i]);
    out->y[i] = double(res.y[i]);
  }
  out->max_drift = double(res.max_drift);
  out->fate = res.fate;
  out->steps = res.steps;
  out->rejected = res.rejected;
}

// Rays that start at starts[i] with coordinate velocity along dirs[i] (rm::coordinate_ray).
extern "C" __global__ void trace_coordinate_rays_kernel(rm::BatchOptions o, const double* starts, const double* dirs,
                                                        int n, rm::RayOut* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Real params[rm::MAX_PARAMS], start[3], dir[3], y0[8];
  for (int k = 0; k < rm::MAX_PARAMS; k++) params[k] = Real(o.params[k]);
  for (int k = 0; k < 3; k++) {
    start[k] = Real(starts[3 * i + k]);
    dir[k] = Real(dirs[3 * i + k]);
  }
  rm::coordinate_ray<Metric>(params, start, dir, y0);
  store_ray(y0, rm::trace_ray<Metric>(params, y0, batch_options(o)), out + i);
}

// Rays leaving a static camera in local directions dirs[i] (right, up, forward).
extern "C" __global__ void trace_camera_rays_kernel(rm::BatchOptions o, rm::CameraFrame cam, const double* dirs, int n,
                                                    rm::RayOut* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Real params[rm::MAX_PARAMS], y0[8];
  for (int k = 0; k < rm::MAX_PARAMS; k++) params[k] = Real(o.params[k]);
  rm::camera_ray<Real>(cam, Real(dirs[3 * i]), Real(dirs[3 * i + 1]), Real(dirs[3 * i + 2]), y0);
  store_ray(y0, rm::trace_ray<Metric>(params, y0, batch_options(o)), out + i);
}

// g and dg at the points x[4 i .. 4 i + 3].
extern "C" __global__ void metric_eval_kernel(const double* params, const double* x, int n, double* g, double* dg) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Real p[rm::MAX_PARAMS], xi[4], gi[16], dgi[64];
  for (int k = 0; k < rm::MAX_PARAMS; k++) p[k] = Real(params[k]);
  for (int k = 0; k < 4; k++) xi[k] = Real(x[4 * i + k]);
  Metric::eval(xi, p, gi, dgi);
  for (int k = 0; k < 16; k++) g[16 * i + k] = double(gi[k]);
  for (int k = 0; k < 64; k++) dg[64 * i + k] = double(dgi[k]);
}
#endif  // RM_BATCH

// --- Composite: bloom, ACES tone mapping and display grading ---------------------

__device__ Rgb sample_bilinear(const float* hdr, int w, int h, float u, float v) {
  // Clamp-to-edge bilinear filtering, as the WebGPU sampler does.
  float x = u * w - 0.5f, y = v * h - 0.5f;
  int x0 = (int)floorf(x), y0 = (int)floorf(y);
  float fx = x - x0, fy = y - y0;
  Rgb acc = rgb(0, 0, 0);
  for (int j = 0; j < 2; j++)
    for (int i = 0; i < 2; i++) {
      int sx = min(max(x0 + i, 0), w - 1), sy = min(max(y0 + j, 0), h - 1);
      float wgt = (i ? fx : 1.0f - fx) * (j ? fy : 1.0f - fy);
      const float* p = hdr + 4 * (sy * w + sx);
      acc.r += p[0] * wgt;
      acc.g += p[1] * wgt;
      acc.b += p[2] * wgt;
    }
  return acc;
}

__device__ Rgb bright(Rgb c) {
  float l = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
  return scale(c, fmaxf(0.0f, l - 0.8f));
}

__device__ float aces(float x) { return clampf((x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f), 0.0f, 1.0f); }

// Writes packed RGBA8. flip_y != 0 puts row 0 at the bottom (OpenGL textures).
extern "C" __global__ void composite_kernel(const float* hdr, int w, int h, float exposure, unsigned* out, int flip_y) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= w || py >= h) return;
  const float u = (px + 0.5f) / w, v = (py + 0.5f) / h;
  Rgb bloom = rgb(0, 0, 0);
  for (int i = 0; i < 12; i++) {
    float a = i * 0.5236f;
    float rad = (i & 1) ? 6.0f : 2.5f;
    Rgb b = bright(sample_bilinear(hdr, w, h, u + cosf(a) * rad / w, v + sinf(a) * rad / h));
    bloom.r += b.r;
    bloom.g += b.g;
    bloom.b += b.b;
  }
  const float* base = hdr + 4 * (py * w + px);
  Rgb m;
  m.r = aces((base[0] + bloom.r / 12.0f * 0.8f) * exposure);
  m.g = aces((base[1] + bloom.g / 12.0f * 0.8f) * exposure);
  m.b = aces((base[2] + bloom.b / 12.0f * 0.8f) * exposure);
  // Colour grading, not physics: restore some saturation after ACES.
  float luma = 0.2126f * m.r + 0.7152f * m.g + 0.0722f * m.b;
  Rgb graded = rgb(clampf(mixf(luma, m.r, 1.7f), 0.0f, 1.0f), clampf(mixf(luma, m.g, 1.7f), 0.0f, 1.0f),
                   clampf(mixf(luma, m.b, 1.7f), 0.0f, 1.0f));
  unsigned r8 = (unsigned)(powf(graded.r, 1.0f / 2.2f) * 255.0f + 0.5f);
  unsigned g8 = (unsigned)(powf(graded.g, 1.0f / 2.2f) * 255.0f + 0.5f);
  unsigned b8 = (unsigned)(powf(graded.b, 1.0f / 2.2f) * 255.0f + 0.5f);
  const int row = flip_y ? (h - 1 - py) : py;
  out[row * w + px] = r8 | (g8 << 8) | (b8 << 16) | (255u << 24);
}
