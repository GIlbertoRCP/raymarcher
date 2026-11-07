// Generic CUDA ray tracer, compiled at runtime with NVRTC (src/nvrtc_tracer.cpp).
// It mirrors web/src/shaders/tracer.wgsl.ts; the geodesic integration itself is
// shared with the host tests through rm/geodesic.h.
//
// Compile-time inputs (NVRTC defines):
//   RM_REAL    float for the interactive view, double for reference renders
//   RM_METRIC  namespace of the generated metric; "metric.cuh" is that header
//   RM_RK45    1: trace_kernel integrates with adaptive RK45 instead of fixed-step RK4
//   RM_BATCH   1: also compile the validation batch kernels
//   RM_TANGENT_P  > 0 (with RM_BATCH): also the tangent batch kernel, P directions (rm/tangent.h)

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
  opt.crossing = rm::CROSSING_LINEAR;  // as the WGSL tracer
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
  opt.crossing = o.crossing;
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

// --- Tangent batch (Phase 4): modules built with RM_BATCH=1 and RM_TANGENT_P > 0 --------------
#if RM_BATCH && defined(RM_TANGENT_P) && RM_TANGENT_P > 0
#include "rm/tangent.h"

// Camera rays from theta[i] (rm::MAX_THETA values: metric parameters, camera
// radius, inclination, yaw, u, v) with tangents along dirs[d] (P x MAX_THETA,
// shared by every ray). MODE 0 (trace_tangent_rays_kernel): symbolic tangents
// (compiler JVP, RK4); MODE 1 (trace_dual_rays_kernel): Dual numbers through
// rm::trace_ray (RK4 or RK45, from o.integrator). Separate kernels, so each gets
// its own register allocation.
// out[(16 + 16 P) i ..] = y0, S0 (8 P), y, S (8 P), S[8 d + j] = dy_j along d;
// meta[2 i] = fate (-1 when there is no static observer), meta[2 i + 1] = steps.
template <int MODE>
__device__ void tangent_ray_record(rm::BatchOptions o, const double* thetas, const double* dirs, int n, double* out, int* meta) {
  constexpr int P = RM_TANGENT_P;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double(*dd)[rm::MAX_THETA] = reinterpret_cast<const double(*)[rm::MAX_THETA]>(dirs);
  const double* theta = thetas + rm::MAX_THETA * i;
  const rm::TraceOptions<Real> opt = batch_options(o);
  double* rec = out + size_t(16 + 16 * P) * i;
  bool ok = false;
  Real y0[8], S0[8 * P];
  if (MODE == 0) {
    rm::TangentResult<Real, P> r;
    rm::tangent_ray_symbolic<Metric, Real, P>(theta, dd, opt, y0, S0, r, ok);
    if (ok) {
      for (int k = 0; k < 8; k++) rec[8 + 8 * P + k] = double(r.y[k]);
      for (int k = 0; k < 8 * P; k++) rec[16 + 8 * P + k] = double(r.S[k]);
      meta[2 * i] = r.fate;
      meta[2 * i + 1] = r.steps;
    }
  } else {
    constexpr int NP = Metric::PARAM_COUNT > 0 ? Metric::PARAM_COUNT : 1;
    rm::Dual<Real, P> dp[NP], dy0[8];
    ok = rm::dual_camera_ray<Metric, Real, P>(theta, dd, dp, dy0);
    if (ok) {
      rm::split_duals<Real, P>(dy0, 8, y0, S0);
      const rm::TraceResult<rm::Dual<Real, P>> r = rm::trace_ray<Metric>(dp, dy0, rm::dual_options<Real, P>(opt));
      for (int k = 0; k < 8; k++) {
        rec[8 + 8 * P + k] = double(r.y[k].v);
        for (int d = 0; d < P; d++) rec[16 + 8 * P + 8 * d + k] = double(r.y[k].d[d]);
      }
      meta[2 * i] = r.fate;
      meta[2 * i + 1] = r.steps;
    }
  }
  if (!ok) {
    meta[2 * i] = -1;
    meta[2 * i + 1] = 0;
    return;
  }
  for (int k = 0; k < 8; k++) rec[k] = double(y0[k]);
  for (int k = 0; k < 8 * P; k++) rec[8 + k] = double(S0[k]);
}

extern "C" __global__ void trace_tangent_rays_kernel(rm::BatchOptions o, const double* thetas, const double* dirs, int n,
                                                     double* out, int* meta) {
  tangent_ray_record<0>(o, thetas, dirs, n, out, meta);
}
extern "C" __global__ void trace_dual_rays_kernel(rm::BatchOptions o, const double* thetas, const double* dirs, int n,
                                                  double* out, int* meta) {
  tangent_ray_record<1>(o, thetas, dirs, n, out, meta);
}

#include "rm/emission.h"

// Model-sky samples (Phase 4, step 4; compiler/src/image-model.ts): sample s is the
// camera ray with (u, v) = xi[2 s .. 2 s + 1] / R for the camera in theta_base
// (rm::MAX_THETA values; R = theta_base[NP + THETA_RADIUS]). Writes the disk
// intensity I[s] (0 unless the ray crosses the disk; rm::thin_disk_intensity),
// meta[3 s ..] = (fate, steps, crossings), and for the dual kernel the
// derivatives dI[P s + d] along dirs[d] (P x MAX_THETA).
extern "C" __global__ void render_samples_dual_kernel(rm::BatchOptions o, const double* theta_base, const double* dirs,
                                                      const double* xi, int n, double taper, double* I, double* dI,
                                                      int* meta) {
  constexpr int P = RM_TANGENT_P;
  constexpr int NP = Metric::PARAM_COUNT;
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n) return;
  double theta[rm::MAX_THETA];
  for (int k = 0; k < rm::MAX_THETA; k++) theta[k] = theta_base[k];
  const double R = theta[NP + rm::THETA_RADIUS];
  theta[NP + rm::THETA_U] = xi[2 * s] / R;
  theta[NP + rm::THETA_V] = xi[2 * s + 1] / R;
  const double(*dd)[rm::MAX_THETA] = reinterpret_cast<const double(*)[rm::MAX_THETA]>(dirs);
  typedef rm::Dual<Real, P> D;
  D dp[NP > 0 ? NP : 1], dy0[8];
  const bool ok = rm::dual_camera_ray<Metric, Real, P>(theta, dd, dp, dy0);
  D val = D(Real(0));
  int fate = -1, steps = 0, crossings = 0;
  if (ok) {
    const rm::TraceResult<D> r = rm::trace_ray<Metric>(dp, dy0, rm::dual_options<Real, P>(batch_options(o)));
    fate = r.fate;
    steps = r.steps;
    crossings = r.crossings;
    if (r.fate == rm::FATE_DISK) val = rm::thin_disk_intensity<Metric, D>(dp, r.y, D(Real(taper)));
  }
  I[s] = double(val.v);
  for (int d = 0; d < P; d++) dI[P * s + d] = double(val.d[d]);
  meta[3 * s] = fate;
  meta[3 * s + 1] = steps;
  meta[3 * s + 2] = crossings;
}

/** The camera of theta_base, in double (as camera_kernel). */
__device__ rm::CameraFrame sample_camera(const double* theta_base, Real params[rm::MAX_PARAMS]) {
  constexpr int NP = Metric::PARAM_COUNT;
  double pd[rm::MAX_PARAMS] = {};
  for (int k = 0; k < NP; k++) pd[k] = theta_base[k];
  for (int k = 0; k < rm::MAX_PARAMS; k++) params[k] = Real(pd[k]);
  return rm::static_camera<Metric>(pd, theta_base[NP + rm::THETA_RADIUS], theta_base[NP + rm::THETA_YAW],
                                   rm::TANGENT_PI / 2 - theta_base[NP + rm::THETA_INCLINATION]);
}

/**
 * Intensity of the sample at model-sky (x, y). branch is 0 for rays that carry
 * no light (escaped, captured, unfinished) and 1 + crossings for disk hits:
 * inside one branch the intensity is a smooth function of position and
 * parameters, so branch changes are where the edge term lives.
 */
__device__ Real plain_sample(const rm::BatchOptions& o, const rm::CameraFrame& cam, const Real* params, double R, double x,
                             double y, double taper, int& fate, int& steps, int& crossings) {
  fate = -1;
  steps = crossings = 0;
  if (!cam.ok) return Real(0);
  Real y0[8];
  rm::camera_ray<Real>(cam, Real(x / R), Real(y / R), Real(1), y0);
  const rm::TraceResult<Real> r = rm::trace_ray<Metric>(params, y0, batch_options(o));
  fate = r.fate;
  steps = r.steps;
  crossings = r.crossings;
  return r.fate == rm::FATE_DISK ? rm::thin_disk_intensity<Metric, Real>(params, r.y, Real(taper)) : Real(0);
}

__device__ inline int branch_of(int fate, int crossings) { return fate == rm::FATE_DISK ? 1 + crossings : 0; }

extern "C" __global__ void render_samples_plain_kernel(rm::BatchOptions o, const double* theta_base, const double* xi, int n,
                                                       double taper, double* I, int* meta) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n) return;
  Real params[rm::MAX_PARAMS];
  const rm::CameraFrame cam = sample_camera(theta_base, params);
  const double R = theta_base[Metric::PARAM_COUNT + rm::THETA_RADIUS];
  int fate, steps, crossings;
  I[s] = double(plain_sample(o, cam, params, R, xi[2 * s], xi[2 * s + 1], taper, fate, steps, crossings));
  meta[3 * s] = fate;
  meta[3 * s + 1] = steps;
  meta[3 * s + 2] = crossings;
}

// Edge location (Phase 4, step 5): for segment i from (seg[4 i], seg[4 i + 2]) to
// (seg[4 i + 1], seg[4 i + 2]) on the model sky, whose ends lie on different
// branches, bisects `iters` times on "same branch as the start" and writes
// out[4 i ..] = (crossing x, intensity at the start side, at the end side, flag).
// seg[4 i + 3] is the expected start branch; flag = 1 when the start or end branch
// differs from what the caller saw (the boundary left the segment).
extern "C" __global__ void edge_bisect_kernel(rm::BatchOptions o, const double* theta_base, const double* seg, int n, int iters,
                                              double taper, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Real params[rm::MAX_PARAMS];
  const rm::CameraFrame cam = sample_camera(theta_base, params);
  const double R = theta_base[Metric::PARAM_COUNT + rm::THETA_RADIUS];
  double a = seg[4 * i], b = seg[4 * i + 1];
  const double y = seg[4 * i + 2];
  int fate, steps, crossings;
  Real Ia = plain_sample(o, cam, params, R, a, y, taper, fate, steps, crossings);
  const int ba = branch_of(fate, crossings);
  Real Ib = plain_sample(o, cam, params, R, b, y, taper, fate, steps, crossings);
  const int bb = branch_of(fate, crossings);
  double flag = (ba != int(seg[4 * i + 3]) || bb == ba) ? 1 : 0;
  for (int it = 0; it < iters && flag == 0; it++) {
    const double m = 0.5 * (a + b);
    const Real Im = plain_sample(o, cam, params, R, m, y, taper, fate, steps, crossings);
    if (branch_of(fate, crossings) == ba) {
      a = m;
      Ia = Im;
    } else {
      b = m;
      Ib = Im;
    }
  }
  out[4 * i] = 0.5 * (a + b);
  out[4 * i + 1] = double(Ia);
  out[4 * i + 2] = double(Ib);
  out[4 * i + 3] = flag;
}

// Critical curve (Phase 4, step 5): along ray i from the model-sky point (ray[4 i], ray[4 i + 1])
// in direction (ray[4 i + 2], ray[4 i + 3]), bisects `iters` times between distances t0 and t1
// on "captured" (call with the disk off) and writes out[2 i ..] = (distance, flag); flag = 1
// unless t0 is captured and t1 is not.
extern "C" __global__ void curve_bisect_kernel(rm::BatchOptions o, const double* theta_base, const double* ray, int n, double t0,
                                               double t1, int iters, double* out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Real params[rm::MAX_PARAMS];
  const rm::CameraFrame cam = sample_camera(theta_base, params);
  const double R = theta_base[Metric::PARAM_COUNT + rm::THETA_RADIUS];
  const double ox = ray[4 * i], oy = ray[4 * i + 1], ex = ray[4 * i + 2], ey = ray[4 * i + 3];
  int fate, steps, crossings;
  auto captured = [&](double t) {
    plain_sample(o, cam, params, R, ox + t * ex, oy + t * ey, 0, fate, steps, crossings);
    return fate == rm::FATE_CAPTURED;
  };
  double a = t0, b = t1;
  const double flag = (captured(a) && !captured(b)) ? 0 : 1;
  for (int it = 0; it < iters && flag == 0; it++) {
    const double m = 0.5 * (a + b);
    if (captured(m)) a = m;
    else b = m;
  }
  out[2 * i] = 0.5 * (a + b);
  out[2 * i + 1] = flag;
}
#endif

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
