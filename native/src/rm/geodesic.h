// Generic null-geodesic integration for any compiled metric. Header-only and
// RM_HD, so the same code runs in the NVRTC tracer (float or double) and in the
// host tests. It mirrors compiler/src/geodesic.ts and the WGSL tracer
// (web/src/shaders/tracer.wgsl.ts):
//
//   state y = (x^mu, p_mu)
//   dx^mu/dl = g^{mu nu} p_nu,   dp_mu/dl = 1/2 (d_mu g_ab) k^a k^b,   k = g^{-1} p
//
// M is a generated rm::<id>::Metric (native/generated/<id>.cuh).
#pragma once

#ifndef RM_HD
#if defined(__CUDACC__)
#define RM_HD __host__ __device__
#else
#define RM_HD
#endif
#endif

#ifndef __CUDACC_RTC__
#include <cmath>
#endif

namespace rm {

#ifndef __CUDACC_RTC__
using std::cos;
using std::fabs;
using std::pow;
using std::sin;
using std::sqrt;
#endif

/** Inverse of a 4x4 matrix stored row-major (same expansion as invert4 in geodesic.ts). */
template <typename Real>
RM_HD inline void invert4(const Real m[16], Real out[16]) {
  const Real a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const Real a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const Real a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const Real a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const Real b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const Real b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const Real b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const Real b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const Real det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  const Real s = Real(1) / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * s;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * s;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * s;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * s;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * s;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * s;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * s;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * s;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * s;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * s;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * s;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * s;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * s;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * s;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * s;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * s;
}

/** g_ab u^a v^b. */
template <typename Real>
RM_HD inline Real dot4(const Real g[16], const Real u[4], const Real v[4]) {
  Real s = Real(0);
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) s += g[4 * i + j] * u[i] * v[j];
  return s;
}

/**
 * Writes dy/dlambda for state y into dy and returns H = 1/2 g^{mu nu} p_mu p_nu,
 * using the fused right-hand side the compiler generates for the metric
 * (symmetric, without the derivatives that are identically zero).
 */
template <class M, typename Real>
RM_HD inline Real geodesic_rhs_generic(const Real* params, const Real y[8], Real dy[8]);

template <class M, typename Real>
RM_HD inline Real geodesic_rhs(const Real* params, const Real y[8], Real dy[8]) {
#if defined(RM_GENERIC_RHS) && RM_GENERIC_RHS
  return geodesic_rhs_generic<M>(params, y, dy);  // for A/B measurements (RM_NVRTC_FLAGS=-DRM_GENERIC_RHS=1)
#else
  return M::rhs(y, params, dy);
#endif
}

/** The same from metric_eval, a numeric inverse and full 4x4 contractions (reference for tests). */
template <class M, typename Real>
RM_HD inline Real geodesic_rhs_generic(const Real* params, const Real y[8], Real dy[8]) {
  Real g[16], dg[64], gi[16];
  M::eval(y, params, g, dg);
  invert4(g, gi);
  for (int mu = 0; mu < 4; mu++) {
    dy[mu] = gi[4 * mu] * y[4] + gi[4 * mu + 1] * y[5] + gi[4 * mu + 2] * y[6] + gi[4 * mu + 3] * y[7];
  }
  for (int c = 0; c < 4; c++) dy[4 + c] = Real(0.5) * dot4(dg + 16 * c, dy, dy);
  return Real(0.5) * (dy[0] * y[4] + dy[1] * y[5] + dy[2] * y[6] + dy[3] * y[7]);
}

enum RayFate : int { FATE_ESCAPED = 0, FATE_CAPTURED = 1, FATE_DISK = 2, FATE_UNFINISHED = 3 };

enum Integrator : int { INTEGRATOR_RK4 = 0, INTEGRATOR_RK45 = 1 };

template <typename Real>
struct TraceOptions {
  int max_steps;       // RK4: steps; RK45: step attempts (accepted + rejected)
  Real step_fraction;  // RK4 step, and the RK45 first step, as a fraction of the coordinate radius
  Real min_step;
  Real escape_radius;
  bool disk;  // stop where the ray crosses the disk (z = 0, inner <= r <= outer)
  int integrator;      // Integrator; RK4 when zero-initialised
  Real rtol;           // RK45 relative tolerance
  Real length_scale;   // RK45 absolute tolerance on x is rtol * length_scale
};

template <typename Real>
struct TraceResult {
  RayFate fate;
  int steps;     // accepted steps
  int rejected;  // RK45 rejected step attempts
  Real max_drift;  // max |H| / p_t^2 over the accepted states (H = 0 for light)
  Real y[8];  // final state; the crossing point for FATE_DISK
  Real u[4];  // gas 4-velocity at the crossing (FATE_DISK only)
};

/**
 * 4-velocity of gas on a circular orbit with angular velocity omega at x:
 * u ~ (1, -omega y, omega x, 0). Returns false where that direction is not
 * timelike (no circular orbit).
 */
template <class M, typename Real>
RM_HD inline bool disk_gas_velocity(const Real* params, const Real x[4], Real u[4]) {
  const Real omega = M::disk_omega(x, params);
  Real v[4] = {Real(1), -omega * x[2], omega * x[1], Real(0)};
  Real g[16], dg[64];
  M::eval(x, params, g, dg);
  const Real norm = dot4(g, v, v);
  if (!(norm < Real(0))) return false;
  const Real s = Real(1) / sqrt(-norm);
  for (int i = 0; i < 4; i++) u[i] = v[i] * s;
  return true;
}

/**
 * Fixed-step RK4 trace of one ray, step by step as in the WGSL tracer:
 * captured inside the horizon or when dt/dlambda diverges, escaped beyond
 * escape_radius, disk at the first crossing of z = 0 inside the disk.
 */
template <class M, typename Real>
RM_HD inline TraceResult<Real> trace_ray_rk4(const Real* params, const Real y0[8], const TraceOptions<Real>& opt) {
  TraceResult<Real> res;
  Real* y = res.y;
  for (int i = 0; i < 8; i++) y[i] = y0[i];
  res.fate = FATE_UNFINISHED;
  res.steps = opt.max_steps;
  res.rejected = 0;
  res.max_drift = Real(0);
  const Real inv_pt2 = Real(1) / (y0[4] * y0[4]);
  Real k1[8], k2[8], k3[8], k4[8], tmp[8];

  for (int step = 0; step < opt.max_steps; step++) {
    if (M::HAS_HORIZON && M::horizon_value(y, params) < Real(0)) {
      res.fate = FATE_CAPTURED;
      res.steps = step;
      return res;
    }
    const Real r = sqrt(y[1] * y[1] + y[2] * y[2] + y[3] * y[3]);
    if (r > opt.escape_radius) {
      res.fate = FATE_ESCAPED;
      res.steps = step;
      return res;
    }
    Real h = opt.step_fraction * r;
    if (h < opt.min_step) h = opt.min_step;

    const Real drift = fabs(geodesic_rhs<M>(params, y, k1)) * inv_pt2;
    // Traced backwards, a ray heading into a horizon piles up against it while
    // dt/dlambda diverges. Treat a diverging or non-finite dt/dlambda as captured.
    if (!(fabs(k1[0]) < Real(1.0e4))) {
      res.fate = FATE_CAPTURED;
      res.steps = step;
      return res;
    }
    if (drift > res.max_drift) res.max_drift = drift;
    for (int i = 0; i < 8; i++) tmp[i] = y[i] + Real(0.5) * h * k1[i];
    geodesic_rhs<M>(params, tmp, k2);
    for (int i = 0; i < 8; i++) tmp[i] = y[i] + Real(0.5) * h * k2[i];
    geodesic_rhs<M>(params, tmp, k3);
    for (int i = 0; i < 8; i++) tmp[i] = y[i] + h * k3[i];
    geodesic_rhs<M>(params, tmp, k4);
    for (int i = 0; i < 8; i++) tmp[i] = y[i] + (h / Real(6)) * (k1[i] + Real(2) * k2[i] + Real(2) * k3[i] + k4[i]);

    if (M::HAS_DISK && opt.disk && y[3] * tmp[3] <= Real(0) && y[3] != tmp[3]) {
      const Real f = y[3] / (y[3] - tmp[3]);
      Real hit[8];
      for (int i = 0; i < 8; i++) hit[i] = y[i] + f * (tmp[i] - y[i]);
      const Real rd = M::disk_radius(hit, params);
      if (rd >= M::disk_inner(hit, params) && rd <= M::disk_outer(hit, params) &&
          disk_gas_velocity<M>(params, hit, res.u)) {
        for (int i = 0; i < 8; i++) y[i] = hit[i];
        res.fate = FATE_DISK;
        res.steps = step + 1;
        return res;
      }
    }
    for (int i = 0; i < 8; i++) y[i] = tmp[i];
  }
  return res;
}

/**
 * Adaptive Dormand-Prince 5(4) trace (Hairer, Norsett & Wanner, Solving ODEs I,
 * with the DOPRI5 dense output), for reference renders in double precision.
 * The local error estimate is held below
 *   |err_i| <= atol_i + rtol * max(|y_i|, |y_new_i|),
 * with atol = rtol * length_scale for x and rtol * |p_t| for p. Steps are
 * accepted or rejected on that test; disk crossings are located on the dense
 * output (4th order) rather than by linear interpolation.
 */
template <class M, typename Real>
RM_HD inline TraceResult<Real> trace_ray_rk45(const Real* params, const Real y0[8], const TraceOptions<Real>& opt) {
  // Butcher tableau
  const Real a21 = Real(1) / 5;
  const Real a31 = Real(3) / 40, a32 = Real(9) / 40;
  const Real a41 = Real(44) / 45, a42 = Real(-56) / 15, a43 = Real(32) / 9;
  const Real a51 = Real(19372) / 6561, a52 = Real(-25360) / 2187, a53 = Real(64448) / 6561, a54 = Real(-212) / 729;
  const Real a61 = Real(9017) / 3168, a62 = Real(-355) / 33, a63 = Real(46732) / 5247, a64 = Real(49) / 176,
             a65 = Real(-5103) / 18656;
  const Real a71 = Real(35) / 384, a73 = Real(500) / 1113, a74 = Real(125) / 192, a75 = Real(-2187) / 6784,
             a76 = Real(11) / 84;
  // 5th-order minus embedded 4th-order weights
  const Real e1 = Real(71) / 57600, e3 = Real(-71) / 16695, e4 = Real(71) / 1920, e5 = Real(-17253) / 339200,
             e6 = Real(22) / 525, e7 = Real(-1) / 40;
  // Dense output
  const Real d1 = Real(-12715105075.0) / Real(11282082432.0), d3 = Real(87487479700.0) / Real(32700410799.0),
             d4 = Real(-10690763975.0) / Real(1880347072.0), d5 = Real(701980252875.0) / Real(199316789632.0),
             d6 = Real(-1453857185.0) / Real(822651844.0), d7 = Real(69997945.0) / Real(29380423.0);

  TraceResult<Real> res;
  Real* y = res.y;
  for (int i = 0; i < 8; i++) y[i] = y0[i];
  res.fate = FATE_UNFINISHED;
  res.steps = 0;
  res.rejected = 0;
  res.max_drift = Real(0);
  const Real pt = fabs(y0[4]);
  const Real inv_pt2 = Real(1) / (pt * pt);
  Real k1[8], k2[8], k3[8], k4[8], k5[8], k6[8], k7[8], tmp[8], ynew[8];

  Real drift = fabs(geodesic_rhs<M>(params, y, k1)) * inv_pt2;
  Real h = opt.step_fraction * sqrt(y[1] * y[1] + y[2] * y[2] + y[3] * y[3]);
  if (h < opt.min_step) h = opt.min_step;

  for (int attempt = 0; attempt < opt.max_steps; attempt++) {
    if (M::HAS_HORIZON && M::horizon_value(y, params) < Real(0)) {
      res.fate = FATE_CAPTURED;
      return res;
    }
    if (sqrt(y[1] * y[1] + y[2] * y[2] + y[3] * y[3]) > opt.escape_radius) {
      res.fate = FATE_ESCAPED;
      return res;
    }
    if (!(fabs(k1[0]) < Real(1.0e4))) {
      res.fate = FATE_CAPTURED;
      return res;
    }
    if (drift > res.max_drift) res.max_drift = drift;

    for (int i = 0; i < 8; i++) tmp[i] = y[i] + h * a21 * k1[i];
    geodesic_rhs<M>(params, tmp, k2);
    for (int i = 0; i < 8; i++) tmp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
    geodesic_rhs<M>(params, tmp, k3);
    for (int i = 0; i < 8; i++) tmp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    geodesic_rhs<M>(params, tmp, k4);
    for (int i = 0; i < 8; i++) tmp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    geodesic_rhs<M>(params, tmp, k5);
    for (int i = 0; i < 8; i++)
      tmp[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
    geodesic_rhs<M>(params, tmp, k6);
    for (int i = 0; i < 8; i++)
      ynew[i] = y[i] + h * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
    const Real drift_new = fabs(geodesic_rhs<M>(params, ynew, k7)) * inv_pt2;  // FSAL: k7 = f(ynew)

    Real err = Real(0);
    for (int i = 0; i < 8; i++) {
      const Real atol = opt.rtol * (i < 4 ? opt.length_scale : pt);
      const Real ay = fabs(y[i]), an = fabs(ynew[i]);
      const Real sc = atol + opt.rtol * (ay > an ? ay : an);
      const Real e = fabs(h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i])) / sc;
      if (!(e <= err)) err = e;  // also propagates NaN
    }

    if (!(err == err)) {  // NaN: the step left the domain (e.g. into the horizon); shrink hard
      h *= Real(0.2);
      res.rejected++;
      continue;
    }
    if (err > Real(1)) {
      Real f = Real(0.9) * pow(err, Real(-0.2));
      h *= f < Real(0.2) ? Real(0.2) : f;
      res.rejected++;
      continue;
    }

    if (M::HAS_DISK && opt.disk && y[3] * ynew[3] <= Real(0) && y[3] != ynew[3]) {
      // Dense output y(theta) = r1 + theta (r2 + (1 - theta) (r3 + theta (r4 + (1 - theta) r5))).
      Real r2[8], r3[8], r4[8], r5[8];
      for (int i = 0; i < 8; i++) {
        r2[i] = ynew[i] - y[i];
        r3[i] = h * k1[i] - r2[i];
        r4[i] = r2[i] - h * k7[i] - r3[i];
        r5[i] = h * (d1 * k1[i] + d3 * k3[i] + d4 * k4[i] + d5 * k5[i] + d6 * k6[i] + d7 * k7[i]);
      }
      // Bisection on z(theta) = 0, which changes sign on [0, 1].
      Real lo = Real(0), hi = Real(1);
      const bool lo_positive = y[3] > Real(0);
      for (int it = 0; it < 60; it++) {
        const Real th = Real(0.5) * (lo + hi);
        const Real z = y[3] + th * (r2[3] + (Real(1) - th) * (r3[3] + th * (r4[3] + (Real(1) - th) * r5[3])));
        if ((z > Real(0)) == lo_positive) lo = th;
        else hi = th;
      }
      const Real th = Real(0.5) * (lo + hi);
      Real hit[8];
      for (int i = 0; i < 8; i++) hit[i] = y[i] + th * (r2[i] + (Real(1) - th) * (r3[i] + th * (r4[i] + (Real(1) - th) * r5[i])));
      const Real rd = M::disk_radius(hit, params);
      if (rd >= M::disk_inner(hit, params) && rd <= M::disk_outer(hit, params) &&
          disk_gas_velocity<M>(params, hit, res.u)) {
        for (int i = 0; i < 8; i++) y[i] = hit[i];
        res.fate = FATE_DISK;
        res.steps++;
        return res;
      }
    }

    for (int i = 0; i < 8; i++) {
      y[i] = ynew[i];
      k1[i] = k7[i];
    }
    drift = drift_new;
    res.steps++;
    Real f = err > Real(0) ? Real(0.9) * pow(err, Real(-0.2)) : Real(5);
    h *= f > Real(5) ? Real(5) : (f < Real(0.2) ? Real(0.2) : f);
  }
  return res;
}

/** Traces one ray with the integrator the options ask for. */
template <class M, typename Real>
RM_HD inline TraceResult<Real> trace_ray(const Real* params, const Real y0[8], const TraceOptions<Real>& opt) {
  if (opt.integrator == INTEGRATOR_RK45) return trace_ray_rk45<M>(params, y0, opt);
  return trace_ray_rk4<M>(params, y0, opt);
}

/**
 * Past-directed null state at `start` whose coordinate velocity is along dir:
 * the path of light that arrives at start travelling along -dir (cameraRay in
 * compiler/test/physics.test.ts).
 */
template <class M, typename Real>
RM_HD inline void coordinate_ray(const Real* params, const Real start[3], const Real dir[3], Real y[8]) {
  Real x[4] = {Real(0), start[0], start[1], start[2]};
  Real g[16], dg[64];
  M::eval(x, params, g, dg);
  Real b = Real(0), c = Real(0);
  for (int i = 1; i < 4; i++) {
    b += Real(2) * g[i] * dir[i - 1];
    for (int j = 1; j < 4; j++) c += g[4 * i + j] * dir[i - 1] * dir[j - 1];
  }
  const Real a = g[0];
  const Real kt = (-b + sqrt(b * b - Real(4) * a * c)) / (Real(2) * a);
  const Real k[4] = {kt, dir[0], dir[1], dir[2]};
  for (int mu = 0; mu < 4; mu++) {
    y[mu] = x[mu];
    y[4 + mu] = Real(0);
    for (int nu = 0; nu < 4; nu++) y[4 + mu] += g[4 * mu + nu] * k[nu];
  }
}

/** Camera placement and covariant frame for one frame, computed in double. */
struct CameraFrame {
  double pos[4];
  /** Covariant frame g e_a: 4-velocity, right, up, forward. */
  double frame[4][4];
  double length_scale;
  int ok;  // 0 when no static observer exists at the camera (inside an ergoregion)
};

/**
 * Static observer on an orbit of radius cam_radius * length_scale around the
 * origin, looking at it, with +z up: orbitCamera (web/src/camera.ts) followed
 * by Spacetime.staticFrame and lowering with g (compiler/src/geodesic.ts).
 */
template <class M>
RM_HD inline CameraFrame static_camera(const double* params, double cam_radius, double yaw, double pitch) {
  CameraFrame cam;
  const double origin[4] = {0, 0, 0, 0};
  cam.length_scale = M::length_scale(origin, params);
  const double radius = cam_radius * cam.length_scale;
  const double pos3[3] = {radius * cos(pitch) * sin(yaw), radius * cos(pitch) * cos(yaw), radius * sin(pitch)};
  double dir[3] = {-pos3[0], -pos3[1], -pos3[2]};
  const double dl = sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
  for (int i = 0; i < 3; i++) dir[i] /= dl;
  const double up[3] = {0, 0, 1};
  cam.pos[0] = 0;
  for (int i = 0; i < 3; i++) cam.pos[i + 1] = pos3[i];

  double g[16], dg[64];
  M::eval(cam.pos, params, g, dg);
  cam.ok = g[0] < 0;
  if (!cam.ok) return cam;

  const double e0[4] = {1 / sqrt(-g[0]), 0, 0, 0};
  const double right[3] = {dir[1] * up[2] - dir[2] * up[1], dir[2] * up[0] - dir[0] * up[2], dir[0] * up[1] - dir[1] * up[0]};
  // Gram-Schmidt in the order forward, up, right (as staticFrame does).
  const double* coords[3] = {dir, up, right};
  double spatial[3][4];
  for (int n = 0; n < 3; n++) {
    double w[4] = {0, coords[n][0], coords[n][1], coords[n][2]};
    const double along0 = dot4(g, w, e0);
    for (int i = 0; i < 4; i++) w[i] += along0 * e0[i];  // e0 . e0 = -1
    for (int m = 0; m < n; m++) {
      const double along = dot4(g, w, spatial[m]);
      for (int i = 0; i < 4; i++) w[i] -= along * spatial[m][i];
    }
    const double norm = sqrt(dot4(g, w, w));
    for (int i = 0; i < 4; i++) spatial[n][i] = w[i] / norm;
  }
  const double* frame[4] = {e0, spatial[2], spatial[1], spatial[0]};  // e0, right, up, forward
  for (int a = 0; a < 4; a++)
    for (int mu = 0; mu < 4; mu++) {
      double s = 0;
      for (int nu = 0; nu < 4; nu++) s += g[4 * mu + nu] * frame[a][nu];
      cam.frame[a][mu] = s;
    }
  return cam;
}

/**
 * Initial state of a ray leaving the camera in local direction d (right, up,
 * forward; normalised here), traced backwards: p = -f0 + d.x f1 + d.y f2 + d.z f3.
 */
template <typename Real>
RM_HD inline void camera_ray(const CameraFrame& cam, Real dx, Real dy, Real dz, Real y[8]) {
  const Real len = sqrt(dx * dx + dy * dy + dz * dz);
  dx /= len;
  dy /= len;
  dz /= len;
  for (int mu = 0; mu < 4; mu++) {
    y[mu] = Real(cam.pos[mu]);
    y[4 + mu] = -Real(cam.frame[0][mu]) + dx * Real(cam.frame[1][mu]) + dy * Real(cam.frame[2][mu]) +
                dz * Real(cam.frame[3][mu]);
  }
}

}  // namespace rm
