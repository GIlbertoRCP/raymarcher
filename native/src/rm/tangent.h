// Tangent (forward-mode) ray tracing for Phase 4 (docs/phase-4-plan.md, step 3).
//
// Two independent ways to get dy_final/dtheta for a ray:
//
//   symbolic  trace_tangent_rk4: carries S = dy/dtheta (8 x P) alongside y,
//             using the compiler's JVP of the metric (M::jvp, tangent.ts):
//               dg^{-1} = -g^{-1} dg g^{-1},  dk = -g^{-1} (dg k - dp),
//               d(dp_c) = 1/2 d(d_c g_ab) k^a k^b + d_c g_ab k^a dk^b,
//             and differentiates each RK4 stage, the step h = step_fraction r
//             and the disk-crossing interpolation exactly. It mirrors the JS
//             reference compiler/src/tangent-geodesic.ts.
//   dual      trace_ray_rk4 / trace_ray_rk45 (geodesic.h) instantiated with
//             Dual<Real, P> (dual.h): no compiler involvement.
//
// theta, the traced inputs: the metric parameters (theta[0 .. NP-1]), then the
// camera radius in units of the length scale, the inclination (pitch =
// pi/2 - inclination), the yaw, and the image-plane direction (u, v, 1) in the
// camera's (right, up, forward) frame. dirs[d][k] is the component of
// direction d along theta[k]. The camera frame's tangent comes from
// static_camera_t on duals in both paths.
//
// trace_tangent_rk4 locates disk crossings by linear interpolation only (the
// reference check); the Hermite crossing (CROSSING_HERMITE) is differentiated
// through the dual path.
#pragma once

#include "rm/dual.h"
#include "rm/geodesic.h"

namespace rm {

/** Metric parameters plus camera radius, inclination, yaw, u and v. */
constexpr int MAX_THETA = 8 + 5;
constexpr double TANGENT_PI = 3.14159265358979323846;

/** Index of each camera input in theta, after the NP metric parameters. */
enum ThetaCamera : int { THETA_RADIUS = 0, THETA_INCLINATION = 1, THETA_YAW = 2, THETA_U = 3, THETA_V = 4 };

/**
 * Initial state of a camera ray as duals, and the metric parameters as duals
 * (params[k].d[d] = dirs[d][k]). Returns false when there is no static
 * observer at the camera.
 */
template <class M, typename Real, int P>
RM_HD inline bool dual_camera_ray(const double* theta, const double (*dirs)[MAX_THETA], Dual<Real, P> params[],
                                  Dual<Real, P> y0[8]) {
  typedef Dual<Real, P> D;
  constexpr int NP = M::PARAM_COUNT;
  auto input = [&](int k) {
    D x = D(Real(theta[k]));
    for (int d = 0; d < P; ++d) x.d[d] = Real(dirs[d][k]);
    return x;
  };
  for (int k = 0; k < NP; ++k) params[k] = input(k);
  const D radius = input(NP + THETA_RADIUS);
  const D pitch = D(Real(TANGENT_PI / 2)) - input(NP + THETA_INCLINATION);
  const D yaw = input(NP + THETA_YAW);
  const CameraFrameT<D> cam = static_camera_t<M, D>(params, radius, yaw, pitch);
  if (!cam.ok) return false;
  camera_ray_t<D>(cam, input(NP + THETA_U), input(NP + THETA_V), D(Real(1)), y0);
  return true;
}

/** Splits duals into values and tangents: v[i], t[8 d + i] (or t[n d + i] with stride n). */
template <typename Real, int P>
RM_HD inline void split_duals(const Dual<Real, P>* x, int n, Real* v, Real* t) {
  for (int i = 0; i < n; ++i) {
    v[i] = x[i].v;
    for (int d = 0; d < P; ++d) t[n * d + i] = x[i].d[d];
  }
}

/** dy/dlambda and dS/dlambda. dparams[NP d + k] is the parameter part of direction d. */
template <class M, typename Real, int P>
RM_HD inline void tangent_rhs(const Real* params, const Real* dparams, const Real y[8], const Real S[8 * P], Real dy[8],
                              Real dS[8 * P]) {
  Real dx[4 * P];
  for (int d = 0; d < P; ++d)
    for (int i = 0; i < 4; ++i) dx[4 * d + i] = S[8 * d + i];
  Real g[16], dg[64], tg[16 * P], tdg[64 * P], gi[16], k[4];
  M::template jvp<Real, P>(y, params, dx, dparams, g, dg, tg, tdg);
  invert4(g, gi);
  for (int mu = 0; mu < 4; ++mu) {
    k[mu] = gi[4 * mu] * y[4] + gi[4 * mu + 1] * y[5] + gi[4 * mu + 2] * y[6] + gi[4 * mu + 3] * y[7];
    dy[mu] = k[mu];
  }
  for (int c = 0; c < 4; ++c) {
    Real s = Real(0);
    for (int a = 0; a < 4; ++a)
      for (int b = 0; b < 4; ++b) s += dg[16 * c + 4 * a + b] * k[a] * k[b];
    dy[4 + c] = Real(0.5) * s;
  }
  for (int d = 0; d < P; ++d) {
    Real m1[4], dk[4];
    for (int a = 0; a < 4; ++a) {
      Real s = Real(0);
      for (int b = 0; b < 4; ++b) s += tg[16 * d + 4 * a + b] * k[b];
      m1[a] = s - S[8 * d + 4 + a];  // dg k - dp
    }
    for (int mu = 0; mu < 4; ++mu) {
      dk[mu] = -(gi[4 * mu] * m1[0] + gi[4 * mu + 1] * m1[1] + gi[4 * mu + 2] * m1[2] + gi[4 * mu + 3] * m1[3]);
      dS[8 * d + mu] = dk[mu];
    }
    for (int c = 0; c < 4; ++c) {
      Real s = Real(0), t = Real(0);
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
          s += tdg[64 * d + 16 * c + 4 * a + b] * k[a] * k[b];
          t += dg[16 * c + 4 * a + b] * k[a] * dk[b];
        }
      dS[8 * d + 4 + c] = Real(0.5) * s + t;
    }
  }
}

template <typename Real, int P>
struct TangentResult {
  RayFate fate;
  int steps;
  Real y[8];
  Real S[8 * P];
};

/**
 * Fixed-step RK4 trace with tangents, step for step as trace_ray_rk4 (same
 * fates, steps and branches), so its S is the derivative of that function.
 */
template <class M, typename Real, int P>
RM_HD inline TangentResult<Real, P> trace_tangent_rk4(const Real* params, const Real* dparams, const Real y0[8],
                                                      const Real S0[8 * P], const TraceOptions<Real>& opt) {
  TangentResult<Real, P> res;
  Real* y = res.y;
  Real* S = res.S;
  for (int i = 0; i < 8; ++i) y[i] = y0[i];
  for (int i = 0; i < 8 * P; ++i) S[i] = S0[i];
  res.fate = FATE_UNFINISHED;
  res.steps = opt.max_steps;
  Real k1[8], k2[8], k3[8], k4[8], ty[8], ny[8];
  Real s1[8 * P], s2[8 * P], s3[8 * P], s4[8 * P], tS[8 * P], nS[8 * P], dh[P];

  for (int step = 0; step < opt.max_steps; ++step) {
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
    if (h < opt.min_step) {
      h = opt.min_step;
      for (int d = 0; d < P; ++d) dh[d] = Real(0);
    } else {
      for (int d = 0; d < P; ++d) dh[d] = opt.step_fraction * (y[1] * S[8 * d + 1] + y[2] * S[8 * d + 2] + y[3] * S[8 * d + 3]) / r;
    }

    tangent_rhs<M, Real, P>(params, dparams, y, S, k1, s1);
    if (!(fabs(k1[0]) < Real(1.0e4))) {
      res.fate = FATE_CAPTURED;
      res.steps = step;
      return res;
    }
    auto stage = [&](Real c, const Real* kp, const Real* sp) {
      for (int i = 0; i < 8; ++i) ty[i] = y[i] + c * h * kp[i];
      for (int d = 0; d < P; ++d)
        for (int i = 0; i < 8; ++i) tS[8 * d + i] = S[8 * d + i] + c * (dh[d] * kp[i] + h * sp[8 * d + i]);
    };
    stage(Real(0.5), k1, s1);
    tangent_rhs<M, Real, P>(params, dparams, ty, tS, k2, s2);
    stage(Real(0.5), k2, s2);
    tangent_rhs<M, Real, P>(params, dparams, ty, tS, k3, s3);
    stage(Real(1), k3, s3);
    tangent_rhs<M, Real, P>(params, dparams, ty, tS, k4, s4);
    for (int i = 0; i < 8; ++i) ny[i] = y[i] + (h / Real(6)) * (k1[i] + Real(2) * k2[i] + Real(2) * k3[i] + k4[i]);
    for (int d = 0; d < P; ++d)
      for (int i = 0; i < 8; ++i) {
        const int o = 8 * d + i;
        nS[o] = S[o] + (dh[d] / Real(6)) * (k1[i] + Real(2) * k2[i] + Real(2) * k3[i] + k4[i]) +
                (h / Real(6)) * (s1[o] + Real(2) * s2[o] + Real(2) * s3[o] + s4[o]);
      }

    if (M::HAS_DISK && opt.disk && y[3] * ny[3] <= Real(0) && y[3] != ny[3]) {
      const Real z = y[3], zn = ny[3], den = z - zn;
      const Real f = z / den;
      Real hit[8];
      for (int i = 0; i < 8; ++i) hit[i] = y[i] + f * (ny[i] - y[i]);
      const Real rd = M::disk_radius(hit, params);
      Real u[4];
      if (rd >= M::disk_inner(hit, params) && rd <= M::disk_outer(hit, params) && disk_gas_velocity<M>(params, hit, u)) {
        for (int d = 0; d < P; ++d) {
          const Real df = (z * nS[8 * d + 3] - zn * S[8 * d + 3]) / (den * den);
          for (int i = 0; i < 8; ++i) {
            const int o = 8 * d + i;
            S[o] = S[o] + df * (ny[i] - y[i]) + f * (nS[o] - S[o]);
          }
        }
        for (int i = 0; i < 8; ++i) y[i] = hit[i];
        res.fate = FATE_DISK;
        res.steps = step + 1;
        return res;
      }
    }
    for (int i = 0; i < 8; ++i) y[i] = ny[i];
    for (int i = 0; i < 8 * P; ++i) S[i] = nS[i];
  }
  return res;
}

/** Converts real trace options to dual ones (constants: zero derivative). */
template <typename Real, int P>
RM_HD inline TraceOptions<Dual<Real, P>> dual_options(const TraceOptions<Real>& o) {
  TraceOptions<Dual<Real, P>> d;
  d.max_steps = o.max_steps;
  d.step_fraction = o.step_fraction;
  d.min_step = o.min_step;
  d.escape_radius = o.escape_radius;
  d.disk = o.disk;
  d.integrator = o.integrator;
  d.rtol = o.rtol;
  d.length_scale = o.length_scale;
  d.crossing = o.crossing;
  return d;
}

/** A camera ray from theta, traced with symbolic tangents. ok is false with no static observer at the camera. */
template <class M, typename Real, int P>
RM_HD inline void tangent_ray_symbolic(const double* theta, const double (*dirs)[MAX_THETA], const TraceOptions<Real>& opt,
                                       Real y0[8], Real S0[8 * P], TangentResult<Real, P>& out, bool& ok) {
  constexpr int NP = M::PARAM_COUNT;
  Dual<Real, P> dp[NP > 0 ? NP : 1], dy0[8];
  ok = dual_camera_ray<M, Real, P>(theta, dirs, dp, dy0);
  if (!ok) return;
  Real params[NP > 0 ? NP : 1], dparams[(NP > 0 ? NP : 1) * P];
  split_duals<Real, P>(dp, NP, params, dparams);
  split_duals<Real, P>(dy0, 8, y0, S0);
  out = trace_tangent_rk4<M, Real, P>(params, dparams, y0, S0, opt);
}

/** The same ray traced on Dual numbers by the ordinary integrator (RK4 or RK45 per opt.integrator). */
template <class M, typename Real, int P>
RM_HD inline TraceResult<Dual<Real, P>> tangent_ray_dual(const double* theta, const double (*dirs)[MAX_THETA],
                                                         const TraceOptions<Real>& opt, bool& ok) {
  constexpr int NP = M::PARAM_COUNT;
  Dual<Real, P> dp[NP > 0 ? NP : 1], dy0[8];
  TraceResult<Dual<Real, P>> res;
  ok = dual_camera_ray<M, Real, P>(theta, dirs, dp, dy0);
  if (!ok) return res;
  return trace_ray<M>(dp, dy0, dual_options<Real, P>(opt));
}

}  // namespace rm
