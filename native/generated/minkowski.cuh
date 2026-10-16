// Generated from spacetime "Flat space" (metrics/minkowski.st) by compiler/src/backend-cuda.ts.
// Do not edit; run `npm run export-cuda` instead.
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
namespace minkowski {

#ifndef __CUDACC_RTC__
using std::cos;
using std::exp;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
#endif

constexpr const char* NAME = "Flat space";
constexpr int PARAM_COUNT = 1;
constexpr bool HAS_HORIZON = false;
constexpr bool HAS_DISK = true;

template <typename Real>
RM_HD inline void metric_eval(const Real x[4], const Real* params, Real g[16], Real dg[64]) {
  (void)x;
  (void)params;
  g[0] = Real(-1.0);
  g[1] = Real(0.0);
  g[2] = Real(0.0);
  g[3] = Real(0.0);
  g[4] = Real(0.0);
  g[5] = Real(1.0);
  g[6] = Real(0.0);
  g[7] = Real(0.0);
  g[8] = Real(0.0);
  g[9] = Real(0.0);
  g[10] = Real(1.0);
  g[11] = Real(0.0);
  g[12] = Real(0.0);
  g[13] = Real(0.0);
  g[14] = Real(0.0);
  g[15] = Real(1.0);
  dg[0] = Real(0.0);
  dg[1] = Real(0.0);
  dg[2] = Real(0.0);
  dg[3] = Real(0.0);
  dg[4] = Real(0.0);
  dg[5] = Real(0.0);
  dg[6] = Real(0.0);
  dg[7] = Real(0.0);
  dg[8] = Real(0.0);
  dg[9] = Real(0.0);
  dg[10] = Real(0.0);
  dg[11] = Real(0.0);
  dg[12] = Real(0.0);
  dg[13] = Real(0.0);
  dg[14] = Real(0.0);
  dg[15] = Real(0.0);
  dg[16] = Real(0.0);
  dg[17] = Real(0.0);
  dg[18] = Real(0.0);
  dg[19] = Real(0.0);
  dg[20] = Real(0.0);
  dg[21] = Real(0.0);
  dg[22] = Real(0.0);
  dg[23] = Real(0.0);
  dg[24] = Real(0.0);
  dg[25] = Real(0.0);
  dg[26] = Real(0.0);
  dg[27] = Real(0.0);
  dg[28] = Real(0.0);
  dg[29] = Real(0.0);
  dg[30] = Real(0.0);
  dg[31] = Real(0.0);
  dg[32] = Real(0.0);
  dg[33] = Real(0.0);
  dg[34] = Real(0.0);
  dg[35] = Real(0.0);
  dg[36] = Real(0.0);
  dg[37] = Real(0.0);
  dg[38] = Real(0.0);
  dg[39] = Real(0.0);
  dg[40] = Real(0.0);
  dg[41] = Real(0.0);
  dg[42] = Real(0.0);
  dg[43] = Real(0.0);
  dg[44] = Real(0.0);
  dg[45] = Real(0.0);
  dg[46] = Real(0.0);
  dg[47] = Real(0.0);
  dg[48] = Real(0.0);
  dg[49] = Real(0.0);
  dg[50] = Real(0.0);
  dg[51] = Real(0.0);
  dg[52] = Real(0.0);
  dg[53] = Real(0.0);
  dg[54] = Real(0.0);
  dg[55] = Real(0.0);
  dg[56] = Real(0.0);
  dg[57] = Real(0.0);
  dg[58] = Real(0.0);
  dg[59] = Real(0.0);
  dg[60] = Real(0.0);
  dg[61] = Real(0.0);
  dg[62] = Real(0.0);
  dg[63] = Real(0.0);
}

template <typename Real>
RM_HD inline Real geodesic_rhs(const Real y[8], const Real* params, Real dy[8]) {
  const Real* x = y;
  (void)x;
  (void)params;
  const Real a00 = Real(-1.0), a01 = Real(0.0), a02 = Real(0.0), a03 = Real(0.0);
  const Real a10 = Real(0.0), a11 = Real(1.0), a12 = Real(0.0), a13 = Real(0.0);
  const Real a20 = Real(0.0), a21 = Real(0.0), a22 = Real(1.0), a23 = Real(0.0);
  const Real a30 = Real(0.0), a31 = Real(0.0), a32 = Real(0.0), a33 = Real(1.0);
  const Real b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const Real b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const Real b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const Real b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const Real s = Real(1) / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);
  const Real i00 = (a11 * b11 - a12 * b10 + a13 * b09) * s;
  const Real i01 = (a02 * b10 - a01 * b11 - a03 * b09) * s;
  const Real i02 = (a31 * b05 - a32 * b04 + a33 * b03) * s;
  const Real i03 = (a22 * b04 - a21 * b05 - a23 * b03) * s;
  const Real i11 = (a00 * b11 - a02 * b08 + a03 * b07) * s;
  const Real i12 = (a32 * b02 - a30 * b05 - a33 * b01) * s;
  const Real i13 = (a20 * b05 - a22 * b02 + a23 * b01) * s;
  const Real i22 = (a30 * b04 - a31 * b02 + a33 * b00) * s;
  const Real i23 = (a21 * b02 - a20 * b04 - a23 * b00) * s;
  const Real i33 = (a20 * b03 - a21 * b01 + a22 * b00) * s;
  const Real k0 = i00 * y[4] + i01 * y[5] + i02 * y[6] + i03 * y[7];
  const Real k1 = i01 * y[4] + i11 * y[5] + i12 * y[6] + i13 * y[7];
  const Real k2 = i02 * y[4] + i12 * y[5] + i22 * y[6] + i23 * y[7];
  const Real k3 = i03 * y[4] + i13 * y[5] + i23 * y[6] + i33 * y[7];
  dy[0] = k0;
  dy[1] = k1;
  dy[2] = k2;
  dy[3] = k3;
  dy[4] = Real(0.0);
  dy[5] = Real(0.0);
  dy[6] = Real(0.0);
  dy[7] = Real(0.0);
  return Real(0.5) * (k0 * y[4] + k1 * y[5] + k2 * y[6] + k3 * y[7]);
}

template <typename Real>
RM_HD inline Real horizon_value(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  return Real(1.0);
}

template <typename Real>
RM_HD inline Real disk_radius(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t9 = x[1] * x[1];
  const Real t10 = x[2] * x[2];
  const Real t11 = t9 + t10;
  const Real t12 = sqrt(t11);
  return t12;
}

template <typename Real>
RM_HD inline Real disk_omega(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t13 = sqrt(params[0]);
  const Real t9 = x[1] * x[1];
  const Real t10 = x[2] * x[2];
  const Real t11 = t9 + t10;
  const Real t12 = sqrt(t11);
  const Real t16 = Real(1.0) / (t12 * sqrt(t12));
  const Real t17 = t13 * t16;
  return t17;
}

template <typename Real>
RM_HD inline Real disk_inner(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t19 = params[0] * Real(6.0);
  return t19;
}

template <typename Real>
RM_HD inline Real disk_outer(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  const Real t21 = params[0] * Real(20.0);
  return t21;
}

template <typename Real>
RM_HD inline Real length_scale(const Real x[4], const Real* params) {
  (void)x;
  (void)params;
  return params[0];
}

// The functions above as one type, so generic code can take the metric as a
// template parameter (rm::trace_ray<rm::kerr::Metric>(...)).
struct Metric {
  static constexpr int PARAM_COUNT = 1;
  static constexpr bool HAS_HORIZON = false;
  static constexpr bool HAS_DISK = true;
  template <typename Real>
  RM_HD static void eval(const Real x[4], const Real* p, Real g[16], Real dg[64]) { metric_eval(x, p, g, dg); }
  template <typename Real>
  RM_HD static Real rhs(const Real y[8], const Real* p, Real dy[8]) { return minkowski::geodesic_rhs(y, p, dy); }
  template <typename Real>
  RM_HD static Real horizon_value(const Real x[4], const Real* p) { return minkowski::horizon_value(x, p); }
  template <typename Real>
  RM_HD static Real disk_radius(const Real x[4], const Real* p) { return minkowski::disk_radius(x, p); }
  template <typename Real>
  RM_HD static Real disk_omega(const Real x[4], const Real* p) { return minkowski::disk_omega(x, p); }
  template <typename Real>
  RM_HD static Real disk_inner(const Real x[4], const Real* p) { return minkowski::disk_inner(x, p); }
  template <typename Real>
  RM_HD static Real disk_outer(const Real x[4], const Real* p) { return minkowski::disk_outer(x, p); }
  template <typename Real>
  RM_HD static Real length_scale(const Real x[4], const Real* p) { return minkowski::length_scale(x, p); }
};

}  // namespace minkowski
}  // namespace rm
