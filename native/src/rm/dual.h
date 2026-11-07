// Forward-mode dual numbers: a value and N directional derivatives.
//
// Phase 4's independent derivative check (docs/phase-4-plan.md, step 3): the
// generated metric code and the integrators in geodesic.h are templated on
// Real, so instantiating them with Dual<double, N> differentiates the whole
// trace (every RK4 stage, the state-dependent step, the disk interpolation)
// with no compiler involvement. The symbolic tangents (tangent.h, from the
// compiler's JVP) must agree with it.
//
// Comparisons look at the value only, so branches (fates, step clamps) follow
// the primal trace exactly as the symbolic tangent integrator does.
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
using std::exp;
using std::fabs;
using std::log;
using std::pow;
using std::sin;
using std::sqrt;
#endif

template <typename T, int N>
struct Dual {
  T v;
  T d[N];

  RM_HD Dual() : v(T(0)) {
    for (int i = 0; i < N; ++i) d[i] = T(0);
  }
  // Implicit, so literals such as Real(2.0) and mixed arithmetic work in generic code.
  RM_HD Dual(T x) : v(x) {
    for (int i = 0; i < N; ++i) d[i] = T(0);
  }
  // Any other scalar (int, or double literals in a float instantiation). The
  // exact Dual(T) and the copy constructor win over this template when they match.
  template <typename U>
  RM_HD Dual(U x) : Dual(T(x)) {}

  /** A variable: value x with derivative 1 along direction k. */
  RM_HD static Dual variable(T x, int k) {
    Dual r(x);
    r.d[k] = T(1);
    return r;
  }

  RM_HD Dual& operator+=(const Dual& b) {
    v += b.v;
    for (int i = 0; i < N; ++i) d[i] += b.d[i];
    return *this;
  }
  RM_HD Dual& operator-=(const Dual& b) {
    v -= b.v;
    for (int i = 0; i < N; ++i) d[i] -= b.d[i];
    return *this;
  }
  RM_HD Dual& operator*=(const Dual& b) {
    for (int i = 0; i < N; ++i) d[i] = d[i] * b.v + v * b.d[i];
    v *= b.v;
    return *this;
  }
  RM_HD Dual& operator/=(const Dual& b) {
    const T inv = T(1) / b.v, q = v * inv;
    for (int i = 0; i < N; ++i) d[i] = (d[i] - q * b.d[i]) * inv;
    v = q;
    return *this;
  }

  // Hidden friends: found by argument-dependent lookup from the generated
  // code's unqualified sqrt(...), pow(...), and they accept T on either side.
  friend RM_HD Dual operator+(Dual a, const Dual& b) { return a += b; }
  friend RM_HD Dual operator-(Dual a, const Dual& b) { return a -= b; }
  friend RM_HD Dual operator*(Dual a, const Dual& b) { return a *= b; }
  friend RM_HD Dual operator/(Dual a, const Dual& b) { return a /= b; }
  friend RM_HD Dual operator-(Dual a) {
    a.v = -a.v;
    for (int i = 0; i < N; ++i) a.d[i] = -a.d[i];
    return a;
  }
  friend RM_HD Dual operator+(const Dual& a) { return a; }

  friend RM_HD bool operator<(const Dual& a, const Dual& b) { return a.v < b.v; }
  friend RM_HD bool operator>(const Dual& a, const Dual& b) { return a.v > b.v; }
  friend RM_HD bool operator<=(const Dual& a, const Dual& b) { return a.v <= b.v; }
  friend RM_HD bool operator>=(const Dual& a, const Dual& b) { return a.v >= b.v; }
  friend RM_HD bool operator==(const Dual& a, const Dual& b) { return a.v == b.v; }
  friend RM_HD bool operator!=(const Dual& a, const Dual& b) { return a.v != b.v; }

  /** a with derivative scaled by f'(a.v) = s. */
  RM_HD Dual chain(T value, T s) const {
    Dual r(value);
    for (int i = 0; i < N; ++i) r.d[i] = s * d[i];
    return r;
  }

  friend RM_HD Dual sqrt(const Dual& a) {
    const T r = sqrt(a.v);
    return a.chain(r, T(0.5) / r);
  }
  friend RM_HD Dual fabs(const Dual& a) { return a.v < T(0) ? -a : a; }
  friend RM_HD Dual sin(const Dual& a) { return a.chain(sin(a.v), cos(a.v)); }
  friend RM_HD Dual cos(const Dual& a) { return a.chain(cos(a.v), -sin(a.v)); }
  friend RM_HD Dual exp(const Dual& a) {
    const T e = exp(a.v);
    return a.chain(e, e);
  }
  friend RM_HD Dual log(const Dual& a) { return a.chain(log(a.v), T(1) / a.v); }
  /**
   * a^b. The generated code passes constant exponents as Real(e), so the
   * log term (undefined for a <= 0) is added only when b carries a derivative.
   */
  friend RM_HD Dual pow(const Dual& a, const Dual& b) {
    const T p = pow(a.v, b.v);
    Dual r = a.chain(p, b.v * pow(a.v, b.v - T(1)));
    bool varying = false;
    for (int i = 0; i < N; ++i) varying = varying || b.d[i] != T(0);
    if (varying) {
      const T l = p * log(a.v);
      for (int i = 0; i < N; ++i) r.d[i] += l * b.d[i];
    }
    return r;
  }
};

}  // namespace rm
