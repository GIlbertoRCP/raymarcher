// Closed-form and quadrature results from the literature, used as references by
// the CPU tests and the fp64 validation suite (raymarcher --validate). Host only.
#pragma once

#include <cmath>
#include <functional>

namespace rm {
namespace analytic {

constexpr double PI = 3.14159265358979323846;

/** Adaptive Simpson quadrature of f on [a, b] to absolute tolerance tol. */
inline double integrate(const std::function<double(double)>& f, double a, double b, double tol = 1e-14) {
  struct S {
    static double step(const std::function<double(double)>& f, double a, double b, double fa, double fm, double fb,
                       double whole, double tol, int depth) {
      const double m = 0.5 * (a + b), lm = 0.5 * (a + m), rm_ = 0.5 * (m + b);
      const double flm = f(lm), frm = f(rm_);
      const double left = (m - a) / 6 * (fa + 4 * flm + fm), right = (b - m) / 6 * (fm + 4 * frm + fb);
      if (depth <= 0 || std::fabs(left + right - whole) <= 15 * tol) return left + right + (left + right - whole) / 15;
      return step(f, a, m, fa, flm, fm, left, tol / 2, depth - 1) + step(f, m, b, fm, frm, fb, right, tol / 2, depth - 1);
    }
  };
  const double fa = f(a), fb = f(b), fm = f(0.5 * (a + b));
  return S::step(f, a, b, fa, fm, fb, (b - a) / 6 * (fa + 4 * fm + fb), tol, 40);
}

/**
 * Schwarzschild light orbit with impact parameter b > sqrt(27) M, in u = 1/r:
 *   (du/dphi)^2 = F(u) = 1/b^2 - u^2 + 2 M u^3.
 * Returns the angle swept by a ray that comes in from radius r_in, passes
 * periapsis and leaves to radius r_out (Schwarzschild r, which is also the
 * Kerr-Schild coordinate radius). For r_in, r_out -> infinity this is
 * pi + the deflection angle.
 */
inline double schwarzschild_sweep(double b, double M, double r_in, double r_out) {
  const auto F = [&](double u) { return 1 / (b * b) - u * u + 2 * M * u * u * u; };
  // Periapsis: the smallest positive root of F, below u = 1/(3M) where F is minimal.
  double lo = 0, hi = 1 / (3 * M);
  for (int i = 0; i < 200; i++) {
    const double mid = 0.5 * (lo + hi);
    (F(mid) > 0 ? lo : hi) = mid;
  }
  const double u0 = 0.5 * (lo + hi);
  // With u = u0 - t^2 the inverse-square-root endpoint singularity goes away.
  // F is cubic and F(u0) = 0, so expanding about u0 is exact:
  //   F(u0 - t^2) / t^2 = (2 u0 - 6 M u0^2) + (6 M u0 - 1) t^2 - 2 M t^4,
  // which avoids the cancellation of evaluating F next to its root.
  const double c0 = 2 * u0 - 6 * M * u0 * u0, c1 = 6 * M * u0 - 1, c2 = -2 * M;
  const auto leg = [&](double r) {
    const double u = 1 / r;
    return integrate([&](double t) {
      const double s = t * t;
      return 2 / std::sqrt(c0 + s * (c1 + s * c2));
    }, 0, std::sqrt(u0 - u));
  };
  return leg(r_in) + leg(r_out);
}

/** Total deflection pi - sweep, as the weak-field series 4M/b + 15 pi M^2 / 4 b^2 approximates it. */
inline double schwarzschild_deflection(double b, double M) { return schwarzschild_sweep(b, M, 1e300, 1e300) - PI; }

/** Angular radius of the Schwarzschild shadow for a static observer at r: sin(alpha) = sqrt(27) M sqrt(1 - 2M/r) / r. */
inline double schwarzschild_shadow_angle(double r, double M) {
  const double s = std::sqrt(27.0) * M * std::sqrt(1 - 2 * M / r) / r;
  // Beyond the photon sphere the shadow is smaller than a hemisphere.
  return r >= 3 * M ? std::asin(s) : PI - std::asin(s);
}

/** Equatorial circular photon orbit radii (Boyer-Lindquist), prograde and retrograde (Bardeen, Press & Teukolsky 1972). */
inline double kerr_photon_orbit_prograde(double a, double M) { return 2 * M * (1 + std::cos(2.0 / 3.0 * std::acos(-a / M))); }
inline double kerr_photon_orbit_retrograde(double a, double M) { return 2 * M * (1 + std::cos(2.0 / 3.0 * std::acos(a / M))); }

/** Innermost stable circular orbit, prograde or retrograde (Bardeen, Press & Teukolsky 1972). */
inline double kerr_isco(double a, double M, bool prograde) {
  const double s = a / M;
  const double z1 = 1 + std::cbrt(1 - s * s) * (std::cbrt(1 + s) + std::cbrt(1 - s));
  const double z2 = std::sqrt(3 * s * s + z1 * z1);
  const double root = std::sqrt((3 - z1) * (3 + z1 + 2 * z2));
  return M * (3 + z2 + (prograde ? -root : root));
}

/**
 * Constants of motion of the spherical photon orbit at Boyer-Lindquist radius
 * r (Bardeen 1973): xi = L_z / E and eta = Q / E^2.
 */
inline double kerr_xi(double r, double a, double M) { return (r * r * (3 * M - r) - a * a * (r + M)) / (a * (r - M)); }
inline double kerr_eta(double r, double a, double M) {
  return r * r * r * (4 * a * a * M - r * (r - 3 * M) * (r - 3 * M)) / (a * a * (r - M) * (r - M));
}

/**
 * Shadow edge seen by a distant observer at inclination theta_o (from the spin
 * axis), in impact-parameter coordinates on the sky (Bardeen 1973):
 *   alpha = -xi / sin(theta_o),  beta = +-sqrt(eta + a^2 cos^2 theta_o - xi^2 cot^2 theta_o).
 * beta2 < 0 means the orbit at r does not reach this observer.
 */
inline void kerr_shadow_point(double r, double a, double M, double theta_o, double& alpha, double& beta2) {
  const double xi = kerr_xi(r, a, M), eta = kerr_eta(r, a, M);
  const double s = std::sin(theta_o), c = std::cos(theta_o);
  alpha = -xi / s;
  beta2 = eta + a * a * c * c - xi * xi * c * c / (s * s);
}

}  // namespace analytic
}  // namespace rm
