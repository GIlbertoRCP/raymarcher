// CPU tests for the physics shared with the CUDA kernel (src/physics.h).
// They need no GPU and no CUDA toolkit, so they run in CI.

#include <cmath>
#include <cstdio>
#include <initializer_list>

#include "physics.h"

static int g_failures = 0;

#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("  [FAIL] %s:%d: ", __FILE__, __LINE__);           \
            std::printf(__VA_ARGS__);                                      \
            std::printf("\n");                                             \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

static bool close_rel(double a, double b, double tol) {
    return std::fabs(a - b) <= tol * std::fmax(std::fabs(a), std::fabs(b));
}

struct Vec { double x, y, z; };
static Vec add(Vec a, Vec b, double s) { return {a.x + s * b.x, a.y + s * b.y, a.z + s * b.z}; }
static double dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static double len(Vec a) { return std::sqrt(dot(a, a)); }

static Vec accel(Vec x, Vec v, double mass) {
    double cx, cv;
    physics::ray_acceleration_coeffs(len(x), dot(x, v), mass, cx, cv);
    return {cx * x.x + cv * v.x, cx * x.y + cv * v.y, cx * x.z + cv * v.z};
}

struct RayResult { bool captured; Vec direction; };

// Ray from x = -start moving in +x with impact parameter b, integrated with
// RK4 in double precision. Same scheme as traceRay in web/src/physics.ts.
static RayResult trace_ray(double b, double mass, double start) {
    const double r_h = mass / 2;
    Vec x = {-start, b, 0};
    Vec v = {1 / physics::refractive_index(len(x), mass), 0, 0};
    for (int step = 0; step < 2000000; ++step) {
        double r = len(x);
        if (r < r_h * 1.001) return {true, {0, 0, 0}};
        if (r > start * 1.01 && x.x > 0 && step > 10) {
            double l = len(v);
            return {false, {v.x / l, v.y / l, v.z / l}};
        }
        double h = 2e-3 * (r - r_h) * physics::refractive_index(r, mass);
        Vec a1 = accel(x, v, mass);
        Vec x2 = add(x, v, h / 2), v2 = add(v, a1, h / 2);
        Vec a2 = accel(x2, v2, mass);
        Vec x3 = add(x, v2, h / 2), v3 = add(v, a2, h / 2);
        Vec a3 = accel(x3, v3, mass);
        Vec x4 = add(x, v3, h), v4 = add(v, a3, h);
        Vec a4 = accel(x4, v4, mass);
        x = add(add(add(add(x, v, h / 6), v2, h / 3), v3, h / 3), v4, h / 6);
        v = add(add(add(add(v, a1, h / 6), a2, h / 3), a3, h / 3), a4, h / 6);
    }
    return {false, {0, 0, 0}};
}

static void test_radius_conversions() {
    const double M = 1.8;
    CHECK(close_rel(physics::schwarzschild_from_isotropic(M / 2, M), 2 * M, 1e-12), "horizon r = M/2 should map to R = 2M");
    CHECK(close_rel(physics::schwarzschild_from_isotropic(physics::kPhotonSphereIso * M, M), 3 * M, 1e-12),
          "photon sphere isotropic radius should map to R = 3M");
    for (double R : {2 * M, 3 * M, 6 * M, 50 * M}) {
        double back = physics::schwarzschild_from_isotropic(physics::isotropic_from_schwarzschild(R, M), M);
        CHECK(close_rel(back, R, 1e-10), "round trip failed for R = %g", R);
    }
}

static void test_refractive_index_gradient() {
    const double M = 1.0;
    for (double r : {0.6, 1.0, 1.866, 5.0, 50.0}) {
        double eps = 1e-6 * r;
        double fd = (physics::refractive_index(r + eps, M) - physics::refractive_index(r - eps, M)) / (2 * eps);
        CHECK(close_rel(physics::refractive_index_gradient(r, M), fd, 1e-6), "dn/dr mismatch at r = %g", r);
    }
}

static void test_float_matches_double() {
    // The kernel runs the float instantiation; it must agree with double away from the horizon.
    for (float r : {1.0f, 3.0f, 10.0f}) {
        float cx_f, cv_f;
        double cx_d, cv_d;
        physics::ray_acceleration_coeffs(r, -0.3f, 1.8f, cx_f, cv_f);
        physics::ray_acceleration_coeffs<double>(r, -0.3, 1.8, cx_d, cv_d);
        CHECK(close_rel(cx_f, cx_d, 1e-5) && close_rel(cv_f, cv_d, 1e-5), "float/double mismatch at r = %g", r);
    }
}

static void test_capture_threshold() {
    const double M = 1.0;
    const double b_crit = std::sqrt(27.0) * M;
    CHECK(trace_ray(0.99 * b_crit, M, 1e4).captured, "ray at 0.99 b_crit should be captured");
    CHECK(!trace_ray(1.01 * b_crit, M, 1e4).captured, "ray at 1.01 b_crit should escape");
}

static void test_weak_field_deflection() {
    const double M = 1.0, b = 1000.0;
    RayResult res = trace_ray(b, M, 1e6);
    double deflection = std::atan2(-res.direction.y, res.direction.x);
    double expected = 4 * M / b + 15 * 3.14159265358979323846 * M * M / (4 * b * b);
    CHECK(close_rel(deflection, expected, 1e-3), "deflection %.6g, expected %.6g", deflection, expected);
}

int main() {
    struct { const char *name; void (*fn)(); } tests[] = {
        {"radius_conversions", test_radius_conversions},
        {"refractive_index_gradient", test_refractive_index_gradient},
        {"float_matches_double", test_float_matches_double},
        {"capture_threshold", test_capture_threshold},
        {"weak_field_deflection", test_weak_field_deflection},
    };
    for (auto &t : tests) {
        int before = g_failures;
        t.fn();
        std::printf("%s %s\n", g_failures == before ? "[PASS]" : "[FAIL]", t.name);
    }
    if (g_failures) {
        std::printf("%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("All tests passed\n");
    return 0;
}
