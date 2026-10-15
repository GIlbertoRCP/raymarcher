#ifndef RAYMARCHER_PHYSICS_H
#define RAYMARCHER_PHYSICS_H

// Schwarzschild spacetime in isotropic coordinates (G = c = 1), shared by the
// CUDA kernel and the CPU test suite. Mirrors web/src/physics.ts.
//
// Light moves like light in a medium with refractive index
//   n(r) = (1 + u)^3 / (1 - u),  u = M / 2r.
// With coordinate time t as the parameter (|v| = 1/n):
//   dv/dt = grad(n) / n^3 - 2 v (grad(n) . v) / n

#include <cmath>

#ifdef __CUDACC__
#define RM_HD __host__ __device__
#else
#define RM_HD
#endif

namespace physics {

// Photon sphere R = 3M as an isotropic radius: r = M (1 + sqrt(3) / 2).
constexpr double kPhotonSphereIso = 1.8660254037844386;

template <typename T>
RM_HD inline T refractive_index(T r, T mass) {
    T u = mass / (T(2) * r);
    return (T(1) + u) * (T(1) + u) * (T(1) + u) / (T(1) - u);
}

// dn/dr for the index above.
template <typename T>
RM_HD inline T refractive_index_gradient(T r, T mass) {
    T u = mass / (T(2) * r);
    return T(-2) * u * (T(1) + u) * (T(1) + u) * (T(2) - u) / ((T(1) - u) * (T(1) - u) * r);
}

// The ray acceleration is a = c_x * x + c_v * v. Returning the two scalars
// keeps this header free of any vector type.
template <typename T>
RM_HD inline void ray_acceleration_coeffs(T r, T x_dot_v, T mass, T &c_x, T &c_v) {
    T n = refractive_index(r, mass);
    T k = refractive_index_gradient(r, mass) / r; // grad(n) = k x
    c_x = k / (n * n * n);
    c_v = T(-2) * k * x_dot_v / n;
}

template <typename T>
RM_HD inline T schwarzschild_from_isotropic(T r, T mass) {
    T k = T(1) + mass / (T(2) * r);
    return r * k * k;
}

// Outer branch, valid for R >= 2M.
template <typename T>
RM_HD inline T isotropic_from_schwarzschild(T R, T mass) {
    T disc = R * R - T(2) * mass * R;
    return (R - mass + std::sqrt(disc > T(0) ? disc : T(0))) / T(2);
}

} // namespace physics

#endif // RAYMARCHER_PHYSICS_H
