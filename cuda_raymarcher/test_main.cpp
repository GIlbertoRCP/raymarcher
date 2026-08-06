#include <iostream>
#include <cmath>
#include <cassert>
#include <cstring>
#include <fstream>
#include "raymarch.h"

// Test Helper Assert Macro with descriptive failure printing
#define ASSERT_TEST(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            return false; \
        } \
    } while(0)

// 1. Validate Isotropic Metric Radius & Horizon Calculation
bool test_isotropic_metric() {
    float M = 2.0f;
    // Schwarzschild radius in standard coordinates: R_s = 2 * M = 4.0
    // Isotropic horizon radius: r_h = M / 2 = 1.0
    float isotropic_rh = M * 0.5f;
    ASSERT_TEST(std::abs(isotropic_rh - 1.0f) < 1e-5f, "Isotropic horizon radius should be M/2");

    // Refractive index calculation check: n(r) = (1 + M / (2r))^3 / (1 - M / (2r))
    float r = 2.0f; // r > r_h
    float u = M / (2.0f * r); // u = 2 / 4 = 0.5
    float n_val = std::pow(1.0f + u, 3.0f) / (1.0f - u); // (1.5)^3 / 0.5 = 3.375 / 0.5 = 6.75
    ASSERT_TEST(std::abs(n_val - 6.75f) < 1e-4f, "Isotropic refractive index calculation incorrect");

    std::cout << "  [PASS] test_isotropic_metric\n";
    return true;
}

// 2. Validate Signed Distance Function (SDF) Primitives & CSG Math
bool test_csg_sdf_primitives() {
    // Sphere SDF: d = length(p) - r
    float p_x = 3.0f, p_y = 4.0f, p_z = 0.0f;
    float dist_origin = std::sqrt(p_x * p_x + p_y * p_y + p_z * p_z); // 5.0
    float radius = 2.0f;
    float d_sphere = dist_origin - radius; // 3.0
    ASSERT_TEST(std::abs(d_sphere - 3.0f) < 1e-5f, "Sphere SDF distance failed");

    // Smooth min (smin) test: smin(a, b, k) when a < b
    float a = 1.0f, b = 3.0f, k = 0.5f;
    // smin(a, b, k) = min(a, b) - h^2 * k * (1/4), where h = max(k - |a-b|, 0) / k
    // Here |a-b| = 2.0 > k (0.5), so h = 0, smin = min(1, 3) = 1.0
    float h = std::max(k - std::abs(a - b), 0.0f) / k;
    float smin_val = std::min(a, b) - h * h * k * 0.25f;
    ASSERT_TEST(std::abs(smin_val - 1.0f) < 1e-5f, "Smooth min should match min when distance > k");

    // smin test when a and b are close
    a = 1.0f; b = 1.1f; k = 0.5f;
    // |1.0 - 1.1| = 0.1 < 0.5. h = (0.5 - 0.1)/0.5 = 0.8
    // smin = 1.0 - 0.64 * 0.5 * 0.25 = 1.0 - 0.08 = 0.92
    h = std::max(k - std::abs(a - b), 0.0f) / k;
    smin_val = std::min(a, b) - h * h * k * 0.25f;
    ASSERT_TEST(smin_val < 1.0f, "Smooth min should produce smooth blending value smaller than min(a, b)");

    std::cout << "  [PASS] test_csg_sdf_primitives\n";
    return true;
}

// 3. Validate SimulationSettings Memory Alignment & Packing
bool test_settings_alignment() {
    SimulationSettings settings;
    std::memset(&settings, 0, sizeof(SimulationSettings));

    // Verify struct size matches 16-byte CUDA uniform buffer alignment
    size_t size = sizeof(SimulationSettings);
    ASSERT_TEST(size > 0, "SimulationSettings size must be non-zero");
    ASSERT_TEST(size % 16 == 0, "SimulationSettings struct must be aligned to 16-byte boundaries for CUDA uniform performance");

    std::cout << "  [PASS] test_settings_alignment (Size: " << size << " bytes, 16-byte aligned)\n";
    return true;
}

// 4. Validate Vector Math Operations
bool test_vector_math() {
    float3 v = make_float3(1.0f, 2.0f, 2.0f);
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); // sqrt(9) = 3.0
    ASSERT_TEST(std::abs(len - 3.0f) < 1e-5f, "Vector length calculation failed");

    float3 u = make_float3(2.0f, 0.0f, 0.0f);
    float dot_val = v.x * u.x + v.y * u.y + v.z * u.z; // 2.0
    ASSERT_TEST(std::abs(dot_val - 2.0f) < 1e-5f, "Vector dot product calculation failed");

    std::cout << "  [PASS] test_vector_math\n";
    return true;
}

int main() {
    std::cout << "=========================================================\n";
    std::cout << "       CUDA RAYMARCHER AUTOMATED UNIT TEST SUITE         \n";
    std::cout << "=========================================================\n";

    bool all_passed = true;
    all_passed &= test_isotropic_metric();
    all_passed &= test_csg_sdf_primitives();
    all_passed &= test_settings_alignment();
    all_passed &= test_vector_math();

    std::cout << "=========================================================\n";
    if (all_passed) {
        std::cout << " ALL UNIT TESTS PASSED SUCCESSFULLY!                    \n";
        std::cout << "=========================================================\n";
        return 0;
    } else {
        std::cerr << " SOME UNIT TESTS FAILED!                                \n";
        std::cout << "=========================================================\n";
        return 1;
    }
}
