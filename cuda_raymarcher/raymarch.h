#ifndef RAYMARCH_H
#define RAYMARCH_H

#include <cuda_runtime.h>

// Mirror the simulation settings from TypeScript
struct SimulationSettings {
    float2 resolution;
    float time;
    float mass;
    float spin;
    int relativity;
    int disk;
    int grid;
    int stars;
    float disk_temp;
    int max_steps;
    float disk_thickness;
    float disk_density;
    float bloom_intensity;
    float disk_speed;
    int show_photon_sphere;
    int shift_visualizer;
    int split_active;
    float split_x;
    int spectrum_mode;
    
    // CSG & Advanced Visual Effects
    int csg_mode;         // 0: None, 1: Union, 2: Smooth Min, 3: Subtraction, 4: Intersection
    float csg_blend;      // Smooth blending factor k
    int soft_shadows;     // 0: Off, 1: Soft shadows enabled
    float shadow_k;       // Penumbra softness factor
    int ao_enabled;       // 0: Off, 1: SDF Ambient Occlusion
    float ao_intensity;   // Ambient Occlusion intensity

    // Gravitational Waves & Relativistic Polarization
    int gw_enabled;       // 0: Off, 1: Gravitational wave metric strain active
    float gw_amplitude;   // Strain wave amplitude A_gw
    float gw_frequency;   // Wave frequency omega_gw
    int polarization_mode;// 0: Off, 1: Synchrotron Stokes linear polarization overlay
    int taa_enabled;      // 0: Off, 1: Temporal Anti-Aliasing (TAA) accumulation
    float taa_blend;      // Reprojection blending alpha

    // Workgroup Block Tuning, Spatial Bounding Acceleration & Diagnostics
    int block_dim_x;      // Thread block width (e.g. 16, 32, 8)
    int block_dim_y;      // Thread block height (e.g. 16, 8, 32)
    int spatial_accel;    // 0: Disabled, 1: Bounding Volume / Spatial Grid acceleration active
    int heatmap_mode;     // 0: Off (Normal render), 1: Ray-Step count false-color heatmap

    float3 cam_pos;
    float3 cam_dir;
    float3 cam_up;
    float3 cam_right;
    float fov_scale;

    // 16-Byte Uniform Buffer Alignment Padding
    float padding0;
    float padding1;
    float padding2;
};

// Benchmark metrics summary structure
struct BenchmarkResult {
    float width;
    float height;
    float avg_frame_time_ms;
    float fps;
    double total_ray_steps;
    double avg_steps_per_pixel;
    double memory_throughput_gbps;
    int block_dim_x;
    int block_dim_y;
    int spatial_accel;
};

// Launch function declaration for the CUDA raymarch kernel
extern "C" void run_raymarch_kernel(
    float4 *d_output,
    unsigned long long *d_step_counter,
    SimulationSettings settings,
    cudaTextureObject_t noiseTex
);

// Launch function declaration for temporal reprojection (TAA) blending
extern "C" void apply_temporal_reprojection(
    float4 *d_current,
    float4 *d_history,
    int width,
    int height,
    float blend_alpha
);

#endif // RAYMARCH_H
