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

    float3 cam_pos;
    float3 cam_dir;
    float3 cam_up;
    float3 cam_right;
    float fov_scale;
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
};

// Launch function declaration for the CUDA kernel
extern "C" void run_raymarch_kernel(
    float4 *d_output,
    unsigned long long *d_step_counter,
    SimulationSettings settings,
    cudaTextureObject_t noiseTex
);

#endif // RAYMARCH_H
