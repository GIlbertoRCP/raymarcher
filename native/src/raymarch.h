#ifndef RAYMARCH_H
#define RAYMARCH_H

#include <cuda_runtime.h>

// Render settings passed to the kernel by value each frame.
struct SimulationSettings {
    float2 resolution;
    float time;
    float mass;
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

    int block_dim_x;   // Thread block width (e.g. 16, 32, 8)
    int block_dim_y;   // Thread block height (e.g. 16, 8, 32)
    int heatmap_mode;  // 1: colour pixels by integration steps taken

    float3 cam_pos;
    float3 cam_dir;
    float3 cam_up;
    float3 cam_right;
    float fov_scale;
};

// Renders one frame into d_output. If d_step_counter is non-null, the total
// number of integration steps taken by all rays is added to it.
// Returns the launch error, if any.
extern "C" cudaError_t run_raymarch_kernel(
    float4 *d_output,
    unsigned long long *d_step_counter,
    SimulationSettings settings,
    cudaTextureObject_t noiseTex
);

#endif // RAYMARCH_H
