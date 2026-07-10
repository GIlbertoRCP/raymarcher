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
    
    float3 cam_pos;
    float3 cam_dir;
    float3 cam_up;
    float3 cam_right;
    float fov_scale;
};

// Launch function declaration for the CUDA kernel
extern "C" void run_raymarch_kernel(
    float4 *d_output,
    SimulationSettings settings,
    cudaTextureObject_t noiseTex
);

#endif // RAYMARCH_H
