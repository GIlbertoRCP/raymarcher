#include "raymarch.h"
#include "physics.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <device_launch_parameters.h>
#include <math.h>

namespace cg = cooperative_groups;

#define PI 3.14159265359f


// Basic vector math helper functions for float3 and float2
__device__ inline float3 fract(float3 v) {
    return make_float3(v.x - floorf(v.x), v.y - floorf(v.y), v.z - floorf(v.z));
}

__device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 mul(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float3 cross_product(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

// Operator overloads for float3 to support natural arithmetic operations like GLSL
__device__ inline float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float3 operator*(float s, float3 a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float3 operator*(float3 a, float3 b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ inline float3& operator+=(float3 &a, float3 b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

__device__ inline float3& operator*=(float3 &a, float s) {
    a.x *= s; a.y *= s; a.z *= s;
    return a;
}


// Pseudo-random hash for stars
__device__ float hash3(float3 p) {
    float3 scale = make_float3(443.8975f, 397.2973f, 491.1871f);
    float3 p_scaled = make_float3(p.x * scale.x, p.y * scale.y, p.z * scale.z);
    p = fract(p_scaled);
    
    float dot_val = dot(p, make_float3(p.y + 19.19f, p.z + 19.19f, p.x + 19.19f));
    p.x += dot_val;
    p.y += dot_val;
    p.z += dot_val;
    
    return p.x * p.y * p.z - floorf(p.x * p.y * p.z);
}

// Procedural starfield
__device__ float3 get_starfield(float3 dir) {
    float3 star_grid = mul(dir, 140.0f);
    float3 grid_id = make_float3(floorf(star_grid.x), floorf(star_grid.y), floorf(star_grid.z));
    float n = hash3(grid_id);
    
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    if (n > 0.985f) {
        float star_intensity = powf(hash3(mul(grid_id, 4.321f)), 6.0f) * 1.5f;
        float tint = hash3(mul(grid_id, 8.765f));
        float3 star_color = make_float3(0.9f, 0.95f, 1.0f);
        if (tint < 0.2f) star_color = make_float3(1.0f, 0.6f, 0.5f); // Red giant
        else if (tint > 0.8f) star_color = make_float3(0.5f, 0.8f, 1.0f); // Blue giant
        
        color = mul(star_color, star_intensity);
    }
    return color;
}

// Calibration grid background
__device__ float3 get_grid(float3 dir) {
    float theta = acosf(fmaxf(-1.0f, fminf(dir.z, 1.0f)));
    float phi = atan2f(dir.y, dir.x);
    
    float grid_theta = (sinf(theta * 18.0f) > 0.03f) ? 1.0f : 0.0f;
    float grid_phi = (sinf(phi * 36.0f) > 0.03f) ? 1.0f : 0.0f;
    float grid = grid_theta * grid_phi;
    
    float3 base = make_float3(0.02f, 0.03f, 0.08f);
    float3 lines = make_float3(0.0f, 0.7f, 1.0f) * 0.8f;
    return make_float3(
        lines.x * (1.0f - grid) + base.x * grid,
        lines.y * (1.0f - grid) + base.y * grid,
        lines.z * (1.0f - grid) + base.z * grid
    );
}

// Fits blackbody temperature to RGB colors
__device__ float3 blackbody(float Temp) {
    Temp = fmaxf(Temp, 0.0f);
    float t = Temp / 1000.0f;
    float r = 1.0f;
    float g = 1.0f;
    float b = 1.0f;
    
    if (t < 6.6f) {
        r = 1.0f;
        g = fmaxf(0.0f, fminf(-3.2f + 2.0f * logf(t + 0.6f) + 0.1f * t, 1.0f));
        b = fmaxf(0.0f, fminf(-10.0f + 4.0f * logf(t + 0.1f), 1.0f));
    } else {
        r = fmaxf(0.0f, fminf(3.0f * powf(t - 0.5f, -0.1f), 1.0f));
        g = fmaxf(0.0f, fminf(2.8f * powf(t - 0.5f, -0.07f), 1.0f));
        b = 1.0f;
    }
    
    if (Temp < 2500.0f) {
        float factor = (2500.0f - Temp) / 1500.0f;
        r = r * (1.0f - factor) + 1.0f * factor;
        g = g * (1.0f - factor) + 0.2f * factor;
        b = b * (1.0f - factor) + 0.0f * factor;
    }
    
    return make_float3(r, g, b);
}

__device__ inline float length3(float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ inline float isotropic_index(float r, float mass) {
    float rr = fmaxf(r, mass * 0.5f + 0.001f);
    float u = fminf(mass / (2.0f * rr), 0.999f);
    return (1.0f + u) * (1.0f + u) * (1.0f + u) / (1.0f - u);
}

// Ray acceleration from physics.h: a = c_x x + c_v v.
__device__ inline float3 compute_force(float3 x, float3 v, float mass) {
    float r = fmaxf(length3(x), mass * 0.5f + 0.001f);
    float c_x, c_v;
    physics::ray_acceleration_coeffs(r, dot(x, v), mass, c_x, c_v);
    return x * c_x + v * c_v;
}

// One RK4 step in coordinate time; |v| is then reset to the exact 1/n.
__device__ void rk4_step(float3 &x, float3 &v, float h, float mass) {
    float3 a1 = compute_force(x, v, mass);

    float3 x2 = x + v * (0.5f * h);
    float3 v2 = v + a1 * (0.5f * h);
    float3 a2 = compute_force(x2, v2, mass);

    float3 x3 = x + v2 * (0.5f * h);
    float3 v3 = v + a2 * (0.5f * h);
    float3 a3 = compute_force(x3, v3, mass);

    float3 x4 = x + v3 * h;
    float3 v4 = v + a3 * h;
    float3 a4 = compute_force(x4, v4, mass);

    x = x + (v + v2 * 2.0f + v3 * 2.0f + v4) * (h / 6.0f);
    v = v + (a1 + a2 * 2.0f + a3 * 2.0f + a4) * (h / 6.0f);

    float v_len = length3(v);
    if (v_len > 0.0f) {
        v = v * (1.0f / (v_len * isotropic_index(length3(x), mass)));
    }
}

__device__ inline float3 mix(float3 a, float3 b, float t) {
    return make_float3(
        a.x * (1.0f - t) + b.x * t,
        a.y * (1.0f - t) + b.y * t,
        a.z * (1.0f - t) + b.z * t
    );
}

// Samples procedural accretion disk density and emission
__device__ void sample_accretion_disk(
    float3 x, float3 v, float h,
    SimulationSettings settings,
    cudaTextureObject_t noiseTex,
    float3 &accum_color, float &accum_trans
) {
    float r_xy = sqrtf(x.x*x.x + x.y*x.y);
    float r_in = 3.2f * settings.mass;
    float r_out = 9.5f * settings.mass;
    
    if (r_xy >= r_in && r_xy <= r_out) {
        float disk_thickness = settings.disk_thickness * settings.mass;
        float vertical_density = expf(-powf(x.z, 2.0f) / (2.0f * powf(disk_thickness, 2.0f)));
        float radial_density = powf(r_in / r_xy, 2.5f);
        float density = settings.disk_density * vertical_density * radial_density;
        
        if (density > 0.001f) {
            float omega = sqrtf(settings.mass) / powf(r_xy, 1.5f);
            float phi = atan2f(x.y, x.x);
            float phi_rotated = phi - settings.time * settings.disk_speed * (0.8f / (powf(r_xy, 1.5f) + 0.1f));
            
            float u1 = r_xy * 0.12f - settings.time * 0.04f;
            float v1 = phi_rotated * 1.5f / (2.0f * PI) + x.z * 0.15f;
            float u2 = r_xy * 0.28f + settings.time * 0.08f;
            float v2 = phi_rotated * 3.0f / (2.0f * PI) - x.z * 0.15f;
            
            float n1 = tex2D<float4>(noiseTex, u1, v1).x;
            float n2 = tex2D<float4>(noiseTex, u2, v2).y * 0.5f;
            float noise = (n1 + n2) / 1.5f;
            
            density *= (0.25f + 0.75f * noise);
            
            float T = settings.disk_temp * powf(r_in / r_xy, 0.75f);
            float3 v_gas = make_float3(-x.y, x.x, 0.0f) * omega;
            
            float gas_speed = sqrtf(v_gas.x*v_gas.x + v_gas.y*v_gas.y + v_gas.z*v_gas.z);
            if (gas_speed > 0.58f) {
                float scale = 0.58f / gas_speed;
                v_gas.x *= scale;
                v_gas.y *= scale;
                v_gas.z *= scale;
            }
            
            float v_len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
            float3 photon_dir = make_float3(0.0f, 0.0f, 0.0f);
            if (v_len > 0.0f) {
                photon_dir = make_float3(v.x / v_len, v.y / v_len, v.z / v_len);
            }
            
            float beta = dot(v_gas, photon_dir);
            float g_doppler = sqrtf(1.0f - beta * beta) / (1.0f - beta);
            
            float x_len = sqrtf(x.x*x.x + x.y*x.y + x.z*x.z);
            float r = fmaxf(x_len, settings.mass * 0.5f + 0.001f);
            float u = fminf(settings.mass / (2.0f * r), 0.999f);
            float g_grav = (1.0f - u) / (1.0f + u);
            
            float g_total = g_doppler * g_grav;
            float T_obs = T * g_total;
            
            float3 emission_color = blackbody(T_obs);
            if (settings.shift_visualizer != 0) {
                if (g_total > 1.0f) {
                    float t = fmaxf(0.0f, fminf((g_total - 1.0f) * 1.5f, 1.0f));
                    emission_color = mix(make_float3(0.1f, 0.7f, 0.1f), make_float3(0.0f, 0.5f, 1.0f), t) * 2.0f;
                } else {
                    float t = fmaxf(0.0f, fminf((1.0f - g_total) * 1.2f, 1.0f));
                    emission_color = mix(make_float3(0.1f, 0.7f, 0.1f), make_float3(1.0f, 0.1f, 0.0f), t) * 2.0f;
                }
            }
            
            float dl = h;
            float opacity = density * 2.5f * dl;
            float trans = expf(-opacity);
            
            float3 emission = mul(emission_color, density * powf(g_total, 3.8f) * 1.8f * dl);
            accum_color.x += accum_trans * emission.x;
            accum_color.y += accum_trans * emission.y;
            accum_color.z += accum_trans * emission.z;
            
            accum_trans *= trans;
        }
    }
}

// Heatmap false-color generator mapping normalized step counts [0.0, 1.0] to Turbo/Viridis palette
__device__ float3 get_step_heatmap_color(float norm_steps) {
    norm_steps = fmaxf(0.0f, fminf(norm_steps, 1.0f));
    if (norm_steps < 0.25f) {
        float t = norm_steps / 0.25f;
        return mix(make_float3(0.05f, 0.05f, 0.30f), make_float3(0.0f, 0.75f, 0.85f), t);
    } else if (norm_steps < 0.50f) {
        float t = (norm_steps - 0.25f) / 0.25f;
        return mix(make_float3(0.0f, 0.75f, 0.85f), make_float3(0.10f, 0.85f, 0.25f), t);
    } else if (norm_steps < 0.75f) {
        float t = (norm_steps - 0.50f) / 0.25f;
        return mix(make_float3(0.10f, 0.85f, 0.25f), make_float3(0.95f, 0.85f, 0.10f), t);
    } else {
        float t = (norm_steps - 0.75f) / 0.25f;
        return mix(make_float3(0.95f, 0.85f, 0.10f), make_float3(0.95f, 0.10f, 0.60f), t);
    }
}

// Raymarching GPU kernel with CUDA Shared Memory Caching & Warp Primitives
__global__ void raymarch_kernel(
    float4 *d_output,
    unsigned long long *d_step_counter,
    SimulationSettings settings,
    cudaTextureObject_t noiseTex
) {
    // 1. CUDA Shared Memory Optimization: Block-level uniform settings caching
    __shared__ SimulationSettings s_settings;
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_settings = settings;
    }
    __syncthreads();

    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    int w = (int)s_settings.resolution.x;
    int h = (int)s_settings.resolution.y;
    
    if (x_idx >= w || y_idx >= h) return;
    
    // Invert vertical axis to match OpenGL UV layout
    float u_val = (float)x_idx / (float)w;
    float v_val = (float)(h - 1 - y_idx) / (float)h;
    
    float px = (u_val * 2.0f - 1.0f);
    float py = (v_val * 2.0f - 1.0f);
    px *= (s_settings.resolution.x / fmaxf(s_settings.resolution.y, 1.0f));
    
    float3 ray_pos = s_settings.cam_pos;
    
    float3 dir_term = make_float3(
        s_settings.cam_dir.x + px * s_settings.fov_scale * s_settings.cam_right.x + py * s_settings.fov_scale * s_settings.cam_up.x,
        s_settings.cam_dir.y + px * s_settings.fov_scale * s_settings.cam_right.y + py * s_settings.fov_scale * s_settings.cam_up.y,
        s_settings.cam_dir.z + px * s_settings.fov_scale * s_settings.cam_right.z + py * s_settings.fov_scale * s_settings.cam_up.z
    );
    float dir_len = sqrtf(dir_term.x*dir_term.x + dir_term.y*dir_term.y + dir_term.z*dir_term.z);
    float3 ray_dir = make_float3(dir_term.x / dir_len, dir_term.y / dir_len, dir_term.z / dir_len);
    
    float3 ray_vel = ray_dir;
    
    float r_cam = sqrtf(ray_pos.x*ray_pos.x + ray_pos.y*ray_pos.y + ray_pos.z*ray_pos.z);
    float r_cam_clamped = fmaxf(r_cam, s_settings.mass * 0.5f + 0.001f);
    float u_cam = fminf(s_settings.mass / (2.0f * r_cam_clamped), 0.999f);
    float n_cam = powf(1.0f + u_cam, 3.0f) / (1.0f - u_cam);
    
    ray_vel.x /= n_cam;
    ray_vel.y /= n_cam;
    ray_vel.z /= n_cam;
    
    float3 accum_color = make_float3(0.0f, 0.0f, 0.0f);
    float accum_trans = 1.0f;
    float r_eh = s_settings.mass * 0.5f;
    
    bool render_newtonian = s_settings.split_active != 0 && u_val > s_settings.split_x;

    unsigned int thread_steps = 0;

    if (!s_settings.relativity || s_settings.mass == 0.0f || render_newtonian) {
        if (s_settings.disk) {
            if (ray_vel.z != 0.0f) {
                float t = -ray_pos.z / ray_vel.z;
                if (t > 0.0f) {
                    float3 hit_pos = make_float3(
                        ray_pos.x + t * ray_vel.x,
                        ray_pos.y + t * ray_vel.y,
                        ray_pos.z + t * ray_vel.z
                    );
                    float r_xy = sqrtf(hit_pos.x*hit_pos.x + hit_pos.y*hit_pos.y);
                    float r_in = 3.2f * s_settings.mass;
                    float r_out = 9.5f * s_settings.mass;
                    if (r_xy >= r_in && r_xy <= r_out && s_settings.mass > 0.0f) {
                        float T = s_settings.disk_temp * powf(r_in / r_xy, 0.75f);
                        float phi = atan2f(hit_pos.y, hit_pos.x);
                        float phi_rotated = phi - s_settings.time * s_settings.disk_speed * (0.8f / (powf(r_xy, 1.5f) + 0.1f));
                        
                        float u1 = r_xy * 0.12f - s_settings.time * 0.04f;
                        float v1 = phi_rotated * 1.5f / (2.0f * PI);
                        float u2 = r_xy * 0.28f + s_settings.time * 0.08f;
                        float v2 = phi_rotated * 3.0f / (2.0f * PI);
                        
                        float n1 = tex2D<float4>(noiseTex, u1, v1).x;
                        float n2 = tex2D<float4>(noiseTex, u2, v2).y * 0.5f;
                        float noise = (n1 + n2) / 1.5f;
                        
                        float3 color = blackbody(T) * (0.25f + 0.75f * noise);
                        if (s_settings.shift_visualizer != 0) {
                            color = make_float3(0.1f, 0.7f, 0.1f) * (0.25f + 0.75f * noise) * 2.0f;
                        }
                        accum_color = color;
                        accum_trans = 0.15f;
                    }
                }
            }
        }
        
        float3 bg_color = make_float3(0.0f, 0.0f, 0.0f);
        if (s_settings.grid) {
            bg_color = get_grid(ray_dir);
        } else if (s_settings.stars) {
            bg_color = get_starfield(ray_dir);
        }
        
        float3 final_color = accum_color + accum_trans * bg_color;
        
        if (s_settings.split_active) {
            float dist_to_line = fabsf(u_val - s_settings.split_x);
            if (dist_to_line < 0.002f) {
                final_color = mix(make_float3(0.0f, 0.9f, 1.0f), final_color, dist_to_line / 0.002f);
            }
        }
        
        float3 color_exposed = final_color * s_settings.bloom_intensity;
        final_color.x = color_exposed.x / (color_exposed.x + 1.0f);
        final_color.y = color_exposed.y / (color_exposed.y + 1.0f);
        final_color.z = color_exposed.z / (color_exposed.z + 1.0f);
        
        final_color.x = powf(final_color.x, 1.0f / 2.2f);
        final_color.y = powf(final_color.y, 1.0f / 2.2f);
        final_color.z = powf(final_color.z, 1.0f / 2.2f);
        
        d_output[y_idx * w + x_idx] = make_float4(final_color.x, final_color.y, final_color.z, 1.0f);
        return;
    }
    
    bool captured = false;
    // Rays that get past both the scene and the camera are treated as escaped.
    float r_escape = fmaxf(30.0f, r_cam + 1.0f);
    float r_ps = s_settings.mass * (float)physics::kPhotonSphereIso;

    for (int step = 0; step < 300; ++step) {
        if (step >= s_settings.max_steps) break;
        thread_steps++;

        float r = length3(ray_pos);

        // Inside the photon sphere (R < 3M) an ingoing ray can only fall in, so
        // it is captured without integrating the stiff region near the horizon.
        if (r <= r_eh + 0.015f || (r < r_ps && dot(ray_pos, ray_vel) < 0.0f)) {
            captured = true;
            break;
        }
        if (r > r_escape) break;

        // Spatial step length: small near the hole, growing quadratically
        // beyond r = 6M where bending is weak.
        float r_dist = fmaxf(0.0f, r - r_eh);
        float ds = 0.05f + 0.35f * fminf(r_dist / 4.0f, 1.0f);
        float r_far = 6.0f * fmaxf(s_settings.mass, 0.1f);
        if (r > r_far) {
            ds += 0.08f * (r - r_far) * (r - r_far);
        }
        ds = fminf(ds, 1.25f);
        // The parameter is coordinate time and light moves at 1/n, so a step
        // of ds in space takes n * ds in time.
        float h_step = ds * isotropic_index(r, s_settings.mass);
        float3 prev_pos = ray_pos;

        rk4_step(ray_pos, ray_vel, h_step, s_settings.mass);

        // Photon sphere boundary visualization crossing
        if (s_settings.show_photon_sphere != 0 && s_settings.mass > 0.0f) {
            float r_prev = length3(prev_pos);
            float r_curr = length3(ray_pos);
            if ((r_prev > r_ps && r_curr <= r_ps) || (r_prev < r_ps && r_curr >= r_ps)) {
                accum_color.x += accum_trans * 0.0f * 0.22f;
                accum_color.y += accum_trans * 1.0f * 0.22f;
                accum_color.z += accum_trans * 0.35f * 0.22f;
            }
        }
        
        // Orbiting Hotspots
        if (s_settings.disk != 0 && s_settings.mass > 0.0f) {
            for (int i = 0; i < 3; ++i) {
                float r_orbit = 4.2f + (float)i * 2.0f;
                float omega = sqrtf(s_settings.mass) / powf(r_orbit, 1.5f);
                float phi_hot = s_settings.time * omega * s_settings.disk_speed + (float)i * 2.0944f;
                float3 p_orbit = make_float3(cosf(phi_hot) * r_orbit, sinf(phi_hot) * r_orbit, 0.0f);
                
                float3 diff = ray_pos - p_orbit;
                float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                
                if (dist < 0.35f) {
                    float3 v_gas = make_float3(-p_orbit.y, p_orbit.x, 0.0f) * omega;
                    float gas_speed = sqrtf(v_gas.x*v_gas.x + v_gas.y*v_gas.y + v_gas.z*v_gas.z);
                    if (gas_speed > 0.58f) {
                        float scale = 0.58f / gas_speed;
                        v_gas.x *= scale;
                        v_gas.y *= scale;
                        v_gas.z *= scale;
                    }
                    
                    float v_len = sqrtf(ray_vel.x*ray_vel.x + ray_vel.y*ray_vel.y + ray_vel.z*ray_vel.z);
                    float3 photon_dir = make_float3(0.0f, 0.0f, 0.0f);
                    if (v_len > 0.0f) {
                        photon_dir = make_float3(ray_vel.x / v_len, ray_vel.y / v_len, ray_vel.z / v_len);
                    }
                    
                    float beta = dot(v_gas, photon_dir);
                    float g_doppler = sqrtf(1.0f - beta * beta) / (1.0f - beta);
                    
                    float g_grav = (1.0f - s_settings.mass / (2.0f * r_orbit)) / (1.0f + s_settings.mass / (2.0f * r_orbit));
                    float g_total = g_doppler * g_grav;
                    
                    float3 p_color = make_float3(1.0f, 0.45f, 0.0f);
                    if (i == 0) { p_color = make_float3(1.0f, 0.2f, 0.1f); }
                    else if (i == 1) { p_color = make_float3(1.0f, 0.7f, 0.1f); }
                    else { p_color = make_float3(0.1f, 0.8f, 1.0f); }
                    
                    if (s_settings.shift_visualizer != 0) {
                        if (g_total > 1.0f) {
                            float t = fmaxf(0.0f, fminf((g_total - 1.0f) * 1.5f, 1.0f));
                            p_color = mix(make_float3(0.1f, 0.7f, 0.1f), make_float3(0.0f, 0.5f, 1.0f), t) * 2.0f;
                        } else {
                            float t = fmaxf(0.0f, fminf((1.0f - g_total) * 1.2f, 1.0f));
                            p_color = mix(make_float3(0.1f, 0.7f, 0.1f), make_float3(1.0f, 0.1f, 0.0f), t) * 2.0f;
                        }
                    } else {
                        p_color = p_color * powf(g_total, 3.8f);
                    }
                    
                    float density_p = expf(-powf(dist / 0.18f, 2.0f));
                    accum_color.x += accum_trans * p_color.x * density_p * 0.45f;
                    accum_color.y += accum_trans * p_color.y * density_p * 0.45f;
                    accum_color.z += accum_trans * p_color.z * density_p * 0.45f;
                }
            }
        }
        
        if (s_settings.disk) {
            float sign_prev = prev_pos.z;
            float sign_curr = ray_pos.z;
            
            if (sign_prev * sign_curr <= 0.0f || fabsf(ray_pos.z) < 0.18f * s_settings.mass) {
                float fraction = fabsf(prev_pos.z) / (fabsf(prev_pos.z) + fabsf(ray_pos.z) + 1e-6f);
                float3 cross_pos = make_float3(
                    prev_pos.x * (1.0f - fraction) + ray_pos.x * fraction,
                    prev_pos.y * (1.0f - fraction) + ray_pos.y * fraction,
                    prev_pos.z * (1.0f - fraction) + ray_pos.z * fraction
                );
                
                sample_accretion_disk(
                    cross_pos, ray_vel, ds,
                    s_settings,
                    noiseTex,
                    accum_color, accum_trans
                );
            }
        }
        
        if (accum_trans < 0.01f) {
            accum_trans = 0.0f;
            break;
        }
    }

    // Sum step counts across the active threads of the warp, then do one
    // atomic per warp instead of one per pixel.
    if (d_step_counter != nullptr) {
        cg::coalesced_group warp = cg::coalesced_threads();
        unsigned long long warp_steps = cg::reduce(warp, (unsigned long long)thread_steps, cg::plus<unsigned long long>());
        if (warp.thread_rank() == 0) {
            atomicAdd(d_step_counter, warp_steps);
        }
    }

    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
    if (!captured) {
        float vel_len = sqrtf(ray_vel.x*ray_vel.x + ray_vel.y*ray_vel.y + ray_vel.z*ray_vel.z);
        float3 escape_dir = make_float3(0.0f, 0.0f, 0.0f);
        if (vel_len > 0.0f) {
            escape_dir = make_float3(ray_vel.x / vel_len, ray_vel.y / vel_len, ray_vel.z / vel_len);
        }
        
        if (s_settings.grid) {
            final_color = get_grid(escape_dir);
        } else if (s_settings.stars) {
            final_color = get_starfield(escape_dir);
        }
    }
    
    float3 color_out = make_float3(
        accum_color.x + accum_trans * final_color.x,
        accum_color.y + accum_trans * final_color.y,
        accum_color.z + accum_trans * final_color.z
    );
    
    if (s_settings.split_active) {
        float dist_to_line = fabsf(u_val - s_settings.split_x);
        if (dist_to_line < 0.002f) {
            color_out = mix(make_float3(0.0f, 0.9f, 1.0f), color_out, dist_to_line / 0.002f);
        }
    }
    
    // Diagnostics Heatmap Mode Override
    if (s_settings.heatmap_mode != 0) {
        float norm_steps = (float)thread_steps / (float)fmaxf(1.0f, (float)s_settings.max_steps);
        color_out = get_step_heatmap_color(norm_steps);
    } else {
        // Tone mapping and gamma correction
        float3 color_exposed = color_out * s_settings.bloom_intensity;
        color_out.x = color_exposed.x / (color_exposed.x + 1.0f);
        color_out.y = color_exposed.y / (color_exposed.y + 1.0f);
        color_out.z = color_exposed.z / (color_exposed.z + 1.0f);
        
        color_out.x = powf(color_out.x, 1.0f / 2.2f);
        color_out.y = powf(color_out.y, 1.0f / 2.2f);
        color_out.z = powf(color_out.z, 1.0f / 2.2f);
    }
    
    d_output[y_idx * w + x_idx] = make_float4(color_out.x, color_out.y, color_out.z, 1.0f);
}

// Kernel launcher with configurable thread block shape
extern "C" cudaError_t run_raymarch_kernel(
    float4 *d_output,
    unsigned long long *d_step_counter,
    SimulationSettings settings,
    cudaTextureObject_t noiseTex
) {
    int bx = (settings.block_dim_x > 0) ? settings.block_dim_x : 32;
    int by = (settings.block_dim_y > 0) ? settings.block_dim_y : 8;
    dim3 block(bx, by);
    dim3 grid(
        ((int)settings.resolution.x + block.x - 1) / block.x,
        ((int)settings.resolution.y + block.y - 1) / block.y
    );
    raymarch_kernel<<<grid, block>>>(d_output, d_step_counter, settings, noiseTex);
    return cudaGetLastError();
}
