#include "raymarch.h"
#include <device_launch_parameters.h>
#include <math.h>

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

// Relativistic equations of motion in isotropic coordinates
__device__ float3 compute_force(float3 x, float3 v, float mass, float spin) {
    float x_len = sqrtf(x.x*x.x + x.y*x.y + x.z*x.z);
    float r = fmaxf(x_len, mass * 0.5f + 0.001f);
    float u = fminf(mass / (2.0f * r), 0.999f);
    float one_plus_u = 1.0f + u;
    float one_minus_u = 1.0f - u;
    
    float num = -mass * powf(one_plus_u, 5.0f) * (2.0f - u);
    float den = powf(r, 3.0f) * powf(one_minus_u, 3.0f);
    float3 f_grav = mul(x, num / den);
    
    if (spin > 0.0f) {
        float j = mass * spin;
        float omega = 2.0f * j / (powf(r, 3.0f) * powf(one_plus_u, 6.0f));
        float3 omega_vec = make_float3(0.0f, 0.0f, omega);
        float3 f_fd = mul(cross_product(v, omega_vec), 2.0f);
        return make_float3(f_grav.x + f_fd.x, f_grav.y + f_fd.y, f_grav.z + f_fd.z);
    }
    
    return f_grav;
}

// Single step of Runge-Kutta 4th order integration
__device__ void rk4_step(float3 &x, float3 &v, float h, float mass, float spin) {
    float3 x1 = x;
    float3 v1 = v;
    float3 a1 = compute_force(x1, v1, mass, spin);
    
    float3 x2 = make_float3(x.x + 0.5f * h * v1.x, x.y + 0.5f * h * v1.y, x.z + 0.5f * h * v1.z);
    float3 v2 = make_float3(v.x + 0.5f * h * a1.x, v.y + 0.5f * h * a1.y, v.z + 0.5f * h * a1.z);
    float3 a2 = compute_force(x2, v2, mass, spin);
    
    float3 x3 = make_float3(x.x + 0.5f * h * v2.x, x.y + 0.5f * h * v2.y, x.z + 0.5f * h * v2.z);
    float3 v3 = make_float3(v.x + 0.5f * h * a2.x, v.y + 0.5f * h * a2.y, v.z + 0.5f * h * a2.z);
    float3 a3 = compute_force(x3, v3, mass, spin);
    
    float3 x4 = make_float3(x.x + h * v3.x, x.y + h * v3.y, x.z + h * v3.z);
    float3 v4 = make_float3(v.x + h * a3.x, v.y + h * a3.y, v.z + h * a3.z);
    float3 a4 = compute_force(x4, v4, mass, spin);
    
    x.x += (h / 6.0f) * (v1.x + 2.0f * v2.x + 2.0f * v3.x + v4.x);
    x.y += (h / 6.0f) * (v1.y + 2.0f * v2.y + 2.0f * v3.y + v4.y);
    x.z += (h / 6.0f) * (v1.z + 2.0f * v2.z + 2.0f * v3.z + v4.z);
    
    v.x += (h / 6.0f) * (a1.x + 2.0f * a2.x + 2.0f * a3.x + a4.x);
    v.y += (h / 6.0f) * (a1.y + 2.0f * a2.y + 2.0f * a3.y + a4.y);
    v.z += (h / 6.0f) * (a1.z + 2.0f * a2.z + 2.0f * a3.z + a4.z);
    
    // Normalization
    float x_len = sqrtf(x.x*x.x + x.y*x.y + x.z*x.z);
    float r = fmaxf(x_len, mass * 0.5f + 0.001f);
    float u = fminf(mass / (2.0f * r), 0.999f);
    float n_ref = powf(1.0f + u, 3.0f) / (1.0f - u);
    float target_v_len = 1.0f / n_ref;
    
    float v_len = sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
    if (v_len > 0.0f) {
        float scale = target_v_len / v_len;
        v.x *= scale;
        v.y *= scale;
        v.z *= scale;
    }
}

// Samples procedural accretion disk density and emission
__device__ void sample_accretion_disk(
    float3 x, float3 v, float h,
    float mass, float spin, float disk_temp, float time,
    cudaTextureObject_t noiseTex,
    float3 &accum_color, float &accum_trans
) {
    float r_xy = sqrtf(x.x*x.x + x.y*x.y);
    float r_in = 3.2f * mass;
    float r_out = 9.5f * mass;
    
    if (r_xy >= r_in && r_xy <= r_out) {
        float disk_thickness = 0.08f * mass;
        float vertical_density = expf(-powf(x.z, 2.0f) / (2.0f * powf(disk_thickness, 2.0f)));
        float radial_density = powf(r_in / r_xy, 2.5f);
        float density = 1.8f * vertical_density * radial_density;
        
        if (density > 0.001f) {
            float omega = 1.0f / (powf(r_xy, 1.5f) + spin);
            float phi = atan2f(x.y, x.x);
            float phi_rotated = phi - time * (0.8f / (powf(r_xy, 1.5f) + 0.1f));
            
            float u1 = r_xy * 0.12f - time * 0.04f;
            float v1 = phi_rotated * 1.5f / (2.0f * PI);
            float u2 = r_xy * 0.28f + time * 0.08f;
            float v2 = phi_rotated * 3.0f / (2.0f * PI);
            
            float n1 = tex2D<float4>(noiseTex, u1, v1).x;
            float n2 = tex2D<float4>(noiseTex, u2, v2).y * 0.5f;
            float noise = (n1 + n2) / 1.5f;
            
            density *= (0.25f + 0.75f * noise);
            
            float T = disk_temp * powf(r_in / r_xy, 0.75f);
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
            float r = fmaxf(x_len, mass * 0.5f + 0.001f);
            float u = fminf(mass / (2.0f * r), 0.999f);
            float g_grav = (1.0f - u) / (1.0f + u);
            
            float g_total = g_doppler * g_grav;
            float T_obs = T * g_total;
            float3 emission_color = blackbody(T_obs);
            
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

// Raymarching GPU kernel
__global__ void raymarch_kernel(
    float4 *d_output,
    SimulationSettings settings,
    cudaTextureObject_t noiseTex
) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    int w = (int)settings.resolution.x;
    int h = (int)settings.resolution.y;
    
    if (x_idx >= w || y_idx >= h) return;
    
    // Invert vertical axis to match OpenGL UV layout (WebGL bottom-up coordinate space)
    float u_val = (float)x_idx / (float)w;
    float v_val = (float)(h - 1 - y_idx) / (float)h;
    
    float px = (u_val * 2.0f - 1.0f);
    float py = (v_val * 2.0f - 1.0f);
    px *= (settings.resolution.x / fmaxf(settings.resolution.y, 1.0f));
    
    float3 ray_pos = settings.cam_pos;
    
    float3 dir_term = make_float3(
        settings.cam_dir.x + px * settings.fov_scale * settings.cam_right.x + py * settings.fov_scale * settings.cam_up.x,
        settings.cam_dir.y + px * settings.fov_scale * settings.cam_right.y + py * settings.fov_scale * settings.cam_up.y,
        settings.cam_dir.z + px * settings.fov_scale * settings.cam_right.z + py * settings.fov_scale * settings.cam_up.z
    );
    float dir_len = sqrtf(dir_term.x*dir_term.x + dir_term.y*dir_term.y + dir_term.z*dir_term.z);
    float3 ray_dir = make_float3(dir_term.x / dir_len, dir_term.y / dir_len, dir_term.z / dir_len);
    
    float3 ray_vel = ray_dir;
    
    float r_cam = sqrtf(ray_pos.x*ray_pos.x + ray_pos.y*ray_pos.y + ray_pos.z*ray_pos.z);
    float r_cam_clamped = fmaxf(r_cam, settings.mass * 0.5f + 0.001f);
    float u_cam = fminf(settings.mass / (2.0f * r_cam_clamped), 0.999f);
    float n_cam = powf(1.0f + u_cam, 3.0f) / (1.0f - u_cam);
    
    ray_vel.x /= n_cam;
    ray_vel.y /= n_cam;
    ray_vel.z /= n_cam;
    
    float3 accum_color = make_float3(0.0f, 0.0f, 0.0f);
    float accum_trans = 1.0f;
    float r_eh = settings.mass * 0.5f;
    
    if (!settings.relativity || settings.mass == 0.0f) {
        if (settings.disk) {
            if (ray_vel.z != 0.0f) {
                float t = -ray_pos.z / ray_vel.z;
                if (t > 0.0f) {
                    float3 hit_pos = make_float3(
                        ray_pos.x + t * ray_vel.x,
                        ray_pos.y + t * ray_vel.y,
                        ray_pos.z + t * ray_vel.z
                    );
                    float r_xy = sqrtf(hit_pos.x*hit_pos.x + hit_pos.y*hit_pos.y);
                    float r_in = 3.2f * settings.mass;
                    float r_out = 9.5f * settings.mass;
                    if (r_xy >= r_in && r_xy <= r_out && settings.mass > 0.0f) {
                        float T = settings.disk_temp * powf(r_in / r_xy, 0.75f);
                        float omega = 1.0f / (powf(r_xy, 1.5f) + settings.spin);
                        float phi = atan2f(hit_pos.y, hit_pos.x);
                        float phi_rotated = phi - settings.time * (0.8f / (powf(r_xy, 1.5f) + 0.1f));
                        
                        float u1 = r_xy * 0.12f - settings.time * 0.04f;
                        float v1 = phi_rotated * 1.5f / (2.0f * PI);
                        float u2 = r_xy * 0.28f + settings.time * 0.08f;
                        float v2 = phi_rotated * 3.0f / (2.0f * PI);
                        
                        float n1 = tex2D<float4>(noiseTex, u1, v1).x;
                        float n2 = tex2D<float4>(noiseTex, u2, v2).y * 0.5f;
                        float noise = (n1 + n2) / 1.5f;
                        
                        accum_color = blackbody(T) * (0.25f + 0.75f * noise);
                        accum_trans = 0.15f;
                    }
                }
            }
        }
        
        float3 bg_color = make_float3(0.0f, 0.0f, 0.0f);
        if (settings.grid) {
            bg_color = get_grid(ray_dir);
        } else if (settings.stars) {
            bg_color = get_starfield(ray_dir);
        }
        
        accum_color.x += accum_trans * bg_color.x;
        accum_color.y += accum_trans * bg_color.y;
        accum_color.z += accum_trans * bg_color.z;
        
        d_output[y_idx * w + x_idx] = make_float4(accum_color.x, accum_color.y, accum_color.z, 1.0f);
        return;
    }
    
    // Geodesic integration loop
    bool captured = false;
    
    for (int step = 0; step < 300; ++step) {
        if (step >= settings.max_steps) break;
        float r = sqrtf(ray_pos.x*ray_pos.x + ray_pos.y*ray_pos.y + ray_pos.z*ray_pos.z);
        
        if (r <= r_eh + 0.015f) {
            captured = true;
            break;
        }
        
        if (r > 30.0f) {
            break;
        }
        
        float h_step = 0.18f * (0.05f + 0.95f * fmaxf(0.0f, fminf((r - r_eh) / 5.0f, 1.0f)));
        float3 prev_pos = ray_pos;
        
        rk4_step(ray_pos, ray_vel, h_step, settings.mass, settings.spin);
        
        if (settings.disk) {
            float sign_prev = prev_pos.z;
            float sign_curr = ray_pos.z;
            
            if (sign_prev * sign_curr <= 0.0f || fabsf(ray_pos.z) < 0.18f * settings.mass) {
                float fraction = fabsf(prev_pos.z) / (fabsf(prev_pos.z) + fabsf(ray_pos.z) + 1e-6f);
                float3 cross_pos = make_float3(
                    prev_pos.x * (1.0f - fraction) + ray_pos.x * fraction,
                    prev_pos.y * (1.0f - fraction) + ray_pos.y * fraction,
                    prev_pos.z * (1.0f - fraction) + ray_pos.z * fraction
                );
                
                sample_accretion_disk(
                    cross_pos, ray_vel, h_step,
                    settings.mass, settings.spin, settings.disk_temp, settings.time,
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
    
    float3 final_color = make_float3(0.0f, 0.0f, 0.0f);
    if (!captured) {
        float vel_len = sqrtf(ray_vel.x*ray_vel.x + ray_vel.y*ray_vel.y + ray_vel.z*ray_vel.z);
        float3 escape_dir = make_float3(0.0f, 0.0f, 0.0f);
        if (vel_len > 0.0f) {
            escape_dir = make_float3(ray_vel.x / vel_len, ray_vel.y / vel_len, ray_vel.z / vel_len);
        }
        
        if (settings.grid) {
            final_color = get_grid(escape_dir);
        } else if (settings.stars) {
            final_color = get_starfield(escape_dir);
        }
    }
    
    float3 color_out = make_float3(
        accum_color.x + accum_trans * final_color.x,
        accum_color.y + accum_trans * final_color.y,
        accum_color.z + accum_trans * final_color.z
    );
    
    // Tone mapping and gamma correction
    color_out.x = color_out.x / (color_out.x + 1.0f);
    color_out.y = color_out.y / (color_out.y + 1.0f);
    color_out.z = color_out.z / (color_out.z + 1.0f);
    
    color_out.x = powf(color_out.x, 1.0f / 2.2f);
    color_out.y = powf(color_out.y, 1.0f / 2.2f);
    color_out.z = powf(color_out.z, 1.0f / 2.2f);
    
    d_output[y_idx * w + x_idx] = make_float4(color_out.x, color_out.y, color_out.z, 1.0f);
}

// Kernel launcher
extern "C" void run_raymarch_kernel(
    float4 *d_output,
    SimulationSettings settings,
    cudaTextureObject_t noiseTex
) {
    dim3 block(16, 16);
    dim3 grid(
        ((int)settings.resolution.x + block.x - 1) / block.x,
        ((int)settings.resolution.y + block.y - 1) / block.y
    );
    raymarch_kernel<<<grid, block>>>(d_output, settings, noiseTex);
    cudaDeviceSynchronize();
}
