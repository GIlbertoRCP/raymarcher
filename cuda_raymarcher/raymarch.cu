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

// Quadrupolar Gravitational Wave metric strain acceleration
__device__ inline float3 evaluate_gw_force(float3 pos, float3 vel, float time, float amp, float freq) {
    if (amp <= 0.0001f) return make_float3(0.0f, 0.0f, 0.0f);
    float phase = freq * time - 0.5f * pos.z;
    float cos_p = cosf(phase);
    float sin_p = sinf(phase);

    float h_plus = amp * cos_p;
    float h_cross = amp * sin_p;

    float3 f;
    f.x = 0.5f * freq * (-h_plus * vel.x + h_cross * vel.y) * sin_p;
    f.y = 0.5f * freq * ( h_cross * vel.x + h_plus * vel.y) * sin_p;
    f.z = -0.25f * freq * (h_plus * (vel.x * vel.x - vel.y * vel.y) + 2.0f * h_cross * vel.x * vel.y) * cos_p;
    return f;
}

// Relativistic equations of motion in isotropic coordinates
__device__ float3 compute_force(float3 x, float3 v, float mass, float spin, SimulationSettings settings) {
    float x_len = sqrtf(x.x*x.x + x.y*x.y + x.z*x.z);
    float r = fmaxf(x_len, mass * 0.5f + 0.001f);
    float u = fminf(mass / (2.0f * r), 0.999f);
    float one_plus_u = 1.0f + u;
    float one_minus_u = 1.0f - u;
    
    float num = -mass * powf(one_plus_u, 5.0f) * (2.0f - u);
    float den = powf(r, 3.0f) * powf(one_minus_u, 3.0f);
    float3 f_total = mul(x, num / den);
    
    if (spin > 0.0f) {
        float j = mass * spin;
        float omega = 2.0f * j / (powf(r, 3.0f) * powf(one_plus_u, 6.0f));
        float3 omega_vec = make_float3(0.0f, 0.0f, omega);
        float3 f_fd = mul(cross_product(v, omega_vec), 2.0f);
        f_total.x += f_fd.x;
        f_total.y += f_fd.y;
        f_total.z += f_fd.z;
    }

    if (settings.gw_enabled) {
        float3 f_gw = evaluate_gw_force(x, v, settings.time, settings.gw_amplitude, settings.gw_frequency);
        f_total.x += f_gw.x;
        f_total.y += f_gw.y;
        f_total.z += f_gw.z;
    }
    
    return f_total;
}

// Single step of Runge-Kutta 4th order integration
__device__ void rk4_step(float3 &x, float3 &v, float h, float mass, float spin, SimulationSettings settings) {
    float3 x1 = x;
    float3 v1 = v;
    float3 a1 = compute_force(x1, v1, mass, spin, settings);
    
    float3 x2 = make_float3(x.x + 0.5f * h * v1.x, x.y + 0.5f * h * v1.y, x.z + 0.5f * h * v1.z);
    float3 v2 = make_float3(v.x + 0.5f * h * a1.x, v.y + 0.5f * h * a1.y, v.z + 0.5f * h * a1.z);
    float3 a2 = compute_force(x2, v2, mass, spin, settings);
    
    float3 x3 = make_float3(x.x + 0.5f * h * v2.x, x.y + 0.5f * h * v2.y, x.z + 0.5f * h * v2.z);
    float3 v3 = make_float3(v.x + 0.5f * h * a2.x, v.y + 0.5f * h * a2.y, v.z + 0.5f * h * a2.z);
    float3 a3 = compute_force(x3, v3, mass, spin, settings);
    
    float3 x4 = make_float3(x.x + h * v3.x, x.y + h * v3.y, x.z + h * v3.z);
    float3 v4 = make_float3(v.x + h * a3.x, v.y + h * a3.y, v.z + h * a3.z);
    float3 a4 = compute_force(x4, v4, mass, spin, settings);
    
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

__device__ inline float3 mix(float3 a, float3 b, float t) {
    return make_float3(
        a.x * (1.0f - t) + b.x * t,
        a.y * (1.0f - t) + b.y * t,
        a.z * (1.0f - t) + b.z * t
    );
}

__device__ float3 get_color_from_spectrum(float Temp, int mode) {
    if (mode == 0) {
        return blackbody(Temp);
    }
    float t = fmaxf(0.0f, fminf(Temp / 12000.0f, 1.0f));
    if (mode == 1) {
        // Nebula Glow: violet-blue to magenta-pink
        return mix(make_float3(0.18f, 0.0f, 0.5f), make_float3(1.0f, 0.0f, 0.55f), t) * 2.2f;
    }
    if (mode == 2) {
        // Plasma Fire: dark red to bright yellow-orange to white
        return mix(make_float3(0.55f, 0.05f, 0.0f), make_float3(1.0f, 0.88f, 0.15f), t) * 2.2f;
    }
    // Quantum Shift: cyan-blue to neon green
    return mix(make_float3(0.0f, 0.35f, 0.8f), make_float3(0.0f, 1.0f, 0.35f), t) * 2.2f;
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
            float omega = 1.0f / (powf(r_xy, 1.5f) + settings.spin);
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
            
            float3 emission_color = get_color_from_spectrum(T_obs, settings.spectrum_mode);
            if (settings.shift_visualizer != 0) {
                if (g_total > 1.0f) {
                    float t = fmaxf(0.0f, fminf((g_total - 1.0f) * 1.5f, 1.0f));
                    emission_color = mix(make_float3(0.1f, 0.7f, 0.1f), make_float3(0.0f, 0.5f, 1.0f), t) * 2.0f;
                } else {
                    float t = fmaxf(0.0f, fminf((1.0f - g_total) * 1.2f, 1.0f));
                    emission_color = mix(make_float3(0.1f, 0.7f, 0.1f), make_float3(1.0f, 0.1f, 0.0f), t) * 2.0f;
                }
            } else if (settings.polarization_mode != 0) {
                // Synchrotron Linear Polarization Direction Vector E_pol = v_gas x B (where B = toroidal -y, x, 0)
                float pol_angle = atan2f(v_gas.y, v_gas.x);
                float pol_t = 0.5f + 0.5f * sinf(pol_angle * 2.0f);
                emission_color = mix(make_float3(1.0f, 0.1f, 0.8f), make_float3(0.0f, 0.9f, 1.0f), pol_t) * 2.5f;
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

// CSG Signed Distance Functions
__device__ inline float sdSphere(float3 p, float r) {
    return sqrtf(p.x*p.x + p.y*p.y + p.z*p.z) - r;
}

__device__ inline float sdTorus(float3 p, float2 t) {
    float q_x = sqrtf(p.x*p.x + p.y*p.y) - t.x;
    return sqrtf(q_x*q_x + p.z*p.z) - t.y;
}

__device__ inline float smin(float a, float b, float k) {
    if (k <= 0.001f) return fminf(a, b);
    float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
    return fminf(a, b) - h * h * k * 0.25f;
}

__device__ inline float smax(float a, float b, float k) {
    if (k <= 0.001f) return fmaxf(a, b);
    float h = fmaxf(k - fabsf(a - b), 0.0f) / k;
    return fmaxf(a, b) + h * h * k * 0.25f;
}

// CSG Evaluation returning distance & material color
__device__ float map_csg(float3 p, int csg_mode, float csg_blend, float mass, float3 &out_color) {
    float d_main = sdSphere(p, 1.8f * mass);
    out_color = make_float3(0.2f, 0.6f, 1.0f); // Default cyan
    
    if (csg_mode == 0) return d_main; // Disabled CSG
    
    float3 p_torus = make_float3(p.x, p.y, p.z - 0.4f * mass);
    float d_torus = sdTorus(p_torus, make_float2(3.5f * mass, 0.4f * mass));
    
    float3 p_sph = make_float3(p.x - 3.2f * mass, p.y, p.z);
    float d_sph = sdSphere(p_sph, 0.9f * mass);
    float d_sub = fminf(d_torus, d_sph);
    
    if (csg_mode == 1) { // Union
        if (d_sub < d_main) out_color = make_float3(1.0f, 0.4f, 0.2f);
        return fminf(d_main, d_sub);
    } else if (csg_mode == 2) { // Smooth Min
        float d_res = smin(d_main, d_sub, csg_blend);
        float h = fmaxf(0.0f, fminf((d_main - d_sub) / (csg_blend + 1e-4f) + 0.5f, 1.0f));
        out_color = mix(make_float3(1.0f, 0.4f, 0.2f), make_float3(0.2f, 0.6f, 1.0f), h);
        return d_res;
    } else if (csg_mode == 3) { // Subtraction
        float d_res = smax(d_main, -d_sub, csg_blend);
        if (-d_sub > d_main) out_color = make_float3(0.9f, 0.1f, 0.3f);
        return d_res;
    } else if (csg_mode == 4) { // Intersection
        float d_res = smax(d_main, d_sub, csg_blend);
        out_color = make_float3(0.8f, 0.2f, 0.9f);
        return d_res;
    }
    return d_main;
}

// Compute normal vector for CSG surface
__device__ float3 calc_normal(float3 p, int csg_mode, float csg_blend, float mass) {
    float eps = 0.002f;
    float3 dummy;
    float d = map_csg(p, csg_mode, csg_blend, mass, dummy);
    float3 n = make_float3(
        map_csg(make_float3(p.x + eps, p.y, p.z), csg_mode, csg_blend, mass, dummy) - d,
        map_csg(make_float3(p.x, p.y + eps, p.z), csg_mode, csg_blend, mass, dummy) - d,
        map_csg(make_float3(p.x, p.y, p.z + eps), csg_mode, csg_blend, mass, dummy) - d
    );
    float len = sqrtf(n.x*n.x + n.y*n.y + n.z*n.z);
    return (len > 0.0001f) ? make_float3(n.x/len, n.y/len, n.z/len) : make_float3(0,0,1);
}

// Soft Shadows approximation via sphere tracing penumbra
__device__ float evaluate_soft_shadow(float3 ro, float3 rd, int csg_mode, float csg_blend, float mass, float k) {
    float res = 1.0f;
    float t = 0.05f;
    float3 dummy;
    for (int i = 0; i < 20; ++i) {
        float h = map_csg(ro + rd * t, csg_mode, csg_blend, mass, dummy);
        if (h < 0.001f) return 0.0f;
        res = fminf(res, k * h / t);
        t += fmaxf(h, 0.04f);
        if (t > 15.0f) break;
    }
    return fmaxf(0.0f, fminf(res, 1.0f));
}

// SDF Ambient Occlusion estimator
__device__ float evaluate_ao(float3 p, float3 n, int csg_mode, float csg_blend, float mass, float intensity) {
    float occ = 0.0f;
    float sca = 1.0f;
    float3 dummy;
    for (int i = 0; i < 5; ++i) {
        float h = 0.01f + 0.12f * (float)i;
        float d = map_csg(p + n * h, csg_mode, csg_blend, mass, dummy);
        occ += (h - d) * sca;
        sca *= 0.75f;
    }
    return fmaxf(0.0f, 1.0f - intensity * occ);
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
                        float omega = 1.0f / (powf(r_xy, 1.5f) + s_settings.spin);
                        float phi = atan2f(hit_pos.y, hit_pos.x);
                        float phi_rotated = phi - s_settings.time * s_settings.disk_speed * (0.8f / (powf(r_xy, 1.5f) + 0.1f));
                        
                        float u1 = r_xy * 0.12f - s_settings.time * 0.04f;
                        float v1 = phi_rotated * 1.5f / (2.0f * PI);
                        float u2 = r_xy * 0.28f + s_settings.time * 0.08f;
                        float v2 = phi_rotated * 3.0f / (2.0f * PI);
                        
                        float n1 = tex2D<float4>(noiseTex, u1, v1).x;
                        float n2 = tex2D<float4>(noiseTex, u2, v2).y * 0.5f;
                        float noise = (n1 + n2) / 1.5f;
                        
                        float3 color = get_color_from_spectrum(T, s_settings.spectrum_mode) * (0.25f + 0.75f * noise);
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
    
    // Geodesic & CSG integration loop
    bool captured = false;
    bool active = true;
    
    // Spatial Bounding Volume Short-Circuit Acceleration
    if (s_settings.spatial_accel != 0 && s_settings.mass > 0.0f) {
        float r_bound = 25.0f * fmaxf(s_settings.mass, 0.1f);
        float r_pos_orig = sqrtf(ray_pos.x * ray_pos.x + ray_pos.y * ray_pos.y + ray_pos.z * ray_pos.z);
        if (r_pos_orig > r_bound) {
            float v_len = sqrtf(ray_vel.x * ray_vel.x + ray_vel.y * ray_vel.y + ray_vel.z * ray_vel.z);
            if (v_len > 0.0f) {
                float3 dir = make_float3(ray_vel.x / v_len, ray_vel.y / v_len, ray_vel.z / v_len);
                float dot_pos_dir = dot(ray_pos, dir);
                // If ray points away from bounding sphere origin, it will never enter gravity zone
                if (dot_pos_dir > 0.0f) {
                    active = false;
                } else {
                    // Check closest approach distance d_perp^2 = |pos|^2 - (pos . dir)^2
                    float d_perp2 = r_pos_orig * r_pos_orig - dot_pos_dir * dot_pos_dir;
                    if (d_perp2 > r_bound * r_bound) {
                        active = false;
                    }
                }
            }
        }
    }

    for (int step = 0; step < 300; ++step) {
        if (!active || step >= s_settings.max_steps) break;
        thread_steps++;
        
        float r = sqrtf(ray_pos.x*ray_pos.x + ray_pos.y*ray_pos.y + ray_pos.z*ray_pos.z);
        
        if (r <= r_eh + 0.015f) {
            captured = true;
            active = false;
            break;
        }
        
        if (r > 30.0f) {
            active = false;
            break;
        }

        // CSG Surface Raymarching Check
        if (s_settings.csg_mode != 0 && s_settings.mass > 0.0f) {
            float3 obj_color;
            float d_csg = map_csg(ray_pos, s_settings.csg_mode, s_settings.csg_blend, s_settings.mass, obj_color);
            if (d_csg < 0.01f) {
                float3 norm = calc_normal(ray_pos, s_settings.csg_mode, s_settings.csg_blend, s_settings.mass);
                float3 light_dir = make_float3(0.577f, 0.577f, 0.577f);
                float diff = fmaxf(0.15f, dot(norm, light_dir));
                
                float shadow = 1.0f;
                if (s_settings.soft_shadows) {
                    shadow = evaluate_soft_shadow(ray_pos + norm * 0.02f, light_dir, s_settings.csg_mode, s_settings.csg_blend, s_settings.mass, s_settings.shadow_k);
                }
                
                float ao = 1.0f;
                if (s_settings.ao_enabled) {
                    ao = evaluate_ao(ray_pos, norm, s_settings.csg_mode, s_settings.csg_blend, s_settings.mass, s_settings.ao_intensity);
                }
                
                float3 surf_color = obj_color * (diff * shadow * ao);
                accum_color.x += accum_trans * surf_color.x;
                accum_color.y += accum_trans * surf_color.y;
                accum_color.z += accum_trans * surf_color.z;
                accum_trans *= 0.10f;
                active = false;
                break;
            }
        }
        
        float r_dist = fmaxf(0.0f, r - r_eh);
        float h_base = 0.05f + 0.35f * fminf(r_dist / 4.0f, 1.0f);
        if (r > 6.0f * fmaxf(s_settings.mass, 0.1f)) {
            float extra_dist = r - 6.0f * fmaxf(s_settings.mass, 0.1f);
            h_base += 0.08f * extra_dist * extra_dist;
        }
        float h_step = fminf(h_base, 1.25f);
        float3 prev_pos = ray_pos;
        
        rk4_step(ray_pos, ray_vel, h_step, s_settings.mass, s_settings.spin, s_settings);
        
        // Photon sphere boundary visualization crossing
        if (s_settings.show_photon_sphere != 0 && s_settings.mass > 0.0f) {
            float r_prev = sqrtf(prev_pos.x*prev_pos.x + prev_pos.y*prev_pos.y + prev_pos.z*prev_pos.z);
            float r_curr = r;
            float r_ps = 3.0f * s_settings.mass;
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
                float omega = 1.0f / (powf(r_orbit, 1.5f) + s_settings.spin);
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
                    cross_pos, ray_vel, h_step,
                    s_settings,
                    noiseTex,
                    accum_color, accum_trans
                );
            }
        }
        
        if (accum_trans < 0.01f) {
            accum_trans = 0.0f;
            active = false;
            break;
        }

        // 2. CUDA Warp-level optimization: Early warp exit check
        unsigned int active_mask = __activemask();
        if (!__any_sync(active_mask, active)) {
            break;
        }
    }
    
    // 3. CUDA Warp Reduction for performance metrics step counter
    if (d_step_counter != nullptr) {
        unsigned int active_mask = __activemask();
        unsigned int warp_steps = thread_steps;
        for (int offset = 16; offset > 0; offset /= 2) {
            warp_steps += __shfl_down_sync(active_mask, warp_steps, offset);
        }
        int lane = (threadIdx.x + threadIdx.y * blockDim.x) % 32;
        if (lane == 0 && warp_steps > 0) {
            atomicAdd(d_step_counter, (unsigned long long)warp_steps);
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

// Kernel launcher with dynamic CUDA thread block tuning
extern "C" void run_raymarch_kernel(
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
    cudaDeviceSynchronize();
}

// Temporal Anti-Aliasing (TAA) reprojection blending kernel
__global__ void taa_blend_kernel(float4* current, float4* history, int width, int height, float alpha) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    float4 currColor = current[idx];
    float4 histColor = history[idx];

    float4 blended;
    blended.x = histColor.x * (1.0f - alpha) + currColor.x * alpha;
    blended.y = histColor.y * (1.0f - alpha) + currColor.y * alpha;
    blended.z = histColor.z * (1.0f - alpha) + currColor.z * alpha;
    blended.w = 1.0f;

    current[idx] = blended;
    history[idx] = blended;
}

extern "C" void apply_temporal_reprojection(
    float4 *d_current,
    float4 *d_history,
    int width,
    int height,
    float blend_alpha
) {
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    taa_blend_kernel<<<grid, block>>>(d_current, d_history, width, height, blend_alpha);
    cudaDeviceSynchronize();
}
