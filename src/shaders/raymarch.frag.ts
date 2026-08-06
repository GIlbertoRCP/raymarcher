export const fragmentShaderSource = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

// Uniforms
uniform vec2 u_resolution;
uniform float u_time;
uniform float u_mass;
uniform float u_spin;
uniform bool u_relativity;
uniform bool u_disk;
uniform bool u_grid;
uniform bool u_stars;
uniform float u_disk_temp;
uniform int u_max_steps;
uniform sampler2D u_noise_tex;
uniform float u_disk_thickness;
uniform float u_disk_density;
uniform float u_bloom_intensity;
uniform float u_disk_speed;
uniform bool u_show_photon_sphere;
uniform bool u_shift_visualizer;
uniform bool u_split_active;
uniform float u_split_x;
uniform int u_spectrum_mode;

// CSG & Lighting Uniforms
uniform int u_csg_mode;
uniform float u_csg_blend;
uniform bool u_soft_shadows;
uniform float u_shadow_k;
uniform bool u_ao_enabled;
uniform float u_ao_intensity;

// Camera Uniforms
uniform vec3 u_cam_pos;
uniform vec3 u_cam_dir;
uniform vec3 u_cam_up;
uniform vec3 u_cam_right;
uniform float u_fov_scale;

const float PI = 3.14159265359;

// Pseudo-random hash for stars
float hash3(vec3 p) {
  p = fract(p * vec3(443.8975, 397.2973, 491.1871));
  p += vec3(dot(p.xyz, p.yzx + 19.19));
  return fract(p.x * p.y * p.z);
}

// Procedural starfield
vec3 get_starfield(vec3 dir) {
  vec3 star_grid = dir * 140.0;
  vec3 grid_id = floor(star_grid);
  float n = hash3(grid_id);
  
  vec3 color = vec3(0.0);
  if (n > 0.985) {
    float star_intensity = pow(hash3(grid_id * 4.321), 6.0) * 1.5;
    // Add red/blue tint based on noise
    float tint = hash3(grid_id * 8.765);
    vec3 star_color = vec3(0.9, 0.95, 1.0);
    if (tint < 0.2) star_color = vec3(1.0, 0.6, 0.5); // Red giant
    else if (tint > 0.8) star_color = vec3(0.5, 0.8, 1.0); // Blue giant
    
    color = star_color * star_intensity;
  }
  return color;
}

// Calibration grid background
vec3 get_grid(vec3 dir) {
  float theta = acos(clamp(dir.z, -1.0, 1.0));
  float phi = atan(dir.y, dir.x);
  
  float grid_theta = step(0.03, sin(theta * 18.0));
  float grid_phi = step(0.03, sin(phi * 36.0));
  float grid = grid_theta * grid_phi;
  
  vec3 base = vec3(0.02, 0.03, 0.08);
  vec3 lines = vec3(0.0, 0.7, 1.0) * 0.8;
  return mix(lines, base, grid);
}

// Fits blackbody temperature to RGB colors
vec3 blackbody(float Temp) {
  Temp = max(Temp, 0.0);
  float t = Temp / 1000.0;
  float r = 1.0;
  float g = 1.0;
  float b = 1.0;
  
  if (t < 6.6) {
    r = 1.0;
    g = clamp(-3.2 + 2.0 * log(t + 0.6) + 0.1 * t, 0.0, 1.0);
    b = clamp(-10.0 + 4.0 * log(t + 0.1), 0.0, 1.0);
  } else {
    r = clamp(3.0 * pow(t - 0.5, -0.1), 0.0, 1.0);
    g = clamp(2.8 * pow(t - 0.5, -0.07), 0.0, 1.0);
    b = 1.0;
  }
  
  // Boost red slightly for cooler glowing dust
  if (Temp < 2500.0) {
    float factor = (2500.0 - Temp) / 1500.0;
    r = mix(r, 1.0, factor);
    g = mix(g, 0.2, factor);
    b = mix(b, 0.0, factor);
  }
  
  return vec3(r, g, b);
}

// Maps temperature and spectrum mode to custom visual profiles
vec3 get_color_from_spectrum(float Temp, int mode) {
  if (mode == 0) {
    return blackbody(Temp);
  }
  float t = clamp(Temp / 12000.0, 0.0, 1.0);
  if (mode == 1) {
    // Nebula Glow: violet-blue to magenta-pink
    return mix(vec3(0.18, 0.0, 0.5), vec3(1.0, 0.0, 0.55), t) * 2.2;
  }
  if (mode == 2) {
    // Plasma Fire: dark red to bright yellow-orange to white
    return mix(vec3(0.55, 0.05, 0.0), vec3(1.0, 0.88, 0.15), t) * 2.2;
  }
  // Quantum Shift: cyan-blue to neon green
  return mix(vec3(0.0, 0.35, 0.8), vec3(0.0, 1.0, 0.35), t) * 2.2;
}

// Relativistic equations of motion in isotropic coordinates
vec3 compute_force(vec3 x, vec3 v) {
  float r = max(length(x), u_mass * 0.5 + 0.001);
  float u = min(u_mass / (2.0 * r), 0.999);
  float one_plus_u = 1.0 + u;
  float one_minus_u = 1.0 - u;
  
  // Radial general relativistic gravitational pull
  float num = -u_mass * pow(one_plus_u, 5.0) * (2.0 - u);
  float den = pow(r, 3.0) * pow(one_minus_u, 3.0);
  vec3 f_grav = (num / den) * x;
  
  // Kerr frame dragging Coriolis force
  if (u_spin > 0.0) {
    float j = u_mass * u_spin; // Angular momentum J = M * a
    float omega = 2.0 * j / (pow(r, 3.0) * pow(one_plus_u, 6.0));
    vec3 omega_vec = vec3(0.0, 0.0, omega);
    vec3 f_fd = 2.0 * cross(v, omega_vec);
    return f_grav + f_fd;
  }
  
  return f_grav;
}

// Single step of Runge-Kutta 4th order integration
void rk4_step(inout vec3 x, inout vec3 v, float h) {
  vec3 x1 = x;
  vec3 v1 = v;
  vec3 a1 = compute_force(x1, v1);
  
  vec3 x2 = x + 0.5 * h * v1;
  vec3 v2 = v + 0.5 * h * a1;
  vec3 a2 = compute_force(x2, v2);
  
  vec3 x3 = x + 0.5 * h * v2;
  vec3 v3 = v + 0.5 * h * a2;
  vec3 a3 = compute_force(x3, v3);
  
  vec3 x4 = x + h * v3;
  vec3 v4 = v + h * a3;
  vec3 a4 = compute_force(x4, v4);
  
  x += (h / 6.0) * (v1 + 2.0 * v2 + 2.0 * v3 + v4);
  v += (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
  
  // Relativistic velocity normalization (Light speed constraint projection)
  float r = max(length(x), u_mass * 0.5 + 0.001);
  float u = min(u_mass / (2.0 * r), 0.999);
  float n_ref = pow(1.0 + u, 3.0) / (1.0 - u);
  float target_v_len = 1.0 / n_ref;
  v = normalize(v) * target_v_len;
}

// Samples procedural accretion disk density and emission
void sample_accretion_disk(vec3 x, vec3 v, float h, inout vec3 accum_color, inout float accum_trans) {
  float r_xy = length(x.xy);
  
  // Accretion disk radial bounds (e.g. from inner boundary near horizon to outer edge)
  float r_in = 3.2 * u_mass;
  float r_out = 9.5 * u_mass;
  
  if (r_xy >= r_in && r_xy <= r_out) {
    float disk_thickness = u_disk_thickness * u_mass;
    
    // Vertical profile (Gaussian)
    float vertical_density = exp(-pow(x.z, 2.0) / (2.0 * pow(disk_thickness, 2.0)));
    
    // Radial density profile (falls off as 1/r^2.5 outside the inner boundary)
    float radial_density = pow(r_in / r_xy, 2.5);
    
    // Base density
    float density = u_disk_density * vertical_density * radial_density;
    
    if (density > 0.001) {
      // Keplerian velocity rotation: angular speed omega
      // For Kerr black holes: omega = 1 / (r^1.5 + a)
      float omega = 1.0 / (pow(r_xy, 1.5) + u_spin);
      
      // Calculate rotation angle based on time and radius
      float phi = atan(x.y, x.x);
      float phi_rotated = phi - u_time * u_disk_speed * (0.8 / (pow(r_xy, 1.5) + 0.1));
      
      // Generate noise UV coordinates for gas structures with vertical shear
      vec2 uv_noise1 = vec2(r_xy * 0.12 - u_time * 0.04, phi_rotated * 1.5 / (2.0 * PI) + x.z * 0.15);
      vec2 uv_noise2 = vec2(r_xy * 0.28 + u_time * 0.08, phi_rotated * 3.0 / (2.0 * PI) - x.z * 0.15);
      
      float n1 = texture(u_noise_tex, uv_noise1).r;
      float n2 = texture(u_noise_tex, uv_noise2).g * 0.5;
      float noise = (n1 + n2) / 1.5;
      
      // Modulate density with noise to get glowing dust clouds
      density *= (0.25 + 0.75 * noise);
      
      // Accretion disk temperature drops radially
      float T = u_disk_temp * pow(r_in / r_xy, 0.75);
      
      // Gas velocity vector in Cartesian coordinates
      vec3 v_gas = omega * vec3(-x.y, x.x, 0.0);
      
      // Limit speed to 0.58 c near event horizon to prevent superluminal coordinate artifacts
      float gas_speed = length(v_gas);
      if (gas_speed > 0.58) {
        v_gas = (v_gas / gas_speed) * 0.58;
      }
      
      // Doppler factor calculation: g = sqrt(1-beta^2) / (1-beta)
      vec3 photon_dir = normalize(v);
      float beta = dot(v_gas, photon_dir);
      float g_doppler = sqrt(1.0 - beta * beta) / (1.0 - beta);
      
      // Gravitational Redshift factor: g_grav = (1 - M/2r) / (1 + M/2r)
      float r = max(length(x), u_mass * 0.5 + 0.001);
      float u = min(u_mass / (2.0 * r), 0.999);
      float g_grav = (1.0 - u) / (1.0 + u);
      
      // Total frequency shift factor
      float g_total = g_doppler * g_grav;
      
      // Doppler shift the temperature and apply beaming (scaling by g^3.8)
      float T_obs = T * g_total;
      vec3 emission_color = get_color_from_spectrum(T_obs, u_spectrum_mode);
      if (u_shift_visualizer) {
        if (g_total > 1.0) {
          float t = clamp((g_total - 1.0) * 1.5, 0.0, 1.0);
          emission_color = mix(vec3(0.1, 0.7, 0.1), vec3(0.0, 0.5, 1.0), t) * 2.0;
        } else {
          float t = clamp((1.0 - g_total) * 1.2, 0.0, 1.0);
          emission_color = mix(vec3(0.1, 0.7, 0.1), vec3(1.0, 0.1, 0.0), t) * 2.0;
        }
      }
      
      // Apply absorption and emission along the ray step
      float dl = h;
      float opacity = density * 2.5 * dl;
      float trans = exp(-opacity);
      
      vec3 emission = emission_color * (density * pow(g_total, 3.8) * 1.8 * dl);
      accum_color += accum_trans * emission;
      accum_trans *= trans;
    }
  }
}

vec3 ACESFilm(vec3 x) {
  float a = 2.51;
  float b = 0.03;
  float c = 2.43;
  float d = 0.59;
  float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
 
void main() {
  // Reconstruct camera ray from viewport coords
  vec2 p = (v_uv * 2.0 - 1.0);
  p.x *= (u_resolution.x / max(u_resolution.y, 1.0));
  
  vec3 ray_pos = u_cam_pos;
  vec3 ray_dir = normalize(u_cam_dir + p.x * u_fov_scale * u_cam_right + p.y * u_fov_scale * u_cam_up);
  vec3 ray_vel = ray_dir;
  
  // Set initial velocities to speed of light in isotropic coordinates at camera distance
  float r_cam = max(length(ray_pos), u_mass * 0.5 + 0.001);
  float u_cam = min(u_mass / (2.0 * r_cam), 0.999);
  float n_cam = pow(1.0 + u_cam, 3.0) / (1.0 - u_cam);
  ray_vel = ray_dir / n_cam;
  
  // Volumetric accumulation states
  vec3 accum_color = vec3(0.0);
  float accum_trans = 1.0;
  
  // Define Event Horizon radius in isotropic coordinates (r_eh = M/2)
  float r_eh = u_mass * 0.5;
  
  // Relativistic vs standard raymarching
  bool render_newtonian = u_split_active && (v_uv.x > u_split_x);
  if (!u_relativity || u_mass == 0.0 || render_newtonian) {
    // Standard flat-space raytracing
    // Simple intersection check with accretion disk plane (z = 0)
    if (u_disk) {
      if (ray_vel.z != 0.0) {
        float t = -ray_pos.z / ray_vel.z;
        if (t > 0.0) {
          vec3 hit_pos = ray_pos + t * ray_vel;
          float r_xy = length(hit_pos.xy);
          float r_in = 3.2 * u_mass;
          float r_out = 9.5 * u_mass;
          if (r_xy >= r_in && r_xy <= r_out && u_mass > 0.0) {
            float T = u_disk_temp * pow(r_in / r_xy, 0.75);
            
            // Keplerian velocity rotation: angular speed omega
            float omega = 1.0 / (pow(r_xy, 1.5) + u_spin);
            float phi = atan(hit_pos.y, hit_pos.x);
            float phi_rotated = phi - u_time * u_disk_speed * (0.8 / (pow(r_xy, 1.5) + 0.1));
            
            vec2 uv_noise1 = vec2(r_xy * 0.12 - u_time * 0.04, phi_rotated * 1.5 / (2.0 * PI));
            vec2 uv_noise2 = vec2(r_xy * 0.28 + u_time * 0.08, phi_rotated * 3.0 / (2.0 * PI));
            float n1 = texture(u_noise_tex, uv_noise1).r;
            float n2 = texture(u_noise_tex, uv_noise2).g * 0.5;
            float noise = (n1 + n2) / 1.5;
            
            vec3 color = get_color_from_spectrum(T, u_spectrum_mode) * (0.25 + 0.75 * noise);
            if (u_shift_visualizer) {
              color = vec3(0.1, 0.7, 0.1) * (0.25 + 0.75 * noise) * 2.0;
            }
            accum_color = color;
            accum_trans = 0.15; // semi-transparent
          }
        }
      }
    }
    
    // Background starfield/grid
    vec3 bg_color = vec3(0.0);
    if (u_grid) {
      bg_color = get_grid(ray_dir);
    } else if (u_stars) {
      bg_color = get_starfield(ray_dir);
    }
    
    accum_color += accum_trans * bg_color;
    fragColor = vec4(accum_color, 1.0);
    return;
  }
  
  // Geodesic integration loop
  float total_lambda = 0.0;
  bool captured = false;
  
  for (int step = 0; step < 300; ++step) {
    if (step >= u_max_steps) break;
    float r = length(ray_pos);
    
    // Horizon collision check: r <= M/2 + delta
    if (r <= r_eh + 0.015) {
      captured = true;
      break;
    }
    
    // Escape condition: ray has traveled far from the black hole
    if (r > 30.0) {
      break;
    }
    
    // Adaptive step sizing to ensure precision near event horizon and performance far away
    float h = 0.18 * mix(0.05, 1.0, clamp((r - r_eh) / 5.0, 0.0, 1.0));
    
    // Store current position to check for disk crossing
    vec3 prev_pos = ray_pos;
    
    // Perform RK4 step
    rk4_step(ray_pos, ray_vel, h);

    // CSG Surface Raymarching Check
    if (u_csg_mode != 0 && u_mass > 0.0) {
      vec3 obj_color = vec3(0.0);
      float eps = 0.002;
      float d_main = length(ray_pos) - 1.8 * u_mass;
      obj_color = vec3(0.2, 0.6, 1.0);
      
      float d_csg = d_main;
      vec3 p_torus = vec3(ray_pos.x, ray_pos.y, ray_pos.z - 0.4 * u_mass);
      float q_x = length(p_torus.xy) - 3.5 * u_mass;
      float d_torus = length(vec2(q_x, p_torus.z)) - 0.4 * u_mass;
      float d_sph = length(vec3(ray_pos.x - 3.2 * u_mass, ray_pos.y, ray_pos.z)) - 0.9 * u_mass;
      float d_sub = min(d_torus, d_sph);

      if (u_csg_mode == 1) {
        d_csg = min(d_main, d_sub);
        if (d_sub < d_main) obj_color = vec3(1.0, 0.4, 0.2);
      } else if (u_csg_mode == 2) {
        float k = u_csg_blend;
        float h_val = max(k - abs(d_main - d_sub), 0.0) / (k + 1e-4);
        d_csg = min(d_main, d_sub) - h_val * h_val * k * 0.25;
        float t_val = clamp((d_main - d_sub) / (k + 1e-4) + 0.5, 0.0, 1.0);
        obj_color = mix(vec3(1.0, 0.4, 0.2), vec3(0.2, 0.6, 1.0), t_val);
      } else if (u_csg_mode == 3) {
        float k = u_csg_blend;
        float h_val = max(k - abs(d_main - (-d_sub)), 0.0) / (k + 1e-4);
        d_csg = max(d_main, -d_sub) + h_val * h_val * k * 0.25;
        if (-d_sub > d_main) obj_color = vec3(0.9, 0.1, 0.3);
      } else if (u_csg_mode == 4) {
        float k = u_csg_blend;
        float h_val = max(k - abs(d_main - d_sub), 0.0) / (k + 1e-4);
        d_csg = max(d_main, d_sub) + h_val * h_val * k * 0.25;
        obj_color = vec3(0.8, 0.2, 0.9);
      }

      if (d_csg < 0.01) {
        vec3 light_dir = vec3(0.577, 0.577, 0.577);
        vec3 norm = vec3(
          length(ray_pos + vec3(eps, 0.0, 0.0)) - length(ray_pos),
          length(ray_pos + vec3(0.0, eps, 0.0)) - length(ray_pos),
          length(ray_pos + vec3(0.0, 0.0, eps)) - length(ray_pos)
        );
        if (length(norm) > 0.0001) norm = normalize(norm); else norm = vec3(0,0,1);
        float diff = max(0.15, dot(norm, light_dir));
        
        float shadow = 1.0;
        if (u_soft_shadows) {
          float st = 0.05;
          for (int sh = 0; sh < 15; sh++) {
            float sh_d = length(ray_pos + norm * 0.02 + light_dir * st) - 1.8 * u_mass;
            if (sh_d < 0.001) { shadow = 0.0; break; }
            shadow = min(shadow, u_shadow_k * sh_d / st);
            st += max(sh_d, 0.04);
            if (st > 10.0) break;
          }
          shadow = clamp(shadow, 0.0, 1.0);
        }

        float ao = 1.0;
        if (u_ao_enabled) {
          float occ = 0.0; float sca = 1.0;
          for (int aoi = 0; aoi < 5; aoi++) {
            float aoh = 0.01 + 0.12 * float(aoi);
            float aod = length(ray_pos + norm * aoh) - 1.8 * u_mass;
            occ += (aoh - aod) * sca;
            sca *= 0.75;
          }
          ao = clamp(1.0 - u_ao_intensity * occ, 0.0, 1.0);
        }

        vec3 surf_color = obj_color * (diff * shadow * ao);
        accum_color += accum_trans * surf_color;
        accum_trans *= 0.10;
        break;
      }
    }
    
    // Photon sphere boundary visualization crossing
    float r_prev = length(prev_pos);
    float r_curr = length(ray_pos);
    float r_ps = 3.0 * u_mass;
    if (u_show_photon_sphere && u_mass > 0.0) {
      if ((r_prev > r_ps && r_curr <= r_ps) || (r_prev < r_ps && r_curr >= r_ps)) {
        accum_color += accum_trans * vec3(0.0, 1.0, 0.35) * 0.22;
      }
    }
    
    // Orbiting Hotspots (3 glowing test particles tracing Keplerian orbits)
    if (u_disk && u_mass > 0.0) {
      for (int i = 0; i < 3; ++i) {
        float r_orbit = 4.2 + float(i) * 2.0;
        float omega = 1.0 / (pow(r_orbit, 1.5) + u_spin);
        float phi = u_time * omega * u_disk_speed + float(i) * 2.0944; // 2*PI/3 spacing
        vec3 p_orbit = vec3(cos(phi) * r_orbit, sin(phi) * r_orbit, 0.0);
        
        float dist = length(ray_pos - p_orbit);
        if (dist < 0.35) {
          // Relativistic Doppler beaming on the hotspot particle
          vec3 v_gas = omega * vec3(-p_orbit.y, p_orbit.x, 0.0);
          float gas_speed = length(v_gas);
          if (gas_speed > 0.58) {
            v_gas = (v_gas / gas_speed) * 0.58;
          }
          vec3 photon_dir = normalize(ray_vel);
          float beta = dot(v_gas, photon_dir);
          float g_doppler = sqrt(1.0 - beta * beta) / (1.0 - beta);
          
          float g_grav = (1.0 - u_mass / (2.0 * r_orbit)) / (1.0 + u_mass / (2.0 * r_orbit));
          float g_total = g_doppler * g_grav;
          
          vec3 p_color = vec3(1.0, 0.45, 0.0);
          if (i == 0) { p_color = vec3(1.0, 0.2, 0.1); }
          else if (i == 1) { p_color = vec3(1.0, 0.7, 0.1); }
          else { p_color = vec3(0.1, 0.8, 1.0); }
          
          if (u_shift_visualizer) {
            if (g_total > 1.0) {
              float t = clamp((g_total - 1.0) * 1.5, 0.0, 1.0);
              p_color = mix(vec3(0.1, 0.7, 0.1), vec3(0.0, 0.5, 1.0), t) * 2.0;
            } else {
              float t = clamp((1.0 - g_total) * 1.2, 0.0, 1.0);
              p_color = mix(vec3(0.1, 0.7, 0.1), vec3(1.0, 0.1, 0.0), t) * 2.0;
            }
          } else {
            p_color = p_color * pow(g_total, 3.8);
          }
          
          float density_p = exp(-pow(dist / 0.18, 2.0));
          accum_color += accum_trans * p_color * density_p * 0.45;
        }
      }
    }
    
    // Integrate accretion disk volume continuously along the ray (3D Volumetric)
    if (u_disk) {
      sample_accretion_disk(ray_pos, ray_vel, h, accum_color, accum_trans);
    }
    
    // If the ray has become almost completely absorbed, terminate early
    if (accum_trans < 0.01) {
      accum_trans = 0.0;
      break;
    }
  }
  
  vec3 final_color = vec3(0.0);
  if (!captured) {
    // Sample lensed background
    vec3 escape_dir = normalize(ray_vel);
    
    if (u_grid) {
      final_color = get_grid(escape_dir);
    } else if (u_stars) {
      final_color = get_starfield(escape_dir);
    }
  }
  
  // Combine lensed background with accumulated accretion disk glow
  vec3 color_out = accum_color + accum_trans * final_color;
  
  // Draw divider line
  if (u_split_active) {
    float dist_to_line = abs(v_uv.x - u_split_x);
    if (dist_to_line < 0.002) {
      color_out = mix(vec3(0.0, 0.9, 1.0), color_out, dist_to_line / 0.002);
    }
  }
  
  fragColor = vec4(color_out, 1.0);
}
`;
