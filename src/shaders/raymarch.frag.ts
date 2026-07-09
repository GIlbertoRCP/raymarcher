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
  p += dot(p.xyz, p.yzx + 19.19);
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

// Relativistic equations of motion in isotropic coordinates
vec3 compute_force(vec3 x, vec3 v) {
  float r = length(x);
  float u = u_mass / (2.0 * r);
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
  float r = length(x);
  float u = u_mass / (2.0 * r);
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
    float disk_thickness = 0.08 * u_mass;
    
    // Vertical profile (Gaussian)
    float vertical_density = exp(-pow(x.z, 2.0) / (2.0 * pow(disk_thickness, 2.0)));
    
    // Radial density profile (falls off as 1/r^2.5 outside the inner boundary)
    float radial_density = pow(r_in / r_xy, 2.5);
    
    // Base density
    float density = 1.8 * vertical_density * radial_density;
    
    if (density > 0.001) {
      // Keplerian velocity rotation: angular speed omega
      // For Kerr black holes: omega = 1 / (r^1.5 + a)
      float omega = 1.0 / (pow(r_xy, 1.5) + u_spin);
      
      // Calculate rotation angle based on time and radius
      float phi = atan(x.y, x.x);
      float phi_rotated = phi - u_time * (0.8 / (pow(r_xy, 1.5) + 0.1));
      
      // Generate noise UV coordinates for gas structures
      vec2 uv_noise1 = vec2(r_xy * 0.12 - u_time * 0.04, phi_rotated * 1.5 / (2.0 * PI));
      vec2 uv_noise2 = vec2(r_xy * 0.28 + u_time * 0.08, phi_rotated * 3.0 / (2.0 * PI));
      
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
      float r = length(x);
      float u = u_mass / (2.0 * r);
      float g_grav = (1.0 - u) / (1.0 + u);
      
      // Total frequency shift factor
      float g_total = g_doppler * g_grav;
      
      // Doppler shift the temperature and apply beaming (scaling by g^3.8)
      float T_obs = T * g_total;
      vec3 emission_color = blackbody(T_obs);
      
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

void main() {
  // Reconstruct camera ray from viewport coords
  vec2 p = (v_uv * 2.0 - 1.0);
  p.x *= (u_resolution.x / u_resolution.y);
  
  vec3 ray_pos = u_cam_pos;
  vec3 ray_dir = normalize(u_cam_dir + p.x * u_fov_scale * u_cam_right + p.y * u_fov_scale * u_cam_up);
  vec3 ray_vel = ray_dir;
  
  // Set initial velocities to speed of light in isotropic coordinates at camera distance
  float r_cam = length(ray_pos);
  float u_cam = u_mass / (2.0 * r_cam);
  float n_cam = pow(1.0 + u_cam, 3.0) / (1.0 - u_cam);
  ray_vel = ray_dir / n_cam;
  
  // Volumetric accumulation states
  vec3 accum_color = vec3(0.0);
  float accum_trans = 1.0;
  
  // Define Event Horizon radius in isotropic coordinates (r_eh = M/2)
  float r_eh = u_mass * 0.5;
  
  // Relativistic vs standard raymarching
  if (!u_relativity || u_mass == 0.0) {
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
            float phi_rotated = phi - u_time * (0.8 / (pow(r_xy, 1.5) + 0.1));
            
            vec2 uv_noise1 = vec2(r_xy * 0.12 - u_time * 0.04, phi_rotated * 1.5 / (2.0 * PI));
            vec2 uv_noise2 = vec2(r_xy * 0.28 + u_time * 0.08, phi_rotated * 3.0 / (2.0 * PI));
            float n1 = texture(u_noise_tex, uv_noise1).r;
            float n2 = texture(u_noise_tex, uv_noise2).g * 0.5;
            float noise = (n1 + n2) / 1.5;
            
            vec3 color = blackbody(T) * (0.25 + 0.75 * noise);
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
  
  for (int step = 0; step < u_max_steps; ++step) {
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
    
    // Integrate accretion disk volume
    if (u_disk) {
      // Check if we crossed the z=0 plane or are currently within the disk volume
      float sign_prev = prev_pos.z;
      float sign_curr = ray_pos.z;
      
      // If we crossed the plane or are very close, sample the disk
      if (sign_prev * sign_curr <= 0.0 || abs(ray_pos.z) < 0.18 * u_mass) {
        // Interpolate position to get exact disk crossing coordinate
        float fraction = abs(prev_pos.z) / (abs(prev_pos.z) + abs(ray_pos.z));
        vec3 cross_pos = mix(prev_pos, ray_pos, fraction);
        
        sample_accretion_disk(cross_pos, ray_vel, h, accum_color, accum_trans);
      }
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
  
  // HDR tone mapping & Gamma correction (makes the glow look extremely rich)
  color_out = color_out / (color_out + vec3(1.0)); // simple Reinhard tone mapping
  color_out = pow(color_out, vec3(1.0 / 2.2));     // Gamma correction
  
  fragColor = vec4(color_out, 1.0);
}
`;
