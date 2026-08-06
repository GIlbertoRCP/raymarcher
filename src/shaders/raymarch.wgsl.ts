export const raymarchWGSL = `
struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOutput {
  var out: VertexOutput;
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  out.position = vec4<f32>(pos[vertex_idx], 0.0, 1.0);
  out.uv = pos[vertex_idx] * 0.5 + 0.5;
  return out;
}

struct Uniforms {
  u_resolution: vec2<f32>,
  u_time: f32,
  u_mass: f32,
  u_spin: f32,
  u_relativity: u32,
  u_disk: u32,
  u_grid: u32,
  u_stars: u32,
  u_disk_temp: f32,
  u_max_steps: u32,
  u_fov_scale: f32,
  u_disk_thickness: f32,
  u_disk_density: f32,
  u_bloom_intensity: f32,
  u_disk_speed: f32,
  u_show_photon_sphere: u32,
  u_shift_visualizer: u32,
  u_split_active: u32,
  u_split_x: f32,
  u_cam_pos: vec3<f32>,
  u_spectrum_mode: f32,
  u_cam_dir: vec3<f32>,
  u_csg_mode: u32,
  u_cam_up: vec3<f32>,
  u_csg_blend: f32,
  u_cam_right: vec3<f32>,
  u_soft_shadows: u32,
  u_shadow_k: f32,
  u_ao_enabled: u32,
  u_ao_intensity: f32,
  u_padding: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var u_noise_tex: texture_2d<f32>;
@group(0) @binding(2) var u_noise_sampler: sampler;
@group(0) @binding(3) var out_texture: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265359;

fn custom_step(edge: f32, x: f32) -> f32 {
  if (x < edge) { return 0.0; }
  return 1.0;
}

fn hash3(p_in: vec3<f32>) -> f32 {
  var p = fract(p_in * vec3<f32>(443.8975, 397.2973, 491.1871));
  p += vec3<f32>(dot(p.xyz, p.yzx + vec3<f32>(19.19)));
  return fract(p.x * p.y * p.z);
}

fn get_starfield(dir: vec3<f32>) -> vec3<f32> {
  let star_grid = dir * 140.0;
  let grid_id = floor(star_grid);
  let n = hash3(grid_id);
  
  var color = vec3<f32>(0.0);
  if (n > 0.985) {
    let star_intensity = pow(hash3(grid_id * 4.321), 6.0) * 1.5;
    let tint = hash3(grid_id * 8.765);
    var star_color = vec3<f32>(0.9, 0.95, 1.0);
    if (tint < 0.2) {
      star_color = vec3<f32>(1.0, 0.6, 0.5);
    } else if (tint > 0.8) {
      star_color = vec3<f32>(0.5, 0.8, 1.0);
    }
    color = star_color * star_intensity;
  }
  return color;
}

fn get_grid(dir: vec3<f32>) -> vec3<f32> {
  let theta = acos(clamp(dir.z, -1.0, 1.0));
  let phi = atan2(dir.y, dir.x);
  
  let grid_theta = custom_step(0.03, sin(theta * 18.0));
  let grid_phi = custom_step(0.03, sin(phi * 36.0));
  let grid = grid_theta * grid_phi;
  
  let base = vec3<f32>(0.02, 0.03, 0.08);
  let lines = vec3<f32>(0.0, 0.7, 1.0) * 0.8;
  return mix(lines, base, grid);
}

fn blackbody(Temp_in: f32) -> vec3<f32> {
  let Temp = max(Temp_in, 0.0);
  let t = Temp / 1000.0;
  var r = 1.0; var g = 1.0; var b = 1.0;
  
  if (t < 6.6) {
    r = 1.0;
    g = clamp(-3.2 + 2.0 * log(t + 0.6) + 0.1 * t, 0.0, 1.0);
    b = clamp(-10.0 + 4.0 * log(t + 0.1), 0.0, 1.0);
  } else {
    r = clamp(3.0 * pow(t - 0.5, -0.1), 0.0, 1.0);
    g = clamp(2.8 * pow(t - 0.5, -0.07), 0.0, 1.0);
    b = 1.0;
  }
  
  if (Temp < 2500.0) {
    let factor = (2500.0 - Temp) / 1500.0;
    r = mix(r, 1.0, factor);
    g = mix(g, 0.2, factor);
    b = mix(b, 0.0, factor);
  }
  return vec3<f32>(r, g, b);
}

fn get_color_from_spectrum(Temp: f32, mode: u32) -> vec3<f32> {
  if (mode == 0u) { return blackbody(Temp); }
  let t = clamp(Temp / 12000.0, 0.0, 1.0);
  if (mode == 1u) { return mix(vec3<f32>(0.18, 0.0, 0.5), vec3<f32>(1.0, 0.0, 0.55), t) * 2.2; }
  if (mode == 2u) { return mix(vec3<f32>(0.55, 0.05, 0.0), vec3<f32>(1.0, 0.88, 0.15), t) * 2.2; }
  return mix(vec3<f32>(0.0, 0.35, 0.8), vec3<f32>(0.0, 1.0, 0.35), t) * 2.2;
}

fn compute_force(x: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
  let r = max(length(x), uniforms.u_mass * 0.5 + 0.001);
  let u = min(uniforms.u_mass / (2.0 * r), 0.999);
  let one_plus_u = 1.0 + u;
  let one_minus_u = 1.0 - u;
  
  let num = -uniforms.u_mass * pow(one_plus_u, 5.0) * (2.0 - u);
  let den = pow(r, 3.0) * pow(one_minus_u, 3.0);
  let f_grav = (num / den) * x;
  
  if (uniforms.u_spin > 0.0) {
    let j = uniforms.u_mass * uniforms.u_spin;
    let omega = 2.0 * j / (pow(r, 3.0) * pow(one_plus_u, 6.0));
    let omega_vec = vec3<f32>(0.0, 0.0, omega);
    let f_fd = 2.0 * cross(v, omega_vec);
    return f_grav + f_fd;
  }
  return f_grav;
}

fn rk4_step(px: ptr<function, vec3<f32>>, pv: ptr<function, vec3<f32>>, h: f32) {
  let x = *px; let v = *pv;
  let a1 = compute_force(x, v);
  
  let x2 = x + 0.5 * h * v;
  let v2 = v + 0.5 * h * a1;
  let a2 = compute_force(x2, v2);
  
  let x3 = x + 0.5 * h * v2;
  let v3 = v + 0.5 * h * a2;
  let a3 = compute_force(x3, v3);
  
  let x4 = x + h * v3;
  let v4 = v + h * a3;
  let a4 = compute_force(x4, v4);
  
  let new_x = x + (h / 6.0) * (v + 2.0 * v2 + 2.0 * v3 + v4);
  var new_v = v + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4);
  
  let r = max(length(new_x), uniforms.u_mass * 0.5 + 0.001);
  let u = min(uniforms.u_mass / (2.0 * r), 0.999);
  let n_ref = pow(1.0 + u, 3.0) / (1.0 - u);
  let target_v_len = 1.0 / n_ref;
  
  let v_len = length(new_v);
  if (v_len > 0.0) {
    new_v = normalize(new_v) * target_v_len;
  }
  *px = new_x; *pv = new_v;
}

// CSG Primitives & Blending
fn sdSphere(p: vec3<f32>, r: f32) -> f32 { return length(p) - r; }
fn sdTorus(p: vec3<f32>, t: vec2<f32>) -> f32 {
  let q = vec2<f32>(length(p.xy) - t.x, p.z);
  return length(q) - t.y;
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
  if (k <= 0.001) { return min(a, b); }
  let h = max(k - abs(a - b), 0.0) / k;
  return min(a, b) - h * h * k * 0.25;
}

fn smax(a: f32, b: f32, k: f32) -> f32 {
  if (k <= 0.001) { return max(a, b); }
  let h = max(k - abs(a - b), 0.0) / k;
  return max(a, b) + h * h * k * 0.25;
}

fn map_csg(p: vec3<f32>, csg_mode: u32, k: f32, mass: f32, color: ptr<function, vec3<f32>>) -> f32 {
  let d_main = sdSphere(p, 1.8 * mass);
  *color = vec3<f32>(0.2, 0.6, 1.0);
  if (csg_mode == 0u) { return d_main; }
  
  let p_torus = vec3<f32>(p.x, p.y, p.z - 0.4 * mass);
  let d_torus = sdTorus(p_torus, vec2<f32>(3.5 * mass, 0.4 * mass));
  let p_sph = vec3<f32>(p.x - 3.2 * mass, p.y, p.z);
  let d_sph = sdSphere(p_sph, 0.9 * mass);
  let d_sub = min(d_torus, d_sph);
  
  if (csg_mode == 1u) { // Union
    if (d_sub < d_main) { *color = vec3<f32>(1.0, 0.4, 0.2); }
    return min(d_main, d_sub);
  } else if (csg_mode == 2u) { // Smooth Min
    let d_res = smin(d_main, d_sub, k);
    let h = clamp((d_main - d_sub) / (k + 1e-4) + 0.5, 0.0, 1.0);
    *color = mix(vec3<f32>(1.0, 0.4, 0.2), vec3<f32>(0.2, 0.6, 1.0), h);
    return d_res;
  } else if (csg_mode == 3u) { // Subtraction
    let d_res = smax(d_main, -d_sub, k);
    if (-d_sub > d_main) { *color = vec3<f32>(0.9, 0.1, 0.3); }
    return d_res;
  } else if (csg_mode == 4u) { // Intersection
    let d_res = smax(d_main, d_sub, k);
    *color = vec3<f32>(0.8, 0.2, 0.9);
    return d_res;
  }
  return d_main;
}

fn calc_normal(p: vec3<f32>, csg_mode: u32, k: f32, mass: f32) -> vec3<f32> {
  let eps = 0.002;
  var dummy = vec3<f32>(0.0);
  let d = map_csg(p, csg_mode, k, mass, &dummy);
  let n = vec3<f32>(
    map_csg(p + vec3<f32>(eps, 0.0, 0.0), csg_mode, k, mass, &dummy) - d,
    map_csg(p + vec3<f32>(0.0, eps, 0.0), csg_mode, k, mass, &dummy) - d,
    map_csg(p + vec3<f32>(0.0, 0.0, eps), csg_mode, k, mass, &dummy) - d
  );
  return normalize(n);
}

fn evaluate_soft_shadow(ro: vec3<f32>, rd: vec3<f32>, csg_mode: u32, blend: f32, mass: f32, k: f32) -> f32 {
  var res = 1.0;
  var t = 0.05;
  var dummy = vec3<f32>(0.0);
  for (var i = 0; i < 20; i++) {
    let h = map_csg(ro + rd * t, csg_mode, blend, mass, &dummy);
    if (h < 0.001) { return 0.0; }
    res = min(res, k * h / t);
    t += max(h, 0.04);
    if (t > 15.0) { break; }
  }
  return clamp(res, 0.0, 1.0);
}

fn evaluate_ao(p: vec3<f32>, n: vec3<f32>, csg_mode: u32, blend: f32, mass: f32, intensity: f32) -> f32 {
  var occ = 0.0; var sca = 1.0;
  var dummy = vec3<f32>(0.0);
  for (var i = 0u; i < 5u; i++) {
    let h = 0.01 + 0.12 * f32(i);
    let d = map_csg(p + n * h, csg_mode, blend, mass, &dummy);
    occ += (h - d) * sca;
    sca *= 0.75;
  }
  return clamp(1.0 - intensity * occ, 0.0, 1.0);
}

fn sample_accretion_disk(
  x: vec3<f32>, v: vec3<f32>, h: f32,
  accum_color: ptr<function, vec3<f32>>, accum_trans: ptr<function, f32>
) {
  let r_xy = length(x.xy);
  let r_in = 3.2 * uniforms.u_mass;
  let r_out = 9.5 * uniforms.u_mass;
  
  if (r_xy >= r_in && r_xy <= r_out) {
    let disk_thickness = uniforms.u_disk_thickness * uniforms.u_mass;
    let vertical_density = exp(-pow(x.z, 2.0) / (2.0 * pow(disk_thickness, 2.0)));
    let radial_density = pow(r_in / r_xy, 2.5);
    var density = uniforms.u_disk_density * vertical_density * radial_density;
    
    if (density > 0.001) {
      let omega = 1.0 / (pow(r_xy, 1.5) + uniforms.u_spin);
      let phi = atan2(x.y, x.x);
      let phi_rotated = phi - uniforms.u_time * uniforms.u_disk_speed * (0.8 / (pow(r_xy, 1.5) + 0.1));
      
      let u1 = r_xy * 0.12 - uniforms.u_time * 0.04;
      let v1 = phi_rotated * 1.5 / (2.0 * PI) + x.z * 0.15;
      let u2 = r_xy * 0.28 + uniforms.u_time * 0.08;
      let v2 = phi_rotated * 3.0 / (2.0 * PI) - x.z * 0.15;
      
      let n1 = textureSampleLevel(u_noise_tex, u_noise_sampler, vec2<f32>(u1, v1), 0.0).r;
      let n2 = textureSampleLevel(u_noise_tex, u_noise_sampler, vec2<f32>(u2, v2), 0.0).g * 0.5;
      let noise = (n1 + n2) / 1.5;
      
      density *= (0.25 + 0.75 * noise);
      
      let T = uniforms.u_disk_temp * pow(r_in / r_xy, 0.75);
      var v_gas = vec3<f32>(-x.y, x.x, 0.0) * omega;
      
      let gas_speed = length(v_gas);
      if (gas_speed > 0.58) {
        v_gas = (v_gas / gas_speed) * 0.58;
      }
      
      let photon_dir = normalize(v);
      let beta = dot(v_gas, photon_dir);
      let g_doppler = sqrt(1.0 - beta * beta) / (1.0 - beta);
      
      let r = max(length(x), uniforms.u_mass * 0.5 + 0.001);
      let u = min(uniforms.u_mass / (2.0 * r), 0.999);
      let g_grav = (1.0 - u) / (1.0 + u);
      
      let g_total = g_doppler * g_grav;
      let T_obs = T * g_total;
      
      var emission_color = get_color_from_spectrum(T_obs, u32(uniforms.u_spectrum_mode));
      if (uniforms.u_shift_visualizer != 0u) {
        if (g_total > 1.0) {
          let t = clamp((g_total - 1.0) * 1.5, 0.0, 1.0);
          emission_color = mix(vec3<f32>(0.1, 0.7, 0.1), vec3<f32>(0.0, 0.5, 1.0), t) * 2.0;
        } else {
          let t = clamp((1.0 - g_total) * 1.2, 0.0, 1.0);
          emission_color = mix(vec3<f32>(0.1, 0.7, 0.1), vec3<f32>(1.0, 0.1, 0.0), t) * 2.0;
        }
      }
      
      let dl = h;
      let opacity = density * 2.5 * dl;
      let trans = exp(-opacity);
      
      let emission = emission_color * (density * pow(g_total, 3.8) * 1.8 * dl);
      *accum_color += *accum_trans * emission;
      *accum_trans *= trans;
    }
  }
}

fn raymarch_core(uv: vec2<f32>) -> vec4<f32> {
  var px = (uv.x * 2.0 - 1.0);
  var py = (uv.y * 2.0 - 1.0);
  px *= (uniforms.u_resolution.x / max(uniforms.u_resolution.y, 1.0));
  
  var ray_pos = uniforms.u_cam_pos;
  
  let dir_term = uniforms.u_cam_dir + px * uniforms.u_fov_scale * uniforms.u_cam_right + py * uniforms.u_fov_scale * uniforms.u_cam_up;
  let ray_dir = normalize(dir_term);
  var ray_vel = ray_dir;
  
  let r_cam = length(ray_pos);
  let r_cam_clamped = max(r_cam, uniforms.u_mass * 0.5 + 0.001);
  let u_cam = min(uniforms.u_mass / (2.0 * r_cam_clamped), 0.999);
  let n_cam = pow(1.0 + u_cam, 3.0) / (1.0 - u_cam);
  
  ray_vel /= n_cam;
  
  var accum_color = vec3<f32>(0.0);
  var accum_trans = 1.0;
  let r_eh = uniforms.u_mass * 0.5;
  
  let render_newtonian = (uniforms.u_split_active != 0u) && (uv.x > uniforms.u_split_x);
  
  if (uniforms.u_relativity == 0u || uniforms.u_mass == 0.0 || render_newtonian) {
    if (uniforms.u_disk != 0u) {
      if (ray_vel.z != 0.0) {
        let t = -ray_pos.z / ray_vel.z;
        if (t > 0.0) {
          let hit_pos = ray_pos + t * ray_vel;
          let r_xy = length(hit_pos.xy);
          let r_in = 3.2 * uniforms.u_mass;
          let r_out = 9.5 * uniforms.u_mass;
          if (r_xy >= r_in && r_xy <= r_out && uniforms.u_mass > 0.0) {
            let T = uniforms.u_disk_temp * pow(r_in / r_xy, 0.75);
            let omega = 1.0 / (pow(r_xy, 1.5) + uniforms.u_spin);
            let phi = atan2(hit_pos.y, hit_pos.x);
            let phi_rotated = phi - uniforms.u_time * uniforms.u_disk_speed * (0.8 / (pow(r_xy, 1.5) + 0.1));
            
            let u1 = r_xy * 0.12 - uniforms.u_time * 0.04;
            let v1 = phi_rotated * 1.5 / (2.0 * PI);
            let u2 = r_xy * 0.28 + uniforms.u_time * 0.08;
            let v2 = phi_rotated * 3.0 / (2.0 * PI);
            
            let n1 = textureSampleLevel(u_noise_tex, u_noise_sampler, vec2<f32>(u1, v1), 0.0).r;
            let n2 = textureSampleLevel(u_noise_tex, u_noise_sampler, vec2<f32>(u2, v2), 0.0).g * 0.5;
            let noise = (n1 + n2) / 1.5;
            
            var color = get_color_from_spectrum(T, u32(uniforms.u_spectrum_mode)) * (0.25 + 0.75 * noise);
            if (uniforms.u_shift_visualizer != 0u) {
              color = vec3<f32>(0.1, 0.7, 0.1) * (0.25 + 0.75 * noise) * 2.0;
            }
            accum_color = color;
            accum_trans = 0.15;
          }
        }
      }
    }
    
    var bg_color = vec3<f32>(0.0);
    if (uniforms.u_grid != 0u) {
      bg_color = get_grid(ray_dir);
    } else if (uniforms.u_stars != 0u) {
      bg_color = get_starfield(ray_dir);
    }
    
    var final_color = accum_color + accum_trans * bg_color;
    if (uniforms.u_split_active != 0u) {
      let dist_to_line = abs(uv.x - uniforms.u_split_x);
      if (dist_to_line < 0.002) {
        final_color = mix(vec3<f32>(0.0, 0.9, 1.0), final_color, dist_to_line / 0.002);
      }
    }
    return vec4<f32>(final_color, 1.0);
  }
  
  var captured = false;
  
  for (var step = 0u; step < 300u; step++) {
    if (step >= uniforms.u_max_steps) { break; }
    let r = length(ray_pos);
    if (r <= r_eh + 0.015) { captured = true; break; }
    if (r > 30.0) { break; }
    
    if (uniforms.u_csg_mode != 0u && uniforms.u_mass > 0.0) {
      var obj_color = vec3<f32>(0.0);
      let d_csg = map_csg(ray_pos, uniforms.u_csg_mode, uniforms.u_csg_blend, uniforms.u_mass, &obj_color);
      if (d_csg < 0.01) {
        let norm = calc_normal(ray_pos, uniforms.u_csg_mode, uniforms.u_csg_blend, uniforms.u_mass);
        let light_dir = vec3<f32>(0.577, 0.577, 0.577);
        let diff = max(0.15, dot(norm, light_dir));
        
        var shadow = 1.0;
        if (uniforms.u_soft_shadows != 0u) {
          shadow = evaluate_soft_shadow(ray_pos + norm * 0.02, light_dir, uniforms.u_csg_mode, uniforms.u_csg_blend, uniforms.u_mass, uniforms.u_shadow_k);
        }
        
        var ao = 1.0;
        if (uniforms.u_ao_enabled != 0u) {
          ao = evaluate_ao(ray_pos, norm, uniforms.u_csg_mode, uniforms.u_csg_blend, uniforms.u_mass, uniforms.u_ao_intensity);
        }
        
        let surf_color = obj_color * (diff * shadow * ao);
        accum_color += accum_trans * surf_color;
        accum_trans *= 0.10;
        break;
      }
    }
    
    let h_step = 0.18 * (0.05 + 0.95 * clamp((r - r_eh) / 5.0, 0.0, 1.0));
    let prev_pos = ray_pos;
    
    rk4_step(&ray_pos, &ray_vel, h_step);
    
    if (uniforms.u_show_photon_sphere != 0u && uniforms.u_mass > 0.0) {
      let r_prev = length(prev_pos);
      let r_curr = r;
      let r_ps = 3.0 * uniforms.u_mass;
      if ((r_prev > r_ps && r_curr <= r_ps) || (r_prev < r_ps && r_curr >= r_ps)) {
        accum_color += accum_trans * vec3<f32>(0.0, 1.0, 0.35) * 0.22;
      }
    }
    
    if (uniforms.u_disk != 0u && uniforms.u_mass > 0.0) {
      for (var i = 0u; i < 3u; i++) {
        let r_orbit = 4.2 + f32(i) * 2.0;
        let omega = 1.0 / (pow(r_orbit, 1.5) + uniforms.u_spin);
        let phi_hot = uniforms.u_time * omega * uniforms.u_disk_speed + f32(i) * 2.0944;
        let p_orbit = vec3<f32>(cos(phi_hot) * r_orbit, sin(phi_hot) * r_orbit, 0.0);
        let dist = length(ray_pos - p_orbit);
        
        if (dist < 0.35) {
          var v_gas = vec3<f32>(-p_orbit.y, p_orbit.x, 0.0) * omega;
          let gas_speed = length(v_gas);
          if (gas_speed > 0.58) { v_gas = (v_gas / gas_speed) * 0.58; }
          
          let photon_dir = normalize(ray_vel);
          let beta = dot(v_gas, photon_dir);
          let g_doppler = sqrt(1.0 - beta * beta) / (1.0 - beta);
          let g_grav = (1.0 - uniforms.u_mass / (2.0 * r_orbit)) / (1.0 + uniforms.u_mass / (2.0 * r_orbit));
          let g_total = g_doppler * g_grav;
          
          var p_color = vec3<f32>(1.0, 0.45, 0.0);
          if (i == 0u) { p_color = vec3<f32>(1.0, 0.2, 0.1); }
          else if (i == 1u) { p_color = vec3<f32>(1.0, 0.7, 0.1); }
          else { p_color = vec3<f32>(0.1, 0.8, 1.0); }
          
          if (uniforms.u_shift_visualizer != 0u) {
            if (g_total > 1.0) {
              let t = clamp((g_total - 1.0) * 1.5, 0.0, 1.0);
              p_color = mix(vec3<f32>(0.1, 0.7, 0.1), vec3<f32>(0.0, 0.5, 1.0), t) * 2.0;
            } else {
              let t = clamp((1.0 - g_total) * 1.2, 0.0, 1.0);
              p_color = mix(vec3<f32>(0.1, 0.7, 0.1), vec3<f32>(1.0, 0.1, 0.0), t) * 2.0;
            }
          } else {
            p_color = p_color * pow(g_total, 3.8);
          }
          
          let density_p = exp(-pow(dist / 0.18, 2.0));
          accum_color += accum_trans * p_color * density_p * 0.45;
        }
      }
    }
    
    if (uniforms.u_disk != 0u) {
      let sign_prev = prev_pos.z;
      let sign_curr = ray_pos.z;
      if (sign_prev * sign_curr <= 0.0 || abs(ray_pos.z) < 0.18 * uniforms.u_mass) {
        let fraction = abs(prev_pos.z) / (abs(prev_pos.z) + abs(ray_pos.z) + 1e-6);
        let cross_pos = mix(prev_pos, ray_pos, fraction);
        sample_accretion_disk(cross_pos, ray_vel, h_step, &accum_color, &accum_trans);
      }
    }
    
    if (accum_trans < 0.01) { accum_trans = 0.0; break; }
  }
  
  var final_color = vec3<f32>(0.0);
  if (!captured) {
    let escape_dir = normalize(ray_vel);
    if (uniforms.u_grid != 0u) { final_color = get_grid(escape_dir); }
    else if (uniforms.u_stars != 0u) { final_color = get_starfield(escape_dir); }
  }
  
  var color_out = accum_color + accum_trans * final_color;
  if (uniforms.u_split_active != 0u) {
    let dist_to_line = abs(uv.x - uniforms.u_split_x);
    if (dist_to_line < 0.002) {
      color_out = mix(vec3<f32>(0.0, 0.9, 1.0), color_out, dist_to_line / 0.002);
    }
  }
  return vec4<f32>(color_out, 1.0);
}

// Compute Shader Entry Point for WebGPU Raymarching
@compute @workgroup_size(16, 16)
fn cs_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let resolution = vec2<u32>(u32(uniforms.u_resolution.x), u32(uniforms.u_resolution.y));
  if (global_id.x >= resolution.x || global_id.y >= resolution.y) {
    return;
  }
  
  let uv = vec2<f32>(
    f32(global_id.x) / uniforms.u_resolution.x,
    f32(global_id.y) / uniforms.u_resolution.y
  );
  
  let color = raymarch_core(uv);
  textureStore(out_texture, vec2<i32>(global_id.xy), color);
}

// Fragment Shader Fallback Entry Point
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  return raymarch_core(in.uv);
}

struct CompVertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_composite(@builtin(vertex_index) vertex_idx: u32) -> CompVertexOutput {
  var out: CompVertexOutput;
  var pos = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 3.0, -1.0),
    vec2<f32>(-1.0,  3.0)
  );
  out.position = vec4<f32>(pos[vertex_idx], 0.0, 1.0);
  out.uv = pos[vertex_idx] * 0.5 + 0.5;
  return out;
}

@group(1) @binding(0) var u_scene_tex: texture_2d<f32>;
@group(1) @binding(1) var u_scene_sampler: sampler;

fn get_brightness(c: vec3<f32>) -> vec3<f32> {
  let luminance = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  let threshold = 0.6;
  return c * max(0.0, luminance - threshold) * 2.0;
}

fn ACESFilm(x_in: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x_in * (a * x_in + b)) / (x_in * (c * x_in + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn sample_tap(uv: vec2<f32>, offset: vec2<f32>) -> vec3<f32> {
  let sample_color = textureSample(u_scene_tex, u_scene_sampler, uv + offset / uniforms.u_resolution).rgb;
  return get_brightness(sample_color);
}

@fragment
fn fs_composite(in: CompVertexOutput) -> @location(0) vec4<f32> {
  let uv = in.uv;
  let base_color = textureSample(u_scene_tex, u_scene_sampler, uv).rgb;
  
  var bloom = vec3<f32>(0.0);
  bloom += sample_tap(uv, vec2<f32>(-1.5, -1.5));
  bloom += sample_tap(uv, vec2<f32>( 1.5, -1.5));
  bloom += sample_tap(uv, vec2<f32>(-1.5,  1.5));
  bloom += sample_tap(uv, vec2<f32>( 1.5,  1.5));
  
  bloom += sample_tap(uv, vec2<f32>(-3.5, -3.5));
  bloom += sample_tap(uv, vec2<f32>( 3.5, -3.5));
  bloom += sample_tap(uv, vec2<f32>(-3.5,  3.5));
  bloom += sample_tap(uv, vec2<f32>( 3.5,  3.5));
  
  bloom += sample_tap(uv, vec2<f32>(-6.0, 0.0));
  bloom += sample_tap(uv, vec2<f32>( 6.0, 0.0));
  bloom += sample_tap(uv, vec2<f32>(0.0, -6.0));
  bloom += sample_tap(uv, vec2<f32>(0.0,  6.0));
  
  bloom = bloom / 12.0;
  
  let blended = base_color + bloom * uniforms.u_bloom_intensity;
  let final_color = ACESFilm(blended);
  let gamma_corrected = pow(final_color, vec3<f32>(1.0 / 2.2));
  
  return vec4<f32>(gamma_corrected, 1.0);
}
`;
