// Generic GPU ray tracer. The metric-specific functions (metric_eval,
// horizon_value, disk_*) come from the compiler's WGSL backend and are spliced
// in by buildTracerShader. The physics mirrors compiler/src/geodesic.ts:
//
//   dx^mu/dl = g^{mu nu} p_nu,   dp_mu/dl = 1/2 (d_mu g_ab) k^a k^b,  k = g^{-1} p

const prelude = /* wgsl */ `
struct Uniforms {
  resolution: vec2<f32>,
  time: f32,
  max_steps: u32,
  cam_pos: vec4<f32>,
  // Covariant camera frame: f0 = g e0 (4-velocity), f1 right, f2 up, f3 forward.
  f0: vec4<f32>,
  f1: vec4<f32>,
  f2: vec4<f32>,
  f3: vec4<f32>,
  params: array<vec4<f32>, 2>,
  fov_scale: f32,
  step_fraction: f32,
  escape_radius: f32,
  exposure: f32,
  disk_temp: f32,
  flags: u32,
  length_scale: f32,
  pad: f32,
}

struct MetricEval {
  g: mat4x4<f32>,
  dg0: mat4x4<f32>,
  dg1: mat4x4<f32>,
  dg2: mat4x4<f32>,
  dg3: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> U: Uniforms;
`;

const runtime = /* wgsl */ `
@group(0) @binding(1) var out_texture: texture_storage_2d<rgba16float, write>;

const FLAG_DISK: u32 = 1u;
const FLAG_GRID: u32 = 2u;
const FLAG_REDSHIFT: u32 = 4u;
const FLAG_STARS: u32 = 8u;
const PI: f32 = 3.14159265359;

fn has_flag(f: u32) -> bool { return (U.flags & f) != 0u; }

fn inverse4(m: mat4x4<f32>) -> mat4x4<f32> {
  let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
  let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
  let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
  let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];
  let b00 = a00 * a11 - a01 * a10; let b01 = a00 * a12 - a02 * a10; let b02 = a00 * a13 - a03 * a10;
  let b03 = a01 * a12 - a02 * a11; let b04 = a01 * a13 - a03 * a11; let b05 = a02 * a13 - a03 * a12;
  let b06 = a20 * a31 - a21 * a30; let b07 = a20 * a32 - a22 * a30; let b08 = a20 * a33 - a23 * a30;
  let b09 = a21 * a32 - a22 * a31; let b10 = a21 * a33 - a23 * a31; let b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  let s = 1.0 / det;
  return mat4x4<f32>(
    vec4<f32>(a11 * b11 - a12 * b10 + a13 * b09, a02 * b10 - a01 * b11 - a03 * b09, a31 * b05 - a32 * b04 + a33 * b03, a22 * b04 - a21 * b05 - a23 * b03) * s,
    vec4<f32>(a12 * b08 - a10 * b11 - a13 * b07, a00 * b11 - a02 * b08 + a03 * b07, a32 * b02 - a30 * b05 - a33 * b01, a20 * b05 - a22 * b02 + a23 * b01) * s,
    vec4<f32>(a10 * b10 - a11 * b08 + a13 * b06, a01 * b08 - a00 * b10 - a03 * b06, a30 * b04 - a31 * b02 + a33 * b00, a21 * b02 - a20 * b04 - a23 * b00) * s,
    vec4<f32>(a11 * b07 - a10 * b09 - a12 * b06, a00 * b09 - a01 * b07 + a02 * b06, a31 * b01 - a30 * b03 - a32 * b00, a20 * b03 - a21 * b01 + a22 * b00) * s
  );
}

struct Deriv {
  dx: vec4<f32>,
  dp: vec4<f32>,
}

fn geodesic_rhs(x: vec4<f32>, p: vec4<f32>) -> Deriv {
  let m = metric_eval(x);
  let k = inverse4(m.g) * p;
  var d: Deriv;
  d.dx = k;
  d.dp = 0.5 * vec4<f32>(dot(k, m.dg0 * k), dot(k, m.dg1 * k), dot(k, m.dg2 * k), dot(k, m.dg3 * k));
  return d;
}

// --- Sky -----------------------------------------------------------------

fn hash3(p_in: vec3<f32>) -> f32 {
  var p = fract(p_in * vec3<f32>(443.8975, 397.2973, 491.1871));
  p += vec3<f32>(dot(p.xyz, p.yzx + vec3<f32>(19.19)));
  return fract(p.x * p.y * p.z);
}

// Point-like stars: each cell of a grid around the sky may hold one star at
// a random position; its brightness falls off with angular distance.
fn starfield(dir: vec3<f32>) -> vec3<f32> {
  let scale = 90.0;
  let p = dir * scale;
  let cell = floor(p);
  if (hash3(cell) < 0.93) { return vec3<f32>(0.0); }
  let star = normalize(cell + vec3<f32>(hash3(cell + 1.7), hash3(cell + 3.1), hash3(cell + 5.3)));
  let d = length(dir - star) * scale;
  let intensity = exp(-d * d * 60.0) * (0.15 + pow(hash3(cell * 4.321), 5.0) * 1.6);
  let tint = hash3(cell * 8.765);
  var c = vec3<f32>(0.95, 0.97, 1.0);
  if (tint < 0.2) { c = vec3<f32>(1.0, 0.75, 0.6); } else if (tint > 0.85) { c = vec3<f32>(0.65, 0.8, 1.0); }
  return c * intensity;
}

// Latitude-longitude grid on the celestial sphere; the standard picture for
// showing how lensing distorts the sky.
fn sky_grid(dir: vec3<f32>) -> vec3<f32> {
  let theta = acos(clamp(dir.z, -1.0, 1.0));
  let phi = atan2(dir.y, dir.x);
  let u = phi / (2.0 * PI) * 24.0;
  let v = theta / PI * 12.0;
  let cu = abs(fract(u) - 0.5);
  let cv = abs(fract(v) - 0.5);
  let line = 1.0 - smoothstep(0.44, 0.49, max(cu, cv));
  let checker = (i32(floor(u)) + i32(floor(v))) & 1;
  var base = vec3<f32>(0.035, 0.04, 0.05);
  if (checker == 1) { base = vec3<f32>(0.06, 0.065, 0.08); }
  return mix(vec3<f32>(0.85, 0.62, 0.32) * 0.55, base, line);
}

fn background(dir: vec3<f32>) -> vec3<f32> {
  if (has_flag(FLAG_GRID)) { return sky_grid(dir); }
  if (has_flag(FLAG_STARS)) { return starfield(dir); }
  return vec3<f32>(0.0);
}

// --- Disk emission -------------------------------------------------------

// Approximate sRGB chromaticity of a blackbody (Tanner Helland's fit), normalised.
fn blackbody(temp_k: f32) -> vec3<f32> {
  let t = clamp(temp_k, 1000.0, 40000.0) / 100.0;
  var c: vec3<f32>;
  if (t <= 66.0) {
    c.r = 1.0;
    c.g = clamp(0.39008158 * log(t) - 0.63184144, 0.0, 1.0);
    c.b = select(clamp(0.54320679 * log(t - 10.0) - 1.19625409, 0.0, 1.0), 0.0, t <= 19.0);
  } else {
    c.r = clamp(1.29293619 * pow(t - 60.0, -0.1332047592), 0.0, 1.0);
    c.g = clamp(1.12989086 * pow(t - 60.0, -0.0755148492), 0.0, 1.0);
    c.b = 1.0;
  }
  return c / max(max(c.r, c.g), c.b);
}

fn value_noise(p: vec2<f32>) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash3(vec3<f32>(i, 1.0));
  let b = hash3(vec3<f32>(i + vec2<f32>(1.0, 0.0), 1.0));
  let c = hash3(vec3<f32>(i + vec2<f32>(0.0, 1.0), 1.0));
  let d = hash3(vec3<f32>(i + vec2<f32>(1.0, 1.0), 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Colour of an optically thick thin disk at crossing point x, for a ray with
// covariant momentum p. The gas follows circular orbits u ~ (1, -omega y, omega x, 0),
// and the observed/emitted energy ratio is g = 1 / (p . u) because p . u = 1 at the camera.
fn disk_color(x: vec4<f32>, p: vec4<f32>) -> vec4<f32> {
  let r = disk_radius(x);
  let r_in = disk_inner(x);
  if (r < r_in || r > disk_outer(x)) { return vec4<f32>(0.0); }

  let omega = disk_omega(x);
  let v = vec4<f32>(1.0, -omega * x.z, omega * x.y, 0.0);
  let gm = metric_eval(x).g;
  let norm = dot(v, gm * v);
  if (norm >= 0.0) { return vec4<f32>(0.0); } // no circular orbit here
  let u = v / sqrt(-norm);
  let shift = 1.0 / dot(p, u);

  // Thin-disk flux profile F ~ (1 - sqrt(r_in / r)) / r^3 (Shakura-Sunyaev),
  // peak-normalised; local temperature T ~ F^(1/4).
  let flux = (1.0 - sqrt(r_in / r)) * pow(r_in / r, 3.0) / 0.05666;
  let temp = U.disk_temp * pow(max(flux, 0.0), 0.25);
  let temp_obs = temp * shift;

  // Artistic turbulence that co-rotates with the gas.
  let phi = atan2(x.z, x.y) - omega * U.time;
  let swirl = value_noise(vec2<f32>(log(r) * 9.0, phi * 6.0)) * 0.6 + value_noise(vec2<f32>(log(r) * 23.0, phi * 15.0)) * 0.4;
  let texture = 0.55 + 0.45 * swirl;

  if (has_flag(FLAG_REDSHIFT)) {
    let s = clamp((shift - 1.0) * 1.6, -1.0, 1.0);
    let c = select(mix(vec3<f32>(0.9, 0.9, 0.85), vec3<f32>(1.0, 0.25, 0.1), -s), mix(vec3<f32>(0.9, 0.9, 0.85), vec3<f32>(0.2, 0.5, 1.0), s), s > 0.0);
    return vec4<f32>(c * texture * 0.8, 1.0);
  }
  // Bolometric brightness scales as T_obs^4 = g^4 T^4.
  let brightness = pow(temp_obs / U.disk_temp, 4.0) * texture;
  return vec4<f32>(blackbody(temp_obs) * brightness, 1.0);
}

// --- Tracer --------------------------------------------------------------

fn trace(uv: vec2<f32>) -> vec3<f32> {
  let aspect = U.resolution.x / max(U.resolution.y, 1.0);
  let ndc = uv * 2.0 - 1.0;
  let d = normalize(vec3<f32>(ndc.x * aspect * U.fov_scale, ndc.y * U.fov_scale, 1.0));
  // Past-directed null momentum leaving the camera: k = -e0 + d (lowered).
  var x = U.cam_pos;
  var p = -U.f0 + d.x * U.f1 + d.y * U.f2 + d.z * U.f3;
  let check_disk = HAS_DISK && has_flag(FLAG_DISK);
  let min_step = 0.002 * U.length_scale;

  for (var i = 0u; i < U.max_steps; i++) {
    if (HAS_HORIZON && horizon_value(x) < 0.0) { return vec3<f32>(0.0); }
    let r = length(x.yzw);
    if (r > U.escape_radius) { break; }

    let h = max(U.step_fraction * r, min_step);
    let k1 = geodesic_rhs(x, p);
    // Traced backwards, a ray heading into a horizon piles up against it while
    // dt/dlambda diverges (in ingoing coordinates it never crosses). Treat a
    // diverging or non-finite time component as captured.
    if (!(abs(k1.dx.x) < 1.0e4)) { return vec3<f32>(0.0); }
    let k2 = geodesic_rhs(x + 0.5 * h * k1.dx, p + 0.5 * h * k1.dp);
    let k3 = geodesic_rhs(x + 0.5 * h * k2.dx, p + 0.5 * h * k2.dp);
    let k4 = geodesic_rhs(x + h * k3.dx, p + h * k3.dp);
    let x_new = x + (h / 6.0) * (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx);
    let p_new = p + (h / 6.0) * (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp);

    if (check_disk && x.w * x_new.w <= 0.0 && x.w != x_new.w) {
      let f = x.w / (x.w - x_new.w);
      let c = disk_color(mix(x, x_new, f), mix(p, p_new, f));
      if (c.a > 0.0) { return c.rgb; }
    }
    x = x_new;
    p = p_new;
  }
  // Escaped (or out of steps): the light arrived from the part of the sky
  // this backwards-traced ray is heading toward.
  let dir = geodesic_rhs(x, p).dx.yzw;
  return background(normalize(dir));
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
  let size = vec2<u32>(U.resolution);
  if (id.x >= size.x || id.y >= size.y) { return; }
  let uv = (vec2<f32>(id.xy) + 0.5) / U.resolution;
  let color = trace(vec2<f32>(uv.x, 1.0 - uv.y));
  textureStore(out_texture, vec2<i32>(id.xy), vec4<f32>(color, 1.0));
}
`;

// Tone mapping and a small bloom, drawn as a fullscreen triangle.
const composite = /* wgsl */ `
struct CompositeOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs_composite(@builtin(vertex_index) i: u32) -> CompositeOut {
  var pos = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var out: CompositeOut;
  out.position = vec4<f32>(pos[i], 0.0, 1.0);
  out.uv = vec2<f32>(pos[i].x * 0.5 + 0.5, 0.5 - pos[i].y * 0.5);
  return out;
}

@group(1) @binding(0) var scene: texture_2d<f32>;
@group(1) @binding(1) var scene_sampler: sampler;

fn bright(c: vec3<f32>) -> vec3<f32> {
  let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
  return c * max(0.0, l - 0.8);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_composite(in: CompositeOut) -> @location(0) vec4<f32> {
  let px = 1.0 / U.resolution;
  var bloom = vec3<f32>(0.0);
  for (var i = 0; i < 12; i++) {
    let a = f32(i) * 0.5236;
    let rad = select(2.5, 6.0, (i & 1) == 1);
    bloom += bright(textureSampleLevel(scene, scene_sampler, in.uv + vec2<f32>(cos(a), sin(a)) * rad * px, 0.0).rgb);
  }
  let base = textureSampleLevel(scene, scene_sampler, in.uv, 0.0).rgb;
  let mapped = aces((base + bloom / 12.0 * 0.8) * U.exposure);
  // Colour grading, not physics: blackbody colours are pale and ACES
  // desaturates them further, so restore some saturation for display.
  let luma = dot(mapped, vec3<f32>(0.2126, 0.7152, 0.0722));
  let graded = clamp(mix(vec3<f32>(luma), mapped, 1.7), vec3<f32>(0.0), vec3<f32>(1.0));
  return vec4<f32>(pow(graded, vec3<f32>(1.0 / 2.2)), 1.0);
}
`;

/** Prelude only: the uniform and metric structs (used to check the layout in tests). */
export const tracerPrelude = prelude;

export function buildTracerShader(metricModule: string): string {
  return [prelude, metricModule, runtime, composite].join('\n');
}
