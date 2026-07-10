export const compositeShaderSource = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene_tex;
uniform vec2 u_resolution;
uniform float u_bloom_intensity;

vec3 get_brightness(vec3 c) {
  float luminance = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float threshold = 0.6;
  return c * max(0.0, luminance - threshold) * 2.0;
}

vec3 sample_tap(vec2 uv, vec2 offset) {
  vec3 sample_color = texture(u_scene_tex, uv + offset / u_resolution).rgb;
  return get_brightness(sample_color);
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
  vec2 uv = v_uv;
  vec3 base_color = texture(u_scene_tex, uv).rgb;
  
  // Multi-tap screen space blur for soft bloom glow
  vec3 bloom = vec3(0.0);
  
  // Ring 1 (Stride 1.5)
  bloom += sample_tap(uv, vec2(-1.5, -1.5));
  bloom += sample_tap(uv, vec2( 1.5, -1.5));
  bloom += sample_tap(uv, vec2(-1.5,  1.5));
  bloom += sample_tap(uv, vec2( 1.5,  1.5));
  
  // Ring 2 (Stride 3.5)
  bloom += sample_tap(uv, vec2(-3.5, -3.5));
  bloom += sample_tap(uv, vec2( 3.5, -3.5));
  bloom += sample_tap(uv, vec2(-3.5,  3.5));
  bloom += sample_tap(uv, vec2( 3.5,  3.5));
  
  // Ring 3 (Stride 6.0)
  bloom += sample_tap(uv, vec2(-6.0, 0.0));
  bloom += sample_tap(uv, vec2( 6.0, 0.0));
  bloom += sample_tap(uv, vec2(0.0, -6.0));
  bloom += sample_tap(uv, vec2(0.0,  6.0));
  
  bloom = bloom / 12.0;
  
  // Add screen space bloom back to raw color
  vec3 blended = base_color + bloom * u_bloom_intensity;
  
  // Apply ACES filmic tone mapping
  vec3 final_color = ACESFilm(blended);
  
  // Output gamma corrected color (sRGB)
  fragColor = vec4(pow(final_color, vec3(1.0 / 2.2)), 1.0);
}
`;
