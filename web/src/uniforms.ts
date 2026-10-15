import type { CameraBasis } from './camera.ts';
import type { Settings } from './settings.ts';

// Byte layout of the WGSL `Uniforms` struct in shaders/raymarch.wgsl.ts.
// vec3 fields are 16-byte aligned, so a scalar is packed into each one's
// trailing 4 bytes. Keep the two in sync; uniforms.test.ts checks this file.
export const UNIFORM_OFFSETS = {
  resolution: 0,
  time: 8,
  mass: 12,
  relativity: 16,
  disk: 20,
  grid: 24,
  stars: 28,
  diskTemp: 32,
  maxSteps: 36,
  fovScale: 40,
  diskThickness: 44,
  diskDensity: 48,
  bloomIntensity: 52,
  diskSpeed: 56,
  showPhotonSphere: 60,
  camPos: 64,
  shiftVisualizer: 76,
  camDir: 80,
  splitActive: 92,
  camUp: 96,
  splitX: 108,
  camRight: 112,
} as const;

export const UNIFORM_BUFFER_SIZE = 128;

export const FOV_SCALE = 0.577; // tan(30 deg): 60 degree vertical field of view

export function packUniforms(
  settings: Settings,
  camera: CameraBasis,
  width: number,
  height: number,
  timeSeconds: number,
): ArrayBuffer {
  const buffer = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
  const view = new DataView(buffer);
  const o = UNIFORM_OFFSETS;
  const f32 = (offset: number, v: number) => view.setFloat32(offset, v, true);
  const u32 = (offset: number, v: number) => view.setUint32(offset, v, true);
  const vec3 = (offset: number, v: readonly number[]) => v.forEach((c, i) => f32(offset + 4 * i, c));

  f32(o.resolution, width);
  f32(o.resolution + 4, height);
  f32(o.time, timeSeconds);
  f32(o.mass, settings.mass);
  u32(o.relativity, settings.relativity ? 1 : 0);
  u32(o.disk, settings.disk ? 1 : 0);
  u32(o.grid, settings.grid ? 1 : 0);
  u32(o.stars, settings.stars ? 1 : 0);
  f32(o.diskTemp, settings.diskTemp);
  u32(o.maxSteps, settings.steps);
  f32(o.fovScale, FOV_SCALE);
  f32(o.diskThickness, settings.diskThickness);
  f32(o.diskDensity, settings.diskDensity);
  f32(o.bloomIntensity, settings.bloomIntensity);
  f32(o.diskSpeed, settings.diskSpeed);
  u32(o.showPhotonSphere, settings.showPhotonSphere ? 1 : 0);
  vec3(o.camPos, camera.pos);
  u32(o.shiftVisualizer, settings.shiftVisualizer ? 1 : 0);
  vec3(o.camDir, camera.dir);
  u32(o.splitActive, settings.splitActive ? 1 : 0);
  vec3(o.camUp, camera.up);
  f32(o.splitX, settings.splitX);
  vec3(o.camRight, camera.right);

  return buffer;
}
