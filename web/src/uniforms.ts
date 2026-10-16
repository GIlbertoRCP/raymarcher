import type { Vec4 } from '../../compiler/src/index.ts';
import type { Settings } from './settings.ts';

// Byte layout of the WGSL `Uniforms` struct in shaders/tracer.wgsl.ts.
// uniforms.test.ts checks that the two agree.
export const UNIFORM_OFFSETS = {
  resolution: 0,
  time: 8,
  maxSteps: 12,
  camPos: 16,
  f0: 32,
  f1: 48,
  f2: 64,
  f3: 80,
  params: 96,
  fovScale: 128,
  stepFraction: 132,
  escapeRadius: 136,
  exposure: 140,
  diskTemp: 144,
  flags: 148,
  lengthScale: 152,
} as const;

export const UNIFORM_BUFFER_SIZE = 160;

export const FLAGS = { disk: 1, grid: 2, redshift: 4, stars: 8, debugFate: 16 } as const;

/**
 * tan of half the vertical field of view. The field-of-view setting applies
 * to the narrower side, so portrait screens are not cropped to a sliver.
 */
export function verticalTan(fovDegrees: number, aspect: number): number {
  return Math.tan((fovDegrees * Math.PI) / 360) / Math.min(1, aspect);
}

export interface FrameInputs {
  width: number;
  height: number;
  timeSeconds: number;
  camPos: Vec4;
  /** Covariant camera frame: 4-velocity, right, up, forward. */
  frame: Vec4[];
  params: readonly number[];
  lengthScale: number;
  escapeRadius: number;
}

export function packUniforms(settings: Settings, f: FrameInputs): ArrayBuffer {
  const buffer = new ArrayBuffer(UNIFORM_BUFFER_SIZE);
  const view = new DataView(buffer);
  const o = UNIFORM_OFFSETS;
  const f32 = (offset: number, v: number) => view.setFloat32(offset, v, true);
  const u32 = (offset: number, v: number) => view.setUint32(offset, v, true);
  const vec = (offset: number, v: readonly number[]) => v.forEach((c, i) => f32(offset + 4 * i, c));

  f32(o.resolution, f.width);
  f32(o.resolution + 4, f.height);
  f32(o.time, f.timeSeconds);
  u32(o.maxSteps, settings.maxSteps);
  vec(o.camPos, f.camPos);
  vec(o.f0, f.frame[0]);
  vec(o.f1, f.frame[1]);
  vec(o.f2, f.frame[2]);
  vec(o.f3, f.frame[3]);
  vec(o.params, f.params);
  f32(o.fovScale, verticalTan(settings.fovDegrees, f.width / f.height));
  f32(o.stepFraction, settings.stepFraction);
  f32(o.escapeRadius, f.escapeRadius);
  f32(o.exposure, settings.exposure);
  f32(o.diskTemp, settings.diskTemp);
  let flags = 0;
  if (settings.disk) flags |= FLAGS.disk;
  if (settings.background === 'grid') flags |= FLAGS.grid;
  if (settings.background === 'stars') flags |= FLAGS.stars;
  if (settings.redshift) flags |= FLAGS.redshift;
  if (settings.debugFate) flags |= FLAGS.debugFate;
  u32(o.flags, flags);
  f32(o.lengthScale, f.lengthScale);
  return buffer;
}
