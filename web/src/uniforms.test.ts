import { describe, expect, it } from 'vitest';
import { orbitCamera } from './camera.ts';
import { defaultSettings } from './settings.ts';
import { raymarchWGSL } from './shaders/raymarch.wgsl.ts';
import { FOV_SCALE, packUniforms, UNIFORM_BUFFER_SIZE, UNIFORM_OFFSETS } from './uniforms.ts';

// Size and alignment of the WGSL types used in the Uniforms struct.
const WGSL_TYPES: Record<string, { size: number; align: number }> = {
  'f32': { size: 4, align: 4 },
  'u32': { size: 4, align: 4 },
  'vec2<f32>': { size: 8, align: 8 },
  'vec3<f32>': { size: 12, align: 16 },
};

const roundUp = (n: number, align: number) => Math.ceil(n / align) * align;
const camelCase = (s: string) => s.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase());

/** Lays out the WGSL `Uniforms` struct using the WGSL alignment rules. */
function wgslUniformLayout() {
  const body = raymarchWGSL.match(/struct Uniforms \{([^}]*)\}/)![1];
  const offsets: Record<string, number> = {};
  let offset = 0;
  let maxAlign = 1;
  for (const [, name, type] of body.matchAll(/(\w+)\s*:\s*([\w<>]+)\s*,/g)) {
    const t = WGSL_TYPES[type];
    if (!t) throw new Error(`Unhandled WGSL type ${type}`);
    offset = roundUp(offset, t.align);
    offsets[camelCase(name.replace(/^u_/, ''))] = offset;
    offset += t.size;
    maxAlign = Math.max(maxAlign, t.align);
  }
  return { offsets, size: roundUp(offset, maxAlign) };
}

describe('uniform buffer layout', () => {
  const wgsl = wgslUniformLayout();

  it('has the same total size as the WGSL struct', () => {
    expect(UNIFORM_BUFFER_SIZE).toBe(wgsl.size);
  });

  it('puts every field at the WGSL offset', () => {
    const { pad: _pad, ...wgslFields } = wgsl.offsets;
    expect(UNIFORM_OFFSETS).toEqual(wgslFields);
  });
});

describe('packUniforms', () => {
  it('writes settings and camera at their offsets', () => {
    const settings = { ...defaultSettings(), mass: 2.5, steps: 120, disk: false, splitX: 0.25 };
    const camera = orbitCamera(10, 0.3, 0.2);
    const view = new DataView(packUniforms(settings, camera, 800, 600, 4.5));
    const o = UNIFORM_OFFSETS;

    expect(view.byteLength).toBe(UNIFORM_BUFFER_SIZE);
    expect(view.getFloat32(o.resolution, true)).toBe(800);
    expect(view.getFloat32(o.resolution + 4, true)).toBe(600);
    expect(view.getFloat32(o.time, true)).toBe(4.5);
    expect(view.getFloat32(o.mass, true)).toBe(2.5);
    expect(view.getUint32(o.maxSteps, true)).toBe(120);
    expect(view.getUint32(o.disk, true)).toBe(0);
    expect(view.getUint32(o.relativity, true)).toBe(1);
    expect(view.getFloat32(o.splitX, true)).toBe(0.25);
    expect(view.getFloat32(o.fovScale, true)).toBeCloseTo(FOV_SCALE, 6);
    for (let i = 0; i < 3; i++) {
      expect(view.getFloat32(o.camPos + 4 * i, true)).toBeCloseTo(camera.pos[i], 5);
      expect(view.getFloat32(o.camRight + 4 * i, true)).toBeCloseTo(camera.right[i], 6);
    }
  });
});
