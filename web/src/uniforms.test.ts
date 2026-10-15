import { describe, expect, it } from 'vitest';
import { tracerPrelude } from './shaders/tracer.wgsl.ts';
import { defaultSettings } from './settings.ts';
import { FLAGS, packUniforms, UNIFORM_BUFFER_SIZE, UNIFORM_OFFSETS } from './uniforms.ts';

// Size and alignment of the WGSL types used in the Uniforms struct.
const WGSL_TYPES: Record<string, { size: number; align: number }> = {
  'f32': { size: 4, align: 4 },
  'u32': { size: 4, align: 4 },
  'vec2<f32>': { size: 8, align: 8 },
  'vec4<f32>': { size: 16, align: 16 },
  'array<vec4<f32>,2>': { size: 32, align: 16 },
};

const roundUp = (n: number, align: number) => Math.ceil(n / align) * align;
const camelCase = (s: string) => s.replace(/_([a-z0-9])/g, (_, c: string) => c.toUpperCase());

/** Lays out the WGSL `Uniforms` struct using the WGSL alignment rules. */
function wgslUniformLayout() {
  const body = tracerPrelude.match(/struct Uniforms \{([^}]*)\}/)![1].replace(/\/\/.*$/gm, '');
  const offsets: Record<string, number> = {};
  let offset = 0;
  let maxAlign = 1;
  for (const [, name, rawType] of body.matchAll(/^\s*(\w+)\s*:\s*(.+?),\s*$/gm)) {
    const type = rawType.replace(/\s+/g, '');
    const t = WGSL_TYPES[type];
    if (!t) throw new Error(`Unhandled WGSL type ${type}`);
    offset = roundUp(offset, t.align);
    offsets[camelCase(name)] = offset;
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
  it('writes the camera frame, parameters and flags', () => {
    const settings = { ...defaultSettings(), background: 'grid' as const, redshift: true, maxSteps: 321 };
    const frame: [number, number, number, number][] = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]];
    const view = new DataView(
      packUniforms(settings, {
        width: 800, height: 600, timeSeconds: 2, camPos: [0, 1, 2, 3], frame,
        params: [1.5, 0.9], lengthScale: 1.5, escapeRadius: 90,
      }),
    );
    const o = UNIFORM_OFFSETS;
    expect(view.getUint32(o.maxSteps, true)).toBe(321);
    expect(view.getFloat32(o.camPos + 12, true)).toBe(3);
    expect(view.getFloat32(o.f2 + 8, true)).toBe(11);
    expect(view.getFloat32(o.params + 4, true)).toBeCloseTo(0.9, 6);
    expect(view.getUint32(o.flags, true)).toBe(FLAGS.disk | FLAGS.grid | FLAGS.redshift);
    expect(view.getFloat32(o.escapeRadius, true)).toBe(90);
  });
});
