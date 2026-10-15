import { describe, expect, it } from 'vitest';
import { orbitCamera, type Vec3 } from './camera.ts';

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: Vec3) => Math.sqrt(dot(a, a));

describe('orbitCamera', () => {
  const cases = [
    [12, 0, 0],
    [12, 1.2, 0.4],
    [5, -2.5, -1.3],
    [30, 3.1, 1.39],
  ];

  it.each(cases)('builds an orthonormal basis (radius %d, yaw %d, pitch %d)', (radius, yaw, pitch) => {
    const { dir, right, up } = orbitCamera(radius, yaw, pitch);
    for (const v of [dir, right, up]) expect(len(v)).toBeCloseTo(1, 12);
    expect(dot(dir, right)).toBeCloseTo(0, 12);
    expect(dot(dir, up)).toBeCloseTo(0, 12);
    expect(dot(right, up)).toBeCloseTo(0, 12);
  });

  it.each(cases)('sits at the requested radius and looks at the origin (radius %d)', (radius, yaw, pitch) => {
    const { pos, dir } = orbitCamera(radius, yaw, pitch);
    expect(len(pos)).toBeCloseTo(radius, 10);
    expect(dot(pos, dir)).toBeCloseTo(-radius, 10);
  });

  it('keeps world +z as up when level', () => {
    const { up } = orbitCamera(12, 0.7, 0);
    expect(up[2]).toBeCloseTo(1, 12);
  });
});
