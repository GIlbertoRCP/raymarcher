import { describe, expect, it } from 'vitest';
import { boundaryMask, distanceTransform, fateAgreement, shadowEdgeOffset } from './compare-lib.ts';

const W = 64, H = 48;

/** Fate map with a captured disc of radius r at (cx, cy) on an escaped background. */
function disc(cx: number, cy: number, r: number): Uint8Array {
  const f = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) f[y * W + x] = 1;
  return f;
}

describe('cross-backend comparison metrics', () => {
  it('reports full agreement for identical maps', () => {
    const a = disc(32, 24, 10);
    const r = fateAgreement(a, a, W, H);
    expect(r.all).toBe(1);
    expect(r.interior).toBe(1);
    expect(shadowEdgeOffset(a, a, W, H)!.maxPx).toBe(0);
  });

  it('excludes the pixels next to a boundary that moved by one pixel', () => {
    const a = disc(32, 24, 10), b = disc(33, 24, 10);
    const r = fateAgreement(a, b, W, H);
    expect(r.all).toBeLessThan(1);
    expect(r.interior).toBe(1);
  });

  it('counts disagreement away from boundaries', () => {
    const a = disc(32, 24, 10), b = Uint8Array.from(a);
    for (let y = 20; y < 29; y++) for (let x = 2; x < 11; x++) b[y * W + x] = 2; // a 9x9 block far from the disc
    const r = fateAgreement(a, b, W, H);
    // The block's 7x7 interior disagrees; its rim is a boundary in b.
    expect(r.interiorPixels).toBe(W * H - boundaryMask(a, b, W, H).reduce((s, v) => s + v, 0));
    expect(1 - r.interior).toBeCloseTo(49 / r.interiorPixels, 12);
    expect(r.confusion[0][2]).toBe(81);
  });

  it('measures a shifted shadow edge in pixels', () => {
    const off = shadowEdgeOffset(disc(30, 24, 12), disc(33, 24, 12), W, H)!;
    expect(off.maxPx).toBeGreaterThanOrEqual(2.9);
    expect(off.maxPx).toBeLessThanOrEqual(3.2);
    expect(off.meanPx).toBeGreaterThan(0.5);
  });

  it('computes exact Euclidean distances', () => {
    const f = new Uint8Array(W * H);
    f[10 * W + 10] = 1;
    const d = distanceTransform(f, W, H);
    expect(d[13 * W + 14]).toBeCloseTo(5, 12);
    expect(d[10 * W + 10]).toBe(0);
  });
});
