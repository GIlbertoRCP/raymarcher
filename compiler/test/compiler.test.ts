import { describe, expect, it } from 'vitest';
import { catalog, type CatalogName } from '../../metrics/index.ts';
import { compileMetric, compileToJs, invert4 } from '../src/index.ts';

// Deterministic pseudo-random points away from the horizon and the axis.
function samplePoints(count: number, seed = 1): number[][] {
  let s = seed;
  const rand = () => ((s = (s * 16807) % 2147483647) / 2147483647);
  const points: number[][] = [];
  while (points.length < count) {
    const p = [rand() * 10, (rand() - 0.5) * 30, (rand() - 0.5) * 30, (rand() - 0.5) * 30];
    if (Math.hypot(p[1], p[2], p[3]) > 4) points.push(p);
  }
  return points;
}

const names = Object.keys(catalog) as CatalogName[];

describe.each(names)('%s metric', name => {
  const metric = compileMetric(catalog[name]);
  const js = compileToJs(metric);
  const params = metric.params.map(p => p.value);
  const g = new Float64Array(16);
  const dg = new Float64Array(64);

  it('generated JS agrees with direct evaluation of the expression graph', () => {
    for (const x of samplePoints(20)) {
      js.evaluate(x, params, g, dg);
      for (let k = 0; k < 16; k++) {
        expect(g[k]).toBeCloseTo(metric.graph.evaluate(metric.g[k], x, params), 12);
      }
    }
  });

  it('has symbolic derivatives that match finite differences', () => {
    const gPlus = new Float64Array(16), gMinus = new Float64Array(16), scratch = new Float64Array(64);
    for (const x of samplePoints(20, 7)) {
      js.evaluate(x, params, g, dg);
      for (let c = 0; c < 4; c++) {
        const eps = 1e-5;
        const xp = [...x], xm = [...x];
        xp[c] += eps;
        xm[c] -= eps;
        js.evaluate(xp, params, gPlus, scratch);
        js.evaluate(xm, params, gMinus, scratch);
        for (let k = 0; k < 16; k++) {
          const fd = (gPlus[k] - gMinus[k]) / (2 * eps);
          expect(Math.abs(dg[16 * c + k] - fd)).toBeLessThan(1e-7 * Math.max(1, Math.abs(fd)));
        }
      }
    }
  });

  it('is symmetric and invertible, with Lorentzian signature', () => {
    for (const x of samplePoints(10, 3)) {
      js.evaluate(x, params, g, dg);
      for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) expect(g[4 * i + j]).toBe(g[4 * j + i]);
      const inv = invert4(g);
      for (let i = 0; i < 4; i++) {
        for (let j = 0; j < 4; j++) {
          let s = 0;
          for (let k = 0; k < 4; k++) s += g[4 * i + k] * inv[4 * k + j];
          expect(s).toBeCloseTo(i === j ? 1 : 0, 10);
        }
      }
      expect(inv[0]).toBeLessThan(0); // the time direction is timelike
    }
  });

  it('is detected as stationary (t is a cyclic coordinate)', () => {
    expect(metric.cyclic).toContain(0);
  });
});

describe('compiler output size', () => {
  it('keeps Kerr small thanks to shared subexpressions', () => {
    const metric = compileMetric(catalog.kerr);
    const roots = [...metric.g, ...metric.dg.flat()];
    const nodes = metric.graph.topologicalOrder(roots).length;
    // 10 components and 30 non-zero derivative components share most of their work.
    expect(nodes).toBeLessThan(400);
  });

  it('marks every coordinate cyclic in flat space', () => {
    expect(compileMetric(catalog.minkowski).cyclic).toEqual([0, 1, 2, 3]);
  });
});
