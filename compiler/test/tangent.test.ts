// Phase 4, step 1: Jacobian-vector products of the metric and the disk.
//
// Three independent checks for every catalog metric:
//   1. The generated JS JVP against central finite differences of the plain
//      JS backend along the same direction (coordinates and parameters).
//   2. The same JVP against explicit symbolic second derivatives
//      (d_e d_c g) and parameter derivatives (d_theta g, d_theta d_c g),
//      contracted with the direction and evaluated by the graph interpreter.
//   3. An analytic sensitivity: the Kerr ISCO slope dr_ISCO/da -> -4 sqrt(2/3) M
//      as a -> 0+ (expansion of Bardeen, Press & Teukolsky 1972).

import { describe, expect, it } from 'vitest';
import { catalog } from '../../metrics/index.ts';
import {
  compileJvpToJs,
  compileMetric,
  compileToJs,
  differentiate,
  differentiateParam,
  metricTangent,
  type CompiledMetric,
} from '../src/index.ts';

function rng(seed: number) {
  let s = seed;
  return () => (s = (s * 16807) % 2147483647) / 2147483647;
}

/** Points at coordinate radius 5..40 in every octant (as export-cuda.ts). */
function samplePoints(count: number, seed: number): number[][] {
  const rand = rng(seed);
  const points: number[][] = [];
  while (points.length < count) {
    const dir = [rand() - 0.5, rand() - 0.5, rand() - 0.5];
    const len = Math.hypot(dir[0], dir[1], dir[2]);
    if (len < 0.1) continue;
    const r = 5 + 35 * rand();
    points.push([rand() * 10, (r * dir[0]) / len, (r * dir[1]) / len, (r * dir[2]) / len]);
  }
  return points;
}

function paramSets(metric: CompiledMetric): number[][] {
  return [0, 0.7, 0.95].map(f =>
    metric.params.map(p => (f === 0 ? p.value : p.min + f * (p.max - p.min))),
  );
}

/** Three directions: coordinates only, parameters only, and both. */
function directions(np: number, seed: number) {
  const rand = rng(seed);
  const r = () => 2 * rand() - 1;
  const dx = [
    [r(), r(), r(), r()],
    [0, 0, 0, 0],
    [r(), r(), r(), r()],
  ];
  const dp = [Array.from({ length: np }, () => 0), Array.from({ length: np }, r), Array.from({ length: np }, r)];
  return { P: 3, dx: dx.flat(), dp: dp.flat(), list: dx.map((v, d) => ({ dx: v, dp: dp[d] })) };
}

/** max_k |a_k - b_k| / max(max_k |b_k|, floor) */
function relError(a: ArrayLike<number>, b: ArrayLike<number>, floor = 1e-12): number {
  let num = 0;
  let den = floor;
  for (let k = 0; k < b.length; k++) {
    num = Math.max(num, Math.abs(a[k] - b[k]));
    den = Math.max(den, Math.abs(b[k]));
  }
  return num / den;
}

const ids = Object.keys(catalog) as (keyof typeof catalog)[];

describe.each(ids)('JVP of %s', id => {
  const metric = compileMetric(catalog[id]);
  const js = compileToJs(metric);
  const jvp = compileJvpToJs(metric);
  const np = metric.params.length;

  it('agrees with central finite differences (relative 1e-6)', () => {
    const h = 1e-4;
    let worst = 0;
    for (const [k, params] of paramSets(metric).entries()) {
      for (const x of samplePoints(12, 101 + k)) {
        const dirs = directions(np, 7 + k);
        const g = new Float64Array(16), dg = new Float64Array(64);
        const tg = new Float64Array(16 * dirs.P), tdg = new Float64Array(64 * dirs.P);
        jvp.evaluate(x, params, dirs.P, dirs.dx, dirs.dp, g, dg, tg, tdg);

        const g0 = new Float64Array(16), dg0 = new Float64Array(64);
        js.evaluate(x, params, g0, dg0);
        expect(relError(g, g0, 1)).toBeLessThan(1e-15);
        expect(relError(dg, dg0, 1)).toBeLessThan(1e-15);

        for (const [d, dir] of dirs.list.entries()) {
          const at = (s: number) => {
            const gs = new Float64Array(16), dgs = new Float64Array(64);
            js.evaluate(x.map((v, i) => v + s * dir.dx[i]), params.map((v, i) => v + s * dir.dp[i]), gs, dgs);
            return { gs, dgs };
          };
          const plus = at(h), minus = at(-h);
          const fdg = plus.gs.map((v, i) => (v - minus.gs[i]) / (2 * h));
          const fddg = plus.dgs.map((v, i) => (v - minus.dgs[i]) / (2 * h));
          const eg = relError(tg.subarray(16 * d, 16 * d + 16), fdg);
          const edg = relError(tdg.subarray(64 * d, 64 * d + 64), fddg);
          worst = Math.max(worst, eg, edg);
        }
      }
    }
    console.log(`${id}: JVP vs finite differences, worst relative ${worst.toExponential(2)} (tolerance 1e-6)`);
    expect(worst).toBeLessThan(1e-6);
  });

  it('agrees with explicit second and parameter derivatives (relative 1e-12)', () => {
    const { graph } = metric;
    // d_e (d_c g) and d_theta (d_c g), d_theta g, built with the plain differentiators.
    const coordMemos = [0, 1, 2, 3].map(() => new Map<number, number>());
    const paramMemos = metric.params.map(() => new Map<number, number>());
    const contract = (node: number) => (x: number[], p: number[], dx: number[], dp: number[]) => {
      let s = 0;
      for (let e = 0; e < 4; e++) {
        if (dx[e] !== 0) s += dx[e] * graph.evaluate(differentiate(graph, node, e, coordMemos[e]), x, p);
      }
      for (let k = 0; k < np; k++) {
        if (dp[k] !== 0) s += dp[k] * graph.evaluate(differentiateParam(graph, node, k, paramMemos[k]), x, p);
      }
      return s;
    };
    let worst = 0;
    for (const [k, params] of paramSets(metric).entries()) {
      for (const x of samplePoints(4, 201 + k)) {
        const dirs = directions(np, 17 + k);
        const g = new Float64Array(16), dg = new Float64Array(64);
        const tg = new Float64Array(16 * dirs.P), tdg = new Float64Array(64 * dirs.P);
        jvp.evaluate(x, params, dirs.P, dirs.dx, dirs.dp, g, dg, tg, tdg);
        for (const [d, dir] of dirs.list.entries()) {
          const wantG = metric.g.map(n => contract(n)(x, params, dir.dx, dir.dp));
          const wantDg = metric.dg.flat().map(n => contract(n)(x, params, dir.dx, dir.dp));
          worst = Math.max(
            worst,
            relError(tg.subarray(16 * d, 16 * d + 16), wantG),
            relError(tdg.subarray(64 * d, 64 * d + 64), wantDg),
          );
        }
      }
    }
    console.log(`${id}: JVP vs explicit second derivatives, worst relative ${worst.toExponential(2)} (tolerance 1e-12)`);
    expect(worst).toBeLessThan(1e-12);
  });

  it('shares symmetric components (g_{mu nu} = g_{nu mu} is one node)', () => {
    const tan = metricTangent(metric);
    for (let mu = 0; mu < 4; mu++) {
      for (let nu = 0; nu < 4; nu++) {
        expect(tan.tg[4 * mu + nu]).toBe(tan.tg[4 * nu + mu]);
        for (let c = 0; c < 4; c++) expect(tan.tdg[16 * c + 4 * mu + nu]).toBe(tan.tdg[16 * c + 4 * nu + mu]);
      }
    }
  });

  it.runIf(metric.disk !== null)('disk JVP agrees with central finite differences (relative 1e-6)', () => {
    const h = 1e-5; // the ISCO near a = 1 is steep; FD truncation at h = 1e-4 is ~5e-7
    let worst = 0;
    for (const [k, params] of paramSets(metric).entries()) {
      for (const x of samplePoints(12, 301 + k)) {
        const dirs = directions(np, 27 + k);
        const out = new Float64Array(4), tout = new Float64Array(4 * dirs.P);
        jvp.disk!(x, params, dirs.P, dirs.dx, dirs.dp, out, tout);
        const f = (xs: number[], ps: number[]) => {
          const d = js.disk!;
          return [d.radius(xs, ps), d.omega(xs, ps), d.inner(xs, ps), d.outer(xs, ps)];
        };
        expect(relError(out, f(x, params), 1)).toBeLessThan(1e-15);
        for (const [d, dir] of dirs.list.entries()) {
          const p = f(x.map((v, i) => v + h * dir.dx[i]), params.map((v, i) => v + h * dir.dp[i]));
          const m = f(x.map((v, i) => v - h * dir.dx[i]), params.map((v, i) => v - h * dir.dp[i]));
          for (let j = 0; j < 4; j++) {
            const fd = (p[j] - m[j]) / (2 * h);
            worst = Math.max(worst, Math.abs(tout[4 * d + j] - fd) / Math.max(Math.abs(fd), Math.abs(out[j]), 1e-12));
          }
        }
      }
    }
    console.log(`${id}: disk JVP vs finite differences, worst relative ${worst.toExponential(2)} (tolerance 1e-6)`);
    expect(worst).toBeLessThan(1e-6);
  });
});

describe('Kerr ISCO sensitivity', () => {
  const metric = compileMetric(catalog.kerr);
  const jvp = compileJvpToJs(metric);
  const innerSlope = (a: number) => {
    const out = new Float64Array(4), tout = new Float64Array(4);
    jvp.disk!([0, 10, 0, 0], [1, a], 1, [0, 0, 0, 0], [0, 1], out, tout);
    return { r: out[2], slope: tout[2] };
  };

  it('dr_ISCO/da -> -4 sqrt(2/3) M as a -> 0+ (series: -4 sqrt(2/3) - (7/9) sqrt(2/3)... )', () => {
    // r_ISCO = 6 - 4 sqrt(2/3) a - (7/18) a^2 + O(a^3) for prograde orbits, M = 1.
    const a = 1e-4;
    const { r, slope } = innerSlope(a);
    const want = -4 * Math.sqrt(2 / 3) - (7 / 9) * a;
    expect(Math.abs(r - (6 - 4 * Math.sqrt(2 / 3) * a))).toBeLessThan(1e-6);
    expect(Math.abs(slope - want)).toBeLessThan(1e-4);
  });

  it('is not finite at exactly a = 0 (the formula has a |a| kink there; priors must avoid a = 0)', () => {
    // sqrt((3 - Z1)(3 + Z1 + 2 Z2)) with 3 - Z1 = O(a^2): its symbolic derivative is 0 * inf.
    // Recorded as a measured fact for step 4 / step 8, not a requirement.
    expect(Number.isFinite(innerSlope(0).slope)).toBe(false);
  });

  it('dr_ISCO/da matches the closed form near extremal spin (a = 0.98)', () => {
    // Independent derivative: central difference of the BPT closed form written out here.
    const bpt = (a: number) => {
      const z1 = 1 + Math.cbrt(1 - a * a) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
      const z2 = Math.sqrt(3 * a * a + z1 * z1);
      return 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
    };
    const a = 0.98, h = 1e-6;
    const fd = (bpt(a + h) - bpt(a - h)) / (2 * h);
    const { r, slope } = innerSlope(a);
    expect(Math.abs(r - bpt(a))).toBeLessThan(1e-13);
    expect(Math.abs(slope - fd) / Math.abs(fd)).toBeLessThan(1e-6);
  });
});
