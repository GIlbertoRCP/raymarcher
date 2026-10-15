// Checks the compiled metrics and the generic geodesic integrator against
// known results from the literature.

import { describe, expect, it } from 'vitest';
import { catalog } from '../../metrics/index.ts';
import { compileMetric, compileToJs, Spacetime } from '../src/index.ts';

function spacetime(source: string, overrides: Record<string, number> = {}): Spacetime {
  const metric = compileMetric(source);
  const params = metric.params.map(p => overrides[p.name] ?? p.value);
  return new Spacetime(compileToJs(metric), params);
}

/**
 * A ray traced the way the renderer traces it: backwards in time from
 * `start`, with coordinate velocity along `dir` (so k^t < 0). It follows the
 * path of light that arrives at `start` travelling along -dir.
 */
function cameraRay(st: Spacetime, start: [number, number, number], dir: [number, number, number]): Float64Array {
  const x = [0, ...start];
  const g = st.metricAt(x);
  const v = [0, ...dir];
  // g_tt kt^2 + 2 (g_ti v^i) kt + g_ij v^i v^j = 0, past-directed root
  let b = 0, c = 0;
  for (let i = 1; i < 4; i++) {
    b += 2 * g[i] * v[i];
    for (let j = 1; j < 4; j++) c += g[4 * i + j] * v[i] * v[j];
  }
  const a = g[0];
  const kt = (-b + Math.sqrt(b * b - 4 * a * c)) / (2 * a);
  const k = [kt, ...dir];
  const p = [0, 1, 2, 3].map(mu => k.reduce((s, kn, nu) => s + g[4 * mu + nu] * kn, 0));
  return Float64Array.from([...x, ...p]);
}

/** Direction of travel (spatial part of dx/dlambda) at the end of a ray. */
function finalDirection(st: Spacetime, state: Float64Array): number[] {
  const d = new Float64Array(8);
  st.derivative(state, d);
  const len = Math.hypot(d[1], d[2], d[3]);
  return [d[1] / len, d[2] / len, d[3] / len];
}

describe('flat space', () => {
  it('leaves rays straight', () => {
    const st = spacetime(catalog.minkowski);
    const res = st.trace(cameraRay(st, [100, 3, -2], [-1, 0, 0]), { escapeRadius: 300 });
    expect(res.fate).toBe('escaped');
    const d = finalDirection(st, res.state);
    expect(d[0]).toBeCloseTo(-1, 12);
    expect(res.state[2]).toBeCloseTo(3, 9);
    expect(res.state[3]).toBeCloseTo(-2, 9);
  });
});

describe('Schwarzschild (Kerr-Schild coordinates)', () => {
  const M = 1;
  const st = spacetime(catalog.schwarzschild);
  const bCrit = Math.sqrt(27) * M;

  it('captures rays just inside b = sqrt(27) M and lets those just outside escape', () => {
    expect(st.trace(cameraRay(st, [1000, 0.99 * bCrit, 0], [-1, 0, 0]), { stepFraction: 0.002 }).fate).toBe('captured');
    expect(st.trace(cameraRay(st, [1000, 1.01 * bCrit, 0], [-1, 0, 0]), { stepFraction: 0.002 }).fate).toBe('escaped');
  });

  it('bends light by 4M/b + 15 pi M^2 / 4b^2 in the weak field', () => {
    const b = 1000 * M;
    const res = st.trace(cameraRay(st, [1e7, b, 0], [-1, 0, 0]), { stepFraction: 0.002 });
    expect(res.fate).toBe('escaped');
    const d = finalDirection(st, res.state);
    const deflection = Math.atan2(-d[1], -d[0]);
    const expected = (4 * M) / b + (15 * Math.PI * M * M) / (4 * b * b);
    expect(Math.abs(deflection - expected) / expected).toBeLessThan(1e-3);
  });

  it('keeps the null constraint H = 0 to high accuracy', () => {
    const res = st.trace(cameraRay(st, [1000, 1.05 * bCrit, 0], [-1, 0, 0]), { stepFraction: 0.002 });
    expect(res.maxConstraintError).toBeLessThan(1e-8);
  });
});

describe('Kerr', () => {
  const M = 1;
  const a = 0.9;
  const st = spacetime(catalog.kerr, { a });
  const metric = compileMetric(catalog.kerr);
  const js = compileToJs(metric);

  it('drags frames in the same sense as the prograde disk orbits', () => {
    // Zero-angular-momentum observers rotate with omega = -g_{t phi} / g_{phi phi},
    // where d/dphi = -y d/dx + x d/dy.
    const x = [0, 10, 0, 0];
    const g = st.metricAt(x);
    const gtphi = -x[2] * g[1] + x[1] * g[2];
    const gphiphi = x[2] * x[2] * g[5] - 2 * x[1] * x[2] * g[6] + x[1] * x[1] * g[10];
    const omegaZamo = -gtphi / gphiphi;
    expect(omegaZamo).toBeGreaterThan(0);
    expect(js.disk!.omega(x, [M, a])).toBeGreaterThan(0);
  });

  it('places the disk inner edge at the innermost stable circular orbit', () => {
    // Bardeen, Press & Teukolsky (1972): r_isco = 6M at a = 0 and about 2.32M at a = 0.9.
    expect(js.disk!.inner([0, 0, 0, 0], [1, 0])).toBeCloseTo(6, 10);
    expect(js.disk!.inner([0, 0, 0, 0], [1, 0.9])).toBeCloseTo(2.3209, 3);
  });

  it('conserves energy and angular momentum about the spin axis along a ray', () => {
    // p_t is conserved because the metric does not depend on t. L_z = x p_y - y p_x
    // is conserved because Kerr is axisymmetric; this is not a coordinate
    // symmetry here, so it is a genuine check of the integration.
    const y0 = cameraRay(st, [200, 6, 2], [-1, 0, 0]);
    const lz = (y: ArrayLike<number>) => y[1] * y[6] - y[2] * y[5];
    const res = st.trace(y0, { stepFraction: 0.002 });
    expect(res.state[4]).toBeCloseTo(y0[4], 10);
    expect(Math.abs(lz(res.state) - lz(y0))).toBeLessThan(1e-6 * Math.abs(lz(y0)));
  });

  it("reproduces Bardeen's analytic shadow outline (a = 0.9, viewed edge-on)", () => {
    // Photon orbits of radius r between the prograde and retrograde circular
    // photon orbits trace the shadow edge (Bardeen 1973). For an observer in
    // the equatorial plane, the edge sits at image coordinates (xi, +-sqrt(eta)):
    const xi = (r: number) => (r * r * (3 * M - r) - a * a * (r + M)) / (a * (r - M));
    const eta = (r: number) => (r ** 3 * (4 * a * a * M - r * (r - 3 * M) ** 2)) / (a * a * (r - M) ** 2);
    const rPro = 2 * M * (1 + Math.cos((2 / 3) * Math.acos(-a)));
    const rRetro = 2 * M * (1 + Math.cos((2 / 3) * Math.acos(a)));

    const D = 1000;
    for (let i = 0; i <= 6; i++) {
      const r = rPro + ((rRetro - rPro) * (i + 0.5)) / 7;
      const edge = [xi(r), Math.sqrt(Math.max(0, eta(r)))];
      // The light reaching an observer at +x travels in +x; photons with
      // L_z > 0 (prograde) then pass on the -y side.
      const fate = (scale: number) =>
        st.trace(cameraRay(st, [D, -scale * edge[0], scale * edge[1]], [-1, 0, 0]), { stepFraction: 0.002 }).fate;
      expect(fate(0.98), `r = ${r.toFixed(3)} just inside the edge`).toBe('captured');
      expect(fate(1.02), `r = ${r.toFixed(3)} just outside the edge`).toBe('escaped');
    }
  });
});
