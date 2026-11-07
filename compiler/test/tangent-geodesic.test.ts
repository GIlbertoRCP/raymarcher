// Phase 4, step 2: the tangent RK4 tracer (compiler/src/tangent-geodesic.ts).
//
//   1. Every ray's dy_final/dtheta against 4-point central finite differences
//      of the same tracer, for all inputs (metric parameters, camera radius,
//      inclination, yaw, image-plane u and v), across a grid of rays and spins.
//   2. Analytic sensitivities:
//      - d(shadow angle)/dR for a static Schwarzschild observer, against the
//        derivative of sin(alpha) = sqrt(27) M sqrt(1 - 2M/R) / R;
//      - d(b_crit)/da for equatorial Kerr rays on both sides, against the
//        derivative of the circular photon-orbit impact parameters;
//      - d(sweep angle)/db for Schwarzschild, against the exact orbit integral.
//   (dr_ISCO/da is checked at the function level in tangent.test.ts.)

import { describe, expect, it } from 'vitest';
import { catalog } from '../../metrics/index.ts';
import { compileJvpToJs, compileMetric, compileToJs, TangentSpacetime, type RayInputs } from '../src/index.ts';

function tracer(id: keyof typeof catalog) {
  const metric = compileMetric(catalog[id]);
  return new TangentSpacetime(compileToJs(metric), compileJvpToJs(metric));
}

/** Unit direction along theta component k (theta = params..., camRadius, inclination, yaw, u, v). */
const unit = (n: number, k: number) => Array.from({ length: n }, (_, i) => (i === k ? 1 : 0));

function thetaOf(inp: RayInputs): number[] {
  return [...inp.params, inp.camRadius, inp.inclination, inp.yaw, inp.u, inp.v];
}
function inputsOf(theta: number[], np: number): RayInputs {
  return {
    params: theta.slice(0, np),
    camRadius: theta[np],
    inclination: theta[np + 1],
    yaw: theta[np + 2],
    u: theta[np + 3],
    v: theta[np + 4],
  };
}

describe('tangent RK4 tracer against finite differences', () => {
  const cases: { id: keyof typeof catalog; params: number[] }[] = [
    { id: 'schwarzschild', params: [1] },
    { id: 'kerr', params: [1, 0.3] },
    { id: 'kerr', params: [1, 0.94] },
    { id: 'reissner-nordstrom', params: [1, 0.6] },
  ];
  const opts = { stepFraction: 0.02, disk: true };

  it.each(cases)('$id $params: every ray and every input (relative 1e-6)', ({ id, params }) => {
    const st = tracer(id);
    const np = params.length;
    const n = np + 5;
    const dirs = Array.from({ length: n }, (_, k) => unit(n, k));
    const grid = [-0.32, -0.17, 0.11, 0.26];
    const fates: Record<string, number> = {};
    let worst = 0, compared = 0, skipped = 0;
    for (const u of grid) {
      for (const v of grid) {
        const inp: RayInputs = { params, camRadius: 50, inclination: (60 * Math.PI) / 180, yaw: 0.3, u, v };
        const ray = st.trace(inp, dirs, opts);
        if (ray.fate !== 'escaped' && ray.fate !== 'disk') continue;
        const theta = thetaOf(inp);
        for (let k = 0; k < n; k++) {
          const h = 1e-4 * Math.max(1, Math.abs(theta[k]));
          const at = (s: number) => {
            const t = theta.slice();
            t[k] += s * h;
            return st.trace(inputsOf(t, np), [], opts);
          };
          const rs = [at(2), at(1), at(-1), at(-2)];
          if (rs.some(r => r.fate !== ray.fate || r.steps !== ray.steps)) {
            skipped++;
            continue;
          }
          let num = 0, den = 1e-12;
          for (let i = 0; i < 8; i++) {
            const fd = (-rs[0].y[i] + 8 * rs[1].y[i] - 8 * rs[2].y[i] + rs[3].y[i]) / (12 * h);
            num = Math.max(num, Math.abs(ray.S[8 * k + i] - fd));
            den = Math.max(den, Math.abs(fd));
          }
          worst = Math.max(worst, num / den);
          compared++;
        }
        fates[ray.fate] = (fates[ray.fate] ?? 0) + 1;
      }
    }
    console.log(
      `${id} ${JSON.stringify(params)}: ${compared} (ray, input) pairs, fates ${JSON.stringify(fates)}, ` +
        `${skipped} skipped (step count or fate changed under the FD step); worst relative ${worst.toExponential(2)} (tolerance 1e-6)`,
    );
    expect(fates.escaped ?? 0).toBeGreaterThan(0);
    if (id !== 'kerr' || params[1] < 0.9) expect(fates.disk ?? 0).toBeGreaterThan(0);
    expect(compared).toBeGreaterThan(40);
    expect(worst).toBeLessThan(1e-6);
  });
});

/** Constants of motion of a ray and their tangents: E = -p_t, L = x p_y - y p_x, b = L / E. */
function impactParameter(y: Float64Array, S: Float64Array, d: number) {
  const E = -y[4], L = y[1] * y[6] - y[2] * y[5];
  const dE = -S[8 * d + 4];
  const dL = S[8 * d + 1] * y[6] + y[1] * S[8 * d + 6] - S[8 * d + 2] * y[5] - y[2] * S[8 * d + 5];
  return { b: L / E, db: (dL * E - L * dE) / (E * E) };
}

describe('Schwarzschild shadow angle against R', () => {
  const st = tracer('schwarzschild');
  // Equatorial camera on +y looking at the origin; u moves the ray in the z = 0 plane.
  const base = (R: number, u: number): RayInputs => ({ params: [1], camRadius: R, inclination: Math.PI / 2, yaw: 0, u, v: 0 });
  const dirs = [unit(6, 1), unit(6, 4)]; // camRadius, u

  it.each([4, 10, 50, 1000])('R = %d M: d(alpha_edge)/dR from ray tangents matches the analytic derivative (relative 1e-9)', R => {
    const bc = Math.sqrt(27);
    // Newton on u for |b(u)| = sqrt(27): the shadow edge.
    let u = 0.5 * bc / R;
    for (let it = 0; it < 30; it++) {
      const { y0, S0 } = st.cameraRay(base(R, u), dirs);
      const { b, db } = impactParameter(y0, S0, 1);
      const f = Math.abs(b) - bc;
      u -= f / (Math.sign(b) * db);
      if (Math.abs(f) < 1e-15) break;
    }
    const { y0, S0 } = st.cameraRay(base(R, u), dirs);
    const bR = impactParameter(y0, S0, 0).db, bu = impactParameter(y0, S0, 1).db;
    const measured = -bR / bu / (1 + u * u); // alpha = atan(u)

    const s = Math.sqrt(1 - 2 / R);
    const sinA = (bc * s) / R;
    const dSin = bc * (-s / (R * R) + 1 / (R * R * R * s));
    // Inside the photon sphere (R < 3M) alpha = pi - asin(...)
    const sign = R >= 3 ? 1 : -1;
    const want = (sign * dSin) / Math.sqrt(1 - sinA * sinA);
    expect(Math.abs(Math.atan(u) - (R >= 3 ? Math.asin(sinA) : Math.PI - Math.asin(sinA)))).toBeLessThan(1e-12);
    const rel = Math.abs(measured - want) / Math.abs(want);
    console.log(`shadow edge R = ${R}: d alpha/dR = ${measured.toExponential(10)}, analytic ${want.toExponential(10)}, relative ${rel.toExponential(2)}`);
    expect(rel).toBeLessThan(1e-9);
  });
});

describe('Kerr critical impact parameter against spin', () => {
  const st = tracer('kerr');
  // Equatorial camera; rays in the z = 0 plane stay there.
  const R = 100;
  const base = (a: number, u: number): RayInputs => ({ params: [1, a], camRadius: R, inclination: Math.PI / 2, yaw: 0, u, v: 0 });
  const dirs = [unit(7, 5), unit(7, 1)]; // u, a

  /** Analytic b_crit and d b_crit / da for prograde (L > 0) and retrograde (L < 0) photon orbits, M = 1. */
  const analytic = (a: number, prograde: boolean) => {
    if (prograde) {
      const t = Math.acos(-a);
      return { b: -a + 6 * Math.cos(t / 3), db: -1 - (2 * Math.sin(t / 3)) / Math.sqrt(1 - a * a) };
    }
    const t = Math.acos(a);
    return { b: -(a + 6 * Math.cos(t / 3)), db: -(1 + (2 * Math.sin(t / 3)) / Math.sqrt(1 - a * a)) };
  };

  /** The implicit-function estimate -(dphi/da|_b) / (dphi/db) from an escaping ray at b = b_c + delta. */
  function estimate(a: number, prograde: boolean, delta: number) {
    const { b: bc } = analytic(a, prograde);
    const target = bc + Math.sign(bc) * delta; // just outside: the ray escapes
    let u = target / R;
    for (let it = 0; it < 40; it++) {
      const { y0, S0 } = st.cameraRay(base(a, u), dirs);
      const { b, db } = impactParameter(y0, S0, 0);
      u -= (b - target) / db;
      if (Math.abs(b - target) < 1e-14 * Math.abs(target)) break;
    }
    const ray = st.trace(base(a, u), dirs, { stepFraction: 0.002 });
    expect(ray.fate).toBe('escaped');
    const bu = impactParameter(ray.y0, ray.S0, 0).db, ba = impactParameter(ray.y0, ray.S0, 1).db;
    const [x, y] = [ray.y[1], ray.y[2]];
    const dphi = (d: number) => (x * ray.S[8 * d + 2] - y * ray.S[8 * d + 1]) / (x * x + y * y);
    const phiB = dphi(0) / bu;
    const phiA = dphi(1) - phiB * ba;
    return -phiA / phiB;
  }

  it.each([
    [0.5, true],
    [0.5, false],
    [0.9, true],
    [0.9, false],
  ])('a = %s, prograde = %s: converges to d b_crit/da linearly in b - b_crit', (a, prograde) => {
    const want = analytic(a, prograde).db;
    const errs = [1e-3, 1e-4, 1e-5].map(delta => Math.abs(estimate(a, prograde, delta) - want) / Math.abs(want));
    console.log(
      `Kerr a = ${a} ${prograde ? 'prograde' : 'retrograde'}: d b_crit/da = ${want.toFixed(8)}; ` +
        `relative error at b - b_crit = 1e-3, 1e-4, 1e-5: ${errs.map(e => e.toExponential(2)).join(', ')}`,
    );
    expect(errs[2]).toBeLessThan(1e-3);
    expect(errs[2]).toBeLessThan(errs[0]);
  });
});

/** Port of rm::analytic::schwarzschild_sweep (native/src/rm/analytic.h): adaptive Simpson, M = 1. */
function integrate(f: (x: number) => number, a: number, b: number, tol = 1e-14): number {
  const step = (a: number, b: number, fa: number, fm: number, fb: number, whole: number, tol: number, depth: number): number => {
    const m = 0.5 * (a + b), lm = 0.5 * (a + m), rm = 0.5 * (m + b);
    const flm = f(lm), frm = f(rm);
    const left = ((m - a) / 6) * (fa + 4 * flm + fm), right = ((b - m) / 6) * (fm + 4 * frm + fb);
    if (depth <= 0 || Math.abs(left + right - whole) <= 15 * tol) return left + right + (left + right - whole) / 15;
    return step(a, m, fa, flm, fm, left, tol / 2, depth - 1) + step(m, b, fm, frm, fb, right, tol / 2, depth - 1);
  };
  const fa = f(a), fb = f(b), fm = f(0.5 * (a + b));
  return step(a, b, fa, fm, fb, ((b - a) / 6) * (fa + 4 * fm + fb), tol, 40);
}
function sweep(b: number, rIn: number, rOut: number): number {
  const F = (u: number) => 1 / (b * b) - u * u + 2 * u * u * u;
  let lo = 0, hi = 1 / 3;
  for (let i = 0; i < 200; i++) {
    const mid = 0.5 * (lo + hi);
    if (F(mid) > 0) lo = mid;
    else hi = mid;
  }
  const u0 = 0.5 * (lo + hi);
  const c0 = 2 * u0 - 6 * u0 * u0, c1 = 6 * u0 - 1, c2 = -2;
  const leg = (r: number) => integrate(t => 2 / Math.sqrt(c0 + t * t * (c1 + t * t * c2)), 0, Math.sqrt(u0 - 1 / r));
  return leg(rIn) + leg(rOut);
}

describe('Schwarzschild sweep angle against impact parameter', () => {
  const st = tracer('schwarzschild');
  const R = 1000;
  const base = (u: number): RayInputs => ({ params: [1], camRadius: R, inclination: Math.PI / 2, yaw: 0, u, v: 0 });
  const dirs = [unit(6, 4)]; // u

  it.each([6, 10, 30, 100])('b ~ %d M: d(sweep)/du matches the exact orbit integral (relative 1e-6)', bTarget => {
    let u = bTarget / R;
    for (let it = 0; it < 30; it++) {
      const { y0, S0 } = st.cameraRay(base(u), dirs);
      const { b, db } = impactParameter(y0, S0, 0);
      u -= (Math.abs(b) - bTarget) / (Math.sign(b) * db);
    }
    const ray = st.trace(base(u), dirs, { stepFraction: 0.002 });
    expect(ray.fate).toBe('escaped');
    const { b, db } = impactParameter(ray.y0, ray.S0, 0);
    const absB = Math.abs(b), dAbsB = Math.sign(b) * db;
    const [x, y] = [ray.y[1], ray.y[2]];
    const rOut = Math.hypot(x, y, ray.y[3]);
    const rIn = Math.hypot(ray.y0[1], ray.y0[2], ray.y0[3]);
    // Traced backwards, p_t > 0, so E = -p_t < 0 and b = L / E is the physical impact
    // parameter while the traced curve itself turns with sign(L) = -sign(b).
    const measured = (-Math.sign(b) * (x * ray.S[2] - y * ray.S[1])) / (x * x + y * y);
    // Exact: dS/du = dS/db db/du + dS/dr_out dr_out/du, with dphi/dr = 1 / (r^2 sqrt(F(1/r))) on the way out.
    const hb = 1e-6 * absB;
    const dSdb = (sweep(absB + hb, rIn, rOut) - sweep(absB - hb, rIn, rOut)) / (2 * hb);
    const uo = 1 / rOut;
    const dSdr = (uo * uo) / Math.sqrt(1 / (absB * absB) - uo * uo + 2 * uo * uo * uo);
    const drOut = (x * ray.S[1] + y * ray.S[2] + ray.y[3] * ray.S[3]) / rOut;
    const want = dSdb * dAbsB + dSdr * drOut;
    const rel = Math.abs(measured - want) / Math.abs(want);
    console.log(`Schwarzschild b = ${absB.toFixed(4)}: d sweep/du = ${measured.toExponential(10)}, exact ${want.toExponential(10)}, relative ${rel.toExponential(2)}`);
    expect(rel).toBeLessThan(1e-6);
  });
});
