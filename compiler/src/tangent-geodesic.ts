// Tangent (forward-mode) RK4 ray tracer in double precision: the JS reference
// for Phase 4 (docs/phase-4-plan.md, step 2).
//
// A ray is a function of its inputs theta: the metric parameters, the camera
// (radius in units of the length scale, inclination, yaw) and the image-plane
// direction (u, v). Alongside the state y = (x^mu, p_mu) this tracer carries
// S = dy/dtheta along P directions in theta space, S[8 d + i] = dy_i along
// direction d. "Discretise, then differentiate": S is the exact derivative of
// the integrator's own output (every RK4 stage, the state-dependent step size
// and the disk-crossing interpolation), not of the continuous geodesic, so
// central finite differences of the same tracer agree with it to their own
// truncation error as long as the step count and the fate do not change.
//
// The integrator mirrors trace_ray_rk4 in native/src/rm/geodesic.h (fixed
// step h = max(min_step, step_fraction r), escape beyond escape_radius,
// captured inside the horizon or when |dt/dlambda| >= 1e4, disk at the first
// z = 0 crossing inside [inner, outer]), and the camera mirrors static_camera
// (orbit position, static observer, Gram-Schmidt on forward, +z, right), so
// step 3 can compare the CUDA tangents with these.
//
// Tangent of the right-hand side, with k = g^{-1} p:
//   dg^{-1} = -g^{-1} dg g^{-1}
//   dk      = dg^{-1} p + g^{-1} dp
//   d(dp_c) = 1/2 d(d_c g_ab) k^a k^b + d_c g_ab k^a dk^b

import type { JsMetric } from './backend-js.ts';
import { invert4 } from './geodesic.ts';
import type { JsMetricJvp } from './tangent.ts';
import { add, cosD, div, dotD, dual, mul, scl, sinD, sqrtD, sub, type D } from './dual.ts';

/** Camera and image-plane inputs, after the metric parameters, in a theta vector. */
export const CAMERA_INPUTS = ['camRadius', 'inclination', 'yaw', 'u', 'v'] as const;

export interface RayInputs {
  /** Metric parameters. */
  params: number[];
  /** Camera distance from the origin in units of the metric's length scale. */
  camRadius: number;
  /** Angle between the line of sight and the +z (spin) axis, in (0, pi). pitch = pi/2 - inclination. */
  inclination: number;
  yaw: number;
  /** Local ray direction (u, v, 1) in the camera's (right, up, forward) frame, before normalising. */
  u: number;
  v: number;
}

export interface TangentTraceOptions {
  /** Disk-crossing interpolation inside an RK4 step (as rm::Crossing): linear (default) or cubic Hermite. */
  crossing?: 'linear' | 'hermite';
  stepFraction?: number;
  minStep?: number;
  maxSteps?: number;
  /** Coordinate radius beyond which a ray has escaped (default 1.01 x the camera's). */
  escapeRadius?: number;
  disk?: boolean;
}

export type Fate = 'captured' | 'escaped' | 'disk' | 'unfinished';

export interface TangentRay {
  fate: Fate;
  steps: number;
  /** Initial and final state; for 'disk' the final state is the crossing point. */
  y0: Float64Array;
  y: Float64Array;
  /** Disk-plane crossings passed through (outside [inner, outer]) before the end. */
  crossings: number;
  /** Their tangents: S[8 d + i]. */
  S0: Float64Array;
  S: Float64Array;
}

/** The theta inputs as duals: component d of the tangent is direction d. */
function inputDuals(inputs: RayInputs, dirs: number[][]) {
  const P = dirs.length;
  const np = inputs.params.length;
  const at = (k: number, v: number) => dual(P, v, dirs.map(d => d[k] ?? 0));
  return {
    params: inputs.params.map((v, k) => at(k, v)),
    camRadius: at(np, inputs.camRadius),
    inclination: at(np + 1, inputs.inclination),
    yaw: at(np + 2, inputs.yaw),
    u: at(np + 3, inputs.u),
    v: at(np + 4, inputs.v),
  };
}

export class TangentSpacetime {
  constructor(readonly metric: JsMetric, readonly jvp: JsMetricJvp) {}

  /**
   * Initial state y0 and its tangents S0 for a camera ray, mirroring
   * static_camera + camera_ray (native/src/rm/geodesic.h). `dirs[d]` is a
   * direction in theta = (params..., camRadius, inclination, yaw, u, v).
   */
  cameraRay(inputs: RayInputs, dirs: number[][]): { y0: Float64Array; S0: Float64Array; dParams: Float64Array } {
    const P = dirs.length;
    const np = inputs.params.length;
    const th = inputDuals(inputs, dirs);
    const dParams = new Float64Array(np * P);
    for (let d = 0; d < P; d++) for (let k = 0; k < np; k++) dParams[np * d + k] = dirs[d][k] ?? 0;

    const origin = [0, 0, 0, 0];
    const lsT = new Float64Array(P);
    const ls = this.jvp.scale(origin, inputs.params, P, new Float64Array(4 * P), dParams, lsT);
    const radius = mul(th.camRadius, dual(P, ls, lsT));
    const pitch = sub(dual(P, Math.PI / 2), th.inclination);
    const cp = cosD(pitch);
    const pos3 = [mul(radius, mul(cp, sinD(th.yaw))), mul(radius, mul(cp, cosD(th.yaw))), mul(radius, sinD(pitch))];
    const dl = sqrtD(add(add(mul(pos3[0], pos3[0]), mul(pos3[1], pos3[1])), mul(pos3[2], pos3[2])));
    const dir = pos3.map(c => scl(div(c, dl), -1));
    const zero = dual(P, 0), one = dual(P, 1);
    const up = [zero, zero, one];
    const right = [
      sub(mul(dir[1], up[2]), mul(dir[2], up[1])),
      sub(mul(dir[2], up[0]), mul(dir[0], up[2])),
      sub(mul(dir[0], up[1]), mul(dir[1], up[0])),
    ];
    const pos = [zero, ...pos3];

    // Metric at the camera and its tangent along (d pos, d params).
    const x = pos.map(c => c[0]);
    const dx = new Float64Array(4 * P);
    for (let d = 0; d < P; d++) for (let i = 0; i < 4; i++) dx[4 * d + i] = pos[i][1 + d];
    const g = new Float64Array(16), dg = new Float64Array(64), tg = new Float64Array(16 * P), tdg = new Float64Array(64 * P);
    this.jvp.evaluate(x, inputs.params, P, dx, dParams, g, dg, tg, tdg);
    if (!(g[0] < 0)) throw new Error('No static observer at the camera');
    const G: D[] = [];
    for (let k = 0; k < 16; k++) G.push(dual(P, g[k], Array.from({ length: P }, (_, d) => tg[16 * d + k])));

    const e0 = [div(one, sqrtD(scl(G[0], -1))), zero, zero, zero];
    const spatial: D[][] = [];
    for (const c of [dir, up, right]) {
      let w = [zero, c[0], c[1], c[2]];
      const along0 = dotD(G, w, e0);
      w = w.map((wi, i) => add(wi, mul(along0, e0[i]))); // e0 . e0 = -1
      for (const e of spatial) {
        const along = dotD(G, w, e);
        w = w.map((wi, i) => sub(wi, mul(along, e[i])));
      }
      const norm = sqrtD(dotD(G, w, w));
      spatial.push(w.map(wi => div(wi, norm)));
    }
    const frame = [e0, spatial[2], spatial[1], spatial[0]]; // e0, right, up, forward
    const lowered = frame.map(e => [0, 1, 2, 3].map(mu => {
      let s = zero;
      for (let nu = 0; nu < 4; nu++) s = add(s, mul(G[4 * mu + nu], e[nu]));
      return s;
    }));

    const len = sqrtD(add(add(mul(th.u, th.u), mul(th.v, th.v)), one));
    const du = div(th.u, len), dv = div(th.v, len), dz = div(one, len);
    const yD: D[] = [];
    for (let mu = 0; mu < 4; mu++) yD.push(pos[mu]);
    for (let mu = 0; mu < 4; mu++) {
      yD.push(add(add(add(scl(lowered[0][mu], -1), mul(du, lowered[1][mu])), mul(dv, lowered[2][mu])), mul(dz, lowered[3][mu])));
    }
    const y0 = new Float64Array(8), S0 = new Float64Array(8 * P);
    for (let i = 0; i < 8; i++) {
      y0[i] = yD[i][0];
      for (let d = 0; d < P; d++) S0[8 * d + i] = yD[i][1 + d];
    }
    return { y0, S0, dParams };
  }

  /**
   * dy/dlambda and its tangents. `dParams[np d + k]` is the metric-parameter
   * part of direction d. Returns dt/dlambda's primal for the capture test via dy.
   */
  rhs(y: Float64Array, S: Float64Array, params: ArrayLike<number>, dParams: Float64Array, P: number, dy: Float64Array, dS: Float64Array, w: Work) {
    const { g, dg, tg, tdg, gi, k, dx } = w;
    for (let d = 0; d < P; d++) for (let i = 0; i < 4; i++) dx[4 * d + i] = S[8 * d + i];
    this.jvp.evaluate(y, params, P, dx, dParams, g, dg, tg, tdg);
    invert4(g, gi);
    for (let mu = 0; mu < 4; mu++) {
      k[mu] = gi[4 * mu] * y[4] + gi[4 * mu + 1] * y[5] + gi[4 * mu + 2] * y[6] + gi[4 * mu + 3] * y[7];
      dy[mu] = k[mu];
    }
    for (let c = 0; c < 4; c++) {
      let s = 0;
      for (let a = 0; a < 4; a++) for (let b = 0; b < 4; b++) s += dg[16 * c + 4 * a + b] * k[a] * k[b];
      dy[4 + c] = 0.5 * s;
    }
    const { m1, dk } = w;
    for (let d = 0; d < P; d++) {
      // m1 = dg g^{-1}; d(g^{-1}) p = -g^{-1} (dg (g^{-1} p)) = -g^{-1} (dg k)
      for (let a = 0; a < 4; a++) {
        let s = 0;
        for (let b = 0; b < 4; b++) s += tg[16 * d + 4 * a + b] * k[b];
        m1[a] = s - S[8 * d + 4 + a]; // dg k - dp
      }
      for (let mu = 0; mu < 4; mu++) {
        dk[mu] = -(gi[4 * mu] * m1[0] + gi[4 * mu + 1] * m1[1] + gi[4 * mu + 2] * m1[2] + gi[4 * mu + 3] * m1[3]);
        dS[8 * d + mu] = dk[mu];
      }
      for (let c = 0; c < 4; c++) {
        let s = 0, t = 0;
        for (let a = 0; a < 4; a++) {
          for (let b = 0; b < 4; b++) {
            s += tdg[64 * d + 16 * c + 4 * a + b] * k[a] * k[b];
            t += dg[16 * c + 4 * a + b] * k[a] * dk[b];
          }
        }
        dS[8 * d + 4 + c] = 0.5 * s + t;
      }
    }
  }

  /** Traces a camera ray and its tangents along `dirs` with RK4. */
  trace(inputs: RayInputs, dirs: number[][], options: TangentTraceOptions = {}): TangentRay {
    const { y0, S0, dParams } = this.cameraRay(inputs, dirs);
    return this.traceState(y0, S0, inputs.params, dParams, dirs.length, {
      escapeRadius: 1.01 * Math.hypot(y0[1], y0[2], y0[3]),
      ...options,
    });
  }

  /** Traces from a given state and tangent; mirrors trace_ray_rk4. */
  traceState(
    y0: Float64Array,
    S0: Float64Array,
    params: ArrayLike<number>,
    dParams: Float64Array,
    P: number,
    options: TangentTraceOptions,
  ): TangentRay {
    const { stepFraction = 0.01, minStep = 1e-4, maxSteps = 200_000, disk = false } = options;
    const escapeRadius = options.escapeRadius ?? 2 * Math.hypot(y0[1], y0[2], y0[3]);
    const w = new Work(P);
    const y = Float64Array.from(y0), S = Float64Array.from(S0);
    const k1 = new Float64Array(8), k2 = new Float64Array(8), k3 = new Float64Array(8), k4 = new Float64Array(8);
    const s1 = new Float64Array(8 * P), s2 = new Float64Array(8 * P), s3 = new Float64Array(8 * P), s4 = new Float64Array(8 * P);
    const ty = new Float64Array(8), tS = new Float64Array(8 * P);
    const ny = new Float64Array(8), nS = new Float64Array(8 * P);
    const dh = new Float64Array(P);
    let crossings = 0;
    const done = (fate: Fate, steps: number): TangentRay => ({ fate, steps, crossings, y0, S0, y, S });
    const hermiteCrossing = options.crossing === 'hermite';
    const k5 = new Float64Array(8), s5 = new Float64Array(8 * P);
    const horizon = this.metric.horizon;
    const diskFns = this.metric.disk;

    for (let step = 0; step < maxSteps; step++) {
      if (horizon && horizon(y, params) < 0) return done('captured', step);
      const r = Math.hypot(y[1], y[2], y[3]);
      if (r > escapeRadius) return done('escaped', step);
      let h = stepFraction * r;
      // dh/dtheta: h = step_fraction * r unless clamped to min_step.
      if (h < minStep) {
        h = minStep;
        dh.fill(0);
      } else {
        for (let d = 0; d < P; d++) dh[d] = (stepFraction * (y[1] * S[8 * d + 1] + y[2] * S[8 * d + 2] + y[3] * S[8 * d + 3])) / r;
      }

      this.rhs(y, S, params, dParams, P, k1, s1, w);
      if (!(Math.abs(k1[0]) < 1e4)) return done('captured', step);
      // Stage c: ty = y + c h kprev, tS = S + c (dh kprev + h sprev)
      const stage = (c: number, kp: Float64Array, sp: Float64Array) => {
        for (let i = 0; i < 8; i++) ty[i] = y[i] + c * h * kp[i];
        for (let d = 0; d < P; d++) for (let i = 0; i < 8; i++) tS[8 * d + i] = S[8 * d + i] + c * (dh[d] * kp[i] + h * sp[8 * d + i]);
      };
      stage(0.5, k1, s1);
      this.rhs(ty, tS, params, dParams, P, k2, s2, w);
      stage(0.5, k2, s2);
      this.rhs(ty, tS, params, dParams, P, k3, s3, w);
      stage(1, k3, s3);
      this.rhs(ty, tS, params, dParams, P, k4, s4, w);
      for (let i = 0; i < 8; i++) ny[i] = y[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
      for (let d = 0; d < P; d++) {
        for (let i = 0; i < 8; i++) {
          const o = 8 * d + i;
          const sumK = k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i];
          const sumS = s1[o] + 2 * s2[o] + 2 * s3[o] + s4[o];
          nS[o] = S[o] + (dh[d] / 6) * sumK + (h / 6) * sumS;
        }
      }

      if (disk && diskFns && y[3] * ny[3] <= 0 && y[3] !== ny[3]) {
        const hit = new Float64Array(8);
        const hitS = new Float64Array(8 * P);
        if (hermiteCrossing) {
          // Cubic Hermite through (y, h k1) and (ny, h k5) on t in [0, 1], as trace_ray_rk4 with
          // CROSSING_HERMITE: bisection on z, one Newton step; the root's tangent from the
          // implicit function theorem, dt = -dz(t) / z'(t).
          this.rhs(ny, nS, params, dParams, P, k5, s5, w);
          const H = (t: number, a0: number, b0: number, a1: number, b1: number) => {
            const t2 = t * t, t3 = t2 * t;
            return (2 * t3 - 3 * t2 + 1) * a0 + (t3 - 2 * t2 + t) * b0 + (3 * t2 - 2 * t3) * a1 + (t3 - t2) * b1;
          };
          const dH = (t: number, a0: number, b0: number, a1: number, b1: number) => {
            const t2 = t * t;
            return (6 * t2 - 6 * t) * a0 + (3 * t2 - 4 * t + 1) * b0 + (6 * t - 6 * t2) * a1 + (3 * t2 - 2 * t) * b1;
          };
          const comp = (i: number) => [y[i], h * k1[i], ny[i], h * k5[i]] as const;
          let lo = 0, hi = 1;
          const loPositive = y[3] > 0;
          for (let it = 0; it < 60; it++) {
            const th = 0.5 * (lo + hi);
            if (H(th, ...comp(3)) > 0 === loPositive) lo = th;
            else hi = th;
          }
          let th = 0.5 * (lo + hi);
          const z = H(th, ...comp(3)), zp = dH(th, ...comp(3));
          if (zp !== 0) th -= z / zp;
          for (let i = 0; i < 8; i++) hit[i] = H(th, ...comp(i));
          const zpt = dH(th, ...comp(3));
          for (let d = 0; d < P; d++) {
            const tc = (i: number) =>
              [S[8 * d + i], dh[d] * k1[i] + h * s1[8 * d + i], nS[8 * d + i], dh[d] * k5[i] + h * s5[8 * d + i]] as const;
            const dth = -H(th, ...tc(3)) / zpt;
            for (let i = 0; i < 8; i++) hitS[8 * d + i] = H(th, ...tc(i)) + dH(th, ...comp(i)) * dth;
          }
        } else {
          // hit = y + f (ny - y), f = z / (z - z_new)
          const z = y[3], zn = ny[3], den = z - zn;
          const f = z / den;
          for (let i = 0; i < 8; i++) hit[i] = y[i] + f * (ny[i] - y[i]);
          for (let d = 0; d < P; d++) {
            const dz = S[8 * d + 3], dzn = nS[8 * d + 3];
            const df = (z * dzn - zn * dz) / (den * den);
            for (let i = 0; i < 8; i++) {
              const o = 8 * d + i;
              hitS[o] = S[o] + df * (ny[i] - y[i]) + f * (nS[o] - S[o]);
            }
          }
        }
        const rd = diskFns.radius(hit, params);
        if (rd >= diskFns.inner(hit, params) && rd <= diskFns.outer(hit, params)) {
          S.set(hitS);
          y.set(hit);
          return done('disk', step + 1);
        }
        crossings++;
      }
      y.set(ny);
      S.set(nS);
    }
    return done('unfinished', maxSteps);
  }
}

/** Scratch arrays for one trace. */
class Work {
  readonly g = new Float64Array(16);
  readonly dg = new Float64Array(64);
  readonly gi = new Float64Array(16);
  readonly k = new Float64Array(4);
  readonly m1 = new Float64Array(4);
  readonly dk = new Float64Array(4);
  readonly tg: Float64Array;
  readonly tdg: Float64Array;
  readonly dx: Float64Array;
  constructor(P: number) {
    this.tg = new Float64Array(16 * Math.max(P, 1));
    this.tdg = new Float64Array(64 * Math.max(P, 1));
    this.dx = new Float64Array(4 * Math.max(P, 1));
  }
}
