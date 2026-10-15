// Double-precision geodesic integrator for any compiled metric.
//
// A ray is the state y = (x^mu, p_mu): position and covariant momentum.
// Hamilton's equations with H = 1/2 g^{mu nu} p_mu p_nu give
//
//   dx^mu / dlambda = g^{mu nu} p_nu
//   dp_mu / dlambda = -1/2 (d_mu g^{ab}) p_a p_b = 1/2 (d_mu g_{ab}) k^a k^b,  k = g^{-1} p
//
// so only g and its first derivatives are needed. H = 0 for light, and its
// drift along a ray measures integration error.

import type { JsMetric } from './backend-js.ts';

export type Vec4 = [number, number, number, number];
export type Vec3 = [number, number, number];

/** Inverse of a 4x4 matrix stored row-major. */
export function invert4(m: ArrayLike<number>, out = new Float64Array(16)): Float64Array {
  const [a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23, a30, a31, a32, a33] = Array.from(m);
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (det === 0) throw new Error('Metric is singular');
  const s = 1 / det;
  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * s;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * s;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * s;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * s;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * s;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * s;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * s;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * s;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * s;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * s;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * s;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * s;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * s;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * s;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * s;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * s;
  return out;
}

export function dot4(g: ArrayLike<number>, a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) s += g[4 * i + j] * a[i] * b[j];
  return s;
}

export function lower(g: ArrayLike<number>, v: ArrayLike<number>): Vec4 {
  return [0, 1, 2, 3].map(i => g[4 * i] * v[0] + g[4 * i + 1] * v[1] + g[4 * i + 2] * v[2] + g[4 * i + 3] * v[3]) as Vec4;
}

export type RayFate = 'captured' | 'escaped' | 'disk' | 'unfinished';

export interface RayResult {
  fate: RayFate;
  /** Final state (x^mu, p_mu); for 'disk' this is the crossing point. */
  state: Float64Array;
  steps: number;
  /** max |H| / (p_t)^2 seen along the ray. */
  maxConstraintError: number;
}

export interface TraceOptions {
  /** Step length as a fraction of the coordinate distance from the origin. */
  stepFraction?: number;
  minStep?: number;
  maxStep?: number;
  maxSteps?: number;
  escapeRadius?: number;
  /** Stop at the disk plane (z = 0) when the crossing lies inside the disk. */
  disk?: boolean;
}

export class Spacetime {
  private readonly g = new Float64Array(16);
  private readonly dg = new Float64Array(64);
  private readonly gInv = new Float64Array(16);
  private readonly k = new Float64Array(4);

  constructor(readonly metric: JsMetric, readonly params: ArrayLike<number>) {}

  /** g_{mu nu} at x (a fresh copy). */
  metricAt(x: ArrayLike<number>): Float64Array {
    this.metric.evaluate(x, this.params, this.g, this.dg);
    return Float64Array.from(this.g);
  }

  /** Writes dy/dlambda for state y into out and returns H. */
  derivative(y: ArrayLike<number>, out: Float64Array): number {
    this.metric.evaluate(y, this.params, this.g, this.dg);
    invert4(this.g, this.gInv);
    const p0 = y[4], p1 = y[5], p2 = y[6], p3 = y[7];
    const gi = this.gInv;
    for (let mu = 0; mu < 4; mu++) {
      this.k[mu] = gi[4 * mu] * p0 + gi[4 * mu + 1] * p1 + gi[4 * mu + 2] * p2 + gi[4 * mu + 3] * p3;
      out[mu] = this.k[mu];
    }
    for (let c = 0; c < 4; c++) {
      out[4 + c] = 0.5 * dot4(this.dg.subarray(16 * c, 16 * c + 16), this.k, this.k);
    }
    return 0.5 * (this.k[0] * p0 + this.k[1] * p1 + this.k[2] * p2 + this.k[3] * p3);
  }

  hamiltonian(y: ArrayLike<number>): number {
    return this.derivative(y, new Float64Array(8));
  }

  insideHorizon(x: ArrayLike<number>): boolean {
    return this.metric.horizon !== null && this.metric.horizon(x, this.params) < 0;
  }

  /**
   * Orthonormal frame of a static observer at x (requires g_tt < 0):
   * e[0] is the 4-velocity, e[1] right, e[2] up and e[3] forward, built by
   * Gram-Schmidt from the coordinate directions.
   */
  staticFrame(x: ArrayLike<number>, forward: Vec3, up: Vec3): Vec4[] {
    const g = this.metricAt(x);
    if (g[0] >= 0) throw new Error('No static observer here (inside an ergoregion or horizon)');
    const e0: Vec4 = [1 / Math.sqrt(-g[0]), 0, 0, 0];
    const right: Vec3 = [
      forward[1] * up[2] - forward[2] * up[1],
      forward[2] * up[0] - forward[0] * up[2],
      forward[0] * up[1] - forward[1] * up[0],
    ];
    const frame: Vec4[] = [e0];
    const spatial: Vec4[] = [];
    for (const c of [forward, up, right]) {
      let w: Vec4 = [0, c[0], c[1], c[2]];
      const along0 = dot4(g, w, e0);
      w = w.map((wi, i) => wi + along0 * e0[i]) as Vec4; // e0 . e0 = -1
      for (const e of spatial) {
        const along = dot4(g, w, e);
        w = w.map((wi, i) => wi - along * e[i]) as Vec4;
      }
      const norm = Math.sqrt(dot4(g, w, w));
      spatial.push(w.map(wi => wi / norm) as Vec4);
    }
    const [eForward, eUp, eRight] = spatial;
    frame.push(eRight, eUp, eForward);
    return frame;
  }

  /**
   * Initial state for a ray leaving the observer in local direction d
   * (components along right, up, forward; need not be normalised).
   * The ray is traced backwards in time: k = -e0 + d.
   */
  rayFromFrame(x: ArrayLike<number>, frame: Vec4[], d: Vec3): Float64Array {
    const len = Math.hypot(d[0], d[1], d[2]);
    const k = frame[0].map((e0, mu) => -e0 + (d[0] * frame[1][mu] + d[1] * frame[2][mu] + d[2] * frame[3][mu]) / len);
    const p = lower(this.metricAt(x), k);
    return Float64Array.from([x[0], x[1], x[2], x[3], ...p]);
  }

  /** Integrates a ray with RK4 until it is captured, escapes or hits the disk. */
  trace(y0: ArrayLike<number>, options: TraceOptions = {}): RayResult {
    const {
      stepFraction = 0.01,
      minStep = 1e-4,
      maxStep = 1e6,
      maxSteps = 200_000,
      disk = false,
    } = options;
    const radius = (y: ArrayLike<number>) => Math.hypot(y[1], y[2], y[3]);
    const escapeRadius = options.escapeRadius ?? 2 * radius(y0);
    const y = Float64Array.from(y0);
    const k1 = new Float64Array(8), k2 = new Float64Array(8), k3 = new Float64Array(8), k4 = new Float64Array(8);
    const tmp = new Float64Array(8);
    const prev = new Float64Array(8);
    const pt0 = Math.abs(y[4]) || 1;
    this.derivative(y, k1);
    const kt0 = Math.abs(k1[0]) || 1; // initial dt/dlambda
    let maxConstraintError = 0;

    for (let steps = 0; steps < maxSteps; steps++) {
      if (this.insideHorizon(y)) return { fate: 'captured', state: y, steps, maxConstraintError };
      const r = radius(y);
      if (r > escapeRadius && steps > 0) return { fate: 'escaped', state: y, steps, maxConstraintError };

      const h = Math.min(maxStep, Math.max(minStep, stepFraction * r));
      const H = this.derivative(y, k1);
      maxConstraintError = Math.max(maxConstraintError, Math.abs(H) / (pt0 * pt0));
      // A ray traced backwards in time into a horizon never crosses it in
      // ingoing coordinates; dt/dlambda diverges instead. Count it as captured.
      if (!(Math.abs(k1[0]) < 1e4 * kt0)) return { fate: 'captured', state: y, steps, maxConstraintError };
      for (let i = 0; i < 8; i++) tmp[i] = y[i] + 0.5 * h * k1[i];
      this.derivative(tmp, k2);
      for (let i = 0; i < 8; i++) tmp[i] = y[i] + 0.5 * h * k2[i];
      this.derivative(tmp, k3);
      for (let i = 0; i < 8; i++) tmp[i] = y[i] + h * k3[i];
      this.derivative(tmp, k4);
      prev.set(y);
      for (let i = 0; i < 8; i++) y[i] += (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);

      if (disk && this.metric.disk && prev[3] * y[3] <= 0 && prev[3] !== y[3]) {
        const f = prev[3] / (prev[3] - y[3]);
        const hit = prev.map((v, i) => v + f * (y[i] - v));
        const rd = this.metric.disk.radius(hit, this.params);
        if (rd >= this.metric.disk.inner(hit, this.params) && rd <= this.metric.disk.outer(hit, this.params)) {
          return { fate: 'disk', state: hit, steps, maxConstraintError };
        }
      }
    }
    return { fate: 'unfinished', state: y, steps: maxSteps, maxConstraintError };
  }
}
