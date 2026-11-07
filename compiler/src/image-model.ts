// Differentiable image formation, JS reference (docs/phase-4-plan.md, step 4).
//
// Model coordinates. The camera sits at R = camRadius length scales; a sample
// at model sky position xi = (xi_x, xi_y), in units of M, is the ray with
// image-plane direction (u, v, 1) = (xi_x / R, xi_y / R, 1) in the camera's
// (right, up, forward) frame. MODELLING CHOICE: xi is defined at the camera's
// tangent plane, so it differs from asymptotic impact parameters by O(M/R)
// (0.1-0.2% at R = 1000 M); theta_g below is defined against xi.
//
// Observed image. With angular size theta_g (uas per M) and position angle PA
// (the projected spin axis, camera "up", turned from image +Y towards -X, the
// sky's north-through-east convention), a sample lands at X = theta_g Rot(PA) xi.
// The observed, beam-convolved image at pixel X_p is the quadrature
//   I_obs(X_p) = A theta_g^2 sum_s w_s B(X_p - theta_g Rot(PA) xi_s) I_M(xi_s)
// over fixed stratified samples xi_s with area weights w_s, where B is a unit-
// integral circular Gaussian and A the flux normalisation. Because xi_s are
// fixed, theta_g, PA and A enter only through the smooth kernel B: their
// derivatives are exact and need no spatial derivatives of I_M. The traced
// parameters (spin, inclination) enter through I_M(xi_s) and its tangents.
//
// Emission (I_M). Rays that are captured or escape carry nothing (black sky).
// A ray that crosses the disk inside [r_in, r_out] carries I = g^4 F(r) tau(r)
// (EmissionModel; thinDisk below), g = 1 / (p . u_gas) (p . u = 1 at the camera).

import type { JsMetric } from './backend-js.ts';
import { add, div, dotD, dual, mul, powD, scl, sqrtD, sub, type D } from './dual.ts';
import type { JsMetricJvp } from './tangent.ts';
import { TangentSpacetime, type RayInputs, type TangentTraceOptions } from './tangent-geodesic.ts';

export interface EmissionModel {
  name: string;
  /** What is a modelling choice, for labels. */
  description: string;
  /** Specific intensity at the disk crossing, from duals: r, redshift g, r_in, r_out. */
  intensity(r: D, g: D, rIn: D, rOut: D): D;
}

/**
 * MODELLING CHOICE. Optically thick thin disk, bolometric: I = g^4 F(r) tau(r),
 * with the renderer's peak-normalised Shakura-Sunyaev-like profile
 * F = (1 - sqrt(r_in/r)) (r_in/r)^3 / 0.05666 (zero at r_in) and a smooth outer
 * taper tau = (1 - s^2)^2, s = (r - r_t)/(r_out - r_t), from r_t = taperStart r_out
 * to r_out (C^1, zero at r_out), so the image is continuous at both edges.
 */
export function thinDisk(taperStart = 0.75): EmissionModel {
  return {
    name: 'thin-disk-g4',
    description: `optically thick thin disk, I = g^4 F(r) tau(r); F = (1 - sqrt(r_in/r)) (r_in/r)^3, taper (1 - s^2)^2 from ${taperStart} r_out to r_out`,
    intensity(r, g, rIn, rOut) {
      const P = r.length - 1;
      const one = dual(P, 1);
      const q = div(rIn, r);
      const F = scl(mul(sub(one, sqrtD(q)), powD(q, 3)), 1 / 0.05666);
      const rt = scl(rOut, taperStart);
      let tau = one;
      if (r[0] >= rOut[0]) tau = dual(P, 0);
      else if (r[0] > rt[0]) {
        const s = div(sub(r, rt), sub(rOut, rt));
        const w = sub(one, mul(s, s));
        tau = mul(w, w);
      }
      return mul(mul(powD(g, 4), F), tau);
    },
  };
}

/** A fixed set of model-sky samples: positions (M units) and area weights. */
export interface Samples {
  xi: Float64Array; // [2 s], [2 s + 1]
  w: Float64Array;
  count: number;
}

/**
 * Stratified samples over the square [-L, L]^2: an n x n grid of cells with k x k
 * jittered points per cell from a fixed seed, so the image is a deterministic,
 * smooth function of the parameters away from fate boundaries.
 */
export function stratifiedSamples(L: number, n: number, k = 1, seed = 1): Samples {
  let s = seed;
  const rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const count = n * n * k * k;
  const xi = new Float64Array(2 * count), w = new Float64Array(count);
  const cell = (2 * L) / n, sub = cell / k;
  let m = 0;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++)
      for (let b = 0; b < k; b++)
        for (let a = 0; a < k; a++) {
          xi[2 * m] = -L + i * cell + (a + rand()) * sub;
          xi[2 * m + 1] = -L + j * cell + (b + rand()) * sub;
          w[m] = sub * sub;
          m++;
        }
  return { xi, w, count };
}

export interface SampleIntensities {
  /** I_M per sample. */
  I: Float64Array;
  /** dI_M/dtheta along each direction: dI[P s + d]. */
  dI: Float64Array;
  /** 'escaped' | 'captured' | 'disk' | 'unfinished' per sample. */
  fate: string[];
  steps: Int32Array;
  /** Disk-plane crossings passed through before the end (with the fate, the sample's branch). */
  crossings: Int32Array;
  P: number;
}

/** Ray tracing plus emission for every sample, with tangents along `dirs` (theta space, see RayInputs). */
export class ImageModel {
  readonly st: TangentSpacetime;

  constructor(readonly metric: JsMetric, readonly jvp: JsMetricJvp, readonly emission: EmissionModel = thinDisk()) {
    this.st = new TangentSpacetime(metric, jvp);
  }

  /** Intensity (dual) of the ray crossing the disk at state y with tangents S. */
  diskIntensity(y: Float64Array, S: Float64Array, params: number[], dParams: Float64Array, P: number): D {
    const dx = new Float64Array(4 * P);
    for (let d = 0; d < P; d++) for (let i = 0; i < 4; i++) dx[4 * d + i] = S[8 * d + i];
    const out = new Float64Array(4), tout = new Float64Array(4 * P);
    this.jvp.disk!(y, params, P, dx, dParams, out, tout);
    const at = (j: number) => dual(P, out[j], Array.from({ length: P }, (_, d) => tout[4 * d + j]));
    const [r, omega, rIn, rOut] = [0, 1, 2, 3].map(at);
    const g = new Float64Array(16), dg = new Float64Array(64), tg = new Float64Array(16 * P), tdg = new Float64Array(64 * P);
    this.jvp.evaluate(y, params, P, dx, dParams, g, dg, tg, tdg);
    const G: D[] = [];
    for (let k = 0; k < 16; k++) G.push(dual(P, g[k], Array.from({ length: P }, (_, d) => tg[16 * d + k])));
    const X = [0, 1, 2, 3].map(i => dual(P, y[i], Array.from({ length: P }, (_, d) => S[8 * d + i])));
    const p = [0, 1, 2, 3].map(i => dual(P, y[4 + i], Array.from({ length: P }, (_, d) => S[8 * d + 4 + i])));
    const zero = dual(P, 0);
    // u ~ (1, -omega y, omega x, 0), as disk_gas_velocity in geodesic.h
    const v = [dual(P, 1), scl(mul(omega, X[2]), -1), mul(omega, X[1]), zero];
    const norm = dotD(G, v, v);
    if (!(norm[0] < 0)) return zero; // no circular orbit here
    const s = div(dual(P, 1), sqrtD(scl(norm, -1)));
    const u = v.map(c => mul(c, s));
    let pu = zero;
    for (let i = 0; i < 4; i++) pu = add(pu, mul(p[i], u[i]));
    const redshift = div(dual(P, 1), pu);
    return this.emission.intensity(r, redshift, rIn, rOut);
  }

  /**
   * I_M and its tangents at every sample, for a camera at `base` (u and v are
   * taken from the samples). `dirs` are directions in theta = (params...,
   * camRadius, inclination, yaw, u, v).
   */
  render(base: RayInputs, samples: Samples, dirs: number[][], options: TangentTraceOptions = {}): SampleIntensities {
    const P = dirs.length;
    const np = base.params.length;
    const dParams = new Float64Array(np * P);
    for (let d = 0; d < P; d++) for (let k = 0; k < np; k++) dParams[np * d + k] = dirs[d][k] ?? 0;
    const I = new Float64Array(samples.count), dI = new Float64Array(samples.count * P);
    const fate: string[] = [], steps = new Int32Array(samples.count), crossings = new Int32Array(samples.count);
    const R = base.camRadius;
    for (let m = 0; m < samples.count; m++) {
      const ray = this.st.trace({ ...base, u: samples.xi[2 * m] / R, v: samples.xi[2 * m + 1] / R }, dirs, { disk: true, ...options });
      fate.push(ray.fate);
      steps[m] = ray.steps;
      crossings[m] = ray.crossings;
      if (ray.fate !== 'disk') continue;
      const val = this.diskIntensity(ray.y, ray.S, base.params, dParams, P);
      I[m] = val[0];
      for (let d = 0; d < P; d++) dI[P * m + d] = val[1 + d];
    }
    return { I, dI, fate, steps, crossings, P };
  }
}

export interface ObservedImageSpec {
  /** n x n pixels of `pixel` uas, centred on the origin; row j = Y, column i = X. */
  n: number;
  pixel: number;
  /** Beam FWHM in uas. */
  beam: number;
}

export interface ObservedImage {
  image: Float64Array;
  /** jac[(P + 3) p + k]: derivative of pixel p along the P traced directions, then theta_g, PA, A. */
  jac: Float64Array;
  K: number;
}

/** The pixel centres of an observed image: X then Y per pixel. */
export function pixelCentres(spec: ObservedImageSpec): Float64Array {
  const c = new Float64Array(2 * spec.n * spec.n);
  for (let j = 0; j < spec.n; j++)
    for (let i = 0; i < spec.n; i++) {
      c[2 * (j * spec.n + i)] = (i + 0.5 - spec.n / 2) * spec.pixel;
      c[2 * (j * spec.n + i) + 1] = (j + 0.5 - spec.n / 2) * spec.pixel;
    }
  return c;
}

/**
 * I_obs at every pixel and its exact derivatives (see the header): along the
 * traced directions through dI, and along theta_g, PA and A through the kernel.
 */
export function observe(
  samples: Samples,
  model: SampleIntensities,
  thetaG: number,
  pa: number,
  A: number,
  spec: ObservedImageSpec,
): ObservedImage {
  const { P } = model;
  const K = P + 3;
  const npx = spec.n * spec.n;
  const centres = pixelCentres(spec);
  const sigma = spec.beam / (2 * Math.sqrt(2 * Math.log(2)));
  const s2 = sigma * sigma;
  const norm = 1 / (2 * Math.PI * s2);
  const c = Math.cos(pa), sn = Math.sin(pa);
  const image = new Float64Array(npx), jac = new Float64Array(npx * K);
  const scale = A * thetaG * thetaG;
  for (let m = 0; m < samples.count; m++) {
    const Im = model.I[m];
    let any = Im !== 0;
    for (let d = 0; d < P && !any; d++) any = model.dI[P * m + d] !== 0;
    if (!any) continue;
    const xx = samples.xi[2 * m], yy = samples.xi[2 * m + 1];
    // Rot(PA) xi and its PA derivative (Rot(PA + pi/2) xi)
    const rx = c * xx - sn * yy, ry = sn * xx + c * yy;
    const qx = -sn * xx - c * yy, qy = c * xx - sn * yy;
    const wm = samples.w[m];
    for (let p = 0; p < npx; p++) {
      const dx = centres[2 * p] - thetaG * rx, dy = centres[2 * p + 1] - thetaG * ry;
      const B = norm * Math.exp(-(dx * dx + dy * dy) / (2 * s2));
      const base = scale * wm * B;
      image[p] += base * Im;
      const o = K * p;
      for (let d = 0; d < P; d++) jac[o + d] += base * model.dI[P * m + d];
      // dB/dtheta_g = B (d . Rot xi) / s2;  dB/dPA = B (d . theta_g Rot' xi) / s2
      const gB = (dx * rx + dy * ry) / s2, pB = (thetaG * (dx * qx + dy * qy)) / s2;
      jac[o + P] += base * Im * (2 / thetaG + gB);
      jac[o + P + 1] += base * Im * pB;
      jac[o + P + 2] += (base / A) * Im;
    }
  }
  return { image, jac, K };
}
