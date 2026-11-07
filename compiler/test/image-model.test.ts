// Phase 4, step 4: differentiable image formation (compiler/src/image-model.ts).
//
//   1. Per-sample disk intensity I_M(xi) and its tangents along spin,
//      inclination and the image-plane position, against 4-point central finite
//      differences of the same model, for samples whose fate and step count do
//      not change under the FD step (fate boundaries are step 5's edge term).
//   2. The observed image's derivatives along theta_g, PA and A (through the
//      beam kernel) against finite differences, and the propagation of the
//      traced tangents through the quadrature.
//   3. Flux bookkeeping: the image sums to A theta_g^2 sum_s w_s I_M(xi_s).
//   4. Continuity: the intensity vanishes at r_in and at r_out.

import { describe, expect, it } from 'vitest';
import { catalog } from '../../metrics/index.ts';
import {
  ImageModel,
  compileJvpToJs,
  compileMetric,
  compileToJs,
  observe,
  stratifiedSamples,
  thinDisk,
  type RayInputs,
  type SampleIntensities,
} from '../src/index.ts';
import { dual } from '../src/dual.ts';

function model(id: keyof typeof catalog) {
  const metric = compileMetric(catalog[id]);
  return new ImageModel(compileToJs(metric), compileJvpToJs(metric), thinDisk());
}

const unit = (n: number, k: number) => Array.from({ length: n }, (_, i) => (i === k ? 1 : 0));

describe('per-sample disk intensity tangents', () => {
  it.each([
    [0.3, 'linear'],
    [0.94, 'linear'],
    [0.3, 'hermite'],
    [0.94, 'hermite'],
  ] as const)('Kerr a = %s, %s crossing: dI/d(a, i, u, v) against finite differences (relative 1e-6)', (a, crossing) => {
    const im = model('kerr');
    const base: RayInputs = { params: [1, a], camRadius: 200, inclination: (60 * Math.PI) / 180, yaw: 0, u: 0, v: 0 };
    // theta = (M, a, R, i, yaw, u, v): directions a, i, u, v
    const ks = [1, 3, 5, 6];
    const dirs = ks.map(k => unit(7, k));
    const samples = stratifiedSamples(18, 6, 1, 3);
    const opts = { stepFraction: 0.02, crossing };
    const res = im.render(base, samples, dirs, opts);
    // Errors are normalised per direction by the largest |dI| over the samples (some derivatives are near zero).
    const errs = ks.map(() => [] as number[]), scale = ks.map(() => 1e-300);
    let compared = 0, skipped = 0, disk = 0;
    for (let m = 0; m < samples.count; m++) {
      if (res.fate[m] !== 'disk') continue;
      disk++;
      const R = base.camRadius;
      const theta = [1, a, R, base.inclination, 0, samples.xi[2 * m] / R, samples.xi[2 * m + 1] / R];
      for (const [d, k] of ks.entries()) {
        const h = 1e-5 * Math.max(1, Math.abs(theta[k]));
        const one = (s: number) => {
          const t = theta.slice();
          t[k] += s * h;
          const inp: RayInputs = { params: [t[0], t[1]], camRadius: t[2], inclination: t[3], yaw: t[4], u: t[5], v: t[6] };
          const ray = im.st.trace(inp, [], { disk: true, ...opts });
          const I = ray.fate === 'disk' ? im.diskIntensity(ray.y, ray.S, inp.params, new Float64Array(0), 0)[0] : 0;
          return { I, fate: ray.fate, steps: ray.steps };
        };
        const r = [one(2), one(1), one(-1), one(-2)];
        if (r.some(x => x.fate !== 'disk' || x.steps !== res.steps[m])) {
          skipped++;
          continue;
        }
        const fd = (-r[0].I + 8 * r[1].I - 8 * r[2].I + r[3].I) / (12 * h);
        const got = res.dI[res.P * m + d];
        errs[d].push(Math.abs(got - fd));
        scale[d] = Math.max(scale[d], Math.abs(fd));
        compared++;
      }
    }
    const worst = Math.max(...errs.map((e, d) => Math.max(0, ...e) / scale[d]));
    console.log(
      `Kerr a = ${a}, ${crossing}: ${disk} disk samples, ${compared} (sample, input) pairs compared, ${skipped} skipped (fate or step count changed); worst error relative to the largest |dI| per input ${worst.toExponential(2)} (tolerance 1e-6)`,
    );
    expect(compared).toBeGreaterThan(40);
    expect(worst).toBeLessThan(1e-6);
  });

  it('vanishes at the inner and outer edges (continuous image)', () => {
    const em = thinDisk();
    const P = 1;
    const at = (r: number) => em.intensity(dual(P, r, [1]), dual(P, 1.2), dual(P, 6), dual(P, 20));
    expect(at(6)[0]).toBe(0);
    expect(at(20)[0]).toBe(0);
    expect(at(20 - 1e-9)[0]).toBeLessThan(1e-15);
    // C^1 at the taper start and at r_out: one-sided slopes agree.
    const slope = (r: number) => at(r)[1];
    expect(Math.abs(slope(15 - 1e-9) - slope(15 + 1e-9))).toBeLessThan(1e-6);
    // The taper's slope vanishes linearly at r_out: ~1e-10 at 1e-9 inside.
    expect(Math.abs(slope(20 - 1e-9))).toBeLessThan(1e-8);
  });
});

describe('observed image', () => {
  // Synthetic sample intensities: the kernel derivatives do not depend on how I_M was made.
  const samples = stratifiedSamples(20, 10, 2, 5);
  const P = 2;
  let s = 99;
  const rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
  const I = new Float64Array(samples.count).map(() => (rand() < 0.4 ? 0 : rand()));
  const dI = new Float64Array(samples.count * P).map(() => rand() - 0.5);
  const model: SampleIntensities = { I, dI, fate: [], steps: new Int32Array(0), crossings: new Int32Array(0), P };
  const spec = { n: 16, pixel: 5, beam: 20 };
  const thetaG = 3.8, pa = 0.6, A = 1.7;

  it('theta_g, PA and A derivatives match finite differences (relative 1e-8)', () => {
    const o = observe(samples, model, thetaG, pa, A, spec);
    const args = [thetaG, pa, A];
    let worst = 0;
    for (let k = 0; k < 3; k++) {
      const h = 1e-5 * Math.abs(args[k]);
      const at = (t: number) => {
        const a = args.slice();
        a[k] += t * h;
        return observe(samples, model, a[0], a[1], a[2], spec).image;
      };
      const [p2, p1, m1, m2] = [at(2), at(1), at(-1), at(-2)];
      let num = 0, den = 0;
      for (let p = 0; p < o.image.length; p++) {
        const fd = (-p2[p] + 8 * p1[p] - 8 * m1[p] + m2[p]) / (12 * h);
        num = Math.max(num, Math.abs(o.jac[o.K * p + P + k] - fd));
        den = Math.max(den, Math.abs(fd));
      }
      worst = Math.max(worst, num / den);
    }
    console.log(`observed image: theta_g, PA, A derivatives vs FD, worst relative ${worst.toExponential(2)} (tolerance 1e-8)`);
    expect(worst).toBeLessThan(1e-8);
  });

  it('propagates traced tangents linearly (exact for a linear perturbation of I_M)', () => {
    const o = observe(samples, model, thetaG, pa, A, spec);
    for (let d = 0; d < P; d++) {
      const t = 1e-3;
      const shifted: SampleIntensities = { ...model, I: I.map((v, m) => v + t * dI[P * m + d]) };
      const o2 = observe(samples, shifted, thetaG, pa, A, spec);
      let num = 0, den = 0;
      for (let p = 0; p < o.image.length; p++) {
        num = Math.max(num, Math.abs((o2.image[p] - o.image[p]) / t - o.jac[o.K * p + d]));
        den = Math.max(den, Math.abs(o.jac[o.K * p + d]));
      }
      expect(num / den).toBeLessThan(1e-9);
    }
  });

  it('conserves flux: the image sums to A theta_g^2 sum w I_M (relative 1e-6)', () => {
    // Field of view +-144 uas covers the rotated +-20 M square (corners at 20 sqrt(2) x 3.8 = 107 uas) plus the beam.
    const wide = { n: 96, pixel: 3, beam: 10 };
    const o = observe(samples, model, thetaG, pa, A, wide);
    const total = o.image.reduce((a, b) => a + b, 0) * wide.pixel * wide.pixel;
    let want = 0;
    for (let m = 0; m < samples.count; m++) want += samples.w[m] * I[m];
    want *= A * thetaG * thetaG;
    expect(Math.abs(total - want) / want).toBeLessThan(1e-6);
  });
});
