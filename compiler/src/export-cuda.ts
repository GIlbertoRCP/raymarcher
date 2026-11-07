// Everything `npm run export-cuda` writes to native/generated/, computed in
// memory so a test can check the committed files are current.
//
//   <id>.cuh                         one header per catalog metric (backend-cuda.ts)
//   catalog.json                     names, labels and parameter ranges for the viewer UI
//   fixtures/metric-agreement.json   JS-backend values at sample points; the native
//                                    test harness checks the C++ headers against them
//   fixtures/jvp-agreement.json      the same for the JVPs (tangent.ts), two directions per point
//   fixtures/image-samples.json      JS image-model disk intensities with d/d(a, i) (image-model.ts), Kerr only
//   fixtures/tangent-rays.json       JS tangent-tracer rays (tangent-geodesic.ts) with dy/dtheta along
//                                    every input, for the C++ tangent integrator (rm/tangent.h)

import { compileToCuda } from './backend-cuda.ts';
import { compileToJs } from './backend-js.ts';
import { compileMetric, type CompiledMetric } from './compile.ts';
import { compileJvpToJs } from './tangent.ts';
import { TangentSpacetime } from './tangent-geodesic.ts';
import { ImageModel, stratifiedSamples, thinDisk } from './image-model.ts';

export interface CatalogEntry {
  id: string;
  name: string;
  description: string;
  header: string;
  hasHorizon: boolean;
  hasDisk: boolean;
  params: { name: string; label: string; value: number; min: number; max: number }[];
}

/** Deterministic points at coordinate radius 5..40, in every octant. */
function samplePoints(count: number, seed: number): number[][] {
  let s = seed;
  const rand = () => (s = (s * 16807) % 2147483647) / 2147483647;
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

function fixtureFor(id: string, metric: CompiledMetric) {
  const js = compileToJs(metric);
  const defaults = metric.params.map(p => p.value);
  const shifted = metric.params.map(p => p.min + 0.7 * (p.max - p.min));
  const cases = [];
  for (const [k, params] of [defaults, shifted].entries()) {
    for (const x of samplePoints(16, 11 + k)) {
      const g = new Float64Array(16);
      const dg = new Float64Array(64);
      js.evaluate(x, params, g, dg);
      cases.push({
        params,
        x,
        g: Array.from(g),
        dg: Array.from(dg),
        horizon: js.horizon ? js.horizon(x, params) : 1,
        disk: js.disk
          ? [js.disk.radius(x, params), js.disk.omega(x, params), js.disk.inner(x, params), js.disk.outer(x, params)]
          : [0, 0, 0, 0],
        scale: js.scale(x, params),
      });
    }
  }
  return { id, cases };
}

/** Two directions per point: one along coordinates and parameters, one along parameters only. */
function jvpFixtureFor(id: string, metric: CompiledMetric) {
  const jvp = compileJvpToJs(metric);
  const np = metric.params.length;
  let s = 977;
  const rand = () => 2 * ((s = (s * 16807) % 2147483647) / 2147483647) - 1;
  const cases = [];
  const sets = [metric.params.map(p => p.value), metric.params.map(p => p.min + 0.95 * (p.max - p.min))];
  for (const [k, params] of sets.entries()) {
    for (const x of samplePoints(8, 31 + k)) {
      const P = 2;
      const dx = [rand(), rand(), rand(), rand(), 0, 0, 0, 0];
      const dp = [...Array.from({ length: np }, rand), ...Array.from({ length: np }, rand)];
      const g = new Float64Array(16), dg = new Float64Array(64);
      const tg = new Float64Array(16 * P), tdg = new Float64Array(64 * P);
      jvp.evaluate(x, params, P, dx, dp, g, dg, tg, tdg);
      const out = new Float64Array(4), tout = new Float64Array(4 * P);
      if (jvp.disk) jvp.disk(x, params, P, dx, dp, out, tout);
      cases.push({ params, x, dx, dp, tg: Array.from(tg), tdg: Array.from(tdg), disk: Array.from(out), tdisk: Array.from(tout) });
    }
  }
  return { id, cases };
}

/** Rays at R = 50 M, inclination 60 deg, traced with tangents along every input (theta = params, R, i, yaw, u, v). */
function tangentRayFixtureFor(id: string, metric: CompiledMetric) {
  const js = compileToJs(metric);
  const st = new TangentSpacetime(js, compileJvpToJs(metric));
  const np = metric.params.length;
  const n = np + 5;
  const dirs = Array.from({ length: n }, (_, k) => Array.from({ length: n }, (_, j) => (j === k ? 1 : 0)));
  const params = metric.params.map(p => p.value);
  const camRadius = 50;
  const options = {
    stepFraction: 0.02,
    minStep: 1e-4,
    maxSteps: 20000,
    escapeRadius: 1.01 * camRadius * js.scale([0, 0, 0, 0], params),
    disk: metric.disk !== null,
  };
  const rays = [];
  for (const [u, v] of [[0.3, 0.05], [-0.25, 0.2], [0.12, -0.18], [-0.05, -0.3], [0.2, 0.25], [0.02, 0.01]]) {
    const inputs = { params, camRadius, inclination: Math.PI / 3, yaw: 0.3, u, v };
    const theta = [...params, camRadius, inputs.inclination, inputs.yaw, u, v];
    const r = st.trace(inputs, dirs, options);
    rays.push({ theta, fate: r.fate, steps: r.steps, y0: Array.from(r.y0), S0: Array.from(r.S0), y: Array.from(r.y), S: Array.from(r.S) });
  }
  return { id, P: n, options, rays };
}

/** Kerr model-sky samples at R = 200 M, i = 60 deg: intensity and d/d(a, i), for rm/emission.h and the sample kernels. */
function imageSampleFixture(metric: CompiledMetric) {
  const im = new ImageModel(compileToJs(metric), compileJvpToJs(metric), thinDisk(0.75));
  const samples = stratifiedSamples(18, 6, 1, 3);
  const dirs = [1, 3].map(k => Array.from({ length: 7 }, (_, j) => (j === k ? 1 : 0)));
  const options = { stepFraction: 0.02, minStep: 1e-4, maxSteps: 20000, escapeRadius: 1.01 * 200 };
  const cases = [];
  for (const [a, crossing] of [[0.6, 'linear'], [0.94, 'linear'], [0.94, 'hermite']] as const) {
    const base = { params: [1, a], camRadius: 200, inclination: Math.PI / 3, yaw: 0, u: 0, v: 0 };
    const r = im.render(base, samples, dirs, { ...options, crossing });
    cases.push({
      crossing,
      theta: [1, a, 200, Math.PI / 3, 0, 0, 0],
      dirs,
      xi: Array.from(samples.xi),
      I: Array.from(r.I),
      dI: Array.from(r.dI),
      fate: r.fate,
      steps: Array.from(r.steps),
      crossings: Array.from(r.crossings),
    });
  }
  return { id: 'kerr', taperStart: 0.75, options, cases };
}

/** Maps paths relative to native/generated/ to file contents. */
export function exportCuda(sources: Record<string, string>): Map<string, string> {
  const files = new Map<string, string>();
  const entries: CatalogEntry[] = [];
  const fixtures = [];
  const jvpFixtures = [];
  const tangentFixtures = [];
  let imageFixture = null;
  for (const id of Object.keys(sources).sort()) {
    const metric = compileMetric(sources[id]);
    const header = `${id}.cuh`;
    files.set(header, compileToCuda(metric, id));
    entries.push({
      id,
      name: metric.name,
      description: metric.description,
      header,
      hasHorizon: metric.horizon !== null,
      hasDisk: metric.disk !== null,
      params: metric.params.map(({ name, label, value, min, max }) => ({ name, label, value, min, max })),
    });
    fixtures.push(fixtureFor(id, metric));
    jvpFixtures.push(jvpFixtureFor(id, metric));
    tangentFixtures.push(tangentRayFixtureFor(id, metric));
    if (id === 'kerr') imageFixture = imageSampleFixture(metric);
  }
  files.set('catalog.json', JSON.stringify({ metrics: entries }, null, 2) + '\n');
  files.set('fixtures/metric-agreement.json', JSON.stringify({ metrics: fixtures }) + '\n');
  files.set('fixtures/jvp-agreement.json', JSON.stringify({ metrics: jvpFixtures }) + '\n');
  files.set('fixtures/tangent-rays.json', JSON.stringify({ metrics: tangentFixtures }) + '\n');
  if (imageFixture) files.set('fixtures/image-samples.json', JSON.stringify(imageFixture) + '\n');
  return files;
}
