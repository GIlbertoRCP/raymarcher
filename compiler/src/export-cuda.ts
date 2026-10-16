// Everything `npm run export-cuda` writes to native/generated/, computed in
// memory so a test can check the committed files are current.
//
//   <id>.cuh                         one header per catalog metric (backend-cuda.ts)
//   catalog.json                     names, labels and parameter ranges for the viewer UI
//   fixtures/metric-agreement.json   JS-backend values at sample points; the native
//                                    test harness checks the C++ headers against them

import { compileToCuda } from './backend-cuda.ts';
import { compileToJs } from './backend-js.ts';
import { compileMetric, type CompiledMetric } from './compile.ts';

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

/** Maps paths relative to native/generated/ to file contents. */
export function exportCuda(sources: Record<string, string>): Map<string, string> {
  const files = new Map<string, string>();
  const entries: CatalogEntry[] = [];
  const fixtures = [];
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
  }
  files.set('catalog.json', JSON.stringify({ metrics: entries }, null, 2) + '\n');
  files.set('fixtures/metric-agreement.json', JSON.stringify({ metrics: fixtures }) + '\n');
  return files;
}
