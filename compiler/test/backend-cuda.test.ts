import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { catalog } from '../../metrics/index.ts';
import { compileMetric, compileToCuda, exportCuda } from '../src/index.ts';

const generatedDir = join(dirname(fileURLToPath(import.meta.url)), '../../native/generated');

describe('CUDA backend', () => {
  const files = exportCuda(catalog);

  // The native build and its agreement test read the committed files, so a
  // metric or backend change must be followed by `npm run export-cuda`.
  it.each([...files.keys()])('native/generated/%s is up to date', path => {
    const committed = readFileSync(join(generatedDir, path), 'utf8').replace(/\r\n/g, '\n');
    expect(committed === files.get(path), `run npm run export-cuda`).toBe(true);
  });

  it('emits typed literals so float instantiations stay in single precision', () => {
    const source = compileToCuda(compileMetric(catalog.kerr), 'kerr');
    expect(source).toContain('namespace kerr {');
    expect(source).toMatch(/Real\(2\.0\)/);
    // Every numeric literal in an expression is wrapped in Real(...).
    const bareLiterals = source
      .split('\n')
      .filter(line => ['  const Real', '  g[', '  dg[', '  dy['].some(prefix => line.startsWith(prefix)))
      .map(line => line.replace(/Real\([^()]*\)/g, '').replace(/\[\d+\]/g, '').replace(/\b[A-Za-z_]\w*\b/g, ''))
      .filter(line => /(?<![A-Za-z_])\d/.test(line));
    expect(bareLiterals).toEqual([]);
  });

  it('lists parameters with their ranges in catalog.json', () => {
    const { metrics } = JSON.parse(files.get('catalog.json')!);
    const kerr = metrics.find((m: { id: string }) => m.id === 'kerr');
    expect(kerr.params.map((p: { name: string }) => p.name)).toEqual(['M', 'a']);
    expect(kerr.params[1]).toMatchObject({ min: 0, max: 0.99, value: 0.6 });
    expect(kerr.hasHorizon).toBe(true);
  });
});
