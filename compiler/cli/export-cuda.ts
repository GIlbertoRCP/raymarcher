// Writes the C++ headers, catalog and test fixture for the native renderer:
//   npm run export-cuda
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { exportCuda } from '../src/export-cuda.ts';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const metricsDir = join(root, 'metrics');
const outDir = join(root, 'native', 'generated');

const sources: Record<string, string> = {};
for (const file of readdirSync(metricsDir).filter(f => f.endsWith('.st'))) {
  sources[file.slice(0, -3)] = readFileSync(join(metricsDir, file), 'utf8');
}

for (const [path, content] of exportCuda(sources)) {
  const target = join(outDir, path);
  mkdirSync(dirname(target), { recursive: true });
  writeFileSync(target, content);
  console.log(`wrote native/generated/${path}`);
}
