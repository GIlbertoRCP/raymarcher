// Times the WebGPU trace pass on scene files, for the CUDA vs WebGPU comparison
// in docs/benchmarks/phase2-3070.json.
//
//   npx vite-node validation/benchmark-web.ts [--size W H] [--frames N] [--headed] scenes/x.json ...
//
// Uses timestamp queries around the compute pass when Chrome exposes them.
// Writes validation/out/webgpu-timing/<scene>.json.
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { appOptions, withApp } from './browser.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, 'validation', 'out', 'webgpu-timing');
const args = process.argv.slice(2);
const opts = appOptions(root, args);
const sizeAt = args.indexOf('--size');
const size = sizeAt >= 0 ? [Number(args[sizeAt + 1]), Number(args[sizeAt + 2])] : null;
const framesAt = args.indexOf('--frames');
const frames = framesAt >= 0 ? Number(args[framesAt + 1]) : 20;
const scenes = args.filter(a => a.endsWith('.json'));

await withApp(opts, async page => {
  mkdirSync(outDir, { recursive: true });
  for (const path of scenes) {
    const scene = JSON.parse(readFileSync(join(root, path), 'utf8'));
    if (size) scene.resolution = size;
    const { ms, method } = (await page.evaluate(([s, f]) => (window as any).raymarcherTime(s, f), [scene, frames] as const)) as {
      ms: number[];
      method: string;
    };
    const sorted = [...ms].sort((a, b) => a - b);
    const result = {
      backend: 'webgpu',
      scene: scene.name,
      width: scene.resolution[0],
      height: scene.resolution[1],
      browser: `${opts.channel} ${opts.headed ? 'headed' : 'headless'}`,
      userAgent: await page.evaluate(() => navigator.userAgent),
      method,
      frames,
      trace_ms_median: sorted[Math.floor(sorted.length / 2)],
      trace_ms_min: sorted[0],
      trace_ms_max: sorted[sorted.length - 1],
    };
    writeFileSync(join(outDir, `${scene.name}-${scene.resolution[0]}x${scene.resolution[1]}.json`), JSON.stringify(result, null, 2) + '\n');
    console.log(`${scene.name.padEnd(28)} ${result.width}x${result.height}  ${result.trace_ms_median.toFixed(3)} ms median (${result.trace_ms_min.toFixed(3)}-${result.trace_ms_max.toFixed(3)}), ${method}`);
  }
});
