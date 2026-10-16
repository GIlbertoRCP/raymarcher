// Renders scene files with the WebGPU app in Chrome and saves what the tracer
// computed, for comparison with the native renderer (validation/compare.ts).
//
//   npx vite-node validation/capture-web.ts [--headed] [--channel chrome|msedge] [scenes/x.json ...]
//
// Starts a Vite dev server, opens the app in Chrome through playwright-core
// (no bundled browser download; uses the installed Chrome) and calls the
// dev-only window.raymarcherCapture hook (web/src/main.ts) for each scene.
// Writes validation/out/webgpu/<scene>.{hdr.f32,fate.u8,json}, in the same
// layout as `raymarcher --render`.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { appOptions, withApp } from './browser.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, 'validation', 'out', 'webgpu');

const args = process.argv.slice(2);
const opts = appOptions(root, args);
const sceneArgs = args.filter((a, i) => a.endsWith('.json') && args[i - 1] !== '--channel');
const scenePaths = sceneArgs.length
  ? sceneArgs.map(p => join(root, p))
  : readdirSync(join(root, 'scenes')).filter(f => f.endsWith('.json')).sort().map(f => join(root, 'scenes', f));

interface CaptureResult {
  width: number;
  height: number;
  gpuMs: number;
  adapter: { vendor: string; architecture: string; description: string } | null;
  userAgent: string;
  hdr: string;
  fate: string;
}

await withApp(opts, async page => {
  mkdirSync(outDir, { recursive: true });
  for (const path of scenePaths) {
    const scene = JSON.parse(readFileSync(path, 'utf8'));
    const res = (await page.evaluate(s => (window as any).raymarcherCapture(s), scene)) as CaptureResult;
    const hdr = Buffer.from(res.hdr, 'base64');
    const fate = Buffer.from(res.fate, 'base64');
    const n = res.width * res.height;
    if (hdr.length !== n * 12 || fate.length !== n) throw new Error(`${scene.name}: unexpected capture size`);
    const counts = [0, 0, 0, 0];
    for (const f of fate) counts[Math.min(f, 3)]++;
    const prefix = join(outDir, scene.name);
    writeFileSync(`${prefix}.hdr.f32`, hdr);
    writeFileSync(`${prefix}.fate.u8`, fate);
    const meta = {
      backend: 'webgpu',
      scene: scene.name,
      scene_file: path.slice(root.length + 1).replace(/\\/g, '/'),
      precision: 'fp32',
      width: res.width,
      height: res.height,
      browser: `${opts.channel} ${opts.headed ? 'headed' : 'headless'}`,
      userAgent: res.userAgent,
      adapter: res.adapter,
      wall_ms_including_readback: Number(res.gpuMs.toFixed(2)),
      fate: { escaped: counts[0] / n, captured: counts[1] / n, disk: counts[2] / n, unfinished: counts[3] / n },
      files: ['hdr.f32', 'fate.u8'],
    };
    writeFileSync(`${prefix}.json`, JSON.stringify(meta, null, 2) + '\n');
    console.log(
      `${scene.name.padEnd(28)} ${res.width}x${res.height}  captured ${(100 * meta.fate.captured).toFixed(2)}% disk ${(100 * meta.fate.disk).toFixed(2)}% escaped ${(100 * meta.fate.escaped).toFixed(2)}% unfinished ${(100 * meta.fate.unfinished).toFixed(3)}%`,
    );
  }
});
