// Cross-backend gate (docs/phase-2-plan.md, step 4): compares the CUDA and
// WebGPU renders of every scene in validation/out/{cuda,webgpu}/.
//
//   npx vite-node validation/compare.ts [--a cuda] [--b webgpu] [--name cross-backend]
//
// --a / --b pick render sets under validation/out/ (for example cuda-fp64);
// --name names the result file and heatmap directory.
//
// For each scene: fate agreement (all pixels, and away from fate boundaries:
// the gate metric, threshold 99.5%), the fate confusion matrix, the
// captured-region (shadow) edge offset in pixels, and the difference between
// the two HDR images after a common display transform. Writes
// validation/results/cross-backend.json and one heatmap per scene to
// validation/results/cross-backend/<scene>.png:
//   left   CUDA fate map; red = fates differ away from a boundary, yellow = differ on a boundary
//   right  |CUDA - WebGPU| after tone mapping, x4
import { execSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { boundaryMask, fateAgreement, FATE_NAMES, imageDifference, shadowEdgeOffset, toneMap } from './compare-lib.ts';
import { encodePng } from './png.ts';

const GATE = 0.995;
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const out = join(root, 'validation', 'out');
const resultsDir = join(root, 'validation', 'results');
const arg = (flag: string, fallback: string) => {
  const i = process.argv.indexOf(flag);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};
const setA = arg('--a', 'cuda');
const setB = arg('--b', 'webgpu');
const resultName = arg('--name', 'cross-backend');
const heatmapDir = join(resultsDir, resultName);

interface Render {
  meta: Record<string, any>;
  fate: Uint8Array;
  hdr: Float32Array;
}

function load(backend: string, scene: string): Render | null {
  const prefix = join(out, backend, scene);
  if (!existsSync(`${prefix}.json`)) return null;
  const hdr = readFileSync(`${prefix}.hdr.f32`);
  return {
    meta: JSON.parse(readFileSync(`${prefix}.json`, 'utf8')),
    fate: new Uint8Array(readFileSync(`${prefix}.fate.u8`)),
    hdr: new Float32Array(hdr.buffer, hdr.byteOffset, hdr.length / 4),
  };
}

const FATE_SHADE: [number, number, number][] = [
  [38, 52, 78], // escaped
  [0, 0, 0], // captured
  [120, 96, 64], // disk
  [110, 40, 110], // unfinished
];

function heatmap(cuda: Render, web: Render, w: number, h: number, a: Uint8Array, b: Uint8Array): Buffer {
  const mask = boundaryMask(cuda.fate, web.fate, w, h);
  const rgb = new Uint8Array(w * 2 * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const left = (y * 2 * w + x) * 3, right = (y * 2 * w + w + x) * 3;
      let c = FATE_SHADE[Math.min(cuda.fate[i], 3)];
      if (cuda.fate[i] !== web.fate[i]) c = mask[i] ? [235, 200, 60] : [255, 40, 30];
      rgb.set(c, left);
      let m = 0;
      for (let k = 0; k < 3; k++) m = Math.max(m, Math.abs(a[3 * i + k] - b[3 * i + k]));
      const v = Math.min(255, m * 4);
      rgb.set([v, Math.round(v * 0.45), Math.round(v * 0.1)], right);
    }
  }
  return encodePng(2 * w, h, rgb);
}

const scenes = existsSync(join(out, setA))
  ? readdirSync(join(out, setA)).filter(f => f.endsWith('.json')).map(f => f.slice(0, -5)).sort()
  : [];
if (!scenes.length) throw new Error(`No renders in validation/out/${setA}`);

mkdirSync(heatmapDir, { recursive: true });
const rows = [];
let missing = 0;
for (const scene of scenes) {
  const cuda = load(setA, scene), web = load(setB, scene);
  if (!cuda || !web) {
    console.log(`${scene}: missing in ${setB}, skipped`);
    missing++;
    continue;
  }
  const { width: w, height: h } = cuda.meta;
  if (web.meta.width !== w || web.meta.height !== h) throw new Error(`${scene}: image sizes differ`);
  const agreement = fateAgreement(cuda.fate, web.fate, w, h);
  const edge = shadowEdgeOffset(cuda.fate, web.fate, w, h);
  const a = toneMap(cuda.hdr), b = toneMap(web.hdr);
  const image = imageDifference(a, b);
  writeFileSync(join(heatmapDir, `${scene}.png`), heatmap(cuda, web, w, h, a, b));
  const pass = agreement.interior >= GATE;
  rows.push({
    scene,
    width: w,
    height: h,
    pass,
    fate_agreement_interior: agreement.interior,
    fate_agreement_all: agreement.all,
    interior_pixels: agreement.interiorPixels,
    boundary_pixels: agreement.boundaryPixels,
    confusion_cuda_rows_webgpu_columns: { order: FATE_NAMES, counts: agreement.confusion },
    shadow_edge_offset_px: edge && { mean: edge.meanPx, p99: edge.p99Px, max: edge.maxPx, edge_pixels: edge.edgePixels },
    image_difference: image,
    cuda_trace_ms: cuda.meta.trace_ms_median,
    cuda_constraint_drift_p99: cuda.meta.constraint_drift_uncaptured?.p99,
  });
  console.log(
    `${pass ? 'PASS' : 'FAIL'} ${scene.padEnd(26)} fate agreement ${(100 * agreement.interior).toFixed(3)}% interior, ${(100 * agreement.all).toFixed(3)}% all` +
      `  shadow edge ${edge ? `${edge.meanPx.toFixed(3)} px mean, ${edge.maxPx.toFixed(2)} px max` : 'n/a'}` +
      `  image ${image.meanAbsLevels.toFixed(3)} levels, ${image.pctOver8.toFixed(3)}% > 8`,
  );
}

function describe(meta: Record<string, any>) {
  return meta.backend === 'webgpu'
    ? { backend: meta.backend, precision: meta.precision, browser: meta.browser, userAgent: meta.userAgent, adapter: meta.adapter }
    : { backend: meta.backend, precision: meta.precision, gpu: meta.gpu, driver: meta.driver, cuda_driver_api: meta.cuda_driver_api, nvrtc: meta.nvrtc };
}

const gitHash = (() => {
  try {
    return execSync('git rev-parse --short HEAD', { cwd: root }).toString().trim();
  } catch {
    return 'unknown';
  }
})();
const dirty = (() => {
  try {
    return execSync('git status --porcelain', { cwd: root }).toString().trim().length > 0;
  } catch {
    return false;
  }
})();
const first = rows.length ? load(setA, rows[0].scene)!.meta : {};
const firstWeb = rows.length ? load(setB, rows[0].scene)!.meta : {};
const worst = rows.reduce((m, r) => Math.min(m, r.fate_agreement_interior), 1);
const result = {
  check: `Fate agreement between render sets '${setA}' (${first.backend}, ${first.precision}) and '${setB}' (${firstWeb.backend}, ${firstWeb.precision}), same scene files`,
  gate: `fate agreement >= ${GATE * 100}% on every scene, excluding pixels within 1 px of a fate boundary (3x3 neighbourhood, either map)`,
  pass: rows.length > 0 && missing === 0 && rows.every(r => r.pass),
  worst_interior_agreement: worst,
  scenes_compared: rows.length,
  scenes_missing: missing,
  sets: { a: setA, b: setB },
  date: new Date().toISOString(),
  git: { commit: gitHash, uncommitted_changes: dirty },
  a: describe(first),
  b: describe(firstWeb),
  image_difference_note: 'HDR images compared after Reinhard x/(1+x) and gamma 2.2 to 8 bits (not the app display path, which adds bloom and grading). WebGPU stores the HDR image as rgba16float, so its values carry half-precision rounding.',
  scenes: rows,
};
writeFileSync(join(resultsDir, `${resultName}.json`), JSON.stringify(result, null, 2) + '\n');
console.log(`\n${result.pass ? 'GATE PASSED' : 'GATE FAILED'}: worst interior fate agreement ${(100 * worst).toFixed(3)}% over ${rows.length} scenes (threshold ${GATE * 100}%)`);
console.log(`Wrote validation/results/${resultName}.json and validation/results/${resultName}/*.png`);
if (!result.pass) process.exitCode = 1;
