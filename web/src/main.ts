import type { CatalogName } from '../../metrics/index.ts';
import { orbitCamera, type Vec3 } from './camera.ts';
import { loadSpacetime } from './scene.ts';
import { defaultSettings, SPACETIMES, type Background } from './settings.ts';
import type { DiskEdgeMark, ShadowEdge } from './shadow.ts';
import type { TraceRequest } from './shadow.worker.ts';
import { verticalTan } from './uniforms.ts';
import { Viewer } from './viewer.ts';

const $ = <T extends Element = HTMLElement>(id: string) => document.getElementById(id) as unknown as T;
const SVG = 'http://www.w3.org/2000/svg';
const DEG = 180 / Math.PI;

const field = $('field');
const canvas = $<HTMLCanvasElement>('view');
const marks = $<SVGSVGElement>('marks');
const sweep = $('sweep');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

let fps = 0;
const viewer = new Viewer(canvas, defaultSettings(), { onFrame: f => (fps = f) });
viewer.animate = !reducedMotion;
const s = viewer.settings;

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, text?: string) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  if (text !== undefined) el.textContent = text;
  return el;
}

const fmt = (v: number, digits = 2) => v.toFixed(digits);

// --- Chart list and generated parameters ----------------------------------

function buildCharts() {
  const list = $('charts');
  SPACETIMES.forEach((name, i) => {
    const { metric } = loadSpacetime(name);
    const li = document.createElement('li');
    li.innerHTML = `<label><input type="radio" name="chart" value="${name}"><span class="num">${i + 1}</span><span class="mark"></span><span class="name"></span></label>`;
    li.querySelector('.name')!.textContent = metric.name;
    const input = li.querySelector('input')!;
    input.checked = name === s.spacetime;
    input.addEventListener('change', () => selectChart(name));
    list.append(li);
  });
}

async function selectChart(name: CatalogName) {
  await viewer.setSpacetime(name);
  buildParams();
  updateTitle();
}

function buildParams() {
  const box = $('params');
  box.replaceChildren();
  const { metric } = viewer.spacetime;
  metric.params.forEach((p, i) => {
    const label = document.createElement('label');
    label.className = 'slider';
    label.innerHTML = `<span class="slider-label"><span></span><output></output></span><input type="range">`;
    label.querySelector('.slider-label span')!.textContent = p.label;
    const input = label.querySelector('input')!;
    const output = label.querySelector('output')!;
    Object.assign(input, { min: p.min, max: p.max, step: (p.max - p.min) / 400, value: viewer.params()[i] });
    graduate(input);
    output.textContent = fmt(viewer.params()[i]);
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      viewer.setParam(i, v);
      output.textContent = fmt(v);
      updateTitle();
    });
    box.append(label);
  });
}

function updateTitle() {
  const { metric, name } = viewer.spacetime;
  $('plate-no').textContent = `Plate ${SPACETIMES.indexOf(name) + 1}`;
  $('chart-name').textContent = metric.name;
  $('chart-desc').textContent = metric.description;
  const params = viewer.params();
  $('chart-params').textContent = metric.params.map((p, i) => `${p.label} ${fmt(params[i])}`).join('   ·   ');
}

// --- Controls bound to settings -------------------------------------------

const ROUND_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 15, 20, 25, 50, 100, 250, 500, 1000, 2000, 5000];

/** Draws graduation ticks under a range input at round values of its own unit, ends included. */
function graduate(input: HTMLInputElement, unit?: number) {
  const min = parseFloat(input.min), max = parseFloat(input.max);
  const span = max - min;
  const step = unit ?? ROUND_STEPS.find(u => span / u <= 10) ?? span;
  const positions = new Set<number>([0, 1]);
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + 1e-9; v += step) positions.add((v - min) / span);
  const tick = (f: number) => `linear-gradient(var(--scale), var(--scale)) calc((100% - 1px) * ${f.toFixed(4)}) 13px / 1px 5px no-repeat`;
  input.style.background = [...positions].map(tick).join(', ') + ', linear-gradient(var(--scale), var(--scale)) 0 50% / 100% 1px no-repeat';
}

function bindSlider(id: string, get: () => number, set: (v: number) => void, show: (v: number) => string, unit?: number) {
  const input = $<HTMLInputElement>(id);
  graduate(input, unit);
  const out = $(`${id}-out`);
  input.value = String(get());
  out.textContent = show(get());
  input.addEventListener('input', () => {
    set(parseFloat(input.value));
    out.textContent = show(get());
    viewer.touch();
  });
  return () => {
    if (document.activeElement !== input) input.value = String(get());
    out.textContent = show(get());
  };
}

const syncDistance = bindSlider('distance', () => s.camRadius, v => (s.camRadius = v), v => `${fmt(v, 1)} M`, 10);
bindSlider('fov', () => s.fovDegrees, v => (s.fovDegrees = v), v => `${v.toFixed(0)}°`, 15);
bindSlider('disk-temp', () => s.diskTemp, v => (s.diskTemp = v), v => `${v.toFixed(0)} K`, 1000);
bindSlider('render-scale', () => s.renderScale, v => (s.renderScale = v), v => `${Math.round(v * 100)}%`, 0.25);

for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="sky"]')) {
  radio.checked = radio.value === s.background;
  radio.addEventListener('change', () => (s.background = radio.value as Background));
}

const diskBox = $<HTMLInputElement>('disk');
diskBox.checked = s.disk;
diskBox.addEventListener('change', () => (s.disk = diskBox.checked));

const redshiftBox = $<HTMLInputElement>('redshift');
redshiftBox.checked = s.redshift;
redshiftBox.addEventListener('change', () => {
  s.redshift = redshiftBox.checked;
  $('gscale').hidden = !s.redshift;
});

const orbitBox = $<HTMLInputElement>('orbit');
orbitBox.addEventListener('change', () => (viewer.autoRotate = orbitBox.checked ? 0.08 : 0));

canvas.addEventListener('keydown', e => {
  const step = e.shiftKey ? 0.15 : 0.05;
  const moves: Record<string, () => void> = {
    ArrowLeft: () => (s.camYaw += step),
    ArrowRight: () => (s.camYaw -= step),
    ArrowUp: () => (s.camPitch = Math.min(1.45, s.camPitch + step)),
    ArrowDown: () => (s.camPitch = Math.max(-1.45, s.camPitch - step)),
    '+': () => (s.camRadius = Math.max(4, s.camRadius / 1.1)),
    '=': () => (s.camRadius = Math.max(4, s.camRadius / 1.1)),
    '-': () => (s.camRadius = Math.min(80, s.camRadius * 1.1)),
  };
  const move = moves[e.key];
  if (move) {
    e.preventDefault();
    move();
    viewer.touch();
  }
});

// --- Margin scales: azimuth and elevation of the view, measured -----------

interface ViewGeometry {
  width: number;
  height: number;
  tanV: number;
  forward: Vec3;
  right: Vec3;
  up: Vec3;
}

function viewGeometry(): ViewGeometry {
  const cam = orbitCamera(1, s.camYaw, s.camPitch);
  return {
    width: field.clientWidth,
    height: field.clientHeight,
    tanV: verticalTan(s.fovDegrees, field.clientWidth / Math.max(1, field.clientHeight)),
    forward: cam.dir,
    right: cam.right,
    up: cam.up,
  };
}

function direction(g: ViewGeometry, x: number, y: number): Vec3 {
  const d = [0, 1, 2].map(i => g.forward[i] + x * g.right[i] + y * g.up[i]) as Vec3;
  const len = Math.hypot(...d);
  return [d[0] / len, d[1] / len, d[2] / len];
}

const NICE_STEPS = [1, 2, 5, 10, 15, 30, 45, 90];

/** Draws ticks where `angleAt(pixel)` crosses round values. */
function drawScale(el: SVGSVGElement, length: number, angleAt: (px: number) => number, horizontal: boolean, axisName: string) {
  el.replaceChildren();
  if (length < 20) return;
  const samples: number[] = [];
  for (let px = 0; px <= length; px++) samples.push(angleAt(px));
  // Unwrap so the angle is continuous across the 0/360 seam.
  for (let i = 1; i < samples.length; i++) {
    while (samples[i] - samples[i - 1] > 180) samples[i] -= 360;
    while (samples[i] - samples[i - 1] < -180) samples[i] += 360;
  }
  const degPerPx = Math.abs(samples[samples.length - 1] - samples[0]) / length;
  // Labelled ticks at least ~80px apart, minor ticks at least 6px apart.
  const major = NICE_STEPS.find(step => step / degPerPx >= 80) ?? 90;
  const minor = major / 5 / degPerPx >= 6 ? major / 5 : major / 2;
  const lo = Math.min(...samples), hi = Math.max(...samples);
  const size = horizontal ? el.clientHeight : el.clientWidth;

  for (let k = Math.ceil(lo / minor); k * minor <= hi; k++) {
    const v = k * minor;
    // Find the pixel where the sampled angle crosses v.
    let px = -1;
    for (let i = 1; i < samples.length; i++) {
      const a = samples[i - 1], b = samples[i];
      if ((a <= v && v <= b) || (b <= v && v <= a)) {
        px = i - 1 + (v - a) / (b - a || 1);
        break;
      }
    }
    if (px < 0) continue;
    const isMajor = Math.abs(v / major - Math.round(v / major)) < 1e-6;
    const tick = isMajor ? 9 : 4;
    const line = horizontal
      ? { x1: px, x2: px, y1: size - tick, y2: size }
      : { y1: px, y2: px, x1: size - tick, x2: size };
    el.append(svg('line', line));
    if (isMajor) {
      const digits = 0;
      const text = horizontal
        ? `${(((v % 360) + 360) % 360).toFixed(digits)}°`
        : `${v > 0 ? '+' : ''}${v.toFixed(digits)}°`;
      const attrs = horizontal
        ? { x: px, y: size - 13, 'text-anchor': 'middle' }
        : { x: size - 12, y: px + 3.5, 'text-anchor': 'end' };
      el.append(svg('text', attrs, text));
    }
  }
  const nameAttrs = horizontal ? { x: 0, y: 11 } : { x: 4, y: -8 };
  el.append(svg('text', { ...nameAttrs, class: 'axis-name' }, axisName));
}

let scaleKey = '';
function updateScales() {
  const g = viewGeometry();
  const key = [g.width, g.height, g.tanV, s.camYaw, s.camPitch].join();
  if (key === scaleKey) return;
  scaleKey = key;
  field.style.setProperty('--field-width', `${g.width}px`);
  const half = g.height / 2;
  const azimuth = (px: number) => {
    const d = direction(g, ((px - g.width / 2) / half) * g.tanV, 0);
    return Math.atan2(d[0], d[1]) * DEG;
  };
  const elevation = (px: number) => {
    const d = direction(g, 0, ((half - px) / half) * g.tanV);
    return Math.asin(d[2]) * DEG;
  };
  drawScale($<SVGSVGElement>('az-scale'), g.width, azimuth, true, 'Azimuth');
  drawScale($<SVGSVGElement>('el-scale'), g.height, elevation, false, 'Elev');
}

// --- Measured marks: the shadow edge traced on the CPU --------------------

let traceKey = '';
let traced: { key: string; edge: ShadowEdge | null; diskEdge: DiskEdgeMark | null } | null = null;
let tracingId = 0; // id of the trace in flight, 0 when idle
let traceProgress = '';
const worker = new Worker(new URL('./shadow.worker.ts', import.meta.url), { type: 'module' });
let pendingKey = '';

worker.onmessage = (e: MessageEvent<{ id: number; type: 'progress' | 'done'; done?: number; total?: number; edge?: ShadowEdge | null; diskEdge?: DiskEdgeMark | null }>) => {
  const msg = e.data;
  if (msg.id !== tracingId) return;
  if (msg.type === 'progress') {
    traceProgress = `tracing ${msg.done}/${msg.total}`;
    return;
  }
  tracingId = 0;
  traceProgress = '';
  traced = { key: pendingKey, edge: msg.edge ?? null, diskEdge: msg.diskEdge ?? null };
  if (pendingKey === traceKey) {
    drawMarks(traced.edge, traced.diskEdge);
    marks.classList.remove('stale');
  }
};

function currentTraceKey(): string {
  return [s.spacetime, s.disk, viewer.params().join(','), s.camYaw.toFixed(4), s.camPitch.toFixed(4), s.camRadius.toFixed(3), s.fovDegrees, field.clientWidth, field.clientHeight].join('|');
}

function predictedShadowDeg(): number | null {
  // Schwarzschild: sin(alpha) = sqrt(27) M sqrt(1 - 2M/R) / R for a static observer at areal radius R.
  if (s.spacetime !== 'schwarzschild') return null;
  const R = s.camRadius; // Kerr-Schild r is the areal radius, in units of M
  if (R <= 3) return null;
  return Math.asin((Math.sqrt(27) * Math.sqrt(1 - 2 / R)) / R) * DEG;
}

/**
 * A measured point with a leader running to a margin note in the field's dark
 * sky: right-aligned, stacked from the top (or from the bottom on narrow fields).
 */
function leader(point: number[], slot: number, title: string, subs: [string, string], g: ViewGeometry) {
  const narrow = g.width < 520;
  const sub = narrow ? subs[1] : subs[0];
  const x = g.width - 16;
  const y = narrow ? g.height - 26 - slot * 40 : 36 + slot * 44;
  // Right-angle leader: straight to the note's row, then along it to the note.
  marks.append(svg('circle', { class: 'dot', cx: point[0], cy: point[1], r: 2.5 }));
  marks.append(svg('path', { class: 'leader', d: `M${point[0]} ${point[1]}V${y + 4}H${x}` }));
  marks.append(svg('text', { x, y, 'text-anchor': 'end' }, title));
  marks.append(svg('text', { x, y: y + 17, 'text-anchor': 'end', class: 'sub' }, sub));
}

function drawMarks(edge: ShadowEdge | null, diskEdge: DiskEdgeMark | null) {
  marks.replaceChildren();
  const g = viewGeometry();
  const half = g.height / 2;
  const toPx = (x: number, y: number) => [g.width / 2 + (x / g.tanV) * half, half - (y / g.tanV) * half];

  const [cx, cy] = toPx(0, 0);
  marks.append(svg('path', { class: 'centre', d: `M${cx - 6} ${cy}h12M${cx} ${cy - 6}v12` }));
  if (!edge) return;

  const pts = edge.points.map(p => toPx(p.x, p.y));
  marks.append(svg('path', { class: 'edge', d: `M${pts.map(p => p.join(' ')).join('L')}Z` }));
  // Notes: the rightmost point takes the row nearest the image so leaders never cross.
  const notes: { point: number[]; title: string; subs: [string, string] }[] = [];
  const anchor = pts.reduce((best, p) => (p[0] - p[1] > best[0] - best[1] ? p : best));
  const predicted = predictedShadowDeg();
  const shadowSub: [string, string] = predicted
    ? [`CPU trace · √27 M predicts ${predicted.toFixed(2)}°`, `√27 M predicts ${predicted.toFixed(2)}°`]
    : s.spacetime === 'kerr'
      ? ['CPU trace · outline validated against Bardeen (1973) in tests', 'Bardeen (1973) outline, in tests']
      : [`CPU trace · ${edge.points.length} rays`, 'CPU trace'];
  notes.push({ point: anchor, title: `Shadow edge and photon ring  ${edge.meanRadiusDeg.toFixed(2)}°`, subs: shadowSub });
  if (diskEdge) {
    notes.push({
      point: toPx(diskEdge.x, diskEdge.y),
      title: `Disk inner edge  r = ${diskEdge.radius.toFixed(2)} M`,
      subs: ['Innermost stable orbit, seen over the hole', 'Innermost stable orbit'],
    });
  }
  notes.sort((a, b) => a.point[0] - b.point[0]);
  notes.forEach((n, slot) => leader(n.point, slot, n.title, n.subs, g));
}

function maybeTrace() {
  const key = currentTraceKey();
  if (key !== traceKey) {
    traceKey = key;
    tracingId = 0; // ignore any result for the old view
    marks.classList.add('stale');
  }
  if (traced?.key === key || tracingId || viewer.idleFor() < 400 || viewer.autoRotate) return;

  const g = viewGeometry();
  tracingId = Math.floor(Math.random() * 1e9) + 1;
  pendingKey = key;
  const request: TraceRequest = {
    id: tracingId,
    spacetime: s.spacetime,
    params: viewer.params(),
    camera: viewer.camera(),
    maxTan: g.tanV * Math.hypot(g.width / g.height, 1),
    disk: s.disk,
  };
  worker.postMessage(request);
}

// --- Readout ---------------------------------------------------------------

function updateReadout() {
  $('ro-frame').textContent = `${fps} fps · ${viewer.previewing ? 'preview' : 'full'}`;
  $('ro-image').textContent = `${canvas.width} × ${canvas.height}`;
  let shadow: string;
  if (!viewer.spacetime.js.horizon) shadow = 'no horizon';
  else if (tracingId) shadow = traceProgress || 'tracing';
  else if (traced && traced.key === traceKey) shadow = traced.edge ? `${traced.edge.meanRadiusDeg.toFixed(2)}° mean` : 'not in view';
  else shadow = 'waiting';
  $('ro-shadow').textContent = shadow;
  sweep.hidden = !(viewer.previewing || tracingId);
}

// --- Start -----------------------------------------------------------------

async function start() {
  if (!(await viewer.init())) {
    $('notice').hidden = false;
    return;
  }
  buildCharts();
  buildParams();
  updateTitle();
  viewer.start();
  const tick = () => {
    updateScales();
    syncDistance();
    maybeTrace();
    updateReadout();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

start().catch(e => {
  $('notice').hidden = false;
  $('notice').querySelector('p')!.textContent = e instanceof Error ? e.message : String(e);
});

// Handy for poking at the renderer from the console during development.
if (import.meta.env.DEV) Object.assign(window, { viewer });
