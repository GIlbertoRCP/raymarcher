// Verification page: every number here is read from the committed result
// files in validation/results/ and docs/benchmarks/, produced on the RTX 3070.

import benchmarks from '../../../docs/benchmarks/phase2-3070.json';
import constraintDrift from '../../../validation/results/constraint-drift.json';
import convergence from '../../../validation/results/convergence.json';
import crossBackend from '../../../validation/results/cross-backend.json';
import deflection from '../../../validation/results/deflection.json';
import fastmath from '../../../validation/results/fastmath-vs-webgpu.json';
import fp32fp64 from '../../../validation/results/fp32-vs-fp64.json';
import suite from '../../../validation/results/fp64-suite.json';
import orbits from '../../../validation/results/orbits.json';
import shadowKerr from '../../../validation/results/shadow-kerr.json';
import shadowSchw from '../../../validation/results/shadow-schwarzschild.json';
import { barChart, lineChart, sci } from './charts.ts';

const tiles = import.meta.glob('../../../validation/results/cross-backend/*.png', { eager: true, query: '?url', import: 'default' }) as Record<string, string>;

const RED = 'var(--red)';
const BLUE = 'var(--blue)';
const NEUTRAL = 'var(--plate-dim)';
const $ = (id: string) => document.getElementById(id)!;
const pct = (v: number, digits = 2) => `${(v * 100).toFixed(digits)}%`;

interface Metric { name: string; value: number; tolerance: number | null; unit: string; pass: boolean; note?: string }

function td(text: string, cls?: string) {
  const cell = document.createElement('td');
  cell.textContent = text;
  if (cls) cell.className = cls;
  return cell;
}

function header(table: HTMLTableElement, cols: string[], caption?: string) {
  if (caption) table.createCaption().textContent = caption;
  const row = table.createTHead().insertRow();
  for (const c of cols) {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = c;
    row.append(th);
  }
  return table.createTBody();
}

function result(pass: boolean, informational: boolean) {
  const cell = document.createElement('td');
  cell.className = informational ? 'result info' : pass ? 'result pass' : 'result fail';
  cell.textContent = informational ? 'Reported' : pass ? 'Pass' : 'Fail';
  return cell;
}

function withUnit(v: number, unit: string) {
  return unit ? `${sci(v)} ${unit}` : sci(v);
}

// --- Ledger -------------------------------------------------------------------

function buildLedger() {
  const body = header($('ledger') as HTMLTableElement, ['Check', 'Measurement', 'Measured', 'Tolerance', 'Result']);
  let passed = 0, total = 0;
  const addRow = (title: string, m: Metric) => {
    const informational = m.tolerance === null;
    if (!informational) {
      total++;
      if (m.pass) passed++;
    }
    const row = body.insertRow();
    row.append(td(title, 'check'), td(m.name), td(withUnit(m.value, m.unit), 'num'), td(informational ? '—' : withUnit(m.tolerance!, m.unit), 'num'), result(m.pass, informational));
  };
  for (const c of suite.checks as { title: string; metrics: Metric[] }[]) for (const m of c.metrics) addRow(c.title, m);
  const fate = (r: { worst_interior_agreement: number; pass: boolean }, title: string, name: string) =>
    addRow(title, { name, value: r.worst_interior_agreement * 100, tolerance: 99.5, unit: '%', pass: r.pass });
  fate(crossBackend, 'CUDA vs WebGPU', `worst fate agreement over ${crossBackend.scenes_compared} scenes`);
  fate(fp32fp64, 'Single vs double precision', `worst fate agreement over ${fp32fp64.scenes_compared} scenes`);
  fate(fastmath, 'Fast-math CUDA vs WebGPU', `worst fate agreement over ${fastmath.scenes_compared} scenes`);
  $('verdict').textContent = `${passed} of ${total} checks pass. Each was run on an ${suite.gpu} on ${new Date(suite.date).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' })}.`;
}

// --- Cross-backend tiles --------------------------------------------------------

function buildTiles() {
  const scenes = crossBackend.scenes;
  const all = scenes.reduce((s, x) => s + x.interior_pixels, 0);
  $('agree-stat').innerHTML = '';
  const big = document.createElement('strong');
  big.textContent = pct(crossBackend.worst_interior_agreement, 1);
  $('agree-stat').append(big, ` of interior pixels agree in the worst of ${scenes.length} scenes, ${(all / 1e6).toFixed(1)} million pixels in all.`);

  const list = $('tiles');
  for (const s of scenes) {
    const url = Object.entries(tiles).find(([path]) => path.endsWith(`/${s.scene}.png`))?.[1];
    const li = document.createElement('li');
    const img = document.createElement('div');
    img.className = 'tile-image';
    if (url) img.style.backgroundImage = `url("${url}")`;
    img.setAttribute('role', 'img');
    img.setAttribute('aria-label', `${describeScene(s.scene)}: fate map and difference image`);
    const cap = document.createElement('p');
    cap.className = 'tile-caption';
    const name = document.createElement('span');
    name.className = 'tile-name';
    name.textContent = describeScene(s.scene);
    const score = document.createElement('span');
    score.className = 'tile-score';
    score.textContent = pct(s.fate_agreement_interior, 2);
    cap.append(name, score);
    const sub = document.createElement('p');
    sub.className = 'tile-sub';
    sub.textContent = `edge offset ${s.shadow_edge_offset_px?.max ?? 0} px max · ${s.cuda_trace_ms.toFixed(1)} ms CUDA`;
    li.append(img, cap, sub);
    list.append(li);
  }
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="tile-view"]')) {
    radio.addEventListener('change', () => {
      list.dataset.view = radio.value;
      $('tile-key').hidden = radio.value !== 'fate';
    });
  }
  $('precision-note').textContent =
    `Single against double precision on CUDA: worst agreement ${pct(fp32fp64.worst_interior_agreement)} over ${fp32fp64.scenes_compared} scenes. ` +
    `CUDA with fast-math intrinsics (the shipped interactive kernel) against WebGPU: ${pct(fastmath.worst_interior_agreement)}.`;
}

function describeScene(id: string): string {
  const m = /^(\w+?)(?:-a(\d+))?-i(\d+)(?:-(disk|nodisk))?$/.exec(id);
  if (!m) return id;
  const name = { kerr: 'Kerr', schwarzschild: 'Schwarzschild', minkowski: 'Flat space' }[m[1]] ?? m[1];
  const spin = m[2] ? ` a ${Number(m[2]) / 100}` : '';
  const disk = m[4] === 'nodisk' ? ', no disk' : '';
  return `${name}${spin}, ${m[3]}°${disk}`;
}

// --- Charts -----------------------------------------------------------------------

function buildDeflection() {
  const d = deflection.data;
  const pairs = (ys: number[]) => d.b_over_M.map((b, i) => [b, ys[i]] as [number, number]);
  lineChart($('chart-deflection'), {
    x: { label: 'impact parameter b / M', scale: 'log', domain: [5, 1200] },
    y: { label: 'deflection, radians', scale: 'log', domain: [3e-3, 8] },
    series: [
      { name: 'Exact integral', kind: 'line', color: NEUTRAL, data: pairs(d.exact_deflection_rad) },
      { name: 'Weak-field series', kind: 'dashed', color: NEUTRAL, data: pairs(d.weak_field_series_rad) },
    ],
    describe: 'Deflection angle against impact parameter, exact integral and weak-field series, log-log.',
  });
  lineChart($('chart-deflection-error'), {
    x: { label: 'impact parameter b / M', scale: 'log', domain: [5, 1200] },
    y: { label: 'error in traced angle, radians', scale: 'log', domain: [1e-15, 1e-3] },
    series: [
      { name: 'RK45 reference, double precision', kind: 'points', color: RED, data: pairs(d.rk45_abs_error_rad) },
      { name: 'RK4 interactive (step 0.03 r)', kind: 'points', color: BLUE, data: pairs(d.rk4_interactive_abs_error_rad) },
    ],
    guides: [{ y: 1e-9, label: 'tolerance 10⁻⁹ rad' }],
    describe: 'Absolute error of the traced deflection angle for both integrators.',
  });
}

function buildShadow() {
  const d = shadowSchw.data;
  const deg = (r: number) => (r * 180) / Math.PI;
  lineChart($('chart-shadow'), {
    x: { label: 'camera radius r / M', scale: 'log', domain: [3.5, 1200] },
    y: { label: 'shadow radius, degrees', scale: 'log', domain: [0.2, 90] },
    series: [
      { name: 'Prediction, √27 M formula', kind: 'line', color: NEUTRAL, data: d.camera_r_over_M.map((r, i) => [r, deg(d.predicted_rad[i])]) },
      { name: 'Traced edge', kind: 'points', color: RED, data: d.camera_r_over_M.map((r, i) => [r, deg(d.traced_rad[i])]) },
    ],
    describe: 'Shadow angular radius against camera radius: prediction line and traced points.',
  });
  $('shadow-caption').textContent = `Traced points against the formula. The largest difference is ${sci(Math.max(...d.abs_error_rad))} radians, at r = ${d.camera_r_over_M[d.abs_error_rad.indexOf(Math.max(...d.abs_error_rad))]} M.`;
}

function bardeenOutline(a: number): [number, number][] {
  // Bardeen (1973), observer in the equatorial plane, M = 1.
  const xi = (r: number) => (r * r * (3 - r) - a * a * (r + 1)) / (a * (r - 1));
  const eta = (r: number) => (r ** 3 * (4 * a * a - r * (r - 3) ** 2)) / (a * a * (r - 1) ** 2);
  const rPro = 2 * (1 + Math.cos((2 / 3) * Math.acos(-a)));
  const rRetro = 2 * (1 + Math.cos((2 / 3) * Math.acos(a)));
  const upper: [number, number][] = [];
  for (let i = 0; i <= 200; i++) {
    const r = rPro + ((rRetro - rPro) * i) / 200;
    upper.push([-xi(r), Math.sqrt(Math.max(0, eta(r)))]);
  }
  return [...upper, ...upper.slice().reverse().map(([x, y]) => [x, -y] as [number, number])];
}

function buildKerr() {
  const host = $('kerr-outlines');
  const spins = [0.3, 0.6, 0.9, 0.99];
  const size = 320;
  const span = 7; // M either side
  const map = (v: number) => size / 2 + (v / span) * (size / 2);
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Bardeen shadow outlines for spins 0.3, 0.6, 0.9 and 0.99, viewed edge-on.');
  const axes = document.createElementNS(ns, 'path');
  axes.setAttribute('class', 'cross');
  axes.setAttribute('d', `M${size / 2} 8V${size - 8}M8 ${size / 2}H${size - 8}`);
  svg.append(axes);
  // Ticks every 1 M along both axes, which cross at the hole.
  const ticks = document.createElementNS(ns, 'path');
  ticks.setAttribute('class', 'cross');
  let d = '';
  for (let v = -span + 1; v < span; v++) {
    if (v === 0) continue;
    d += `M${map(v)} ${size / 2 - 3}v6M${size / 2 - 3} ${map(v)}h6`;
  }
  ticks.setAttribute('d', d);
  svg.append(ticks);
  spins.forEach((a, i) => {
    const outline = bardeenOutline(a);
    const path = document.createElementNS(ns, 'path');
    path.setAttribute('class', 'outline');
    path.style.setProperty('opacity', String(0.35 + (0.65 * i) / (spins.length - 1)));
    path.setAttribute('d', `M${outline.map(([x, y]) => `${map(x)} ${map(-y)}`).join('L')}Z`);
    svg.append(path);
    // Label just outside the flattened (prograde) edge, staggered downwards.
    const label = document.createElementNS(ns, 'text');
    label.setAttribute('x', String(map(outline[0][0]) - 5));
    label.setAttribute('y', String(size / 2 + 16 + i * 14));
    label.setAttribute('text-anchor', 'end');
    label.textContent = `a ${a}`;
    svg.append(label);
  });
  const unit = document.createElementNS(ns, 'text');
  unit.setAttribute('x', String(size - 10));
  unit.setAttribute('y', String(size - 12));
  unit.setAttribute('text-anchor', 'end');
  unit.setAttribute('class', 'unit');
  unit.textContent = 'ticks every 1 M';
  svg.append(unit);
  host.append(svg);

  const table = $('kerr-table') as HTMLTableElement;
  const configs = shadowKerr.data.configurations;
  const incl = [...new Set(configs.map(c => c.inclination_deg))];
  const body = header(table, ['Spin a / M', ...incl.map(i => `Inclination ${i}°`)], 'Largest distance between traced edge and Bardeen\'s outline, in M');
  for (const a of [...new Set(configs.map(c => c.a))]) {
    const row = body.insertRow();
    row.append(td(String(a), 'check'));
    for (const i of incl) {
      const c = configs.find(c => c.a === a && c.inclination_deg === i)!;
      row.append(td(sci(c.max_abs_deviation_M), 'num'));
    }
  }
}

function buildOrbits() {
  const body = header($('orbit-table') as HTMLTableElement, ['Spacetime', 'Spin', 'Orbit', 'Measured', 'Exact', 'Difference'], 'Radii in M (Boyer–Lindquist)');
  const rows = orbits.data.rows as Record<string, unknown>[];
  for (const r of rows) {
    if (r.metric === 'kerr' && r.a === 0) continue; // same as Schwarzschild
    const entries: [string, { traced?: number; from_metric?: number; analytic: number }][] = [
      ['Photon orbit, prograde', r.photon_prograde as never],
      ['Photon orbit, retrograde', r.photon_retrograde as never],
      ['ISCO, prograde', r.isco_prograde as never],
      ['ISCO, retrograde', r.isco_retrograde as never],
    ];
    entries.forEach(([label, v], i) => {
      if (Number(r.a) === 0 && i % 2 === 1) return; // no spin: prograde = retrograde
      const measured = v.traced ?? v.from_metric!;
      const row = body.insertRow();
      row.append(
        td(i === 0 ? (r.metric === 'kerr' ? 'Kerr' : 'Schwarzschild') : '', 'check'),
        td(i === 0 ? String(r.a) : '', 'num'),
        td(Number(r.a) === 0 ? label.replace(', prograde', '') : label),
        td(measured.toFixed(9), 'num'),
        td(v.analytic.toFixed(9), 'num'),
        td(sci(Math.abs(measured - v.analytic)), 'num'),
      );
    });
  }
}

function buildIntegrator() {
  const d = convergence.data;
  const markers = [RED, BLUE, NEUTRAL];
  lineChart($('chart-rk4'), {
    x: { label: 'step size (fraction of r)', scale: 'log', domain: [0.002, 0.2] },
    y: { label: 'relative error', scale: 'log', domain: [1e-14, 1e-4] },
    series: d.rk4.map((s, i) => ({
      name: `b = ${s.b0_over_M} M`,
      kind: 'line' as const,
      color: markers[i],
      data: d.rk4_step_fractions.map((h, j) => [h, s.relative_error[j]] as [number, number]),
    })),
    describe: 'RK4 relative error against step size for three impact parameters, log-log.',
  });
  const orders = d.rk4.map(s => s.fitted_order_finest_4);
  $('rk4-caption').textContent = `RK4 error against step size for three impact parameters. Fitted slopes: ${orders.map(o => o.toFixed(3)).join(', ')}. Fourth order means a slope of 4.`;

  const all = d.rk45.flatMap(s => d.rk45_rtol.map((t, j) => [t, s.relative_error[j]] as [number, number]));
  lineChart($('chart-rk45'), {
    x: { label: 'requested tolerance', scale: 'log', domain: [5e-14, 2e-4] },
    y: { label: 'relative error', scale: 'log', domain: [1e-15, 1e-3] },
    series: [
      { name: 'Error equals tolerance', kind: 'dashed', color: NEUTRAL, data: [[5e-14, 5e-14], [2e-4, 2e-4]] },
      { name: 'RK45, three impact parameters', kind: 'points', color: RED, data: all },
    ],
    describe: 'RK45 relative error against requested tolerance, with the error-equals-tolerance line.',
  });

  const runs = constraintDrift.data.runs;
  const decades = constraintDrift.data.decades;
  const fraction = (h: number[]) => {
    const n = h.reduce((a, b) => a + b, 0);
    return decades.map((e, i) => [10 ** e, h[i] / n] as [number, number]).filter(([, f]) => f > 0);
  };
  lineChart($('chart-drift'), {
    x: { label: 'largest |H| / p_t² along the ray', scale: 'log', domain: [1e-13, 1e-3] },
    y: { label: 'share of rays', scale: 'linear', domain: [0, 0.9], format: v => `${Math.round(v * 100)}%` },
    series: [
      { name: 'RK45 reference', kind: 'line', color: RED, data: fraction(runs[1].histogram) },
      { name: 'RK4 interactive', kind: 'line', color: BLUE, data: fraction(runs[0].histogram) },
    ],
    describe: 'Distribution of the null-constraint drift over a full Kerr frame for both integrators.',
  });
  $('drift-caption').textContent = `Kerr a = 0.95, one 960 × 540 frame, rays sorted by their worst drift (by decade). 99% of rays stay below ${sci(runs[1].p99)} with RK45 and ${sci(runs[0].p99)} with the interactive RK4.`;
}

function buildSpeed() {
  const scenes = benchmarks.cuda_vs_webgpu_same_gpu.scenes;
  barChart(
    $('chart-backends'),
    scenes.map(s => ({
      label: describeScene(s.scene),
      values: [
        { name: 'CUDA (shipped kernel)', value: s.cuda_trace_ms, color: RED },
        { name: 'WebGPU in Chrome', value: s.webgpu_trace_ms, color: BLUE },
      ],
    })),
    { label: 'ms per 1920 × 1080 frame', scale: 'linear', domain: [0, 140], format: v => `${Number(v.toFixed(1))}` },
    'Trace time per 1080p frame for CUDA and WebGPU on four scenes.',
  );
  const speedups = scenes.map(s => s.cuda_speedup_over_webgpu);
  $('backends-caption').textContent = `CUDA is ${Math.min(...speedups).toFixed(2)}–${Math.max(...speedups).toFixed(2)}× faster than WebGPU on the same GPU. Lower is faster.`;

  const body = header($('precision-table') as HTMLTableElement, ['Spacetime', 'Single, RK4', 'Double, RK4', 'Double, RK45', 'Double ÷ single'], 'Trace time per 1080p frame, ms');
  for (const p of benchmarks.precision_and_integrator) {
    const row = body.insertRow();
    const name = { kerr: 'Kerr', schwarzschild: 'Schwarzschild', minkowski: 'Flat space' }[p.metric] ?? p.metric;
    row.append(td(name, 'check'), td(p.fp32_rk4.trace_ms.toFixed(1), 'num'), td(p.fp64_rk4.trace_ms.toFixed(0), 'num'), td(p.fp64_rk45_rtol_1e10.trace_ms.toFixed(0), 'num'), td(`${p.fp64_over_fp32_time.toFixed(0)}×`, 'num'));
  }
}

function buildProvenance() {
  const dl = $('provenance');
  const items: [string, string][] = [
    ['GPU', `${suite.gpu} (${suite.arch})`],
    ['Driver', `${suite.driver}, CUDA driver API ${suite.cuda_driver_api}, NVRTC ${suite.nvrtc}`],
    ['System', `${benchmarks.software.os}`],
    ['Browser', 'Headless Chrome 155, WebGPU on D3D12'],
    ['Source', `commit ${suite.git.commit}`],
  ];
  for (const [k, v] of items) {
    const dt = document.createElement('dt');
    dt.textContent = k;
    const dd = document.createElement('dd');
    dd.textContent = v;
    dl.append(dt, dd);
  }
  $('provenance-note').textContent = suite.git.uncommitted_changes
    ? 'These results were produced from a working tree with uncommitted changes on top of that commit. Raw results: validation/results/ and docs/benchmarks/ in the repository.'
    : 'Raw results: validation/results/ and docs/benchmarks/ in the repository.';
}

buildLedger();
buildTiles();
buildDeflection();
buildShadow();
buildKerr();
buildOrbits();
buildIntegrator();
buildSpeed();
buildProvenance();
