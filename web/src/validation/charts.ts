// Small SVG charts in the atlas style: graduated axes like the playground's
// margin scales, red for measured series, a neutral line for predictions.
// Every chart has a hover readout and redraws to its container's width.

const SVG = 'http://www.w3.org/2000/svg';

export type Scale = 'linear' | 'log';

export interface Axis {
  label: string;
  scale: Scale;
  domain: [number, number];
  format?: (v: number) => string;
}

export interface Series {
  name: string;
  kind: 'line' | 'points' | 'dashed';
  /** CSS colour token, e.g. 'var(--red)'. */
  color: string;
  data: [number, number][];
  /** Direct label drawn at the series' last point. */
  label?: string;
}

export interface LineChartSpec {
  x: Axis;
  y: Axis;
  series: Series[];
  /** Horizontal guide lines, e.g. a tolerance. */
  guides?: { y: number; label: string }[];
  height?: number;
  describe: string;
}

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}, text?: string) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text !== undefined) node.textContent = text;
  return node;
}

function scaler(axis: Axis, from: number, to: number) {
  const [a, b] = axis.domain;
  if (axis.scale === 'log') {
    const la = Math.log10(a), lb = Math.log10(b);
    return (v: number) => from + ((Math.log10(v) - la) / (lb - la)) * (to - from);
  }
  return (v: number) => from + ((v - a) / (b - a)) * (to - from);
}

/** Round tick values: decades for log axes, 1-2-5 steps for linear ones. */
function ticks(axis: Axis, maxTicks: number): { value: number; major: boolean }[] {
  const [a, b] = axis.domain;
  const out: { value: number; major: boolean }[] = [];
  if (axis.scale === 'log') {
    const lo = Math.floor(Math.log10(a)), hi = Math.ceil(Math.log10(b));
    const decades = hi - lo;
    const every = Math.max(1, Math.ceil(decades / maxTicks));
    for (let e = lo; e <= hi; e++) {
      for (const m of decades <= 4 ? [1, 2, 3, 4, 5, 6, 7, 8, 9] : [1]) {
        const v = m * 10 ** e;
        if (v < a * 0.999 || v > b * 1.001) continue;
        out.push({ value: v, major: m === 1 && (e - lo) % every === 0 });
      }
    }
    return out;
  }
  const span = b - a;
  const raw = span / maxTicks;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(s => s * mag).find(s => s >= raw) ?? raw;
  for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push({ value: v, major: true });
  return out;
}

export function sci(v: number): string {
  if (v === 0) return '0';
  const e = Math.floor(Math.log10(Math.abs(v)));
  if (e >= -2 && e <= 3) return String(Number(v.toPrecision(3)));
  const m = v / 10 ** e;
  const mant = Math.abs(m - 1) < 1e-9 ? '' : `${Number(m.toFixed(1))}×`;
  return `${mant}10${superscript(e)}`;
}

function superscript(n: number): string {
  const map: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
  return String(n).split('').map(c => map[c] ?? c).join('');
}

/** Renders a line/point chart into `host`, redrawing when the host resizes. */
export function lineChart(host: HTMLElement, spec: LineChartSpec) {
  host.classList.add('chart');
  const tip = document.createElement('div');
  tip.className = 'chart-tip';
  tip.hidden = true;

  const draw = () => {
    const width = Math.max(280, host.clientWidth);
    const height = spec.height ?? Math.round(Math.min(340, Math.max(220, width * 0.5)));
    const m = { top: 14, right: 18, bottom: 40, left: 58 };
    const x = scaler(spec.x, m.left, width - m.right);
    const y = scaler(spec.y, height - m.bottom, m.top);
    const fmtX = spec.x.format ?? sci;
    const fmtY = spec.y.format ?? sci;

    const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': spec.describe });
    // Axes: hairline frame on two sides with graduations, like the plate margins.
    const axes = el('g', { class: 'axes' });
    axes.append(el('line', { x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom }));
    axes.append(el('line', { x1: m.left, x2: m.left, y1: m.top, y2: height - m.bottom }));
    for (const t of ticks(spec.x, Math.floor(width / 90))) {
      const px = x(t.value);
      axes.append(el('line', { x1: px, x2: px, y1: height - m.bottom, y2: height - m.bottom + (t.major ? 7 : 3) }));
      if (t.major) {
        axes.append(el('line', { class: 'grid', x1: px, x2: px, y1: m.top, y2: height - m.bottom }));
        axes.append(el('text', { x: px, y: height - m.bottom + 19, 'text-anchor': 'middle' }, fmtX(t.value)));
      }
    }
    for (const t of ticks(spec.y, Math.floor(height / 48))) {
      const py = y(t.value);
      axes.append(el('line', { x1: m.left - (t.major ? 7 : 3), x2: m.left, y1: py, y2: py }));
      if (t.major) {
        axes.append(el('line', { class: 'grid', x1: m.left, x2: width - m.right, y1: py, y2: py }));
        axes.append(el('text', { x: m.left - 11, y: py + 3.5, 'text-anchor': 'end' }, fmtY(t.value)));
      }
    }
    axes.append(el('text', { class: 'axis-label', x: width - m.right, y: height - 6, 'text-anchor': 'end' }, spec.x.label));
    svg.append(axes);
    host.querySelector('.chart-ylabel')?.remove();
    const ylabel = document.createElement('p');
    ylabel.className = 'chart-ylabel';
    ylabel.textContent = spec.y.label;

    for (const g of spec.guides ?? []) {
      const py = y(g.y);
      svg.append(el('line', { class: 'guide', x1: m.left, x2: width - m.right, y1: py, y2: py }));
      svg.append(el('text', { class: 'guide-label', x: width - m.right, y: py - 5, 'text-anchor': 'end' }, g.label));
    }

    const points: { px: number; py: number; s: Series; d: [number, number] }[] = [];
    for (const s of spec.series) {
      const pts = s.data.map(d => [x(d[0]), y(d[1])] as const);
      if (s.kind === 'points') {
        const g = el('g', { class: 'points', style: `--c:${s.color}` });
        for (const [px, py] of pts) g.append(el('circle', { cx: px, cy: py, r: 4 }));
        svg.append(g);
      } else {
        svg.append(el('path', { class: `line ${s.kind}`, style: `--c:${s.color}`, d: `M${pts.map(p => p.join(' ')).join('L')}` }));
      }
      s.data.forEach((d, i) => points.push({ px: pts[i][0], py: pts[i][1], s, d }));
      if (s.label) {
        const last = pts[pts.length - 1];
        svg.append(el('text', { class: 'direct', x: last[0] - 6, y: last[1] - 8, 'text-anchor': 'end' }, s.label));
      }
    }

    const cross = el('line', { class: 'crosshair', y1: m.top, y2: height - m.bottom, visibility: 'hidden' });
    const ring = el('circle', { class: 'hover-ring', r: 6, visibility: 'hidden' });
    svg.append(cross, ring);

    const hit = el('rect', { x: m.left, y: m.top, width: width - m.left - m.right, height: height - m.top - m.bottom, fill: 'transparent' });
    hit.addEventListener('pointermove', e => {
      const box = svg.getBoundingClientRect();
      const mx = ((e.clientX - box.left) / box.width) * width;
      const my = ((e.clientY - box.top) / box.height) * height;
      let best = points[0], bestD = Infinity;
      for (const p of points) {
        const d = Math.hypot(p.px - mx, (p.py - my) * 0.35);
        if (d < bestD) { bestD = d; best = p; }
      }
      cross.setAttribute('x1', String(best.px));
      cross.setAttribute('x2', String(best.px));
      cross.setAttribute('visibility', 'visible');
      ring.setAttribute('cx', String(best.px));
      ring.setAttribute('cy', String(best.py));
      ring.setAttribute('visibility', 'visible');
      tip.hidden = false;
      tip.innerHTML = '';
      const name = document.createElement('strong');
      name.textContent = best.s.name;
      const row = document.createElement('span');
      row.textContent = `${spec.x.label} ${fmtX(best.d[0])} · ${fmtY(best.d[1])}`;
      tip.append(name, row);
      const left = (best.px / width) * box.width;
      tip.style.left = `${Math.min(left, box.width - 200)}px`;
      tip.style.top = `${(best.py / height) * box.height}px`;
    });
    hit.addEventListener('pointerleave', () => {
      tip.hidden = true;
      cross.setAttribute('visibility', 'hidden');
      ring.setAttribute('visibility', 'hidden');
    });
    svg.append(hit);

    host.replaceChildren(ylabel, svg, tip);
    if (spec.series.length > 1) host.append(legend(spec.series));
  };

  redrawOnResize(host, draw);
}

/** Draws now, then again whenever the host's width changes. */
function redrawOnResize(host: HTMLElement, draw: () => void) {
  let lastWidth = host.clientWidth;
  draw();
  new ResizeObserver(() => {
    if (host.clientWidth === lastWidth) return;
    lastWidth = host.clientWidth;
    draw();
  }).observe(host);
}

export function legend(series: { name: string; color: string; kind?: string }[]) {
  const ul = document.createElement('ul');
  ul.className = 'chart-legend';
  for (const s of series) {
    const li = document.createElement('li');
    const swatch = document.createElement('span');
    swatch.className = `swatch ${s.kind ?? 'line'}`;
    swatch.style.setProperty('--c', s.color);
    li.append(swatch, s.name);
    ul.append(li);
  }
  return ul;
}

export interface BarGroup {
  label: string;
  values: { name: string; value: number; color: string }[];
}

/** Horizontal grouped bars on a shared axis (one measure, one unit). */
export function barChart(host: HTMLElement, groups: BarGroup[], axis: Axis, describe: string) {
  host.classList.add('chart');
  const tip = document.createElement('div');
  tip.className = 'chart-tip';
  tip.hidden = true;
  const names = [...new Map(groups.flatMap(g => g.values.map(v => [v.name, v.color] as const))).entries()];

  const draw = () => {
    const width = Math.max(280, host.clientWidth);
    // Narrow screens: group labels sit above their bars instead of beside them.
    const stacked = width < 520;
    const labelW = stacked ? 8 : Math.min(190, width * 0.3);
    const labelH = stacked ? 20 : 0;
    const barH = 12, gap = 2, groupGap = 18;
    const perGroup = labelH + names.length * (barH + gap) - gap;
    const m = { top: 8, right: 54, bottom: 34, left: labelW };
    const height = m.top + groups.length * (perGroup + groupGap) - groupGap + m.bottom;
    const x = scaler(axis, m.left, width - m.right);
    const fmt = axis.format ?? sci;
    const svg = el('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': describe });
    const axes = el('g', { class: 'axes' });
    for (const t of ticks(axis, Math.floor((width - labelW) / 80))) {
      const px = x(t.value);
      if (!t.major) continue;
      axes.append(el('line', { class: 'grid', x1: px, x2: px, y1: m.top, y2: height - m.bottom }));
      axes.append(el('line', { x1: px, x2: px, y1: height - m.bottom, y2: height - m.bottom + 6 }));
      axes.append(el('text', { x: px, y: height - m.bottom + 18, 'text-anchor': 'middle' }, fmt(t.value)));
    }
    axes.append(el('line', { x1: m.left, x2: width - m.right, y1: height - m.bottom, y2: height - m.bottom }));
    axes.append(el('text', { class: 'axis-label', x: width - m.right, y: height - 4, 'text-anchor': 'end' }, axis.label));
    svg.append(axes);

    groups.forEach((g, gi) => {
      const top = m.top + gi * (perGroup + groupGap);
      svg.append(
        stacked
          ? el('text', { class: 'group-label', x: m.left, y: top + 13 }, g.label)
          : el('text', { class: 'group-label', x: m.left - 10, y: top + perGroup / 2 + 4, 'text-anchor': 'end' }, g.label),
      );
      g.values.forEach(v => {
        const row = names.findIndex(([n]) => n === v.name);
        const yTop = top + labelH + row * (barH + gap);
        const x0 = x(axis.domain[0]);
        const w = Math.max(1, x(v.value) - x0);
        const bar = el('rect', { class: 'bar', x: x0, y: yTop, width: w, height: barH, style: `--c:${v.color}` });
        bar.addEventListener('pointerenter', () => {
          tip.hidden = false;
          tip.innerHTML = '';
          const s = document.createElement('strong');
          s.textContent = `${g.label} · ${v.name}`;
          const r = document.createElement('span');
          r.textContent = `${fmt(v.value)} ${axis.label}`;
          tip.append(s, r);
          const box = svg.getBoundingClientRect();
          tip.style.left = `${Math.min(((x0 + w) / width) * box.width, box.width - 200)}px`;
          tip.style.top = `${(yTop / height) * box.height}px`;
        });
        bar.addEventListener('pointerleave', () => (tip.hidden = true));
        svg.append(bar);
        svg.append(el('text', { class: 'bar-value', x: x0 + w + 6, y: yTop + barH - 2 }, fmt(v.value)));
      });
    });
    host.replaceChildren(svg, tip);
    if (names.length > 1) host.append(legend(names.map(([name, color]) => ({ name, color, kind: 'bar' }))));
  };
  redrawOnResize(host, draw);
}
