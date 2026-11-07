// Jacobian-vector products of a compiled metric (Phase 4, step 1).
//
// For a tangent direction (dx, dtheta) in coordinates and metric parameters,
// the JVP of metric_eval returns, alongside g and d_c g,
//   tg[mu nu]       = dg_{mu nu}                = (d_c g) dx^c + (d g / d theta_k) dtheta_k
//   tdg[c, mu nu]   = d(d_c g_{mu nu})          = (d_c d_e g) dx^e + (d_c d g / d theta_k) dtheta_k
// and the JVP of the disk functions (radius, omega, inner, outer) likewise.
// These are what the tangent integrator needs: dg^{-1} = -g^{-1} dg g^{-1},
// dk = dg^{-1} p + g^{-1} dp, d(dp_c) = 1/2 (d(d_c g) kk + 2 d_c g k dk).
//
// The JVP is built symbolically in the same hash-consed graph as the metric
// (diff.ts, `directional`), so it shares subexpressions with the plain
// evaluation, and symmetric components (g_{mu nu} = g_{nu mu}) are the same
// node. emitSplit puts everything that does not depend on the direction
// (g, d g and the partial derivatives the chain rule needs) before a loop over
// directions, so P directions cost one primal evaluation plus P linear passes.
//
// Layouts (shared by the JS and C++ code):
//   dx[4 d + i], dtheta[NP d + k]          direction d = 0 .. P-1
//   tg[16 d + 4 mu + nu], tdg[64 d + 16 c + 4 mu + nu]
//   disk out[4] = (radius, omega, inner, outer), tout[4 d + j]

import type { CompiledMetric } from './compile.ts';
import { directional } from './diff.ts';
import { emitSplit, type Dialect, type SplitBlock } from './emit.ts';

export interface MetricTangent {
  /** Tangent node of g_{mu nu}, index 4 mu + nu. */
  tg: number[];
  /** Tangent node of d_c g_{mu nu}, index 16 c + 4 mu + nu. */
  tdg: number[];
  /** Tangent nodes of (radius, omega, inner, outer), or null with no disk. */
  disk: number[] | null;
  /** Tangent node of the length scale. */
  scale: number;
}

const cache = new WeakMap<CompiledMetric, MetricTangent>();

/** Builds (once per compiled metric) the symbolic tangent of every metric output. */
export function metricTangent(metric: CompiledMetric): MetricTangent {
  const cached = cache.get(metric);
  if (cached) return cached;
  const { graph } = metric;
  const memo = new Map<number, number>();
  const t = (id: number) => directional(graph, id, memo);
  const result: MetricTangent = {
    tg: metric.g.map(t),
    tdg: metric.dg.flat().map(t),
    disk: metric.disk && [metric.disk.radius, metric.disk.omega, metric.disk.inner, metric.disk.outer].map(t),
    scale: 0,
  };
  // Built last so the node numbering of everything above (and of the C++ code) is unchanged.
  result.scale = t(metric.scale);
  cache.set(metric, result);
  return result;
}

function diskRoots(metric: CompiledMetric): number[] | null {
  return metric.disk && [metric.disk.radius, metric.disk.omega, metric.disk.inner, metric.disk.outer];
}

/** Split emission of the metric JVP: primal roots are g then dg, tangent roots tg then tdg. */
export function emitMetricJvp(metric: CompiledMetric, dialect: Dialect): SplitBlock {
  const tan = metricTangent(metric);
  return emitSplit(metric.graph, [...metric.g, ...metric.dg.flat()], [...tan.tg, ...tan.tdg], dialect);
}

/** Split emission of the disk JVP, or null when the metric has no disk. */
export function emitDiskJvp(metric: CompiledMetric, dialect: Dialect): SplitBlock | null {
  const roots = diskRoots(metric);
  const tan = metricTangent(metric);
  if (!roots || !tan.disk) return null;
  return emitSplit(metric.graph, roots, tan.disk, dialect);
}

/** Size of the emitted JVP, for the step 1 code-size measurement. */
export function jvpStats(metric: CompiledMetric, dialect: Dialect) {
  const m = emitMetricJvp(metric, dialect);
  return { primalStatements: m.primal.length, tangentStatements: m.tangent.length };
}

// --- JavaScript ------------------------------------------------------------

type Vec = ArrayLike<number>;
type Out = Float64Array | number[];

export interface JsMetricJvp {
  /**
   * Writes g, dg (as JsMetric.evaluate) and, for each of `P` directions,
   * tg[16 d + ...] and tdg[64 d + ...].
   */
  evaluate(x: Vec, p: Vec, P: number, dx: Vec, dp: Vec, g: Out, dg: Out, tg: Out, tdg: Out): void;
  /** out[4] = (radius, omega, inner, outer); tout[4 d + j] their tangents. Null with no disk. */
  disk: ((x: Vec, p: Vec, P: number, dx: Vec, dp: Vec, out: Out, tout: Out) => void) | null;
  /** Returns the length scale and writes its tangents into tout[d]. */
  scale: (x: Vec, p: Vec, P: number, dx: Vec, dp: Vec, tout: Out) => number;
  source: string;
}

const jsTangent: Dialect = {
  number: v => (v < 0 ? `(${v})` : `${v}`),
  coord: i => `x[${i}]`,
  param: i => `p[${i}]`,
  tangent: (src, i) => (src === 'coord' ? `dx[o4 + ${i}]` : `dp[on + ${i}]`),
  fn: (f, a) => `Math.${f}(${a})`,
  pow: (b, e) => `Math.pow(${b}, ${e})`,
  declare: (name, value) => `const ${name} = ${value};`,
};

export function compileJvpToJs(metric: CompiledMetric): JsMetricJvp {
  const np = metric.params.length;
  const m = emitMetricJvp(metric, jsTangent);
  const source = [
    ...m.primal,
    ...m.primalResults.map((e, k) => (k < 16 ? `g[${k}] = ${e};` : `dg[${k - 16}] = ${e};`)),
    'for (let d = 0; d < P; d++) {',
    `const o4 = 4 * d, on = ${np} * d, o16 = 16 * d, o64 = 64 * d;`,
    ...m.tangent,
    ...m.tangentResults.map((e, k) => (k < 16 ? `tg[o16 + ${k}] = ${e};` : `tdg[o64 + ${k - 16}] = ${e};`)),
    '}',
  ].join('\n');
  const evaluate = new Function('x', 'p', 'P', 'dx', 'dp', 'g', 'dg', 'tg', 'tdg', source) as JsMetricJvp['evaluate'];

  const d = emitDiskJvp(metric, jsTangent);
  const disk =
    d &&
    (new Function(
      'x',
      'p',
      'P',
      'dx',
      'dp',
      'out',
      'tout',
      [
        ...d.primal,
        ...d.primalResults.map((e, k) => `out[${k}] = ${e};`),
        'for (let d = 0; d < P; d++) {',
        `const o4 = 4 * d, on = ${np} * d;`,
        ...d.tangent,
        ...d.tangentResults.map((e, k) => `tout[4 * d + ${k}] = ${e};`),
        '}',
      ].join('\n'),
    ) as NonNullable<JsMetricJvp['disk']>);
  const tan = metricTangent(metric);
  const sc = emitSplit(metric.graph, [metric.scale], [tan.scale], jsTangent);
  const scale = new Function(
    'x',
    'p',
    'P',
    'dx',
    'dp',
    'tout',
    [
      ...sc.primal,
      'for (let d = 0; d < P; d++) {',
      `const o4 = 4 * d, on = ${np} * d;`,
      ...sc.tangent,
      `tout[d] = ${sc.tangentResults[0]};`,
      '}',
      `return ${sc.primalResults[0]};`,
    ].join('\n'),
  ) as JsMetricJvp['scale'];
  return { evaluate, disk, scale, source };
}

// --- CUDA / C++ ------------------------------------------------------------

/** C++ templates appended to the generated header by backend-cuda.ts. */
export function cudaJvpFunctions(metric: CompiledMetric, cpp: Dialect): string[] {
  const dialect: Dialect = {
    ...cpp,
    tangent: (src, i) => (src === 'coord' ? `dx[4 * d + ${i}]` : `dparams[NP * d + ${i}]`),
  };
  const indent = (lines: string[]) => lines.map(l => `  ${l}`);
  const np = metric.params.length;
  const m = emitMetricJvp(metric, dialect);
  const out: string[] = [
    '// Jacobian-vector product of metric_eval along P directions (compiler/src/tangent.ts):',
    '// dx[4 d + i], dparams[PARAM_COUNT d + k] -> tg[16 d + 4 mu + nu], tdg[64 d + 16 c + 4 mu + nu].',
    'template <typename Real, int P>',
    'RM_HD inline void metric_jvp(const Real x[4], const Real* params, const Real* dx, const Real* dparams,',
    '                             Real g[16], Real dg[64], Real* tg, Real* tdg) {',
    '  (void)x;',
    '  (void)params;',
    '  (void)dx;',
    '  (void)dparams;',
    `  constexpr int NP = ${np};`,
    '  (void)NP;',
    ...m.primal,
    ...m.primalResults.map((e, k) => (k < 16 ? `  g[${k}] = ${e};` : `  dg[${k - 16}] = ${e};`)),
    'RM_UNROLL',
    '  for (int d = 0; d < P; ++d) {',
    ...indent(m.tangent),
    ...m.tangentResults.map((e, k) => (k < 16 ? `    tg[16 * d + ${k}] = ${e};` : `    tdg[64 * d + ${k - 16}] = ${e};`)),
    '  }',
    '}',
    '',
    '// The disk functions and their tangents: out = (radius, omega, inner, outer), tout[4 d + j].',
    'template <typename Real, int P>',
    'RM_HD inline void disk_jvp(const Real x[4], const Real* params, const Real* dx, const Real* dparams,',
    '                           Real out[4], Real* tout) {',
    '  (void)x;',
    '  (void)params;',
    '  (void)dx;',
    '  (void)dparams;',
    `  constexpr int NP = ${np};`,
    '  (void)NP;',
  ];
  const d = emitDiskJvp(metric, dialect);
  if (d === null) {
    out.push('  for (int j = 0; j < 4; ++j) out[j] = Real(0.0);', '  for (int j = 0; j < 4 * P; ++j) tout[j] = Real(0.0);');
  } else {
    out.push(
      ...d.primal,
      ...d.primalResults.map((e, k) => `  out[${k}] = ${e};`),
      'RM_UNROLL',
      '  for (int d = 0; d < P; ++d) {',
      ...indent(d.tangent),
      ...d.tangentResults.map((e, k) => `    tout[4 * d + ${k}] = ${e};`),
      '  }',
    );
  }
  out.push('}', '');
  return out;
}
