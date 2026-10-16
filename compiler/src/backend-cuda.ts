// CUDA/C++ backend: compiles a metric into a header of function templates.
//
// The generated header defines, inside namespace rm::<id>,
//   metric_eval<Real>(const Real x[4], const Real* params, Real g[16], Real dg[64])
//   geodesic_rhs<Real>(const Real y[8], const Real* params, Real dy[8]) -> H   (fused Hamilton's equations)
//   horizon_value / disk_radius / disk_omega / disk_inner / disk_outer / length_scale
//     <Real>(const Real x[4], const Real* params) -> Real
//   constexpr NAME, PARAM_COUNT, HAS_HORIZON, HAS_DISK
// Every function is RM_HD (__host__ __device__ under nvcc or NVRTC), so the same
// header serves the NVRTC tracer (float or double) and plain host C++ (tests,
// camera setup). The layout of g and dg matches the JS backend:
// g[4 mu + nu], dg[16 c + 4 mu + nu] = d g_{mu nu} / d x^c.

import type { CompiledMetric } from './compile.ts';
import { emitBlock, type Dialect } from './emit.ts';

function doubleLiteral(v: number): string {
  if (!Number.isFinite(v)) throw new Error(`Cannot emit non-finite constant ${v}`);
  // String(v) is the shortest representation that round-trips, so the C++
  // compiler parses back exactly the double the JS backend uses.
  let s = String(v);
  if (!/[.eE]/.test(s)) s += '.0';
  return s;
}

const cpp: Dialect = {
  number: v => `Real(${doubleLiteral(v)})`,
  coord: i => `x[${i}]`,
  param: i => `params[${i}]`,
  fn: (f, a) => `${f}(${a})`,
  pow: (b, e) => `pow(${b}, ${e})`,
  declare: (name, value) => `  const Real ${name} = ${value};`,
};

const PRELUDE = `#ifndef RM_HD
#if defined(__CUDACC__)
#define RM_HD __host__ __device__
#else
#define RM_HD
#endif
#endif

#ifndef __CUDACC_RTC__
#include <cmath>
#endif`;

function cString(s: string): string {
  return JSON.stringify(s);
}

/** Namespace-safe identifier for a catalog entry (the .st file name). */
export function cudaIdentifier(id: string): string {
  const ident = id.replace(/[^A-Za-z0-9_]/g, '_');
  return /^[0-9]/.test(ident) ? `_${ident}` : ident;
}

function scalarFn(metric: CompiledMetric, name: string, root: number | null, fallback: string): string[] {
  const head = `template <typename Real>\nRM_HD inline Real ${name}(const Real x[4], const Real* params) {`;
  if (root === null) return [head, '  (void)x;', '  (void)params;', `  return Real(${fallback});`, '}', ''];
  const { lines, results } = emitBlock(metric.graph, [root], cpp);
  return [head, '  (void)x;', '  (void)params;', ...lines, `  return ${results[0]};`, '}', ''];
}

/** Upper triangle of a symmetric 4x4 matrix, row-major. */
const UPPER: [number, number][] = [[0, 0], [0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 3]];

// Cofactor expansion of the inverse (same as invert4 in compiler/src/geodesic.ts),
// for the entries of the upper triangle; a symmetric matrix has a symmetric inverse.
const INVERSE_UPPER: Record<string, string> = {
  '00': '(a11 * b11 - a12 * b10 + a13 * b09)',
  '01': '(a02 * b10 - a01 * b11 - a03 * b09)',
  '02': '(a31 * b05 - a32 * b04 + a33 * b03)',
  '03': '(a22 * b04 - a21 * b05 - a23 * b03)',
  '11': '(a00 * b11 - a02 * b08 + a03 * b07)',
  '12': '(a32 * b02 - a30 * b05 - a33 * b01)',
  '13': '(a20 * b05 - a22 * b02 + a23 * b01)',
  '22': '(a30 * b04 - a31 * b02 + a33 * b00)',
  '23': '(a21 * b02 - a20 * b04 - a23 * b00)',
  '33': '(a20 * b03 - a21 * b01 + a22 * b00)',
};

/**
 * Hamilton's equations for this metric as one function: dx = g^{-1} p and
 * dp_c = 1/2 (d_c g_ab) k^a k^b. Unlike the generic version in
 * native/src/rm/geodesic.h, it uses the symmetry of g (10 distinct entries, a
 * symmetric inverse, each k^a k^b pair once) and leaves out every d_c g_ab that
 * is identically zero (all of d_t g for a stationary metric). Floating-point
 * code cannot drop a multiplication by a literal 0 by itself, so this is where
 * the zeros are known.
 */
function fusedRhs(metric: CompiledMetric): string[] {
  const { graph } = metric;
  const terms: { c: number; a: number; b: number; id: number }[] = [];
  for (let c = 0; c < 4; c++) {
    for (const [a, b] of UPPER) {
      const id = metric.dg[c][4 * a + b];
      if (graph.constant(id) !== 0) terms.push({ c, a, b, id });
    }
  }
  const roots = [...UPPER.map(([a, b]) => metric.g[4 * a + b]), ...terms.map(t => t.id)];
  const { lines, results } = emitBlock(graph, roots, cpp);
  const upperIndex = (i: number, j: number) => UPPER.findIndex(([a, b]) => a === Math.min(i, j) && b === Math.max(i, j));

  const out: string[] = [
    'template <typename Real>',
    'RM_HD inline Real geodesic_rhs(const Real y[8], const Real* params, Real dy[8]) {',
    '  const Real* x = y;',
    '  (void)x;',
    '  (void)params;',
    ...lines,
  ];
  for (let i = 0; i < 4; i++) {
    out.push(`  const Real ${[0, 1, 2, 3].map(j => `a${i}${j} = ${results[upperIndex(i, j)]}`).join(', ')};`);
  }
  out.push(
    '  const Real b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;',
    '  const Real b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;',
    '  const Real b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;',
    '  const Real b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;',
    '  const Real s = Real(1) / (b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06);',
    ...UPPER.map(([i, j]) => `  const Real i${i}${j} = ${INVERSE_UPPER[`${i}${j}`]} * s;`),
  );
  const inv = (i: number, j: number) => `i${Math.min(i, j)}${Math.max(i, j)}`;
  for (let mu = 0; mu < 4; mu++) {
    out.push(`  const Real k${mu} = ${[0, 1, 2, 3].map(nu => `${inv(mu, nu)} * y[${4 + nu}]`).join(' + ')};`);
  }
  out.push('  dy[0] = k0;', '  dy[1] = k1;', '  dy[2] = k2;', '  dy[3] = k3;');
  const pairs = new Set(terms.map(t => `${t.a}${t.b}`));
  for (const p of [...pairs].sort()) out.push(`  const Real kk${p} = k${p[0]} * k${p[1]};`);
  for (let c = 0; c < 4; c++) {
    const parts = terms
      .map((t, n) => ({ t, expr: results[UPPER.length + n] }))
      .filter(({ t }) => t.c === c)
      .map(({ t, expr }) => (t.a === t.b ? `Real(0.5) * ${expr} * kk${t.a}${t.b}` : `${expr} * kk${t.a}${t.b}`));
    out.push(`  dy[${4 + c}] = ${parts.length ? parts.join(' + ') : 'Real(0.0)'};`);
  }
  out.push('  return Real(0.5) * (k0 * y[4] + k1 * y[5] + k2 * y[6] + k3 * y[7]);', '}', '');
  return out;
}

export function compileToCuda(metric: CompiledMetric, id: string): string {
  const ns = cudaIdentifier(id);
  const roots = [...metric.g, ...metric.dg.flat()];
  const { lines, results } = emitBlock(metric.graph, roots, cpp);
  const assignments = results.map((expr, k) => (k < 16 ? `  g[${k}] = ${expr};` : `  dg[${k - 16}] = ${expr};`));

  return [
    `// Generated from spacetime "${metric.name}" (metrics/${id}.st) by compiler/src/backend-cuda.ts.`,
    '// Do not edit; run `npm run export-cuda` instead.',
    '#pragma once',
    '',
    PRELUDE,
    '',
    'namespace rm {',
    `namespace ${ns} {`,
    '',
    '#ifndef __CUDACC_RTC__',
    'using std::cos;',
    'using std::exp;',
    'using std::log;',
    'using std::pow;',
    'using std::sin;',
    'using std::sqrt;',
    '#endif',
    '',
    `constexpr const char* NAME = ${cString(metric.name)};`,
    `constexpr int PARAM_COUNT = ${metric.params.length};`,
    `constexpr bool HAS_HORIZON = ${metric.horizon !== null};`,
    `constexpr bool HAS_DISK = ${metric.disk !== null};`,
    '',
    'template <typename Real>',
    'RM_HD inline void metric_eval(const Real x[4], const Real* params, Real g[16], Real dg[64]) {',
    '  (void)x;',
    '  (void)params;',
    ...lines,
    ...assignments,
    '}',
    '',
    ...fusedRhs(metric),
    ...scalarFn(metric, 'horizon_value', metric.horizon, '1.0'),
    ...scalarFn(metric, 'disk_radius', metric.disk?.radius ?? null, '0.0'),
    ...scalarFn(metric, 'disk_omega', metric.disk?.omega ?? null, '0.0'),
    ...scalarFn(metric, 'disk_inner', metric.disk?.inner ?? null, '0.0'),
    ...scalarFn(metric, 'disk_outer', metric.disk?.outer ?? null, '0.0'),
    ...scalarFn(metric, 'length_scale', metric.scale, '1.0'),
    '// The functions above as one type, so generic code can take the metric as a',
    '// template parameter (rm::trace_ray<rm::kerr::Metric>(...)).',
    'struct Metric {',
    `  static constexpr int PARAM_COUNT = ${metric.params.length};`,
    `  static constexpr bool HAS_HORIZON = ${metric.horizon !== null};`,
    `  static constexpr bool HAS_DISK = ${metric.disk !== null};`,
    '  template <typename Real>',
    '  RM_HD static void eval(const Real x[4], const Real* p, Real g[16], Real dg[64]) { metric_eval(x, p, g, dg); }',
    '  template <typename Real>',
    `  RM_HD static Real rhs(const Real y[8], const Real* p, Real dy[8]) { return ${ns}::geodesic_rhs(y, p, dy); }`,
    ...['horizon_value', 'disk_radius', 'disk_omega', 'disk_inner', 'disk_outer', 'length_scale'].flatMap(f => [
      '  template <typename Real>',
      `  RM_HD static Real ${f}(const Real x[4], const Real* p) { return ${ns}::${f}(x, p); }`,
    ]),
    '};',
    '',
    `}  // namespace ${ns}`,
    '}  // namespace rm',
    '',
  ].join('\n');
}
