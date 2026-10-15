// WGSL backend: compiles a metric into shader functions for the GPU tracer.
//
// The generated module defines
//   fn metric_eval(x: vec4<f32>) -> MetricEval   g and its four derivatives
//   fn horizon_value(x: vec4<f32>) -> f32         negative inside the capture region
//   fn disk_radius / disk_omega / disk_inner / disk_outer (x: vec4<f32>) -> f32
//   const HAS_HORIZON: bool, const HAS_DISK: bool
// and reads metric parameters from `U.params`, declared by the tracer.

import type { CompiledMetric } from './compile.ts';
import { emitBlock, type Dialect } from './emit.ts';

export const MAX_PARAMS = 8;

function floatLiteral(v: number): string {
  let s = String(v);
  if (!/[.eE]/.test(s)) s += '.0';
  return v < 0 ? `(${s})` : s;
}

const wgsl: Dialect = {
  number: floatLiteral,
  coord: i => `x.${'xyzw'[i]}`,
  param: i => `U.params[${i >> 2}].${'xyzw'[i & 3]}`,
  fn: (f, a) => `${f}(${a})`,
  pow: (b, e) => `pow(${b}, ${e})`,
  declare: (name, value) => `  let ${name} = ${value};`,
};

function scalarFn(metric: CompiledMetric, name: string, root: number | null, fallback: string): string {
  if (root === null) return `fn ${name}(x: vec4<f32>) -> f32 { return ${fallback}; }`;
  const { lines, results } = emitBlock(metric.graph, [root], wgsl);
  return [`fn ${name}(x: vec4<f32>) -> f32 {`, ...lines, `  return ${results[0]};`, '}'].join('\n');
}

function mat4(entries: string[]): string {
  // Column-major constructor; the matrices are symmetric so the order only
  // matters for readability.
  return `mat4x4<f32>(${entries.join(', ')})`;
}

export function compileToWgsl(metric: CompiledMetric): string {
  if (metric.params.length > MAX_PARAMS) {
    throw new Error(`The GPU tracer supports at most ${MAX_PARAMS} parameters`);
  }
  const roots = [...metric.g, ...metric.dg.flat()];
  const { lines, results } = emitBlock(metric.graph, roots, wgsl);
  const g = results.slice(0, 16);
  const dg = [0, 1, 2, 3].map(c => results.slice(16 + 16 * c, 32 + 16 * c));

  return [
    `// Generated from spacetime "${metric.name}". Do not edit.`,
    `const HAS_HORIZON: bool = ${metric.horizon !== null};`,
    `const HAS_DISK: bool = ${metric.disk !== null};`,
    '',
    'fn metric_eval(x: vec4<f32>) -> MetricEval {',
    ...lines,
    '  var m: MetricEval;',
    `  m.g = ${mat4(g)};`,
    ...dg.map((d, c) => `  m.dg${c} = ${mat4(d)};`),
    '  return m;',
    '}',
    '',
    scalarFn(metric, 'horizon_value', metric.horizon, '1.0'),
    scalarFn(metric, 'disk_radius', metric.disk?.radius ?? null, '0.0'),
    scalarFn(metric, 'disk_omega', metric.disk?.omega ?? null, '0.0'),
    scalarFn(metric, 'disk_inner', metric.disk?.inner ?? null, '0.0'),
    scalarFn(metric, 'disk_outer', metric.disk?.outer ?? null, '0.0'),
    '',
  ].join('\n');
}
