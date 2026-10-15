// JavaScript backend: compiles a metric into double-precision functions.
// Used as the reference implementation in tests and for camera setup on the CPU.

import type { CompiledMetric } from './compile.ts';
import { emitBlock, type Dialect } from './emit.ts';

const js: Dialect = {
  number: v => (v < 0 ? `(${v})` : `${v}`),
  coord: i => `x[${i}]`,
  param: i => `p[${i}]`,
  fn: (f, a) => `Math.${f}(${a})`,
  pow: (b, e) => `Math.pow(${b}, ${e})`,
  declare: (name, value) => `const ${name} = ${value};`,
};

type Vec = ArrayLike<number>;
type Scalar = (x: Vec, p: Vec) => number;

export interface JsMetric {
  /** Writes g_{mu nu} into g[16] and d_c g_{mu nu} into dg[16 c + 4 mu + nu]. */
  evaluate(x: Vec, p: Vec, g: Float64Array, dg: Float64Array): void;
  /** Negative inside the capture region; null when the metric has no horizon. */
  horizon: Scalar | null;
  disk: { radius: Scalar; omega: Scalar; inner: Scalar; outer: Scalar } | null;
  scale: Scalar;
  source: string;
}

function scalarFunction(metric: CompiledMetric, root: number): Scalar {
  const { lines, results } = emitBlock(metric.graph, [root], js);
  return new Function('x', 'p', `${lines.join('\n')}\nreturn ${results[0]};`) as Scalar;
}

export function compileToJs(metric: CompiledMetric): JsMetric {
  const roots = [...metric.g, ...metric.dg.flat()];
  const { lines, results } = emitBlock(metric.graph, roots, js);
  const assignments = results.map((expr, k) => (k < 16 ? `g[${k}] = ${expr};` : `dg[${k - 16}] = ${expr};`));
  const source = `${lines.join('\n')}\n${assignments.join('\n')}`;
  const evaluate = new Function('x', 'p', 'g', 'dg', source) as JsMetric['evaluate'];

  return {
    evaluate,
    horizon: metric.horizon === null ? null : scalarFunction(metric, metric.horizon),
    disk: metric.disk && {
      radius: scalarFunction(metric, metric.disk.radius),
      omega: scalarFunction(metric, metric.disk.omega),
      inner: scalarFunction(metric, metric.disk.inner),
      outer: scalarFunction(metric, metric.disk.outer),
    },
    scale: scalarFunction(metric, metric.scale),
    source,
  };
}
