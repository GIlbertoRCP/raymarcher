import { differentiate } from './diff.ts';
import { parseMetric, type ParsedMetric } from './parser.ts';

export interface CompiledMetric extends ParsedMetric {
  /** dg[c][4 * mu + nu] is the node for d g_{mu nu} / d x^c. */
  dg: number[][];
  /**
   * Coordinates the metric does not depend on. Each one gives a conserved
   * momentum along every geodesic (p_t = -E for a stationary spacetime).
   */
  cyclic: number[];
}

export function compileMetric(source: string): CompiledMetric {
  const parsed = parseMetric(source);
  const { graph, g } = parsed;
  const dg = [0, 1, 2, 3].map(c => {
    const memo = new Map<number, number>();
    return g.map(component => differentiate(graph, component, c, memo));
  });
  const cyclic = [0, 1, 2, 3].filter(c => dg[c].every(id => graph.constant(id) === 0));
  return { ...parsed, dg, cyclic };
}
