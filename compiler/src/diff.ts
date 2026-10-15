import type { Graph } from './expr.ts';

/**
 * Symbolic derivative of node `id` with respect to coordinate `coord`.
 * Results are memoised per graph and coordinate, so differentiating all ten
 * metric components shares work, and hash-consing shares the resulting nodes.
 */
export function differentiate(graph: Graph, id: number, coord: number, memo: Map<number, number>): number {
  const cached = memo.get(id);
  if (cached !== undefined) return cached;

  const d = (child: number) => differentiate(graph, child, coord, memo);
  const n = graph.node(id);
  let result: number;
  switch (n.k) {
    case 'num':
    case 'param':
      result = graph.num(0);
      break;
    case 'coord':
      result = graph.num(n.i === coord ? 1 : 0);
      break;
    case 'add':
      result = graph.add(...n.a.map(d));
      break;
    case 'mul':
      // Product rule: sum over i of (d a_i) * prod_{j != i} a_j.
      result = graph.add(
        ...n.a.map((factor, i) => graph.mul(d(factor), ...n.a.filter((_, j) => j !== i))),
      );
      break;
    case 'pow':
      result = graph.mul(graph.num(n.e), graph.pow(n.b, n.e - 1), d(n.b));
      break;
    case 'fn': {
      const inner = d(n.a);
      switch (n.f) {
        case 'sqrt':
          result = graph.mul(graph.num(0.5), inner, graph.pow(id, -1));
          break;
        case 'sin':
          result = graph.mul(graph.fn('cos', n.a), inner);
          break;
        case 'cos':
          result = graph.mul(graph.num(-1), graph.fn('sin', n.a), inner);
          break;
        case 'exp':
          result = graph.mul(id, inner);
          break;
        case 'log':
          result = graph.mul(inner, graph.pow(n.a, -1));
          break;
      }
      break;
    }
  }
  memo.set(id, result);
  return result;
}
