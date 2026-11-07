import type { Graph, Node } from './expr.ts';

type Leaf = Extract<Node, { k: 'coord' | 'param' | 'tan' }>;

/**
 * Forward-mode symbolic differentiation. `leaf` gives the derivative of each
 * input node; every other rule is the chain rule. Results are memoised per
 * memo map (one map per derivative being taken), so differentiating many roots
 * shares work, and hash-consing shares the resulting nodes.
 */
export function derive(graph: Graph, id: number, leaf: (n: Leaf) => number, memo: Map<number, number>): number {
  const cached = memo.get(id);
  if (cached !== undefined) return cached;

  const d = (child: number) => derive(graph, child, leaf, memo);
  const n = graph.node(id);
  let result: number;
  switch (n.k) {
    case 'num':
      result = graph.num(0);
      break;
    case 'coord':
    case 'param':
    case 'tan':
      result = leaf(n);
      break;
    case 'add':
      result = graph.add(...n.a.map(d));
      break;
    case 'mul':
      // Product rule: sum over i of (d a_i) * prod_{j != i} a_j.
      // (Node creation order is kept as it was before tangents existed, so the
      // temporaries in already-generated code keep their names.)
      result = graph.add(...n.a.map((factor, i) => graph.mul(d(factor), ...n.a.filter((_, j) => j !== i))));
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

/** Symbolic derivative of node `id` with respect to coordinate `coord`. */
export function differentiate(graph: Graph, id: number, coord: number, memo: Map<number, number>): number {
  return derive(graph, id, n => graph.num(n.k === 'coord' && n.i === coord ? 1 : 0), memo);
}

/** Symbolic derivative of node `id` with respect to parameter `param`. */
export function differentiateParam(graph: Graph, id: number, param: number, memo: Map<number, number>): number {
  return derive(graph, id, n => graph.num(n.k === 'param' && n.i === param ? 1 : 0), memo);
}

/**
 * Directional derivative of node `id` along (dx, dtheta): the result is a
 * node linear in the tangent inputs graph.tan('coord', i) and
 * graph.tan('param', i). Tangent inputs are themselves held constant, so the
 * result is a first-order JVP.
 */
export function directional(graph: Graph, id: number, memo: Map<number, number>): number {
  return derive(graph, id, n => (n.k === 'tan' ? graph.num(0) : graph.tan(n.k, n.i)), memo);
}
