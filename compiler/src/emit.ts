import type { Graph, FnName } from './expr.ts';

/** What differs between target languages. */
export interface Dialect {
  number(v: number): string;
  coord(i: number): string;
  param(i: number): string;
  /** A tangent input (dx^i or dtheta_i) for the current direction; needed only by JVP code. */
  tangent?(src: 'coord' | 'param', i: number): string;
  fn(f: FnName, arg: string): string;
  /** General power with a non-integer, non-half-integer exponent. */
  pow(base: string, e: string): string;
  /** Declares a temporary: `const t1 = ...;` or `let t1 = ...;`. */
  declare(name: string, value: string): string;
}

export interface EmittedBlock {
  /** Statements that compute every temporary, in dependency order. */
  lines: string[];
  /** Expression for each root, in the order the roots were given. */
  results: string[];
}

function product(name: string, n: number): string {
  return new Array(n).fill(name).join(' * ');
}

/** Generated names for emitted nodes; shared by blocks emitted in sequence. */
export type Names = Map<number, string>;

function emitter(graph: Graph, dialect: Dialect, names: Names) {
  const ref = (id: number): string => {
    const named = names.get(id);
    if (named) return named;
    const n = graph.node(id);
    if (n.k === 'num') return dialect.number(n.v);
    if (n.k === 'coord') return dialect.coord(n.i);
    if (n.k === 'param') return dialect.param(n.i);
    if (n.k === 'tan') {
      if (!dialect.tangent) throw new Error('This dialect has no tangent inputs');
      return dialect.tangent(n.src, n.i);
    }
    throw new Error(`Node ${id} used before it was emitted`);
  };

  /** The statement defining node `id`, or null for nodes referenced inline. */
  const statement = (id: number, prefix: string): string | null => {
    const n = graph.node(id);
    let value: string;
    switch (n.k) {
      case 'num':
      case 'coord':
      case 'param':
      case 'tan':
        return null;
      case 'add':
        value = n.a.map(ref).join(' + ');
        break;
      case 'mul':
        value = n.a.map(ref).join(' * ');
        break;
      case 'fn':
        value = dialect.fn(n.f, ref(n.a));
        break;
      case 'pow': {
        const b = ref(n.b);
        const e = n.e;
        const mag = Math.abs(e);
        if (Number.isInteger(e) && mag <= 4) {
          value = e > 0 ? product(b, mag) : `${dialect.number(1)} / (${product(b, mag)})`;
        } else if (Number.isInteger(e * 2) && mag <= 4.5) {
          // x^(k + 1/2) = x^k * sqrt(x)
          const k = Math.floor(mag);
          const root = dialect.fn('sqrt', b);
          const whole = k > 0 ? `${product(b, k)} * ${root}` : root;
          value = e > 0 ? whole : `${dialect.number(1)} / (${whole})`;
        } else {
          value = dialect.pow(b, dialect.number(e));
        }
        break;
      }
    }
    const name = `${prefix}${id}`;
    names.set(id, name);
    return dialect.declare(name, value);
  };
  return { ref, statement };
}

/**
 * Emits straight-line code computing `roots`. Each shared node becomes one
 * temporary, so common subexpressions are evaluated once.
 */
export function emitBlock(graph: Graph, roots: readonly number[], dialect: Dialect, prefix = 't'): EmittedBlock {
  const names: Names = new Map();
  const { ref, statement } = emitter(graph, dialect, names);
  const lines: string[] = [];
  for (const id of graph.topologicalOrder(roots)) {
    const line = statement(id, prefix);
    if (line !== null) lines.push(line);
  }
  return { lines, results: roots.map(ref) };
}

export interface SplitBlock {
  /** Statements that do not depend on tangent inputs: computed once. */
  primal: string[];
  /** Statements that depend on tangent inputs: computed once per direction. */
  tangent: string[];
  primalResults: string[];
  tangentResults: string[];
}

/**
 * Emits a JVP in two parts. Nodes that do not depend on any tangent input
 * (the plain evaluation and the partial derivatives the chain rule needs) go
 * in `primal`; the rest go in `tangent`, which the caller wraps in a loop over
 * directions. Tangent temporaries use the prefix `d`.
 */
export function emitSplit(
  graph: Graph,
  primalRoots: readonly number[],
  tangentRoots: readonly number[],
  dialect: Dialect,
): SplitBlock {
  const order = graph.topologicalOrder([...primalRoots, ...tangentRoots]);
  const dependsOnTangent = new Set<number>();
  for (const id of order) {
    const n = graph.node(id);
    const children = n.k === 'add' || n.k === 'mul' ? n.a : n.k === 'pow' ? [n.b] : n.k === 'fn' ? [n.a] : [];
    if (n.k === 'tan' || children.some(c => dependsOnTangent.has(c))) dependsOnTangent.add(id);
  }
  const names: Names = new Map();
  const { ref, statement } = emitter(graph, dialect, names);
  const primal: string[] = [];
  const tangent: string[] = [];
  for (const id of order) {
    const inner = dependsOnTangent.has(id);
    const line = statement(id, inner ? 'd' : 't');
    if (line !== null) (inner ? tangent : primal).push(line);
  }
  return { primal, tangent, primalResults: primalRoots.map(ref), tangentResults: tangentRoots.map(ref) };
}
