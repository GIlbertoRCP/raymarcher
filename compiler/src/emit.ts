import type { Graph, FnName } from './expr.ts';

/** What differs between target languages. */
export interface Dialect {
  number(v: number): string;
  coord(i: number): string;
  param(i: number): string;
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

/**
 * Emits straight-line code computing `roots`. Each shared node becomes one
 * temporary, so common subexpressions are evaluated once.
 */
export function emitBlock(graph: Graph, roots: readonly number[], dialect: Dialect, prefix = 't'): EmittedBlock {
  const names = new Map<number, string>();
  const lines: string[] = [];

  const ref = (id: number): string => {
    const named = names.get(id);
    if (named) return named;
    const n = graph.node(id);
    if (n.k === 'num') return dialect.number(n.v);
    if (n.k === 'coord') return dialect.coord(n.i);
    if (n.k === 'param') return dialect.param(n.i);
    throw new Error(`Node ${id} used before it was emitted`);
  };

  for (const id of graph.topologicalOrder(roots)) {
    const n = graph.node(id);
    let value: string;
    switch (n.k) {
      case 'num':
      case 'coord':
      case 'param':
        continue; // referenced inline
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
    lines.push(dialect.declare(name, value));
    names.set(id, name);
  }
  return { lines, results: roots.map(ref) };
}
