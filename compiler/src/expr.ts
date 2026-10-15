// Expression graph for metric components.
//
// Every node is hash-consed: building the same expression twice returns the
// same id, so shared subexpressions (and their derivatives) are computed once
// in the generated code. Constructors also fold constants and apply the
// identities x + 0 = x, x * 1 = x and x * 0 = 0, which keeps symbolic
// derivatives small.

export type FnName = 'sqrt' | 'sin' | 'cos' | 'exp' | 'log';
export const FUNCTIONS: readonly FnName[] = ['sqrt', 'sin', 'cos', 'exp', 'log'];

export type Node =
  | { k: 'num'; v: number }
  | { k: 'coord'; i: number }
  | { k: 'param'; i: number }
  | { k: 'add'; a: number[] }
  | { k: 'mul'; a: number[] }
  | { k: 'pow'; b: number; e: number } // base node, constant exponent
  | { k: 'fn'; f: FnName; a: number };

const FN_IMPL: Record<FnName, (x: number) => number> = {
  sqrt: Math.sqrt,
  sin: Math.sin,
  cos: Math.cos,
  exp: Math.exp,
  log: Math.log,
};

export class Graph {
  readonly nodes: Node[] = [];
  private readonly index = new Map<string, number>();

  private intern(node: Node): number {
    const key = JSON.stringify(node);
    let id = this.index.get(key);
    if (id === undefined) {
      id = this.nodes.length;
      this.nodes.push(node);
      this.index.set(key, id);
    }
    return id;
  }

  node(id: number): Node {
    return this.nodes[id];
  }

  /** The constant value of a node, or undefined if it is not a constant. */
  constant(id: number): number | undefined {
    const n = this.nodes[id];
    return n.k === 'num' ? n.v : undefined;
  }

  num(v: number): number {
    if (!Number.isFinite(v)) throw new Error(`Non-finite constant ${v}`);
    return this.intern({ k: 'num', v: Object.is(v, -0) ? 0 : v });
  }

  coord(i: number): number {
    return this.intern({ k: 'coord', i });
  }

  param(i: number): number {
    return this.intern({ k: 'param', i });
  }

  add(...terms: number[]): number {
    let constant = 0;
    const rest: number[] = [];
    for (const t of terms) {
      const n = this.nodes[t];
      if (n.k === 'num') constant += n.v;
      else if (n.k === 'add') {
        for (const s of n.a) {
          const c = this.constant(s);
          if (c !== undefined) constant += c;
          else rest.push(s);
        }
      } else rest.push(t);
    }
    if (constant !== 0) rest.push(this.num(constant));
    if (rest.length === 0) return this.num(0);
    if (rest.length === 1) return rest[0];
    return this.intern({ k: 'add', a: rest.sort((x, y) => x - y) });
  }

  mul(...factors: number[]): number {
    let constant = 1;
    const rest: number[] = [];
    for (const f of factors) {
      const n = this.nodes[f];
      if (n.k === 'num') constant *= n.v;
      else if (n.k === 'mul') {
        for (const s of n.a) {
          const c = this.constant(s);
          if (c !== undefined) constant *= c;
          else rest.push(s);
        }
      } else rest.push(f);
    }
    if (constant === 0) return this.num(0);
    if (constant !== 1) rest.push(this.num(constant));
    if (rest.length === 0) return this.num(constant);
    if (rest.length === 1) return rest[0];
    return this.intern({ k: 'mul', a: rest.sort((x, y) => x - y) });
  }

  neg(a: number): number {
    return this.mul(this.num(-1), a);
  }

  sub(a: number, b: number): number {
    return this.add(a, this.neg(b));
  }

  div(a: number, b: number): number {
    const cb = this.constant(b);
    if (cb === 0) throw new Error('Division by the constant 0');
    if (cb !== undefined) return this.mul(a, this.num(1 / cb));
    return this.mul(a, this.pow(b, -1));
  }

  /** base^e for a constant exponent e. */
  pow(b: number, e: number): number {
    if (e === 0) return this.num(1);
    if (e === 1) return b;
    const cb = this.constant(b);
    if (cb !== undefined) return this.num(cb ** e);
    const n = this.nodes[b];
    // (x^a)^e = x^(a e) is safe when e is an integer.
    if (n.k === 'pow' && Number.isInteger(e)) return this.pow(n.b, n.e * e);
    if (e === 0.5) return this.fn('sqrt', b);
    return this.intern({ k: 'pow', b, e });
  }

  fn(f: FnName, a: number): number {
    const c = this.constant(a);
    if (c !== undefined) return this.num(FN_IMPL[f](c));
    return this.intern({ k: 'fn', f, a });
  }

  /** Evaluates a node in double precision (used by tests and the analyser). */
  evaluate(id: number, coords: ArrayLike<number>, params: ArrayLike<number>, memo = new Map<number, number>()): number {
    const cached = memo.get(id);
    if (cached !== undefined) return cached;
    const n = this.nodes[id];
    let v: number;
    switch (n.k) {
      case 'num': v = n.v; break;
      case 'coord': v = coords[n.i]; break;
      case 'param': v = params[n.i]; break;
      case 'add': v = n.a.reduce((s, t) => s + this.evaluate(t, coords, params, memo), 0); break;
      case 'mul': v = n.a.reduce((s, t) => s * this.evaluate(t, coords, params, memo), 1); break;
      case 'pow': v = this.evaluate(n.b, coords, params, memo) ** n.e; break;
      case 'fn': v = FN_IMPL[n.f](this.evaluate(n.a, coords, params, memo)); break;
    }
    memo.set(id, v);
    return v;
  }

  /** All nodes reachable from the roots, children before parents. */
  topologicalOrder(roots: readonly number[]): number[] {
    const seen = new Set<number>();
    const order: number[] = [];
    const visit = (id: number) => {
      if (seen.has(id)) return;
      seen.add(id);
      const n = this.nodes[id];
      if (n.k === 'add' || n.k === 'mul') n.a.forEach(visit);
      else if (n.k === 'pow') visit(n.b);
      else if (n.k === 'fn') visit(n.a);
      order.push(id);
    };
    roots.forEach(visit);
    return order;
  }
}
