// Parser for the metric language.
//
// A metric file is a list of one-line statements:
//
//   spacetime "Kerr"
//   description "A rotating black hole in Kerr-Schild coordinates"
//   coords t x y z
//   param M = 1 in [0.2, 3] "Mass"
//   let r2 = x^2 + y^2 + z^2
//   g t t = -1 + f
//   g t x = f * lx            # one entry per pair; the metric is symmetric
//   horizon r < 2*M           # rays that satisfy this are captured
//   disk radius = r           # optional thin disk in the z = 0 plane
//   disk omega = sqrt(M) / r^1.5
//   disk inner = 6*M
//   disk outer = 20*M
//   scale = M                 # length unit for the camera distance
//
// Components that are not given are zero. Expressions support + - * / ^,
// parentheses, numbers, pi, and sqrt sin cos exp log. Exponents must be
// constants. `#` starts a comment.

import { FUNCTIONS, Graph, type FnName } from './expr.ts';

export class MetricSyntaxError extends Error {
  constructor(message: string, readonly line: number, readonly column: number) {
    super(`${line}:${column}: ${message}`);
    this.name = 'MetricSyntaxError';
  }
}

export interface ParamSpec {
  name: string;
  label: string;
  value: number;
  min: number;
  max: number;
}

export interface DiskSpec {
  radius: number;
  omega: number;
  inner: number;
  outer: number;
}

export interface ParsedMetric {
  name: string;
  description: string;
  coords: string[];
  params: ParamSpec[];
  graph: Graph;
  /** Node ids of g_{mu nu}, row-major 4x4, symmetric. */
  g: number[];
  /** Node whose value is negative inside the capture region, or null. */
  horizon: number | null;
  disk: DiskSpec | null;
  scale: number;
}

type Token =
  | { t: 'num'; v: number; col: number }
  | { t: 'ident'; v: string; col: number }
  | { t: 'str'; v: string; col: number }
  | { t: 'op'; v: string; col: number };

function tokenize(text: string, line: number): Token[] {
  const tokens: Token[] = [];
  let i = 0;
  while (i < text.length) {
    const c = text[i];
    const col = i + 1;
    if (c === '#') break;
    if (/\s/.test(c)) { i++; continue; }
    const num = /^(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?/.exec(text.slice(i));
    if (num) {
      tokens.push({ t: 'num', v: parseFloat(num[0]), col });
      i += num[0].length;
      continue;
    }
    const ident = /^[A-Za-z_][A-Za-z0-9_]*/.exec(text.slice(i));
    if (ident) {
      tokens.push({ t: 'ident', v: ident[0], col });
      i += ident[0].length;
      continue;
    }
    if (c === '"') {
      const end = text.indexOf('"', i + 1);
      if (end < 0) throw new MetricSyntaxError('Unterminated string', line, col);
      tokens.push({ t: 'str', v: text.slice(i + 1, end), col });
      i = end + 1;
      continue;
    }
    const two = text.slice(i, i + 2);
    if (two === '<=' || two === '>=') {
      tokens.push({ t: 'op', v: two, col });
      i += 2;
      continue;
    }
    if ('+-*/^()=<>[],'.includes(c)) {
      tokens.push({ t: 'op', v: c, col });
      i++;
      continue;
    }
    throw new MetricSyntaxError(`Unexpected character '${c}'`, line, col);
  }
  return tokens;
}

class LineParser {
  private pos = 0;

  constructor(
    private readonly tokens: Token[],
    private readonly line: number,
    private readonly graph: Graph,
    private readonly scope: Map<string, number>,
  ) {}

  private error(message: string, tok = this.tokens[this.pos]): never {
    const col = tok ? tok.col : (this.tokens.at(-1)?.col ?? 1);
    throw new MetricSyntaxError(message, this.line, col);
  }

  peek(): Token | undefined {
    return this.tokens[this.pos];
  }

  done(): boolean {
    return this.pos >= this.tokens.length;
  }

  expectEnd() {
    if (!this.done()) this.error(`Unexpected '${this.peek()!.v}'`);
  }

  isOp(v: string): boolean {
    const t = this.peek();
    return t?.t === 'op' && t.v === v;
  }

  op(v: string) {
    if (!this.isOp(v)) this.error(`Expected '${v}'`);
    this.pos++;
  }

  ident(): string {
    const t = this.peek();
    if (t?.t !== 'ident') this.error('Expected a name');
    this.pos++;
    return t.v;
  }

  str(): string {
    const t = this.peek();
    if (t?.t !== 'str') this.error('Expected a quoted string');
    this.pos++;
    return t.v;
  }

  number(): number {
    const negative = this.isOp('-');
    if (negative) this.pos++;
    const t = this.peek();
    if (t?.t !== 'num') this.error('Expected a number');
    this.pos++;
    return negative ? -t.v : t.v;
  }

  expr(): number {
    let left = this.term();
    while (this.isOp('+') || this.isOp('-')) {
      const op = this.peek()!.v;
      this.pos++;
      const right = this.term();
      left = op === '+' ? this.graph.add(left, right) : this.graph.sub(left, right);
    }
    return left;
  }

  private term(): number {
    let left = this.unary();
    while (this.isOp('*') || this.isOp('/')) {
      const op = this.peek()!.v;
      this.pos++;
      const right = this.unary();
      left = op === '*' ? this.graph.mul(left, right) : this.graph.div(left, right);
    }
    return left;
  }

  private unary(): number {
    if (this.isOp('-')) {
      this.pos++;
      return this.graph.neg(this.unary());
    }
    if (this.isOp('+')) {
      this.pos++;
      return this.unary();
    }
    return this.power();
  }

  private power(): number {
    const base = this.atom();
    if (!this.isOp('^')) return base;
    const opTok = this.peek()!;
    this.pos++;
    const exponent = this.unary();
    const e = this.graph.constant(exponent);
    if (e === undefined) this.error('Exponents must be constants', opTok);
    return this.graph.pow(base, e);
  }

  private atom(): number {
    const t = this.peek();
    if (!t) this.error('Expected an expression');
    if (t.t === 'num') {
      this.pos++;
      return this.graph.num(t.v);
    }
    if (this.isOp('(')) {
      this.pos++;
      const inner = this.expr();
      this.op(')');
      return inner;
    }
    if (t.t === 'ident') {
      this.pos++;
      if (this.isOp('(')) {
        if (!(FUNCTIONS as readonly string[]).includes(t.v)) this.error(`Unknown function '${t.v}'`, t);
        this.pos++;
        const arg = this.expr();
        this.op(')');
        return this.graph.fn(t.v as FnName, arg);
      }
      if (t.v === 'pi') return this.graph.num(Math.PI);
      const id = this.scope.get(t.v);
      if (id === undefined) this.error(`Unknown name '${t.v}'`, t);
      return id;
    }
    this.error(`Unexpected '${t.v}'`);
  }
}

export function parseMetric(source: string): ParsedMetric {
  const graph = new Graph();
  const scope = new Map<string, number>();
  const result: ParsedMetric = {
    name: 'Untitled',
    description: '',
    coords: [],
    params: [],
    graph,
    g: new Array(16).fill(graph.num(0)),
    horizon: null,
    disk: null,
    scale: graph.num(1),
  };
  const disk: Partial<DiskSpec> = {};
  const seen = new Set<string>();
  const defineName = (name: string, id: number, line: number, col: number) => {
    if (scope.has(name) || FUNCTIONS.includes(name as FnName) || name === 'pi') {
      throw new MetricSyntaxError(`'${name}' is already defined`, line, col);
    }
    scope.set(name, id);
  };

  source.split('\n').forEach((text, index) => {
    const line = index + 1;
    const tokens = tokenize(text, line);
    if (tokens.length === 0) return;
    const p = new LineParser(tokens, line, graph, scope);
    const keywordTok = tokens[0];
    const keyword = p.ident();
    const needCoords = () => {
      if (result.coords.length === 0) throw new MetricSyntaxError(`'coords' must come before '${keyword}'`, line, keywordTok.col);
    };

    switch (keyword) {
      case 'spacetime':
        result.name = p.peek()?.t === 'str' ? p.str() : p.ident();
        break;
      case 'description':
        result.description = p.str();
        break;
      case 'coords': {
        if (result.coords.length) throw new MetricSyntaxError('coords given twice', line, keywordTok.col);
        while (!p.done()) {
          const col = p.peek()!.col;
          const name = p.ident();
          defineName(name, graph.coord(result.coords.length), line, col);
          result.coords.push(name);
        }
        if (result.coords.length !== 4) throw new MetricSyntaxError('coords needs exactly 4 names', line, keywordTok.col);
        break;
      }
      case 'param': {
        const col = p.peek()?.col ?? keywordTok.col;
        const name = p.ident();
        p.op('=');
        const value = p.number();
        const inTok = p.peek();
        if (inTok?.t !== 'ident' || inTok.v !== 'in') throw new MetricSyntaxError("Expected 'in [min, max]'", line, inTok?.col ?? col);
        p.ident();
        p.op('[');
        const min = p.number();
        p.op(',');
        const max = p.number();
        p.op(']');
        const label = p.peek()?.t === 'str' ? p.str() : name;
        if (!(min <= value && value <= max)) throw new MetricSyntaxError(`Default ${value} is outside [${min}, ${max}]`, line, col);
        defineName(name, graph.param(result.params.length), line, col);
        result.params.push({ name, label, value, min, max });
        break;
      }
      case 'let': {
        const col = p.peek()?.col ?? keywordTok.col;
        const name = p.ident();
        p.op('=');
        defineName(name, p.expr(), line, col);
        break;
      }
      case 'g': {
        needCoords();
        const aTok = p.peek();
        const a = result.coords.indexOf(p.ident());
        const bTok = p.peek();
        const b = result.coords.indexOf(p.ident());
        if (a < 0) throw new MetricSyntaxError('Expected a coordinate name', line, aTok!.col);
        if (b < 0) throw new MetricSyntaxError('Expected a coordinate name', line, bTok!.col);
        const key = `g${Math.min(a, b)}${Math.max(a, b)}`;
        if (seen.has(key)) throw new MetricSyntaxError(`g ${result.coords[a]} ${result.coords[b]} given twice`, line, keywordTok.col);
        seen.add(key);
        p.op('=');
        const value = p.expr();
        result.g[4 * a + b] = value;
        result.g[4 * b + a] = value;
        break;
      }
      case 'horizon': {
        needCoords();
        const lhs = p.expr();
        const opTok = p.peek();
        if (opTok?.t !== 'op' || !['<', '>', '<=', '>='].includes(opTok.v)) {
          throw new MetricSyntaxError("Expected '<' or '>'", line, opTok?.col ?? keywordTok.col);
        }
        p.op(opTok.v);
        const rhs = p.expr();
        result.horizon = opTok.v.startsWith('<') ? graph.sub(lhs, rhs) : graph.sub(rhs, lhs);
        break;
      }
      case 'disk': {
        needCoords();
        const fieldTok = p.peek();
        const field = p.ident();
        if (!['radius', 'omega', 'inner', 'outer'].includes(field)) {
          throw new MetricSyntaxError(`Unknown disk property '${field}'`, line, fieldTok!.col);
        }
        p.op('=');
        disk[field as keyof DiskSpec] = p.expr();
        break;
      }
      case 'scale':
        p.op('=');
        result.scale = p.expr();
        break;
      default:
        throw new MetricSyntaxError(`Unknown statement '${keyword}'`, line, keywordTok.col);
    }
    p.expectEnd();
  });

  if (result.coords.length === 0) throw new MetricSyntaxError("Missing 'coords' statement", 1, 1);
  const diskFields = Object.keys(disk);
  if (diskFields.length > 0) {
    const missing = ['radius', 'omega', 'inner', 'outer'].filter(f => !(f in disk));
    if (missing.length) throw new MetricSyntaxError(`Disk is missing: ${missing.join(', ')}`, 1, 1);
    result.disk = disk as DiskSpec;
  }
  return result;
}
