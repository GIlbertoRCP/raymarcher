import { describe, expect, it } from 'vitest';
import { MetricSyntaxError, parseMetric } from '../src/index.ts';

const header = 'coords t x y z\n';

function errorOf(source: string): MetricSyntaxError {
  try {
    parseMetric(source);
  } catch (e) {
    if (e instanceof MetricSyntaxError) return e;
    throw e;
  }
  throw new Error('Expected a syntax error');
}

describe('parseMetric', () => {
  it('reads name, params, components and fills the rest with zero', () => {
    const m = parseMetric(`spacetime "Test"\n${header}param M = 1 in [0, 2] "Mass"\ng t t = -(1 - 2*M/x)\ng x y = 3`);
    expect(m.name).toBe('Test');
    expect(m.params).toEqual([{ name: 'M', label: 'Mass', value: 1, min: 0, max: 2 }]);
    const at = (i: number) => m.graph.evaluate(m.g[i], [0, 4, 0, 0], [1]);
    expect(at(0)).toBeCloseTo(-0.5, 12);
    expect(at(1 * 4 + 2)).toBe(3);
    expect(at(2 * 4 + 1)).toBe(3); // symmetric
    expect(at(3 * 4 + 3)).toBe(0);
  });

  it('respects precedence: -x^2 is -(x^2) and ^ binds right to left', () => {
    const m = parseMetric(`${header}g t t = -x^2\ng x x = 2^3^2`);
    expect(m.graph.evaluate(m.g[0], [0, 3, 0, 0], [])).toBe(-9);
    expect(m.graph.evaluate(m.g[5], [0, 0, 0, 0], [])).toBe(512);
  });

  it('turns horizon conditions into a function that is negative inside', () => {
    const m = parseMetric(`${header}horizon x < 2\n`);
    expect(m.graph.evaluate(m.horizon!, [0, 1, 0, 0], [])).toBeLessThan(0);
    expect(m.graph.evaluate(m.horizon!, [0, 3, 0, 0], [])).toBeGreaterThan(0);
    const flipped = parseMetric(`${header}horizon 2 > x\n`);
    expect(flipped.graph.evaluate(flipped.horizon!, [0, 1, 0, 0], [])).toBeLessThan(0);
  });

  it('ignores comments and blank lines', () => {
    expect(() => parseMetric(`# a comment\n\n${header}g t t = -1 # trailing\n`)).not.toThrow();
  });

  it.each([
    ['g t t = -1', 1, 1, "'coords' must come before 'g'"],
    [`${header}g t t = -1 + q`, 2, 14, "Unknown name 'q'"],
    [`${header}g t t = x^y`, 2, 10, 'Exponents must be constants'],
    [`${header}g t t = -1\ng t t = -2`, 3, 1, 'given twice'],
    [`${header}param M = 5 in [0, 2]`, 2, 7, 'outside'],
    [`${header}g t w = 1`, 2, 5, 'Expected a coordinate name'],
    [`${header}g t t = foo(x)`, 2, 9, "Unknown function 'foo'"],
    [`${header}let x = 2`, 2, 5, "'x' is already defined"],
    [`${header}g t t = (1 + x`, 2, 14, "Expected ')'"],
    ['coords t x y', 1, 1, 'exactly 4'],
    [`${header}disk inner = 3`, 1, 1, 'Disk is missing'],
    [`${header}frobnicate`, 2, 1, "Unknown statement 'frobnicate'"],
  ])('reports errors with their position: %s', (source, line, column, message) => {
    const e = errorOf(source);
    expect(e.message).toContain(message);
    expect([e.line, e.column]).toEqual([line, column]);
  });
});
