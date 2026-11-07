// Small forward-mode dual numbers for the JS reference: a Float64Array
// [value, t_0 .. t_{P-1}]. Used for the camera frame (tangent-geodesic.ts)
// and the disk intensity (image-model.ts); the ray itself is differentiated
// with the compiler's symbolic JVP.

export type D = Float64Array;

export function dual(P: number, v: number, t?: ArrayLike<number>): D {
  const d = new Float64Array(1 + P);
  d[0] = v;
  if (t) for (let i = 0; i < P; i++) d[1 + i] = t[i];
  return d;
}
export const add = (a: D, b: D): D => a.map((x, i) => x + b[i]);
export const sub = (a: D, b: D): D => a.map((x, i) => x - b[i]);
export const scl = (a: D, s: number): D => a.map(x => x * s);
export const mul = (a: D, b: D): D => a.map((x, i) => (i === 0 ? a[0] * b[0] : a[0] * b[i] + x * b[0]));
export const div = (a: D, b: D): D => a.map((x, i) => (i === 0 ? a[0] / b[0] : (x * b[0] - a[0] * b[i]) / (b[0] * b[0])));
export const sqrtD = (a: D): D => {
  const r = Math.sqrt(a[0]);
  return a.map((x, i) => (i === 0 ? r : x / (2 * r)));
};
export const sinD = (a: D): D => a.map((x, i) => (i === 0 ? Math.sin(a[0]) : Math.cos(a[0]) * x));
export const cosD = (a: D): D => a.map((x, i) => (i === 0 ? Math.cos(a[0]) : -Math.sin(a[0]) * x));
/** a^n for a constant n. */
export const powD = (a: D, n: number): D => {
  const p = Math.pow(a[0], n);
  const s = n * Math.pow(a[0], n - 1);
  return a.map((x, i) => (i === 0 ? p : s * x));
};

/** g(a, b) for dual 4-vectors and a dual metric (16 entries). */
export function dotD(g: D[], a: D[], b: D[]): D {
  const P = a[0].length - 1;
  let s = dual(P, 0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) s = add(s, mul(g[4 * i + j], mul(a[i], b[j])));
  return s;
}
