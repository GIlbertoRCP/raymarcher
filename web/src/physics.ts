// Schwarzschild spacetime in isotropic coordinates (G = c = 1).
//
// The renderer works in isotropic radius r, where light rays follow the
// optical metric with refractive index n(r). Physical radii quoted in the
// literature (horizon at 2M, photon sphere at 3M, ISCO at 6M) are areal
// Schwarzschild radii R, related by R = r (1 + M / 2r)^2.

/** Areal (Schwarzschild) radius R for an isotropic radius r. */
export function schwarzschildFromIsotropic(r: number, mass: number): number {
  const k = 1 + mass / (2 * r);
  return r * k * k;
}

/** Isotropic radius r for an areal radius R (outer branch, valid for R >= 2M). */
export function isotropicFromSchwarzschild(R: number, mass: number): number {
  if (R < 2 * mass) {
    throw new RangeError(`R = ${R} is inside the horizon (2M = ${2 * mass})`);
  }
  return (R - mass + Math.sqrt(Math.max(0, R * R - 2 * mass * R))) / 2;
}

/** Effective refractive index of the optical metric, n(r) = (1 + u)^3 / (1 - u), u = M / 2r. */
export function refractiveIndex(r: number, mass: number): number {
  const u = mass / (2 * r);
  return Math.pow(1 + u, 3) / (1 - u);
}

/** Event horizon: R = 2M, which is r = M / 2 in isotropic coordinates. */
export function horizonRadius(mass: number) {
  return { schwarzschild: 2 * mass, isotropic: mass / 2 };
}

/** Photon sphere: R = 3M, which is r = M (1 + sqrt(3) / 2) in isotropic coordinates. */
export function photonSphereRadius(mass: number) {
  return { schwarzschild: 3 * mass, isotropic: mass * (1 + Math.sqrt(3) / 2) };
}

/** Critical impact parameter b = sqrt(27) M: rays with b below this are captured. */
export function criticalImpactParameter(mass: number): number {
  return Math.sqrt(27) * mass;
}

// --- Reference ray integrator -------------------------------------------
//
// A double-precision mirror of the shader's ray equations, used by the tests
// to check the physics against known results.
//
// With coordinate time t as the parameter, light in a medium of index n(r)
// obeys (from the optical Hamiltonian H = (p^2 - n^2) / 2):
//
//   dx/dt = v,  |v| = 1/n
//   dv/dt = grad(n) / n^3 - 2 v (grad(n) . v) / n

type V3 = [number, number, number];

/** dn/dr for n(r) = (1 + u)^3 / (1 - u), u = M / 2r. */
export function refractiveIndexGradient(r: number, mass: number): number {
  const u = mass / (2 * r);
  return (-2 * u * (1 + u) ** 2 * (2 - u)) / ((1 - u) ** 2 * r);
}

export function rayAcceleration(x: V3, v: V3, mass: number): V3 {
  const r = Math.hypot(x[0], x[1], x[2]);
  const n = refractiveIndex(r, mass);
  const k = refractiveIndexGradient(r, mass) / r; // grad(n) = k x
  const gradDotV = k * (x[0] * v[0] + x[1] * v[1] + x[2] * v[2]);
  return [0, 1, 2].map(i => (k * x[i]) / n ** 3 - (2 * v[i] * gradDotV) / n) as V3;
}

export interface RayResult {
  captured: boolean;
  /** Final unit direction of travel (meaningful when not captured). */
  direction: V3;
  steps: number;
}

/**
 * Traces a ray that starts far away at x = -start, moving in +x with impact
 * parameter b along y. Uses RK4 with a step proportional to distance from the horizon.
 */
export function traceRay(impactParameter: number, mass: number, start = 1e4 * mass, stepFraction = 2e-3): RayResult {
  const rHorizon = mass / 2;
  let x: V3 = [-start, impactParameter, 0];
  const n0 = refractiveIndex(Math.hypot(...x), mass);
  let v: V3 = [1 / n0, 0, 0];
  const add = (a: V3, b: V3, s: number): V3 => [a[0] + s * b[0], a[1] + s * b[1], a[2] + s * b[2]];

  for (let steps = 0; steps < 1_000_000; steps++) {
    const r = Math.hypot(...x);
    if (r < rHorizon * 1.001) return { captured: true, direction: [0, 0, 0], steps };
    if (r > start * 1.01 && x[0] > 0 === v[0] > 0 && steps > 10) {
      const len = Math.hypot(...v);
      return { captured: false, direction: [v[0] / len, v[1] / len, v[2] / len], steps };
    }
    const h = stepFraction * (r - rHorizon) * refractiveIndex(r, mass);
    const a1 = rayAcceleration(x, v, mass);
    const x2 = add(x, v, h / 2), v2 = add(v, a1, h / 2);
    const a2 = rayAcceleration(x2, v2, mass);
    const x3 = add(x, v2, h / 2), v3 = add(v, a2, h / 2);
    const a3 = rayAcceleration(x3, v3, mass);
    const x4 = add(x, v3, h), v4 = add(v, a3, h);
    const a4 = rayAcceleration(x4, v4, mass);
    x = [0, 1, 2].map(i => x[i] + (h / 6) * (v[i] + 2 * v2[i] + 2 * v3[i] + v4[i])) as V3;
    v = [0, 1, 2].map(i => v[i] + (h / 6) * (a1[i] + 2 * a2[i] + 2 * a3[i] + a4[i])) as V3;
  }
  throw new Error('traceRay did not terminate');
}
