import { describe, expect, it } from 'vitest';
import {
  criticalImpactParameter,
  horizonRadius,
  isotropicFromSchwarzschild,
  photonSphereRadius,
  refractiveIndex,
  refractiveIndexGradient,
  schwarzschildFromIsotropic,
  traceRay,
} from './physics.ts';

describe('isotropic <-> Schwarzschild radius', () => {
  it('maps the horizon r = M/2 to R = 2M', () => {
    expect(schwarzschildFromIsotropic(0.5, 1)).toBeCloseTo(2, 12);
    expect(isotropicFromSchwarzschild(2, 1)).toBeCloseTo(0.5, 12);
  });

  it('round-trips for radii outside the horizon', () => {
    for (const M of [0.5, 1, 1.8, 3.5]) {
      for (const R of [2 * M, 3 * M, 6 * M, 20 * M, 1000 * M]) {
        expect(schwarzschildFromIsotropic(isotropicFromSchwarzschild(R, M), M)).toBeCloseTo(R, 9);
      }
    }
  });

  it('approaches r = R far from the mass', () => {
    const R = 1e6;
    expect(isotropicFromSchwarzschild(R, 1) / R).toBeCloseTo(1, 5);
  });

  it('rejects radii inside the horizon', () => {
    expect(() => isotropicFromSchwarzschild(1.5, 1)).toThrow(RangeError);
  });
});

describe('characteristic radii', () => {
  it('places the horizon at R = 2M', () => {
    expect(horizonRadius(1.8)).toEqual({ schwarzschild: 3.6, isotropic: 0.9 });
  });

  it('places the photon sphere at R = 3M with a consistent isotropic radius', () => {
    const M = 1.8;
    const ps = photonSphereRadius(M);
    expect(ps.schwarzschild).toBeCloseTo(3 * M, 12);
    expect(schwarzschildFromIsotropic(ps.isotropic, M)).toBeCloseTo(3 * M, 12);
    expect(ps.isotropic).toBeCloseTo(isotropicFromSchwarzschild(3 * M, M), 12);
  });

  it('gives the critical impact parameter sqrt(27) M', () => {
    expect(criticalImpactParameter(1)).toBeCloseTo(5.196152422706632, 12);
  });
});

describe('refractive index', () => {
  it('tends to 1 far away', () => {
    expect(refractiveIndex(1e8, 1)).toBeCloseTo(1, 6);
  });

  it('matches the weak-field limit n ~ 1 + 2M/r', () => {
    const r = 1e4;
    expect(refractiveIndex(r, 1) - 1).toBeCloseTo(2 / r, 7);
  });

  it('diverges at the horizon', () => {
    expect(refractiveIndex(0.5 + 1e-9, 1)).toBeGreaterThan(1e8);
  });
});

describe('reference ray integrator', () => {
  const M = 1;
  const bCrit = criticalImpactParameter(M);

  it('matches dn/dr against a finite difference', () => {
    for (const r of [0.6, 1, 1.866, 5, 50]) {
      const eps = 1e-6 * r;
      const fd = (refractiveIndex(r + eps, M) - refractiveIndex(r - eps, M)) / (2 * eps);
      expect(refractiveIndexGradient(r, M) / fd).toBeCloseTo(1, 6);
    }
  });

  it('captures rays just inside the critical impact parameter sqrt(27) M', () => {
    expect(traceRay(0.99 * bCrit, M).captured).toBe(true);
  });

  it('lets rays just outside the critical impact parameter escape', () => {
    expect(traceRay(1.01 * bCrit, M).captured).toBe(false);
  });

  it('reproduces the weak-field deflection 4M/b + 15 pi M^2 / 4b^2', () => {
    const b = 1000 * M;
    const { direction } = traceRay(b, M, 1e6 * M);
    const deflection = Math.atan2(-direction[1], direction[0]);
    const expected = (4 * M) / b + (15 * Math.PI * M * M) / (4 * b * b);
    expect(Math.abs(deflection - expected) / expected).toBeLessThan(1e-3);
  });
});
