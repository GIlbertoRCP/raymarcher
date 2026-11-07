// Emission at the disk crossing for Phase 4's forward model (docs/phase-4-plan.md,
// step 4). Mirrors compiler/src/image-model.ts (thinDisk, ImageModel.diskIntensity).
// Templated on Real, so with Dual numbers (dual.h) the intensity's derivatives
// with respect to every traced input come along.
//
// MODELLING CHOICE: optically thick thin disk, bolometric, I = g^4 F(r) tau(r):
//   g   = 1 / (p . u_gas)  (p . u = 1 at the camera; u_gas from disk_gas_velocity)
//   F   = (1 - sqrt(r_in / r)) (r_in / r)^3 / 0.05666   (peak-normalised, zero at r_in)
//   tau = (1 - s^2)^2, s = (r - r_t) / (r_out - r_t), r_t = taper_start r_out  (C^1, zero at r_out)
#pragma once

#include "rm/geodesic.h"

namespace rm {

template <class M, typename Real>
RM_HD inline Real thin_disk_intensity(const Real* params, const Real y[8], Real taper_start) {
  Real u[4];
  if (!disk_gas_velocity<M>(params, y, u)) return Real(0);
  const Real r = M::disk_radius(y, params);
  const Real r_in = M::disk_inner(y, params);
  const Real r_out = M::disk_outer(y, params);
  const Real pu = y[4] * u[0] + y[5] * u[1] + y[6] * u[2] + y[7] * u[3];
  const Real g = Real(1) / pu;
  const Real q = r_in / r;
  const Real F = (Real(1) - sqrt(q)) * (q * q * q) / Real(0.05666);
  const Real rt = taper_start * r_out;
  Real tau = Real(1);
  if (!(r < r_out)) {
    tau = Real(0);
  } else if (r > rt) {
    const Real s = (r - rt) / (r_out - rt);
    const Real w = Real(1) - s * s;
    tau = w * w;
  }
  const Real g2 = g * g;
  return g2 * g2 * F * tau;
}

}  // namespace rm
