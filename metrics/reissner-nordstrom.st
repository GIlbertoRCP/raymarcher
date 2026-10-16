spacetime "Reissner–Nordström"
description "A non-rotating black hole with electric charge. Charge pulls the horizon and the photon orbit inward."
coords t x y z
param M = 1 in [0.2, 3] "Mass"
param q = 0.6 in [0, 0.95] "Charge Q/M"

# Kerr-Newman with no spin, in outgoing Kerr-Schild form:
#   g = eta + f l l,  l = (-1, x/r, y/r, z/r),  f = (2 M r - Q^2) / r^2
# Light is uncharged, so it feels the charge only through the metric.
let Q = q*M
let r = sqrt(x^2 + y^2 + z^2)
let f = (2*M*r - Q^2) / r^2
let lx = x / r
let ly = y / r
let lz = z / r

g t t = -1 + f
g t x = -f*lx
g t y = -f*ly
g t z = -f*lz
g x x = 1 + f*lx^2
g x y = f*lx*ly
g x z = f*lx*lz
g y y = 1 + f*ly^2
g y z = f*ly*lz
g z z = 1 + f*lz^2

let rplus = M + sqrt(M^2 - Q^2)
horizon r < rplus*1.01

# Circular geodesics have Omega^2 = (M r - Q^2) / r^4. The innermost stable
# orbit is the real root of rho^3 - 6 rho^2 + 9 q^2 rho - 4 q^4 = 0 (rho = r/M):
# 6M at q = 0, 4M at q = 1. Cardano's formula with s = q^2, whose discriminant
# factors exactly as s^2 (5 - 4 s)(1 - s) >= 0, so it stays finite at both ends.
let s = q^2
let A = 8 - 9*s + 2*s^2
let rootD = s * sqrt((5 - 4*s) * (1 - s))
let rho = 2 + (A + rootD)^(1/3) + (A - rootD)^(1/3)
disk radius = r
disk omega = sqrt(M*r - Q^2) / r^2
disk inner = rho*M
disk outer = 20*M
scale = M
