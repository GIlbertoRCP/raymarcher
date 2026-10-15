spacetime "Schwarzschild"
description "A non-rotating black hole, in outgoing Kerr-Schild coordinates."
coords t x y z
param M = 1 in [0.2, 3] "Mass"

# g = eta + f l l with l = (-1, x/r, y/r, z/r): outgoing Kerr-Schild form.
# The renderer traces rays backwards in time from the camera; in outgoing
# coordinates those rays cross the (past) horizon smoothly instead of piling
# up against it.
let r = sqrt(x^2 + y^2 + z^2)
let f = 2*M / r
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

horizon r < 2*M*1.01

# Thin disk of circular geodesic orbits, from the innermost stable orbit outward
disk radius = r
disk omega = sqrt(M) / r^1.5
disk inner = 6*M
disk outer = 20*M
scale = M
