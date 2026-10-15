spacetime "Kerr"
description "A rotating black hole in outgoing Kerr-Schild coordinates. Spin drags spacetime around with it."
coords t x y z
param M = 1 in [0.2, 3] "Mass"
param a = 0.6 in [0, 0.99] "Spin a/M"

# Outgoing Kerr-Schild form g = eta + f l l: the time reverse of the ingoing
# form in Visser, arXiv:0706.0622, with a -> -a so the hole still spins
# toward +phi. Rays traced backwards from the camera cross the past horizon
# smoothly in these coordinates.
# The Boyer-Lindquist radius r is defined implicitly by
#   r^4 - (x^2 + y^2 + z^2 - A^2) r^2 - A^2 z^2 = 0
let A = a*M
let w = x^2 + y^2 + z^2 - A^2
let r2 = (w + sqrt(w^2 + 4*A^2*z^2)) / 2
let r = sqrt(r2)
let f = 2*M*r^3 / (r2^2 + A^2*z^2)
let lx = (r*x - A*y) / (r2 + A^2)
let ly = (r*y + A*x) / (r2 + A^2)
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

let rplus = M + sqrt(M^2 - A^2)
horizon r < rplus*1.01

# Prograde circular orbits; inner edge at the innermost stable orbit (Bardeen, Press & Teukolsky 1972)
let s = a
let Z1 = 1 + (1 - s^2)^(1/3) * ((1 + s)^(1/3) + (1 - s)^(1/3))
let Z2 = sqrt(3*s^2 + Z1^2)
disk radius = r
disk omega = sqrt(M) / (r^1.5 + A*sqrt(M))
disk inner = M*(3 + Z2 - sqrt((3 - Z1)*(3 + Z1 + 2*Z2)))
disk outer = 20*M
scale = M
