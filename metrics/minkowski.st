spacetime "Flat space"
description "No gravity: light travels in straight lines. The disk orbits as if a mass M were present, for comparison."
coords t x y z
param M = 1 in [0.2, 3] "Disk mass scale"

g t t = -1
g x x = 1
g y y = 1
g z z = 1

let r = sqrt(x^2 + y^2)
disk radius = r
disk omega = sqrt(M) / r^1.5
disk inner = 6*M
disk outer = 20*M
scale = M
