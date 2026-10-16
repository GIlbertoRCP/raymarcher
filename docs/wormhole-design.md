# Design spike: the Ellis–Bronnikov wormhole

**Recommendation:** model the wormhole in a conformally flat Cartesian chart that covers both universes. It needs two small additions and no change to how metrics are written:

- an `exit` statement in the metric language, which names a second sky;
- a step-size rule based on coordinate displacement, shared by all three integrators.

A prototype with today's compiler already finds the analytic crossing threshold exactly.

## The problem

The tracer assumes things that a wormhole breaks:

1. **Space is one copy of R³ in x, y, z.** The camera orbits the origin, and a ray "escapes" when |x| grows past a radius. A wormhole has two asymptotically flat regions joined at a throat.
2. **There is one sky.** Rays that pass through the throat should see the *other* universe's sky. That's the whole visual point of the *Interstellar* wormhole (James, von Tunzelmann, Franklin & Thorne 2015, *Am. J. Phys.* 83, 486).

## The chart

The massless Ellis–Bronnikov wormhole with throat radius a is

```
ds² = −dt² + dl² + (l² + a²)(dθ² + sin²θ dφ²),   l ∈ (−∞, ∞)
```

Substituting l = ρ − a²/(4ρ) with ρ ∈ (0, ∞) gives

```
dl² + (l² + a²) dΩ² = W² (dρ² + ρ² dΩ²),   W = 1 + a² / (4ρ²)
```

This is conformally flat. In Cartesian coordinates with ρ = |x|, the spatial metric is simply W² δ_ij:

| Region | Meaning |
| --- | --- |
| ρ → ∞ | infinity of our universe (l → +∞); W → 1, so physical and coordinate lengths agree |
| ρ = a/2 | the throat (l = 0); the circumference radius ρW reaches its minimum, a |
| ρ → 0 | infinity of the other universe (l → −∞); the inversion ρ → a²/(4ρ) maps it to the outside |

So the existing metric language describes the wormhole as it is:

```
spacetime "Ellis wormhole"
coords t x y z
param a = 1 in [0.2, 3] "Throat radius"
let rho2 = x^2 + y^2 + z^2
let W = 1 + a^2 / (4*rho2)
g t t = -1
g x x = W^2
g y y = W^2
g z z = W^2
scale = a
```

## Evidence from the spike

The metric above was compiled with the current compiler and traced with the current double-precision integrator. Rays start 2000a away, as camera rays (backwards in time):

| Impact parameter | Result | Constraint drift |
| --- | --- | --- |
| 0.90 a | reaches ρ ≈ 0.03a: crossed to the other side | 1e−12 |
| 0.99 a | crossed | 1e−12 |
| 1.01 a | escaped back to our universe | 9e−13 |
| 1.10 a | escaped | 6e−13 |

The analytic result is that a photon crosses exactly when its impact parameter is below the throat radius a, because the photon sphere is the throat itself. The chart reproduces it without modification.

The spike also shows the two gaps. Crossing rays never terminate: they end "unfinished" after the step budget. This is because:
- coordinate speed near the origin falls as 1/W² ~ 16ρ⁴/a⁴;
- the current step h = f·r in affine parameter therefore moves them ever more slowly.

## What to build

### 1. `exit` statements (compiler + all three backends)

```
exit rho2 < (0.02*a)^2 "other side"
```

`exit` generalises `horizon`: a boolean condition plus a sky label.
- **Termination:** a ray that meets it stops, like capture, and records which exit it took.
- **Sky lookup:** the renderer looks up the second sky using the ray's *position* direction, normalize(x). Near ρ → 0, the inversion makes the position direction the far side's celestial direction. This differs from the escape case, which uses the velocity direction.
- **Fate values:** the fate map gains an "exit k" value per statement, so the cross-backend gate covers it.
- **Emission:** `compileToWgsl`, `compileToCuda` and `compileToJs` each emit `exit_value_k(x)`, using the same pattern as `horizon_value`.

### 2. Step size from coordinate displacement (WGSL, CUDA, JS together)

Replace h = f·r with h = f·r / |dx/dλ|_coord, so each step moves a fixed fraction of the coordinate radius whatever the metric's conformal factor.
- For Kerr-Schild Schwarzschild and Kerr, |dx/dλ| ≈ 1 far away, so most rays behave as before. Close to the horizon, however, steps shrink, which changes step counts and timings.
- **This touches the cross-backend gate and the fp64 suite**, so it lands as one change across all three integrators. Then re-run on the 3070:
  - `npm run export-cuda`
  - `--validate`
  - the 27-scene comparison
  - the benchmarks

### 3. Two skies

- The sky functions take a sky index: 0 is our universe, k is exit k.
- The second sky needs to be visibly different: the *Interstellar* paper uses two photographs. Without image assets, use a different star seed and a tinted lat-long grid, labelled in the key.
- Later, an equirectangular image per sky can be loaded into a texture array.

### 4. Validation

All analytic, in the style of the existing suite:

- **Crossing threshold:** rays with b < a cross, rays with b > a return (the spike's test, made exact by bisection: the edge should be at a to 1e−9).
- **Deflection:** the deflection angle for b > a has a closed form in complete elliptic integrals (Ellis wormhole lensing, e.g. Nakajima & Asada 2012, *Phys. Rev. D* 85, 107501). Compare it across b.
- **Visual comparison:** James et al. 2015, Figures 1 and 6, at matched throat length and camera distance. The *Interstellar* wormhole adds a lensing length and a smoothed mass term beyond Ellis's metric; that version is a second metric file, not a change to this one.

## Effort estimate

The compiler and language changes (`exit`) are small. The step rule is a one-line change per backend, plus re-running the gate and validation on the 3070. The two-sky shading and the wormhole metric file are about a day of work. Validation adds the elliptic-integral reference.

## Not chosen

- **Spherical coordinates (t, l, θ, φ):** they cover both universes and the throat directly, but put coordinate singularities on the polar axis that every off-equator ray crosses. That needs special handling in a generic tracer.
- **Per-ray 2D integration using spherical symmetry** (the DNEG approach): it's fast and accurate, but specific to symmetric metrics, which defeats the compiler.
- **Two coordinate patches with a sheet flag:** this works, but adds patch transitions to every backend. The conformal chart gets the same result with one chart.
