# Phase 4: the inverse problem

Given an image of a black hole, infer spin a, inclination i, angular size θ_g = GM/(c²D) and position angle, with honest, calibrated uncertainties. The tracer becomes differentiable, fits are done by gradients, and the posterior is sampled and checked for calibration.

The demo is not "we recover a = 0.94 ± 0.03". It is: recover parameters from synthetic data with known noise, show the full posterior including degeneracies, show how the spin uncertainty shrinks with resolution and signal-to-noise, and say what data like the EHT's can and cannot tell you.

Ground rules are unchanged: the user commits and pushes; every physics or statistics claim has a test with a number, a tolerance and a pass/fail; modelling choices are labelled; benchmarks state hardware, driver, resolution, scene and integrator.

## Where Phase 3 actually stands (checked 2026-11-06)

| Item | State |
| --- | --- |
| RK4 in three backends, RK45 fp64 on CUDA | done |
| `horizon`, `disk` (optically thick), Reissner–Nordström | done, validated |
| Cross-backend gate | 27 scenes, passes |
| Ellis wormhole (`exit`, second sky) | design only (`docs/wormhole-design.md`) |
| Optically thin disk | not found |
| Essay (`web/essay.html`) | not started |

Phase 4 steps 1–8 need none of the missing items. They stay open as Phase 3 work; the step 0 gate is 27 scenes until the wormhole lands.

## Gates

1. **Gradients are right.** Per-ray sensitivities agree with central finite differences in fp64 (relative 10⁻⁶); symbolic tangents agree with an independent dual-number implementation (10⁻¹⁰); the image-loss gradient agrees with finite differences of supersampled images to 1%.
2. **Posteriors are calibrated.** Simulation-based calibration passes for (a, i, θ_g, PA) on synthetic image-domain data over at least 200 prior draws; 68% and 95% coverage within the binomial band.
3. **Results are published** on the verification page and in an essay chapter, every number read from `validation/results/`.

## Key design decisions

- Only a and i are traced with tangents. θ_g and position angle are image-plane transforms with exact, free derivatives. Rendering in units of M at ≥ 1000 M decouples M.
- Forward mode, not adjoint: P ≈ 2–4 traced parameters, cost ≈ (1 + P)× a render, full per-pixel Jacobian (needed by Gauss–Newton and Laplace).
- Discretise, then differentiate: gradients of the integrator's actual output.
- Two independent derivative implementations: compiler-emitted symbolic JVPs (all backends) and `Dual<Real, N>` in C++.
- An image continuous in θ: black sky, flux → 0 at r_in, smooth taper at r_out. The remaining discontinuity (critical curve) gets an edge term, measured in step 5.
- Sensitivities grow like e^(γn) near the photon ring; measure clipping vs fp64 on the loss gradient.

## Amendments (from review, 2026-11-06)

1. **Pluggable emission model from step 4.** Define an emission interface (emissivity, its tangents, its parameters) instead of hard-wiring the thin disk. Add a *misspecification test* to step 8: simulate with one emission model, fit with another, report the bias. SBC alone only proves self-consistency.
2. **Render API shaped for a later adjoint.** `render(θ)` takes a batch dimension and offers both J and Jᵀv (Jᵀv from J for now), so a pixelated or neural emission map can use an adjoint without changing the Python side.
3. **fp64 on the 3070 is 1/64 of fp32.** Measure fp64-vs-clipped-fp32 near the ring at the *start* of step 4; it decides the production kernel.
4. **Budget SBC early.** Estimate 200 datasets × NUTS gradient evaluations × render cost at the end of step 3. Pin the Python stack; consider NumPyro or BlackJAX alongside Pyro (the likelihood is an external call either way).
5. **Step 9 made concrete.**
   - CNN embedding → conditional normalising flow (`sbi` or `zuko`); neural ratio estimation as a baseline.
   - Gradient-augmented training (joint score matching, Brehmer et al. "Mining gold"), using the Jacobians the tracer already computes; measure the simulation-count saving.
   - SBC on ≥ 10⁴ datasets (cheap when amortised) plus TARP expected coverage (Lemos et al. 2023).
   - Train on a family of emission models; evaluate on a held-out family; add an OOD score.
   - Datasets versioned by generator git hash and seed.
6. **Later phase:** neural-field emission fitted jointly with spin, via the adjoint.

## Steps

0. Phase 3 landed and validated on the GPU (see table above).
1. **Compiler tangents.** `compiler/src/tangent.ts`: JVP of `metric_eval` (g, ∂g, δg, δ∂g) and of `disk_radius/omega/inner/outer`, for JS and CUDA first, WGSL after step 3. Tests against finite differences for every catalog metric; generated C++ JVP checked against JS to 10⁻¹². Measure code size, NVRTC compile time and registers for Kerr.
2. **Tangent integrator in JS (fp64).** Exact tangent of each RK4 stage, of the disk-crossing interpolation and of the camera frame. Per-ray FD checks; analytic sensitivities (shadow angle vs R, ∂b_crit/∂a, dr_ISCO/da, Schwarzschild d(deflection)/db).
3. **CUDA tangents + dual check.** `native/src/rm/tangent.h`, `native/src/rm/dual.h`; image kernel with P tangent channels; WGSL tangent path; fp64 GPU "sensitivities" check.
4. **Differentiable image formation.** I = g⁴ F(r) τ(r); image-plane scale/rotation; Gaussian beam; fixed stratified supersampling.
5. **Critical-curve edge term**, analytic (Bardeen outline) and generic (bisection + implicit function theorem). Gate 1.
6. **Observation model.** Image domain first (χ²); then visibilities, closure phases, log closure amplitudes, station gains; EHT 2017 (u, v) tracks.
7. **Inference engine** (`inference/`, Python): C API + ctypes, PyTorch autograd.Function, multi-start L-BFGS, Laplace (Gauss–Newton Fisher), NUTS, emcee cross-check.
8. **Calibration and information curves.** SBC ≥ 200 datasets; posterior width vs beam, SNR, inclination; misspecification test. Gate 2.
9. **Amortised inference** (see amendment 5).
10. **Real data (stretch):** EHT 2017 M87* with a declared optically thin emission model; compare θ_g with EHT Paper VI (3.8 ± 0.4 µas).
11. **Publish:** verification-page sections, essay chapter 7 with an in-browser fit, README section.
12. **Benchmarks:** gradient cost vs plain render (symbolic vs dual, fp32 vs fp64), registers/occupancy, rays/s, NUTS samples/min.

Stop and report at each checkpoint (after steps 0, 2, 3, 5, 8, 12) with numbers and plots.

## Step 1 results (2026-11-06)

Code: `compiler/src/tangent.ts` (JVP graph + JS and C++ emitters), `diff.ts` (`derive`, `differentiateParam`, `directional`), `emit.ts` (`emitSplit`), tests in `compiler/test/tangent.test.ts`, native check `generated_jvp_matches_js_backend`, measurement tool `native/tools/jvp_cost.cpp`.

| Check | Measured | Tolerance | Result |
| --- | --- | --- | --- |
| JS JVP vs central FD (g, ∂g), worst over 4 metrics × 3 param sets × 12 points × 3 directions | 6.4e-9 (RN), 5.4e-9 (Kerr) | 1e-6 | pass |
| JS JVP vs explicit symbolic second/parameter derivatives | 4.9e-15 (Kerr) | 1e-12 | pass |
| Disk JVP vs central FD (h = 1e-5) | FD error falls as h² (4.6e-7 → 4.6e-9 → 4.6e-11 for h = 1e-4 → 1e-6, Kerr ISCO at a = 0.95) | 1e-6 | pass |
| dr_ISCO/da at a = 1e-4 vs −4√(2/3) − (7/9)a | < 1e-4 | 1e-4 | pass |
| dr_ISCO/da at a = 0.98 vs FD of an independent BPT implementation | < 1e-6 | 1e-6 | pass |
| Generated C++ JVP vs JS (P = 2, 16 points per metric) | 0 (bit-identical) | 1e-12 | pass |
| Mutation test (direction offset of dθ broken) | 10 of 19 tests fail | — | tests are not vacuous |

**Finding: dr_ISCO/da is not finite at exactly a = 0.** The BPT formula has a |a| kink there (3 − Z₁ = O(a²) under a square root). Priors must exclude a = 0 or the disk inner edge must be reparametrised before step 8.

**GPU cost** (RTX 3070, driver 617.42, NVRTC 13.3, 2²⁰ points, no fast math, no register cap; `docs/benchmarks/phase4-step1-jvp-3070.json`):

| Metric | Precision | eval | JVP P = 1 | JVP P = 2 | JVP P = 2 spill (stack) |
| --- | --- | --- | --- | --- | --- |
| Schwarzschild | fp64 | 1.97 ns, 70 regs | 8.1 ns | 11.6 ns | 1,024 B |
| Kerr | fp32 | 0.079 ns, 93 regs | 3.39 ns | 12.4 ns | 2,632 B |
| Kerr | fp64 | 3.55 ns, 168 regs | 27.3 ns | 44.5 ns | 7,432 B |

NVRTC compile time for Kerr: 0.65 s (P = 1) to 1.7 s (P = 4). Emitted Kerr code: 241 statements for eval, 245 shared + 986 per direction for the JVP. The fp32 eval kernel is memory-bound, so fp32 ratios overstate the arithmetic cost.

The spills come from liveness, not size. Emitting the directions interleaved (unrolled, dependency-ordered, no loop) instead of hoisting all direction-independent temporaries cuts Kerr's stack frame from 2,632 to 768 B (fp32, P = 2) and from 7,432 to 3,984 B (fp64), measured with `nvcc --resource-usage`. The rest is the 80(1 + P) outputs the metric-level interface must hold. **Decision for step 3:** the production CUDA kernel should use a fused tangent right-hand side (8(1 + P) outputs, contracting δg and δ∂g with k inside the generated code, like `geodesic_rhs`), compared against `Dual<Real, N>`. `metric_jvp` stays as the reference and the building block for the JS integrator.

## Step 2 results (2026-11-06)

Code: `compiler/src/tangent-geodesic.ts` (`TangentSpacetime`: camera in dual numbers mirroring `static_camera`, tangent RHS, tangent RK4 mirroring `trace_ray_rk4` including dh/dθ of the state-dependent step and the disk-crossing interpolation), a length-scale JVP in `tangent.ts`, tests in `compiler/test/tangent-geodesic.test.ts`. θ = (metric params, camRadius, inclination, yaw, u, v), all traced at once.

| Check | Measured | Tolerance | Result |
| --- | --- | --- | --- |
| Per-ray dy_final/dθ vs 4-point central FD, Schwarzschild; Kerr a = 0.3 and 0.94; RN q = 0.6 (R = 50 M, i = 60°, 16 rays each, escaped and disk) | 3.4e-11, 4.3e-9, 1.1e-8, 6.9e-8 over 95–109 (ray, input) pairs; 1–6 pairs skipped where the FD step changed the step count | 1e-6 | pass |
| d(shadow angle)/dR, Schwarzschild static observer, R = 4, 10, 50, 1000 M | ≤ 1.2e-15 | 1e-9 | pass |
| ∂b_crit/∂a, equatorial Kerr, a = 0.5 and 0.9, both sides, implicit-function estimate at b − b_crit = 1e-3, 1e-4, 1e-5 | converges linearly; 1.5e-5 to 1.1e-4 at 1e-5 | 1e-3 and decreasing | pass |
| d(sweep)/du vs the exact Schwarzschild orbit integral (finite radii), b = 6, 10, 30, 100 M | 9e-11 to 7.5e-9 | 1e-6 | pass |
| Mutations: drop dh/dθ; drop the disk-interpolation df/dθ | FD error 1.1–1.3 on every case | — | tests catch both |

dr_ISCO/da is checked at the function level (step 1). Sign convention worth remembering: rays are traced backwards, so p_t > 0 and E = −p_t < 0; b = L/E is the physical impact parameter, and the traced curve turns with sign(L) = −sign(b).

The ∂b_crit/∂a check is the first use of the implicit function theorem on fate boundaries (step 5's generic edge velocity): −(∂φ/∂a)/(∂φ/∂b) from escaping rays near the critical curve converges to the boundary velocity with error ∝ (b − b_crit).

## Step 3 results (2026-11-06)

Code: `native/src/rm/dual.h` (`Dual<T, N>`), `native/src/rm/tangent.h` (symbolic tangent RK4 from `M::jvp`; dual camera rays; helpers), `static_camera_t` / `camera_ray_t` templated on Real in `geodesic.h` (the double `static_camera` now calls them), tangent batch kernels `trace_tangent_rays_kernel` / `trace_dual_rays_kernel` in `kernels/tracer.cu` (`ModuleOptions.tangents` → `-DRM_TANGENT_P`), fp64 check `sensitivities` in `validate.cpp`, `raymarcher --bench-tangents`. JS fixture `native/generated/fixtures/tangent-rays.json` (from `export-cuda`).

**Fix found by the dual check:** RK45 located the disk crossing by bisection, which on duals gives the crossing parameter zero derivative (duals vs FD: 0.92 relative). One Newton step after the bisection carries dθ = −δz/z′ (implicit function theorem); in double it only polishes the root. After the fix: 1.2e-8.

| Check | Measured | Tolerance | Result |
| --- | --- | --- | --- |
| CPU: C++ symbolic tangents vs JS reference, all inputs, 4 metrics × 5–6 rays | ≤ 2.3e-14 | 1e-9 | pass |
| CPU: duals vs symbolic tangents (RK4) | ≤ 5.5e-15 | 1e-10 | pass |
| CPU: duals through RK45 (rtol 1e-9, Kerr a = 0.9, disk hit, 13 rejected steps) vs 4-point FD | 1.2e-8 | 1e-6 | pass |
| GPU fp64: symbolic vs JS, 21 rays | 4.7e-14 | 1e-9 | pass |
| GPU fp64: duals vs symbolic | 2.6e-14 | 1e-10 | pass |
| GPU fp64: d(shadow angle)/dR, R = 4–1000 M | 2.1e-15 | 1e-9 | pass |
| GPU fp64: ∂b_crit/∂a at b − b_crit = 1e-5 (four cases) | 1.06e-4; error ratio 1e-4 → 1e-5 ≥ 7.9 | 1e-3; ≥ 5 | pass |
| GPU fp64: d(sweep)/du vs exact integral, b = 6–100 M | 7.5e-9 | 1e-6 | pass |

**Speed** (RTX 3070, driver 617.42, NVRTC 13.3; Kerr a = 0.94, i = 60°, camera at 1000 M, 128 × 128 rays over ±15 M, disk on, RK4 step fraction 0.01; GPU at 16–30% background load; `docs/benchmarks/phase4-step3-tangents-3070.json`):

| | plain | dual P = 1 | dual P = 2 | dual P = 4 | symbolic P = 2 |
| --- | --- | --- | --- | --- | --- |
| fp32 | 4.3 ms (3.8 M rays/s) | 19 ms (4.4×) | 192 ms (44×) | 323 ms (75×) | 639 ms (148×) |
| fp64 | 324 ms (51 k rays/s) | 855 ms (2.6×) | 1,622 ms (5.0×) | 2,891 ms (8.9×) | 2,897 ms (9.0×) |

- **Decision (open question 2): duals are the production path**, 1.8–12× faster than the symbolic JVP. The symbolic tangents stay as the independent check, and are what the WGSL path needs.
- fp64 cost is close to the (1 + P)× expectation (5.0× at P = 2). fp32 is far from it: at P ≥ 2 the dual arrays live in local memory (stack 960 B at P = 1, 3,424 B at P = 2; 0 spills), and the plain fp32 kernel is unusually cheap. Lifting the fp32 128-register cap for tangent modules made them 2–4× faster (measured; now the default for those modules).
- NVRTC compile of a tangent module (both kernels): 9–49 s for Kerr.

**SBC budget estimate (amendment 4).** Assumptions, not measurements: 64 × 64 pixels × 4 stratified samples = 16,384 rays per gradient, P = 2 traced (a, i); NUTS with 500 warm-up + 500 draws at ~16 gradient evaluations per draw ≈ 16,000 gradients per dataset. From the measured dual P = 2 throughput: fp32 ≈ 0.19 s per gradient, ≈ 51 min per dataset, **≈ 7 days for 200 datasets** (fp64: ≈ 60 days). Too slow as it stands. Levers, to measure in steps 4–8: fewer steps per ray (most of the ~1,000 steps are spent between 1000 M and 50 M, where a larger step fraction may be accurate enough); fewer rays (supersample only near the ring); a fused tangent right-hand side for fp32; SBC on the Laplace approximation for the full 200 and NUTS on a subset; and amortised NPE (step 9), whose SBC is cheap.

**Full fp64 suite re-run** (2026-11-06, `raymarcher --validate`, uncommitted tree on 931c44e): all 8 checks pass, including the new `sensitivities`; every metric value of the seven existing checks is identical to the committed results, so the RK45 Newton step changed no reported number.

**Not done in step 3:** the WGSL tangent path (needed only for the in-browser fit, step 11) and the image kernel with tangent channels (it needs the intensity model, so it moves to step 4).

## Step 4 results (2026-11-06)

Code: `compiler/src/image-model.ts` (EmissionModel interface + `thinDisk`, `ImageModel` per-sample intensity with tangents, `stratifiedSamples`, `observe`), `compiler/src/dual.ts` (shared JS duals), `native/src/rm/emission.h` (`thin_disk_intensity`, templated), `render_samples_dual_kernel` / `render_samples_plain_kernel`, `raymarcher --image-gradients`, fixture `native/generated/fixtures/image-samples.json`.

**Forward model.** Model sky ξ in units of M at the camera's tangent plane (R = 1000 M; O(M/R) ≈ 0.1–0.2% from asymptotic impact parameters, a labelled choice). Observed image I_obs(X) = A θ_g² Σ_s w_s B(X − θ_g Rot(PA) ξ_s) I_M(ξ_s) over fixed stratified samples, B a unit Gaussian beam. θ_g, PA and A enter only through B, so their derivatives are exact and need no spatial derivatives of I_M; spin and inclination enter through I_M and its tangents. Emission (labelled choice): thin disk I = g⁴ F(r) τ(r), F = (1 − √(r_in/r))(r_in/r)³ (zero at r_in), C¹ taper τ = (1 − s²)² from 0.75 r_out to r_out; black sky.

| Check | Measured | Tolerance | Result |
| --- | --- | --- | --- |
| JS per-sample dI/d(a, i, u, v) vs 4-point FD, Kerr a = 0.3 / 0.94, R = 200 M | 9.3e-10 / 2.1e-10 (relative to the largest per input; 2 of 154 pairs skipped for a step-count change) | 1e-6 | pass |
| Observed image: ∂/∂θ_g, ∂/∂PA, ∂/∂A vs FD | 8.0e-11 | 1e-8 | pass |
| Flux bookkeeping Σ image = A θ_g² Σ w I_M | < 1e-6 | 1e-6 | pass |
| Intensity zero at r_in and r_out; taper C¹ | yes | — | pass |
| CPU: C++ duals vs JS image model (40 Kerr disk samples) | I 2.6e-15, dI 1.1e-14 | 1e-10 | pass |
| GPU fp64: duals vs JS image model | I 2.4e-15, dI 2.0e-14 | 1e-10 | pass (in `sensitivities`) |

**Measurement: total-flux gradient vs supersampled FD** (`validation/results/image-gradients.json`; Kerr, R = 1000 M, ±22 M, RK4 0.01, fp64; quadrature on 128² and 1024² samples, FD of the 1024² flux at h = 4e-3 and 2e-3):

| Scene | Parameter | Quadrature 1024² vs FD | from fate flips (edge) | from crossing-step changes | Quadrature 128² vs 1024² |
| --- | --- | --- | --- | --- | --- |
| a = 0.94, i = 60° | spin | +0.5% | −0.43% | +0.07% | 4% |
| a = 0.94, i = 60° | inclination | +64% | −68% | +4.2% | 2.8× |
| a = 0.5, i = 30° | spin | −1.0% | −1.0% | +1.5% | 10% |
| a = 0.5, i = 30° | inclination | −17% | +17% | +2.6% | 3.7× |

What this says:
1. **The edge term is not negligible.** For spin it is ~0.4–1% of the flux gradient; for inclination it dominates (17–68%), because the flux depends weakly on inclination while the n ≥ 1 images move across the critical curve. Step 5 is required, not optional.
2. **Crossing-step changes are a second discontinuity.** The disk crossing is located by linear interpolation inside an RK4 step, so the image jumps by O(h²) when the crossing moves to another step; FD sees this (0.07–4.2%), tangents do not. Fix in step 5: cubic Hermite crossing (uses k at both ends, O(h⁴) jumps), measured the same way.
3. **The quadrature is noisy near the ring.** Per-sample tangents there are 700–1,600× the median (the e^{γn} growth), so 128² samples misestimate the inclination gradient by 3–4× and the spin gradient by 4–10%. Step 5 needs a near-ring treatment (more samples near the critical curve, or the analytic ring handled separately), measured against sample count.

**Open question 1 answered (measured):** fp32 tangents are not good enough near the ring, and clipping is worse. fp32 total-gradient error: 4e-5 and 1.4e-3 (a = 0.94) but 8% and 140% (a = 0.5, i = 30°), all from the top 5% of samples by step count (per-sample error up to 3× the largest tangent; the median sample is fine at 2–4e-8). Clipping |dI| at the 99th / 99.9th percentile changes the gradient by 3–160%. So near-ring samples need fp64 (or a different treatment); fp32 is fine for the bulk.

## Step 5 results (2026-11-06): gate 1 passed

Code (all in `native/`):
- `RingFlow`, `assemble(..., flow)`, `add_edge_term`, `run_gate1` and `run_gate1_diag` in `src/validate.cpp`;
- `edge_bisect_kernel` and `curve_bisect_kernel` in `kernels/tracer.cu`;
- the cubic Hermite disk crossing (`CROSSING_HERMITE`) in `src/rm/geodesic.h`;
- the CLI flags `raymarcher --gate1` and `--gate1-diag`.

Result: `validation/results/image-gradient-gate.json`.

**Setup.**
- **Scene:** Kerr, camera at 1000 M, model sky ±22 M, thin disk g⁴Fτ, RK4 with step fraction 0.01 and the Hermite crossing, fp64. RTX 3070, driver 617.42, NVRTC 13.3.
- **Observation:** 64 × 64 pixels of 3 µas, θ_g = 3.8 µas/M, PA = 0.4, Gaussian beam of 20 and 7 µas FWHM.
- **Data:** the 2048²-sample image at (a + 0.04, i + 3°, 1.03 θ_g, PA + 0.1).
- **Loss:** ½ Σ (I − D)²/σ² with σ = 1% of the data peak (a test choice).
- **Scenes:** (a, i) = (0.94, 60°) and (0.5, 30°).

**The method.**
- **Samples:** 512² stratified samples with dual tangents along (a, i, u, v). Cells near a branch change, or in the top 3% of |dI|, are refined 8 × 8, giving 1.2–1.34 M samples.
- **Ring flow:** samples move with the critical curve (below).
- **Edge term:** a sum over row crossings of Δy (I_L − I_R) times the crossing's velocity relative to the moving samples, dx*/dθ + (dx*/dy) V_y − V_x. Crossings are found on a 2048 × 1024 grid by 48 bisections; dx*/dθ and dx*/dy come from bisection at ±1e-5.
- **Cost:** 288 s and 342 s per scene on the 3070, all fp64.

**The truth.**
- **Finite differences:** central differences of the loss of 2048²-sample images, on two independent jitter sets.
- **Spin and inclination:** Richardson extrapolation of h = 3.2e-2 and 1.6e-2, checked against the 1.6e-2 and 8e-3 pair.
- **θ_g and PA:** 5e-5 relative.
- **Standard error:** |t₁ − t₂|/2. The gate also requires it to be at most 0.3%.

| Scene | Parameter | Error, 20 µas | Error, 7 µas | Truth s.e. (20 / 7) | Plain estimator (no flow), 20 / 7 |
| --- | --- | --- | --- | --- | --- |
| a = 0.94, i = 60° | spin | 0.092% | 0.18% | 0.0007% / 0.006% | 1,950% / 141% |
| a = 0.94, i = 60° | inclination | 0.12% | 0.009% | 0.18% / 0.04% | 4,220% / 248% |
| a = 0.94, i = 60° | θ_g | 0.012% | 0.006% | ≤ 0.01% | same (exact) |
| a = 0.94, i = 60° | PA | 0.011% | 0.041% | ≤ 0.003% | same (exact) |
| a = 0.5, i = 30° | spin | 0.075% | 0.15% | 0.07% / 0.07% | 12.8% / 8.7% |
| a = 0.5, i = 30° | inclination | 0.23% | 0.23% | 0.13% / 0.09% | 3.4% / 1.9% |
| a = 0.5, i = 30° | θ_g | 0.031% | 0.089% | ≤ 0.008% | same (exact) |
| a = 0.5, i = 30° | PA | 0.048% | 0.082% | ≤ 0.004% | same (exact) |

All 16 are within the 1% tolerance, so **gate 1 passes**. The largest error is 0.23%, and the largest truth standard error is 0.18%. The truth's Richardson check moves it by at most 0.55%. Other numbers from the same run:
- **The unrefined 512² base with the flow** is within 0.43% everywhere, so the refinement is now a margin, not a necessity.
- **The edge term** is up to 2.5% of the gradient (spin, a = 0.5, 20 µas). Halving the edge rows changes the result by ≤ 0.2%.
- **The critical-curve velocity** at h = 1e-4 vs 1e-3 agrees to 3e-5 of its maximum, with 0 failed bisections out of 10 × 1440 per scene.

**How we got here (four runs and a diagnostic; logs are in the session scratchpad, numbers here).**
1. **Run 1** (256² base, fixed-sample quadrature plus the edge term): spin and inclination off by 0.9–5.7%; θ_g and PA fine.
2. **Run 2** (512² base) added diagnostics:
   - The error was in the Jacobian itself: "Jacobian with dense residuals" was about equal to the total error.
   - It was not in the edge resolution or the residuals.
   - Scene 2's inclination got *worse* with more samples: 3.7% at 256², 11.9% at 512².
3. **Diagnostic** (`--gate1-diag`): dual tangents on all 4.2 M dense samples, with per-sample FD at four step sizes.
   - **Per-sample derivatives are right:** 8e-7 to 2.4e-3 weighted disagreement with FD at h = 1e-5.
   - **Their sum is still 28% off** for spin and 6% for inclination at (0.5, 30°).
   - **All of the disagreement sits on the critical curve:** in radial bins at 4.5–5.0 M for (0.5, 30°), and at 3.0 M on the prograde side for (0.94, 60°).
   - **Cause:** near the curve the n-th subring has width ~e^(−γn), so dI/dθ at a fixed sky point grows like 1/(distance to the curve). Its integral converges but its square does not, so the sample sum has unbounded variance. More samples don't help, and can make it worse.
4. **Fix: the ring flow,** a change of variables of the image integral, the reparameterisation trick from differentiable rendering.
   - **Definition:** V_d(ξ) = (∂ρ_c/∂θ_d)(φ) · g(ρ − ρ_c(φ)) · e_ρ, about the curve's centroid, with g(x) = (1 − (x/1.5 M)²)².
   - **Measuring the curve:** ρ_c(φ) comes from bisection on capture along 1440 directions, on the tracer itself; ∂ρ_c/∂θ from central differences at 1e-4.
   - **Per-sample terms:** each sample contributes dI/dθ + V·∇I (from the u, v tangents), plus I ∇B·V and B I ∇·V.
   - **Why it works:** in moving coordinates the 1/distance growth becomes log(distance), whose square is integrable.
   - **Exactness:** any Lipschitz V gives the same gradient in exact arithmetic, so V changes only the variance.
   - **Run 3:** 15 of 16 passed. The one failure (spin at (0.5, 30°), 20 µas, 1.34%) had a truth whose FD spread was 0.9%.
5. **Run 4:** a stronger truth (larger h, two jitter sets, standard error required to be ≤ 0.3%) passes all 16.

**Labelled choices.**
- **Estimator choices** (they change only the variance): flow width W = 1.5 M, centre at the curve's centroid, 1440 directions with linear interpolation in φ, refinement rule.
- **Test choices:** σ = 1% of peak, data offsets, beam sizes and pixel grid.

**What this means for later steps.**
- **The plain fixed-sample gradient is unusable near the ring,** at any sample count we can afford. Every gradient-based step (6: MAP, 7: Laplace / NUTS, 8: SBC) must use the flow estimator.
- **The flow needs the critical curve and its velocity per θ:** 10 × 1440 bisections of 52 steps each. Their time is inside the 288–342 s per scene but was not measured on its own; time it in step 6 before deciding how often to recompute it. Options for step 7: reuse the curve across nearby θ (V only needs to be close, not exact), or use the analytic Kerr critical curve (Bardeen), whose O(M/R) mismatch only costs variance.
- **The SBC budget** in step 3 assumed 16,384 rays per gradient. The flow's base-only error (≤ 0.43% at 262,144 samples) is the number to re-measure at lower sample counts in step 6.

**Not done in step 5.**
- **Gate 1 is not in the `--validate` suite;** a full run takes ~80 min. A cheaper version for the suite, with one scene, one beam and a 1024² truth, is left for later.
- **No `--image-gradients` re-run** to show the Hermite crossing removed the crossing-step term. Gate 1 uses Hermite and passes, but the step-4 table still shows the linear crossing.
- **No WGSL path** (step 11).

## Open questions

- ~~fp64 near the ring in the interactive path, or clipping enough for MAP?~~ Neither fp32 nor clipping is adequate near the ring (step 4 measurement); fp64 or a separate near-ring treatment there.
- ~~Symbolic JVP or duals for production CUDA?~~ Duals (step 3 measurement); symbolic stays as the check.
- Inclination prior: isotropic (uniform in cos i) by default, stated wherever posteriors are shown.
