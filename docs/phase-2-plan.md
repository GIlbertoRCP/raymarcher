# Raymarcher: Phase 2 plan (CUDA backend + validation)

## Context

Raymarcher compiles spacetime metric files (`metrics/*.st`) into GPU ray tracers. Phase 1 shipped:
- **Compiler** (`compiler/`, TypeScript): a parser builds a hash-consed expression graph, then symbolic derivatives are taken and two backends are emitted:
  - WGSL, for the WebGPU tracer in `web/`;
  - double-precision JavaScript, for camera setup, the CPU shadow cross-check and tests.
- **Tests:** 50, including the √27 M capture threshold, weak-field deflection, Bardeen's Kerr shadow outline, the ISCO, and conservation of E and L_z.
- **Catalog:** Schwarzschild and Kerr in **outgoing** Kerr-Schild Cartesian coordinates, plus flat space. Outgoing coordinates are used because rays are traced backwards from the camera, and they end on the past horizon, which these coordinates make regular.
- **Physics:**
  - Rays follow Hamilton's equations: dx/dλ = g⁻¹p, dp_μ/dλ = ½ (∂_μ g_ab) kᵃ kᵇ.
  - The camera is a static observer, with an orthonormal frame built by Gram–Schmidt against g.
  - The disk is an optically thick thin disk on circular orbits. Redshift is g = 1/(p·u_gas) and brightness scales as g⁴.
- **Web UI:** a field-star-atlas plate (see DESIGN.md and PRODUCT.md).

The native CUDA app (`native/`) still runs the Phase 0 hand-written isotropic Schwarzschild kernel. Phase 2 moves it onto the compiler.

**Ground rules:**
- Never commit or push: the user does that.
- Physics claims must be backed by a test or a measurement.
- Artistic choices (disk texture, colour grading) are labelled as such.
- Benchmarks always state hardware, driver, resolution, scene and integrator.

## Gate for Phase 2

1. The CUDA image of the same scene agrees with the WebGPU image. The ray fate (captured / escaped / disk) must agree on at least 99.5% of pixels, excluding pixels within 1 px of a fate boundary.
2. The validation suite runs in fp64 on CUDA, and its results are published on a validation page.

## Steps

### 0. Baseline (on the 3070)
- Build `native/` in Release.
- Run `ctest` with `-DRAYMARCHER_BUILD_VIEWER=OFF`.
- Run the viewer, and run `--benchmark`.
- Save the output to `docs/benchmarks/phase0-3070.json`, with GPU, driver and CUDA version. This is the "before" number.

**Done when:** the numbers are recorded honestly (steps per ray should now be well above 1).

### 1. CUDA/C++ backend in the compiler
- **New file:** `compiler/src/backend-cuda.ts`, using the same `emitBlock` as WGSL. The dialect differences:
  - the generated code is a function template on `Real`, so it serves both float and double;
  - functions are `RM_HD` (`__host__ __device__`);
  - it uses `sqrt` / `pow`.
- **Generated function names:**
  - `metric_eval<Real>(const Real x[4], const Real* params, Real g[16], Real dg[64])`
  - `horizon_value`, `disk_radius`, `disk_omega`, `disk_inner`, `disk_outer`, `length_scale`
- **CLI:** `npm run export-cuda`, via a Node script at `compiler/cli/export-cuda.ts`. It writes `native/generated/<name>.cuh` for every catalog metric, plus a `catalog.json` listing names, labels and parameter ranges.

**Test (runs in CI, no GPU):** the generated code is plain C++ with `RM_HD`, so compile it with g++/clang for the host. A small harness evaluates `metric_eval<double>` at random points and compares it with the JS backend, through a JSON fixture of points and expected values written by vitest. They must agree to 1e-12.

### 2. Native tracer on generated code (NVRTC)
- **New kernel:** replace `native/src/raymarch.cu` with a generic tracer template that mirrors `web/src/shaders/tracer.wgsl.ts`:
  - Hamiltonian right-hand side;
  - numeric 4×4 inverse;
  - termination on the horizon, on escape, and on diverging dt/dλ;
  - thin-disk crossing, exact redshift, g⁴ brightness;
  - stars and grid sky.
- **Runtime compilation:** the kernel source plus the chosen `native/generated/*.cuh` are compiled at runtime with **NVRTC** and launched through the driver API (`cuModuleLoadData`, `cuLaunchKernel`). Switching metrics needs no C++ rebuild.
- **Keep:** GLFW, ImGui and PBO interop.
- **ImGui panel:** the parameter sliders come from `catalog.json`.
- **Camera frame:** computed on the host in double, using the generated `metric_eval<double>` (port `Spacetime.staticFrame` from `compiler/src/geodesic.ts`).
- **Precision switch:** `Real = float` for the interactive view, `Real = double` for reference renders. Set it with an NVRTC define.

**Done when:** Schwarzschild, Kerr and flat space render interactively on the 3070, and the viewer's "scene" readout matches the web app's parameters.

### 3. Adaptive integrator for reference renders
- Add Dormand–Prince RK45 with error control in fp64: tolerances on x and p, step rejection, and a maximum step count.
- Keep fixed-step RK4 as the interactive path.

**Done when:** a convergence study shows RK4 error falling at order 4 and RK45 meeting the requested tolerance.

### 4. Headless scene renders and cross-backend comparison
- **Shared scene format** (`scenes/*.json`): spacetime, parameters, camera radius / yaw / pitch / fov, resolution, sky, disk and integrator.
  - Web side: a dev-only "Export scene" action, plus URL query params that load a scene.
- **Native:** `raymarcher --render scenes/x.json --out out/x` writes:
  - an RGB float image;
  - a per-pixel fate map (captured / escaped / disk / unfinished);
  - steps per pixel;
  - |H| drift per pixel.
- **WebGPU:** a headless capture script (Playwright with Chrome, WebGPU enabled; run headed on Windows if headless lacks WebGPU) renders the same scene and dumps the image and the fate map. Add a `debugFate` flag to the tracer that outputs the fate as a colour.
- **Comparison:** `validation/compare.ts` produces:
  - fate agreement %;
  - an image difference heatmap;
  - the shadow-edge offset in pixels.

**Gate check:** at least 99.5% fate agreement on the scene set:
- Schwarzschild and Kerr (a = 0.6 and 0.95), at 3 inclinations, disk on and off;
- flat space.

### 5. Validation suite (CUDA, fp64)
Put the code in `validation/`. Results go to `validation/results/*.json`, including GPU name, git hash and date. The suite covers:
- **Deflection angle vs impact parameter:** compare with the exact Schwarzschild integral, not only the weak-field series.
- **Shadow outline:**
  - Schwarzschild: radius vs camera distance, against sin α = √27 M √(1−2M/R)/R.
  - Kerr: the general-inclination Bardeen outline, (α, β) for an observer at inclination θ_o, for a = 0.3 / 0.6 / 0.9 / 0.99 and θ_o = 17° / 45° / 90°. Report the maximum deviation.
- **ISCO and photon orbit radii** vs the analytic formulas.
- **Constraint drift:** a histogram of |H| over a full frame, for RK4 vs RK45.
- **Convergence plots:** error vs step size.

**Done when:** every check reports a number, a tolerance and a pass/fail.

### 6. Honest benchmarks
- **Profile:** use Nsight Compute on the interactive kernel. Record occupancy, register count, achieved FLOP/s, and warp divergence (rays finish at different times).
- **Report** `docs/benchmarks/phase2-3070.json` with:
  - rays/s and mean / p95 steps per ray;
  - fp32 vs fp64, and RK4 vs RK45;
  - CUDA vs WebGPU on the same scene on the same 3070.
- **Optimisations:** only optimisations backed by measurements, for example:
  - register pressure from the inverse;
  - fusing `metric_eval` with the inverse;
  - persistent threads / ray compaction for divergence.

**Done when:** every number in the README comes from these files.

### 7. New spacetimes and user-defined metrics
- **Reissner–Nordström** in Kerr-Schild form, with charge Q. Validation: the analytic photon-sphere radius r = (3M + √(9M² − 8Q²))/2.
- **User-defined metrics** in the web app: a metric editor in the key band.
  - It compiles live, and shows parse errors with line and column (MetricSyntaxError already provides these).
  - WGSL compile errors are reported (Renderer already throws ShaderCompileError).
  - Limits: at most 8 parameters, and a step budget.
- **Ellis–Bronnikov wormhole:** design spike first. It has two asymptotic regions, so the Cartesian-coordinates assumption in the tracer (spatial coords = x, y, z; escape by radius) needs a "which side" flag or a different chart. Write up the approach before building it. Compare against James et al. 2015 (the *Interstellar* paper).

### 8. Validation page
Build `web/validation.html` in the atlas world (DESIGN.md). Each check gets:
- a plate;
- the measured value against the prediction;
- a chart (convergence, deflection curve, shadow overlay of Bardeen vs traced);
- the cross-backend comparison images.

The data comes from `validation/results/*.json`. Link it from the playground's imprint.

### 9. CI
Add these jobs:
- compile the generated C++ host-side;
- run the JS-vs-C++ metric agreement test;
- run `compare.ts` on any committed render fixtures (small resolution).

CUDA itself stays off CI; results are committed from the 3070.

## Order and checkpoints

1. Steps 0–2: CUDA renders the catalog through the compiler.
2. Step 4: the cross-backend gate.
3. Steps 3, 5 and 6: fp64 validation and benchmarks.
4. Steps 7–9: new metrics, the editor and the validation page.

Stop and report at each checkpoint, with screenshots and numbers.
