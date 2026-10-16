# Raymarcher

A field atlas of curved spacetime. Raymarcher compiles a spacetime's metric into a GPU ray tracer and renders what light does around it, in real time, in the browser. The physics is checked against published results.

![A Kerr black hole rendered as an atlas plate, with the CPU-traced shadow edge and disk inner edge marked in red](docs/playground-kerr.jpg)

Every pixel traces one light ray backwards from the camera through curved spacetime. Rays that pass close to a black hole bend, loop around it or fall in. That is what produces the dark shadow, the thin photon ring at its edge, and the image of the disk's far side wrapped over the top.

## How it works

You describe a spacetime in a small metric language:

```
spacetime "Schwarzschild"
coords t x y z
param M = 1 in [0.2, 3] "Mass"
let r = sqrt(x^2 + y^2 + z^2)
let f = 2*M / r
g t t = -1 + f
g t x = -f*x/r
...
horizon r < 2*M*1.01
```

The compiler (`compiler/`) does the rest:

1. **Parse** the file into an expression graph. Identical subexpressions are stored once (hash-consing), and constants fold as the graph is built.
2. **Differentiate** every metric component symbolically with respect to every coordinate.
3. **Analyse** which coordinates the metric does not depend on. Each gives a conserved momentum.
4. **Emit** straight-line code to two backends:
   - **WGSL**, for the WebGPU tracer;
   - **JavaScript** (double precision), for camera setup, the CPU cross-check and the tests.

The generated Kerr module is about 350 lines of WGSL.

Rays follow Hamilton's equations, which need only the metric g and its first derivatives. The inverse metric is computed numerically on the GPU.

```
dx^μ/dλ = g^{μν} p_ν          dp_μ/dλ = ½ (∂_μ g_{αβ}) k^α k^β,   k = g⁻¹ p
```

The camera is a static observer whose orthonormal frame is built from the local metric. Disk colours come from the exact redshift g = (p·u_camera) / (p·u_gas) for gas on circular orbits. Brightness scales as g⁴.

Built-in spacetimes (`metrics/`):
- **Schwarzschild**: a non-rotating black hole.
- **Kerr**: a spinning black hole, with spin a/M up to 0.99.
- **Flat space**: for comparison.

The black holes use outgoing Kerr-Schild coordinates. Rays traced backwards from a camera end on the past horizon, and these coordinates make that horizon regular.

## The playground

The page is laid out as one chart plate of a field star atlas:
- The render sits inside a ruled neatline. The margins carry live azimuth and elevation scales for the view.
- The key along the foot holds every control. Parameter sliders are generated from the metric file.
- The red marks are measured, not drawn. A Web Worker traces the shadow edge on the CPU, in double precision, with the same compiled metric, and draws it over the GPU image. It also locates the lensed inner edge of the disk.

For Schwarzschild the traced edge agrees with the analytic √27 M prediction to within 0.01°, which makes the overlay a live cross-check between the two backends.

![The Schwarzschild plate with the sky grid: nested Einstein rings and the disk wrapped over the shadow](docs/playground-schwarzschild-grid.jpg)

## Verification

| Check | Expected | Where |
| --- | --- | --- |
| Symbolic derivatives | Match finite differences for every catalog metric | `compiler/test/compiler.test.ts` |
| Generated code | Agrees with direct evaluation of the expression graph | `compiler/test/compiler.test.ts` |
| Metric sanity | Symmetric, invertible (g g⁻¹ = I), Lorentzian | `compiler/test/compiler.test.ts` |
| Capture threshold | b < √27 M falls in, b > √27 M escapes | `compiler/test/physics.test.ts`, `native/tests` |
| Weak-field deflection | 4M/b + 15πM²/4b², within 0.1% | `compiler/test/physics.test.ts`, `native/tests` |
| Generated C++ | Agrees with the JS backend to 1e-12 at 96 sample points (g, ∂g, horizon, disk) | `native/tests` |
| Kerr shadow | Bardeen's (1973) analytic outline at a = 0.9, ±2% | `compiler/test/physics.test.ts` |
| Kerr ISCO | 6M at a = 0, 2.32M at a = 0.9 (Bardeen, Press & Teukolsky 1972) | `compiler/test/physics.test.ts` |
| Conservation | Energy and L_z conserved along rays in Kerr | `compiler/test/physics.test.ts` |
| Null constraint | H = ½ g^{μν} p_μ p_ν stays below 10⁻⁸ | `compiler/test/physics.test.ts` |
| Parser errors | Reported with line and column | `compiler/test/parser.test.ts` |
| GPU uniform layout | TypeScript offsets match the WGSL struct | `web/src/uniforms.test.ts` |

The Kerr shadow test is not trivially satisfiable: a mirrored outline (prograde and retrograde swapped) fails it.

### fp64 validation on the GPU

`raymarcher --validate` runs these checks on the CUDA tracer in double precision with the adaptive Dormand–Prince RK45 integrator, and writes each result, with GPU, driver, git commit and date, to `validation/results/`. Measured on an RTX 3070:

| Check | Reference | Largest deviation | Tolerance |
| --- | --- | --- | --- |
| Deflection, b = 5.21 to 1000 M | Exact Schwarzschild orbit integral | 4.6 × 10⁻¹² rad | 10⁻⁹ rad |
| Shadow size, camera at 4 to 1000 M | sin α = √27 M √(1 − 2M/r) / r | 1.3 × 10⁻¹³ rad | 10⁻⁹ rad |
| Kerr shadow outline, a = 0.3 to 0.99, inclination 17°, 45°, 90° | Bardeen (1973) | 2.5 × 10⁻¹⁰ M | 10⁻⁴ M |
| Circular photon orbits, a = 0 to 0.99 | Bardeen, Press & Teukolsky (1972) | 9.7 × 10⁻¹² M | 10⁻⁸ M |
| ISCO from the metric's derivatives | Bardeen, Press & Teukolsky (1972) | 3.1 × 10⁻⁷ M | 10⁻⁶ M |
| Null constraint over a full Kerr frame (p99) | H = 0 | 1.5 × 10⁻¹⁰ | 10⁻⁸ |
| RK4 convergence | Order 4 | Fitted order 3.990 to 3.995 | 3.8 to 4.2 |
| RK45 error control, rtol 10⁻⁴ to 10⁻¹² | Requested tolerance | 1.7 × rtol | 10 × rtol |

The interactive integrator (RK4, step 0.03 r) is less accurate by design: its swept-angle error reaches 3.5 × 10⁻⁵ rad for rays grazing the photon sphere, and the p99 constraint drift over a frame is 2.9 × 10⁻⁷.

### CUDA and WebGPU agree

The two renderers trace the same 21 scenes (`scenes/`: Schwarzschild and Kerr at a = 0.6 and 0.95, three inclinations, disk on and off, plus flat space). Every ray ends the same way in both: captured, escaped or on the disk. Across 10.9 million pixels no fate differs (`validation/results/cross-backend.json`; the gate is 99.5%). The fp32 and fp64 CUDA renders agree just as exactly. Rounding moves a fate boundary by far less than a pixel; the integrator's own error, which both backends share, is what the fp64 checks above measure.

## Running it

Needs a browser with WebGPU (recent Chrome or Edge, or Safari 26+):

```bash
npm install
npm run dev
```

```bash
npm test
```

**Native viewer** (`native/`, CUDA 11.2+, CMake 3.18+, an NVIDIA GPU). The tracer (`native/kernels/tracer.cu`) is compiled at runtime with NVRTC for the chosen metric and precision, from the headers `npm run export-cuda` writes to `native/generated/`. Switching spacetime or precision needs no rebuild.

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --config Release
```

Run `native/build/raymarcher` (`native/build/Release/raymarcher.exe` on Windows). `--benchmark` times every catalog metric at 1080p, 1440p and 4K; `--snapshot <id> <file.png>` renders one frame headless. After changing a metric or the compiler, regenerate the headers:

```bash
npm run export-cuda
```

CPU-only tests (generated C++ vs the JS backend, and the physics checks):

```bash
cmake -S native -B native/build-tests -DRAYMARCHER_BUILD_VIEWER=OFF && cmake --build native/build-tests && ctest --test-dir native/build-tests
```

### Cross-backend check

Scene files (`scenes/*.json`) describe a view completely, so the same image can be rendered by both backends. The web app loads one with `?scene=<name>` (add `&debugFate=1` to colour pixels by ray fate). To render the scene set with both and compare:

```bash
for f in scenes/*.json; do native/build/Release/raymarcher.exe --render "$f" --out "validation/out/cuda/$(basename "$f" .json)"; done
```

```bash
npx vite-node validation/capture-web.ts
```

```bash
npx vite-node validation/compare.ts
```

The capture script drives the installed Chrome through playwright-core. `compare.ts` writes `validation/results/cross-backend.json`: ray-fate agreement per scene (the gate is 99.5% away from fate boundaries), the shadow-edge offset in pixels, and image differences, with a heatmap per scene.

## Performance

Measured on an RTX 3070 (driver 617.42, CUDA 13.3), trace kernel only, fp32, at 1920 × 1080; details in `docs/benchmarks/phase2-3070.json`.

| Scene | CUDA | WebGPU (Chrome) |
| --- | --- | --- |
| Kerr a = 0.6, disk, near edge-on | 62 ms | 83 ms |
| Kerr a = 0.95, no disk, 45° | 96 ms | 129 ms |
| Schwarzschild, disk, near edge-on | 39 ms | 50 ms |
| Flat space, disk | 2.0 ms | 3.2 ms |

The CUDA kernel got 1.6× faster on Kerr during Phase 2: the compiler now emits a fused right-hand side that uses the symmetry of g and drops derivatives that are identically zero (8–10% faster on the two Kerr scenes), the fp32 kernel is capped at 128 registers so two blocks share an SM (12–15%), and the fp32 kernel uses fast math (18–19%), which changed no ray fate in the cross-backend set. Rays in a warp take almost the same number of steps (97–99% of lanes busy), so there is little divergence to recover. Nsight Compute shows the Kerr kernel is arithmetic-bound: the FMA pipe is busy 82% of the time, memory throughput is under 3%, and it reaches 10.4 TFLOP/s, 53% of the card’s fp32 peak at the profiled clock. The gap is mostly unfused multiplies in the generated code. Double precision runs about 60× slower than single on this card, close to its 1:64 hardware ratio, and is used for reference renders and validation.

## Repository layout

```
compiler/   metric language: parser, expression graph, derivatives, WGSL + JS backends, geodesic integrator, tests
metrics/    the spacetime catalog (.st files)
web/        WebGPU playground: tracer shader, viewer, atlas UI, CPU shadow worker
native/     CUDA viewer (NVRTC), runtime kernel, generated metric headers, CPU tests
scenes/     scene files shared by both renderers
validation/ cross-backend capture and comparison, results
docs/       images for this README
```

## Known limitations

- **Disk:** an optically thick thin disk with a Shakura–Sunyaev-shaped temperature profile. The swirl texture and colour grading are illustrative.
- **Precision:** the interactive views work in single precision (fast math on CUDA). Reference renders and the validation suite use double.
- **Performance:** each pixel integrates a full geodesic. Integrated laptop GPUs manage a few frames per second at full resolution, and the view drops to half resolution while you move.
- **Metrics:** only the built-in catalog; user-editable metrics are planned.
