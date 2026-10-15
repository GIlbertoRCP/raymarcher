# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Curious readers first: students, engineers and Hacker News readers who want to understand how gravity bends light, arriving from a shared link. Recruiters and technical reviewers see the same surface and judge it as a portfolio piece; they are served by clarity and correctness, not by a separate mode.

## Product Purpose

Raymarcher renders what light does around black holes and other spacetimes, in real time, in the browser. A spacetime compiler turns a metric file into a GPU ray tracer, so every image comes from the actual geodesic equations. Success: a visitor understands something new about curved spacetime within a minute, and a reviewer can verify the physics is correct.

## Positioning

The images are computed from the metric, not drawn: one compiler produces the WebGPU tracer, a CUDA reference renderer and a double-precision test oracle, and the results are checked against published results (√27 M capture radius, weak-field deflection, Bardeen's Kerr shadow outline).

## Operating Context

- Opened from a link on a laptop or desktop with WebGPU (Chrome, Edge, Safari 26+); phones are secondary.
- Rendering cost is high (one geodesic per pixel); mid-range laptops run at low single-digit to ~30 fps depending on resolution.
- The app will grow into an explorable essay (8 chapters, live figures) plus a validation page. The current surface is the playground.

## Capabilities and Constraints

- Spacetimes: Schwarzschild, Kerr (spin a/M up to 0.99), flat space. Parameters are generated from each metric file (name, label, range).
- Views: optically thick thin disk (on/off), sky as stars / lat-long grid / none, redshift colouring, exposure, peak disk temperature.
- Camera: orbit drag, wheel zoom (4–80 M), field of view; quality via resolution scale, max steps, step fraction.
- Live metric editing (user-defined spacetimes) is planned, not built.
- Disk texture and colour grading are artistic; the geometry, lensing, redshift and g⁴ beaming are physical.

## Brand Commitments

Name: Raymarcher. No other visual commitments; the earlier glassmorphic cyan dashboard is an anti-reference.

## Evidence on Hand

- Automated tests: compiler/test (parser, derivatives vs finite differences, Bardeen shadow, conserved quantities), native/tests.
- Render: docs/web-render.jpg (Phase 0, isotropic renderer).
- No benchmarks, users, testimonials or press yet; do not invent any.

## Product Principles

1. The image is the argument: the render leads, chrome recedes.
2. Every visual claim is physically honest, and artistic choices are labelled as such.
3. Explain in plain language; equations are available, never required.
4. Works from a link: no install, graceful when WebGPU is missing or slow.

## Accessibility & Inclusion

Keyboard-operable controls with visible focus; text contrast WCAG AA against the dark render; respect prefers-reduced-motion (pause auto-rotation and disk animation).
