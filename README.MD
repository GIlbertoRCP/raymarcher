# Relativistic Spacetime & Black Hole Raymarcher

An interactive, real-time web visualizer of Einstein's General Relativity. Rather than using standard Newtonian approximations or pre-rendered textures, this simulator integrates light ray paths (null geodesics) directly on the GPU using a custom WebGL2 fragment shader.

## Features

- **Relativistic Raymarching Core**: Performs 4th-order Runge-Kutta (RK4) integration inside a parallelized fragment shader for every screen pixel at 60 FPS.
- **Isotropic Metric Formulation**: Solves geodesic equations in **Cartesian Isotropic Coordinates** to avoid Boyer-Lindquist coordinate poles and horizon singularities, allowing free camera rotation.
- **Kerr Spacetime Simulation**: Approximates a rotating black hole using post-Newtonian frame-dragging (Lense-Thirring Coriolis corrections).
- **Volumetric Accretion Disk**: Simulates orbiting gas density profiles, Keplerian rotation dynamics ($\omega \propto r^{-3/2}$), layered noise dust structures, and key relativistic phenomena:
  - **Doppler Beaming**: Asymmetrical brightness scaling ($g^{3.8}$) based on rotation direction.
  - **Gravitational Redshift**: Frequencies shift dynamically based on distance from the horizon.
  - **Blackbody Thermal Colors**: Temperature changes radially and Doppler-shifts to map to the RGB spectrum.
- **Environment and Lensing Options**:
  - Distorted starfield or calibration grid background.
  - A toggle to switch instantly between standard flat Newtonian space and curved Relativistic spacetime.
- **Glassmorphic Interactive Control Panel**: Custom styled preset triggers ("Gargantua", "Micro Hole", "Kerr Extremum", "Newtonian Mode"), spacetime sliders (Mass, Spin), quality controls, telemetry dashboard, and a physics guide.

## Physics Formulation

### 1. Light Ray Geodesics
In Cartesian isotropic coordinates, the Schwarzschild metric reduces the problem of null geodesics ($ds^2 = 0$) to a radially-varying effective index of refraction $n(r)$:
$$n(r) = \frac{\left(1 + \frac{M}{2r}\right)^3}{1 - \frac{M}{2r}}$$

The acceleration vector of a photon is:
$$\frac{d\mathbf{p}}{d\lambda} = \frac{1}{2} \nabla \left( n(r)^2 \right) = - \frac{M \left(1 + \frac{M}{2r}\right)^5 \left(2 - \frac{M}{2r}\right)}{r^3 \left(1 - \frac{M}{2r}\right)^3} \mathbf{x}$$

### 2. Frame Dragging
For a rotating black hole with spin parameter $a = J/M$ ($0 \le a < M$), frame-dragging introduces a Coriolis-like force:
$$\frac{d\mathbf{p}}{d\lambda} = \mathbf{F}_{grav} + 2 \mathbf{p} \times \boldsymbol{\Omega}_{fd}$$
where:
$$\boldsymbol{\Omega}_{fd} = \left(0, 0, \frac{2 J}{r^3 \left(1 + \frac{M}{2r}\right)^6}\right)$$

---

## Setup & Running (Windows / macOS / Linux)

### Prerequisites
Make sure you have [Node.js](https://nodejs.org/) installed.

### Setup
1. Clone the repository.
2. In the project directory, install the package dependencies:
   ```bash
   npm install
   ```

### Running Locally
Run the local Vite development server:
```bash
npm run dev
```
Open the provided local URL (typically `http://localhost:5173`) in your WebGL2-compatible browser.

### Building for Production
Compile TypeScript and bundle the assets into static HTML/JS/CSS:
```bash
npm run build
```
The output will be placed in the `dist/` directory.

---

## High-Performance WebGPU Renderer (Browser)

The frontend visualizer is powered by a high-performance **WebGPU** graphics pipeline written in **WGSL (WebGPU Shading Language)**. This allows the simulator to communicate directly with native Vulkan/DirectX12 device drivers on your GPU with near-zero driver overhead.

### Dual-Pipeline Architecture
To ensure compatibility across all devices, the application implements an automatic fallback mechanism:
1. **WebGPU (High Performance)**: Default mode for modern browsers (Chrome, Edge, Opera) with native GPU performance.
2. **WebGL2 (Compatibility Fallback)**: Used automatically if WebGPU is unsupported or disabled (e.g., Firefox, Safari without flags).
3. **WebGL1 (Legacy Fallback)**: Translates shaders dynamically to support older devices and browsers.

You can verify which pipeline is active in the **Telemetry** dashboard under the **Pipeline** indicator.

---

## Offline CUDA C++ Renderer (Optional / Reference)

A native C++ and NVIDIA CUDA C++ implementation is located in the [cuda_raymarcher](file:///c:/Users/huevi/Documents/raymarcher/cuda_raymarcher/) subdirectory. This runs outside the browser context, rendering high-resolution frames of the black hole directly to `.bmp` image files.

For build and running instructions of the C++/CUDA project, refer to the [CMake configuration](file:///c:/Users/huevi/Documents/raymarcher/cuda_raymarcher/CMakeLists.txt) and the source files within that directory.

