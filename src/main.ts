import { Raymarcher } from './raymarcher.ts';

// Global Simulation Settings
export interface Settings {
  mass: number;
  spin: number;
  relativity: boolean;
  disk: boolean;
  grid: boolean;
  stars: boolean;
  diskTemp: number;
  steps: number;

  // Camera Settings
  camRadius: number;
  camPitch: number; // in radians
  camYaw: number;   // in radians
}

const settings: Settings = {
  mass: 1.80,
  spin: 0.60,
  relativity: true,
  disk: true,
  grid: false,
  stars: true,
  diskTemp: 6500,
  steps: 160,

  // Camera State
  camRadius: 12.0,
  camPitch: 0.15, // slightly tilted down
  camYaw: 0.0,
};

// UI Elements
const massInput = document.getElementById('param-mass') as HTMLInputElement;
const massBubble = document.getElementById('val-mass') as HTMLSpanElement;
const spinInput = document.getElementById('param-spin') as HTMLInputElement;
const spinBubble = document.getElementById('val-spin') as HTMLSpanElement;
const diskTempInput = document.getElementById('param-disk-temp') as HTMLInputElement;
const diskTempBubble = document.getElementById('val-disk-temp') as HTMLSpanElement;
const stepsInput = document.getElementById('param-steps') as HTMLInputElement;
const stepsBubble = document.getElementById('val-steps') as HTMLSpanElement;

const relativityToggle = document.getElementById('toggle-relativity') as HTMLInputElement;
const diskToggle = document.getElementById('toggle-disk') as HTMLInputElement;
const gridToggle = document.getElementById('toggle-grid') as HTMLInputElement;
const starsToggle = document.getElementById('toggle-stars') as HTMLInputElement;

const fpsLabel = document.getElementById('stat-fps') as HTMLSpanElement;
const horizonLabel = document.getElementById('stat-horizon') as HTMLSpanElement;
const photonLabel = document.getElementById('stat-photon-sphere') as HTMLSpanElement;

// Preset buttons
const presetGargantua = document.getElementById('preset-gargantua') as HTMLButtonElement;
const presetMicro = document.getElementById('preset-micro') as HTMLButtonElement;
const presetKerr = document.getElementById('preset-kerr') as HTMLButtonElement;
const presetNewton = document.getElementById('preset-newton') as HTMLButtonElement;

// Guide Tabs
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// Helper to update numeric display bubbles
function updateBubbles() {
  massBubble.textContent = settings.mass.toFixed(2);
  spinBubble.textContent = settings.spin.toFixed(2);
  diskTempBubble.textContent = `${settings.diskTemp} K`;
  stepsBubble.textContent = settings.steps.toString();
  
  // Calculate and display physical constants in Schwarzschild coordinates (scaled to isotropic)
  // Event horizon in isotropic: r_eh = M/2
  // Let's display the Schwarzschild event horizon r_s = 2M for user familiarity
  const rs = 2.0 * settings.mass;
  const rps = 3.0 * settings.mass;
  horizonLabel.textContent = `${rs.toFixed(2)} m`;
  photonLabel.textContent = `${rps.toFixed(2)} m`;
}

// Preset Handler
function applyPreset(presetName: string) {
  const buttons = [presetGargantua, presetMicro, presetKerr, presetNewton];
  buttons.forEach(btn => btn.classList.remove('active'));

  if (presetName === 'gargantua') {
    presetGargantua.classList.add('active');
    settings.mass = 1.80;
    settings.spin = 0.60;
    settings.relativity = true;
    settings.disk = true;
    settings.diskTemp = 6500;
  } else if (presetName === 'micro') {
    presetMicro.classList.add('active');
    settings.mass = 3.50;
    settings.spin = 0.0;
    settings.relativity = true;
    settings.disk = true;
    settings.diskTemp = 11000;
  } else if (presetName === 'kerr') {
    presetKerr.classList.add('active');
    settings.mass = 1.50;
    settings.spin = 0.98;
    settings.relativity = true;
    settings.disk = true;
    settings.diskTemp = 9500;
  } else if (presetName === 'newtonian') {
    presetNewton.classList.add('active');
    settings.mass = 0.0; // effectively zero gravity
    settings.spin = 0.0;
    settings.relativity = false;
    settings.disk = true;
    settings.diskTemp = 6000;
  }

  // Update inputs to match settings
  massInput.value = settings.mass.toString();
  spinInput.value = settings.spin.toString();
  diskTempInput.value = settings.diskTemp.toString();
  relativityToggle.checked = settings.relativity;
  diskToggle.checked = settings.disk;
  
  updateBubbles();
}

presetGargantua.addEventListener('click', () => applyPreset('gargantua'));
presetMicro.addEventListener('click', () => applyPreset('micro'));
presetKerr.addEventListener('click', () => applyPreset('kerr'));
presetNewton.addEventListener('click', () => applyPreset('newtonian'));

// Bind inputs
massInput.addEventListener('input', () => {
  settings.mass = parseFloat(massInput.value);
  updateBubbles();
});

spinInput.addEventListener('input', () => {
  settings.spin = parseFloat(spinInput.value);
  updateBubbles();
});

diskTempInput.addEventListener('input', () => {
  settings.diskTemp = parseInt(diskTempInput.value);
  updateBubbles();
});

stepsInput.addEventListener('input', () => {
  settings.steps = parseInt(stepsInput.value);
  updateBubbles();
});

relativityToggle.addEventListener('change', () => {
  settings.relativity = relativityToggle.checked;
});

diskToggle.addEventListener('change', () => {
  settings.disk = diskToggle.checked;
});

gridToggle.addEventListener('change', () => {
  settings.grid = gridToggle.checked;
});

starsToggle.addEventListener('change', () => {
  settings.stars = starsToggle.checked;
});

// Guide Tab Switching
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    tabBtns.forEach(b => b.classList.remove('active'));
    tabContents.forEach(c => c.classList.add('hidden'));

    btn.classList.add('active');
    const tabName = btn.getAttribute('data-tab');
    const contentEl = document.getElementById(`tab-${tabName}`);
    if (contentEl) {
      contentEl.classList.remove('hidden');
    }
  });
});

// Initialize canvas and Raymarcher engine
const canvas = document.getElementById('webgl-canvas') as HTMLCanvasElement;
const raymarcher = new Raymarcher(canvas);

// Orbit Camera Drag Controls
let isDragging = false;
let startX = 0;
let startY = 0;
let startYaw = 0;
let startPitch = 0;

canvas.addEventListener('mousedown', (e) => {
  isDragging = true;
  startX = e.clientX;
  startY = e.clientY;
  startYaw = settings.camYaw;
  startPitch = settings.camPitch;
});

window.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  const dx = e.clientX - startX;
  const dy = e.clientY - startY;
  
  // Convert mouse drag to angle changes
  settings.camYaw = startYaw - dx * 0.005;
  settings.camPitch = Math.max(-1.4, Math.min(1.4, startPitch + dy * 0.005));
});

window.addEventListener('mouseup', () => {
  isDragging = false;
});

// Touch controls for mobile
canvas.addEventListener('touchstart', (e) => {
  if (e.touches.length === 1) {
    isDragging = true;
    startX = e.touches[0].clientX;
    startY = e.touches[0].clientY;
    startYaw = settings.camYaw;
    startPitch = settings.camPitch;
  }
});

canvas.addEventListener('touchmove', (e) => {
  if (!isDragging || e.touches.length !== 1) return;
  const dx = e.touches[0].clientX - startX;
  const dy = e.touches[0].clientY - startY;
  settings.camYaw = startYaw - dx * 0.007;
  settings.camPitch = Math.max(-1.4, Math.min(1.4, startPitch + dy * 0.007));
});

canvas.addEventListener('touchend', () => {
  isDragging = false;
});

// Zoom Controls (Mouse Wheel)
canvas.addEventListener('wheel', (e) => {
  settings.camRadius = Math.max(5.0, Math.min(30.0, settings.camRadius + e.deltaY * 0.01));
});

// FPS telemetry variables
let lastFrameTime = performance.now();
let frameCount = 0;
let fpsTimer = 0;

// Main Render Loop
function renderLoop(time: number) {
  // Rotate camera slowly in yaw when not dragging
  if (!isDragging) {
    // Autoincrement yaw for subtle movement
    settings.camYaw += 0.001;
  }

  // Render using WebGL engine
  raymarcher.render(settings);

  // Compute FPS
  frameCount++;
  const delta = time - lastFrameTime;
  lastFrameTime = time;
  fpsTimer += delta;

  if (fpsTimer >= 1000) {
    fpsLabel.textContent = `${frameCount} FPS`;
    frameCount = 0;
    fpsTimer = 0;
  }

  requestAnimationFrame(renderLoop);
}

// Initial update and start loop
updateBubbles();
requestAnimationFrame(renderLoop);
