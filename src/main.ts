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
  diskThickness: number;
  diskDensity: number;
  bloomIntensity: number;
  diskSpeed: number;
  showPhotonSphere: boolean;
  shiftVisualizer: boolean;
  splitActive: boolean;
  splitX: number;
  spectrumMode: number;

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
  diskThickness: 0.08,
  diskDensity: 1.80,
  bloomIntensity: 1.00,
  diskSpeed: 1.00,
  showPhotonSphere: false,
  shiftVisualizer: false,
  splitActive: false,
  splitX: 0.5,
  spectrumMode: 0,

  // Camera State
  camRadius: 12.0,
  camYaw: 0.0,
  camPitch: 0.0,
};

let currentFlightMode: 'manual' | 'orbit' | 'plunge' | 'flyby' = 'manual';

// UI Elements
const massInput = document.getElementById('param-mass') as HTMLInputElement;
const massBubble = document.getElementById('val-mass') as HTMLSpanElement;
const spinInput = document.getElementById('param-spin') as HTMLInputElement;
const spinBubble = document.getElementById('val-spin') as HTMLSpanElement;
const diskTempInput = document.getElementById('param-disk-temp') as HTMLInputElement;
const diskTempBubble = document.getElementById('val-disk-temp') as HTMLSpanElement;
const stepsInput = document.getElementById('param-steps') as HTMLInputElement;
const stepsBubble = document.getElementById('val-steps') as HTMLSpanElement;

const diskThicknessInput = document.getElementById('param-disk-thickness') as HTMLInputElement;
const diskThicknessBubble = document.getElementById('val-disk-thickness') as HTMLSpanElement;
const diskDensityInput = document.getElementById('param-disk-density') as HTMLInputElement;
const diskDensityBubble = document.getElementById('val-disk-density') as HTMLSpanElement;
const bloomIntensityInput = document.getElementById('param-bloom-intensity') as HTMLInputElement;
const bloomIntensityBubble = document.getElementById('val-bloom-intensity') as HTMLSpanElement;

const diskSpeedInput = document.getElementById('param-disk-speed') as HTMLInputElement;
const diskSpeedBubble = document.getElementById('val-disk-speed') as HTMLSpanElement;
const photonSphereToggle = document.getElementById('toggle-photon-sphere') as HTMLInputElement;
const shiftVisualizerToggle = document.getElementById('toggle-shift-visualizer') as HTMLInputElement;
const splitScreenToggle = document.getElementById('toggle-split-screen') as HTMLInputElement;
const spectrumModeSelect = document.getElementById('param-spectrum-mode') as HTMLSelectElement;

const flightManual = document.getElementById('flight-manual') as HTMLButtonElement;
const flightOrbit = document.getElementById('flight-orbit') as HTMLButtonElement;
const flightPlunge = document.getElementById('flight-plunge') as HTMLButtonElement;
const flightFlyby = document.getElementById('flight-flyby') as HTMLButtonElement;

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
  diskThicknessBubble.textContent = settings.diskThickness.toFixed(2);
  diskDensityBubble.textContent = settings.diskDensity.toFixed(2);
  bloomIntensityBubble.textContent = settings.bloomIntensity.toFixed(2);
  diskSpeedBubble.textContent = settings.diskSpeed.toFixed(2);
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
    settings.diskThickness = 0.08;
    settings.diskDensity = 1.80;
    settings.diskSpeed = 1.00;
    settings.spectrumMode = 0;
  } else if (presetName === 'micro') {
    presetMicro.classList.add('active');
    settings.mass = 3.50;
    settings.spin = 0.0;
    settings.relativity = true;
    settings.disk = true;
    settings.diskTemp = 11000;
    settings.diskThickness = 0.04;
    settings.diskDensity = 3.00;
    settings.diskSpeed = 1.60;
    settings.spectrumMode = 1;
  } else if (presetName === 'kerr') {
    presetKerr.classList.add('active');
    settings.mass = 1.50;
    settings.spin = 0.98;
    settings.relativity = true;
    settings.disk = true;
    settings.diskTemp = 9500;
    settings.diskThickness = 0.12;
    settings.diskDensity = 1.20;
    settings.diskSpeed = 0.60;
    settings.spectrumMode = 2;
  } else if (presetName === 'newtonian') {
    presetNewton.classList.add('active');
    settings.mass = 0.0; // effectively zero gravity
    settings.spin = 0.0;
    settings.relativity = false;
    settings.disk = true;
    settings.diskTemp = 6000;
    settings.diskThickness = 0.08;
    settings.diskDensity = 1.50;
    settings.diskSpeed = 1.00;
    settings.spectrumMode = 0;
  }

  // Update inputs to match settings
  massInput.value = settings.mass.toString();
  spinInput.value = settings.spin.toString();
  diskThicknessInput.value = settings.diskThickness.toString();
  diskDensityInput.value = settings.diskDensity.toString();
  bloomIntensityInput.value = settings.bloomIntensity.toString();
  diskSpeedInput.value = settings.diskSpeed.toString();
  diskTempInput.value = settings.diskTemp.toString();
  relativityToggle.checked = settings.relativity;
  diskToggle.checked = settings.disk;
  photonSphereToggle.checked = settings.showPhotonSphere;
  shiftVisualizerToggle.checked = settings.shiftVisualizer;
  splitScreenToggle.checked = settings.splitActive;
  spectrumModeSelect.value = settings.spectrumMode.toString();
  
  applyFlightMode('manual');
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

diskThicknessInput.addEventListener('input', () => {
  settings.diskThickness = parseFloat(diskThicknessInput.value);
  updateBubbles();
});

diskDensityInput.addEventListener('input', () => {
  settings.diskDensity = parseFloat(diskDensityInput.value);
  updateBubbles();
});

bloomIntensityInput.addEventListener('input', () => {
  settings.bloomIntensity = parseFloat(bloomIntensityInput.value);
  updateBubbles();
});

diskSpeedInput.addEventListener('input', () => {
  settings.diskSpeed = parseFloat(diskSpeedInput.value);
  updateBubbles();
});

photonSphereToggle.addEventListener('change', () => {
  settings.showPhotonSphere = photonSphereToggle.checked;
});

shiftVisualizerToggle.addEventListener('change', () => {
  settings.shiftVisualizer = shiftVisualizerToggle.checked;
});

splitScreenToggle.addEventListener('change', () => {
  settings.splitActive = splitScreenToggle.checked;
});

spectrumModeSelect.addEventListener('change', () => {
  settings.spectrumMode = parseInt(spectrumModeSelect.value);
});

const canvasElement = document.getElementById('canvas') as HTMLCanvasElement;
if (canvasElement) {
  canvasElement.addEventListener('mousemove', (e) => {
    if (settings.splitActive) {
      const rect = canvasElement.getBoundingClientRect();
      const x = e.clientX - rect.left;
      settings.splitX = Math.max(0.0, Math.min(1.0, x / rect.width));
    }
  });
}

function applyFlightMode(mode: 'manual' | 'orbit' | 'plunge' | 'flyby') {
  currentFlightMode = mode;
  flightManual.classList.remove('active');
  flightOrbit.classList.remove('active');
  flightPlunge.classList.remove('active');
  flightFlyby.classList.remove('active');
  
  if (mode === 'manual') flightManual.classList.add('active');
  else if (mode === 'orbit') flightOrbit.classList.add('active');
  else if (mode === 'plunge') flightPlunge.classList.add('active');
  else if (mode === 'flyby') flightFlyby.classList.add('active');
}

flightManual.addEventListener('click', () => applyFlightMode('manual'));
flightOrbit.addEventListener('click', () => applyFlightMode('orbit'));
flightPlunge.addEventListener('click', () => applyFlightMode('plunge'));
flightFlyby.addEventListener('click', () => applyFlightMode('flyby'));

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
  // Handle camera flight modes
  if (currentFlightMode === 'orbit') {
    settings.camYaw += 0.004;
    settings.camPitch = Math.sin(time * 0.0003) * 0.35;
    settings.camRadius = 12.0;
  } else if (currentFlightMode === 'plunge') {
    const cycle = (time * 0.0005) % (2.0 * Math.PI);
    settings.camRadius = 9.8 + Math.cos(cycle) * 4.8;
    settings.camYaw += 0.006;
    settings.camPitch = Math.sin(cycle) * 0.3;
  } else if (currentFlightMode === 'flyby') {
    const cycle = (time * 0.0004) % (2.0 * Math.PI);
    settings.camRadius = 8.5 + Math.sin(cycle * 2.0) * 3.5;
    settings.camYaw = cycle;
    settings.camPitch = Math.cos(cycle) * 0.55;
  } else if (!isDragging) {
    // Autoincrement yaw for subtle movement in manual mode
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

// Physics Guide Info Modal Events
const infoButton = document.getElementById('btn-info') as HTMLButtonElement;
const infoModal = document.getElementById('info-modal') as HTMLDivElement;
const modalClose = document.getElementById('modal-close') as HTMLButtonElement;

if (infoButton && infoModal && modalClose) {
  infoButton.addEventListener('click', () => {
    infoModal.classList.remove('hidden');
  });
  
  modalClose.addEventListener('click', () => {
    infoModal.classList.add('hidden');
  });
  
  infoModal.addEventListener('click', (e) => {
    if (e.target === infoModal) {
      infoModal.classList.add('hidden');
    }
  });
}

