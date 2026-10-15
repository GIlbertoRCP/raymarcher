import { horizonRadius, photonSphereRadius } from './physics.ts';
import { Renderer } from './renderer.ts';
import { defaultSettings, type Settings } from './settings.ts';

const settings: Settings = defaultSettings();

type FlightMode = 'manual' | 'orbit' | 'plunge' | 'flyby';
let currentFlightMode: FlightMode = 'manual';

function byId<T extends HTMLElement>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element #${id}`);
  return el as T;
}

// Sliders: [input id, bubble id, settings key, formatter]
type NumericKey = 'renderScale' | 'mass' | 'diskTemp' | 'steps' | 'diskThickness' | 'diskDensity' | 'bloomIntensity' | 'diskSpeed';
const sliders: [string, string, NumericKey, (v: number) => string][] = [
  ['param-render-scale', 'val-render-scale', 'renderScale', v => `${Math.round(v * 100)}%`],
  ['param-mass', 'val-mass', 'mass', v => v.toFixed(2)],
  ['param-disk-temp', 'val-disk-temp', 'diskTemp', v => `${v} K`],
  ['param-steps', 'val-steps', 'steps', v => v.toString()],
  ['param-disk-thickness', 'val-disk-thickness', 'diskThickness', v => v.toFixed(2)],
  ['param-disk-density', 'val-disk-density', 'diskDensity', v => v.toFixed(2)],
  ['param-bloom-intensity', 'val-bloom-intensity', 'bloomIntensity', v => v.toFixed(2)],
  ['param-disk-speed', 'val-disk-speed', 'diskSpeed', v => v.toFixed(2)],
];

// Toggles: [checkbox id, settings key]
type BooleanKey = 'relativity' | 'disk' | 'grid' | 'stars' | 'showPhotonSphere' | 'shiftVisualizer' | 'splitActive';
const toggles: [string, BooleanKey][] = [
  ['toggle-relativity', 'relativity'],
  ['toggle-disk', 'disk'],
  ['toggle-grid', 'grid'],
  ['toggle-stars', 'stars'],
  ['toggle-photon-sphere', 'showPhotonSphere'],
  ['toggle-shift-visualizer', 'shiftVisualizer'],
  ['toggle-split-screen', 'splitActive'],
];

const fpsLabel = byId<HTMLSpanElement>('stat-fps');
const horizonLabel = byId<HTMLSpanElement>('stat-horizon');
const photonLabel = byId<HTMLSpanElement>('stat-photon-sphere');

function syncControls() {
  for (const [inputId, bubbleId, key, format] of sliders) {
    byId<HTMLInputElement>(inputId).value = settings[key].toString();
    byId<HTMLSpanElement>(bubbleId).textContent = format(settings[key]);
  }
  for (const [inputId, key] of toggles) {
    byId<HTMLInputElement>(inputId).checked = settings[key];
  }
  // Areal (Schwarzschild) radii, in geometric units where G = c = 1.
  horizonLabel.textContent = horizonRadius(settings.mass).schwarzschild.toFixed(2);
  photonLabel.textContent = photonSphereRadius(settings.mass).schwarzschild.toFixed(2);
}

for (const [inputId, , key] of sliders) {
  const input = byId<HTMLInputElement>(inputId);
  input.addEventListener('input', () => {
    settings[key] = parseFloat(input.value);
    syncControls();
  });
}

for (const [inputId, key] of toggles) {
  const input = byId<HTMLInputElement>(inputId);
  input.addEventListener('change', () => {
    settings[key] = input.checked;
  });
}

// Presets
const presets: Record<string, Partial<Settings>> = {
  'preset-standard': { mass: 1.8, relativity: true, disk: true, diskTemp: 6500, diskThickness: 0.08, diskDensity: 1.8, diskSpeed: 1.0 },
  'preset-massive': { mass: 3.5, relativity: true, disk: true, diskTemp: 11000, diskThickness: 0.04, diskDensity: 3.0, diskSpeed: 1.6 },
  'preset-flat': { mass: 0.0, relativity: false, disk: true, diskTemp: 6000, diskThickness: 0.08, diskDensity: 1.5, diskSpeed: 1.0 },
};

for (const [id, preset] of Object.entries(presets)) {
  byId<HTMLButtonElement>(id).addEventListener('click', () => {
    for (const other of Object.keys(presets)) byId(other).classList.remove('active');
    byId(id).classList.add('active');
    Object.assign(settings, preset);
    applyFlightMode('manual');
    syncControls();
  });
}

// Flight modes
const flightButtons: Record<FlightMode, HTMLButtonElement> = {
  manual: byId('flight-manual'),
  orbit: byId('flight-orbit'),
  plunge: byId('flight-plunge'),
  flyby: byId('flight-flyby'),
};

function applyFlightMode(mode: FlightMode) {
  currentFlightMode = mode;
  for (const [m, btn] of Object.entries(flightButtons)) {
    btn.classList.toggle('active', m === mode);
  }
}

for (const [mode, btn] of Object.entries(flightButtons)) {
  btn.addEventListener('click', () => applyFlightMode(mode as FlightMode));
}

// Guide tabs
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');
tabBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    tabBtns.forEach(b => b.classList.remove('active'));
    tabContents.forEach(c => c.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.getAttribute('data-tab')}`)?.classList.remove('hidden');
  });
});

// Info modal
const infoModal = byId<HTMLDivElement>('info-modal');
byId('btn-info').addEventListener('click', () => infoModal.classList.remove('hidden'));
byId('modal-close').addEventListener('click', () => infoModal.classList.add('hidden'));
infoModal.addEventListener('click', e => {
  if (e.target === infoModal) infoModal.classList.add('hidden');
});

// Camera controls
const canvas = byId<HTMLCanvasElement>('gpu-canvas');
let isDragging = false;
let startX = 0;
let startY = 0;
let startYaw = 0;
let startPitch = 0;

function beginDrag(x: number, y: number) {
  isDragging = true;
  startX = x;
  startY = y;
  startYaw = settings.camYaw;
  startPitch = settings.camPitch;
}

function drag(x: number, y: number, sensitivity: number) {
  settings.camYaw = startYaw - (x - startX) * sensitivity;
  settings.camPitch = Math.max(-1.4, Math.min(1.4, startPitch + (y - startY) * sensitivity));
}

canvas.addEventListener('mousedown', e => beginDrag(e.clientX, e.clientY));
window.addEventListener('mousemove', e => {
  if (isDragging) drag(e.clientX, e.clientY, 0.005);
  if (settings.splitActive) {
    const rect = canvas.getBoundingClientRect();
    settings.splitX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  }
});
window.addEventListener('mouseup', () => (isDragging = false));

canvas.addEventListener('touchstart', e => {
  if (e.touches.length === 1) beginDrag(e.touches[0].clientX, e.touches[0].clientY);
});
canvas.addEventListener('touchmove', e => {
  if (isDragging && e.touches.length === 1) drag(e.touches[0].clientX, e.touches[0].clientY, 0.007);
});
canvas.addEventListener('touchend', () => (isDragging = false));

canvas.addEventListener('wheel', e => {
  settings.camRadius = Math.max(5.0, Math.min(40.0, settings.camRadius + e.deltaY * 0.01));
});

function updateFlight(time: number) {
  if (currentFlightMode === 'orbit') {
    settings.camYaw += 0.004;
    settings.camPitch = Math.sin(time * 0.0003) * 0.35;
    settings.camRadius = 22.0;
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
    settings.camYaw += 0.001;
  }
}

async function start() {
  syncControls();

  const renderer = new Renderer(canvas);
  if (!(await renderer.init())) {
    byId('webgpu-unsupported').classList.remove('hidden');
    return;
  }

  let lastFrameTime = performance.now();
  let frameCount = 0;
  let fpsTimer = 0;

  const frame = (time: number) => {
    updateFlight(time);
    renderer.render(settings);

    frameCount++;
    fpsTimer += time - lastFrameTime;
    lastFrameTime = time;
    if (fpsTimer >= 1000) {
      fpsLabel.textContent = `${frameCount} FPS`;
      frameCount = 0;
      fpsTimer = 0;
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

start();
