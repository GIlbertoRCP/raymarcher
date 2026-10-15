export interface Settings {
  renderScale: number; // 0.25 to 1.0 internal resolution
  mass: number;
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

  camRadius: number;
  camPitch: number; // radians
  camYaw: number; // radians
}

export function defaultSettings(): Settings {
  return {
    renderScale: 0.5,
    mass: 1.8,
    relativity: true,
    disk: true,
    grid: false,
    stars: true,
    diskTemp: 6500,
    steps: 150,
    diskThickness: 0.08,
    diskDensity: 1.0,
    bloomIntensity: 1.0,
    diskSpeed: 1.0,
    showPhotonSphere: false,
    shiftVisualizer: false,
    splitActive: false,
    splitX: 0.5,
    camRadius: 22.0,
    camYaw: 0.0,
    camPitch: 0.12,
  };
}
