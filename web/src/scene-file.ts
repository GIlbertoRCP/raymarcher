// The shared scene format (scenes/*.json): everything needed to render the same
// image in the WebGPU app and the native CUDA renderer. Read by the web app
// (?scene=<name>), the native renderer (raymarcher --render) and validation/.

import { catalog, type CatalogName } from '../../metrics/index.ts';
import { loadSpacetime } from './scene.ts';
import { defaultSettings, type Background, type Settings } from './settings.ts';

export interface SceneFile {
  /** File name without .json; also names the render outputs. */
  name: string;
  spacetime: CatalogName;
  /** Metric parameters by name; missing ones take the metric file's default. */
  params: Record<string, number>;
  camera: {
    /** Distance from the origin in units of the metric's length scale (usually M). */
    radius: number;
    yaw: number; // radians
    /** Elevation above the disk plane in radians; the inclination from the spin axis is pi/2 - pitch. */
    pitch: number;
    fovDegrees: number; // applies to the narrower image side
  };
  resolution: [number, number];
  sky: Background;
  disk: boolean;
  redshift: boolean;
  exposure: number;
  diskTemp: number; // peak disk temperature in kelvin
  /** Seconds; only moves the artistic disk texture. */
  time: number;
  integrator: { method: 'rk4'; maxSteps: number; stepFraction: number };
}

function fail(message: string): never {
  throw new Error(`Invalid scene: ${message}`);
}

const num = (v: unknown, what: string): number => (typeof v === 'number' && Number.isFinite(v) ? v : fail(`${what} must be a number`));

/** Validates a parsed scene file. */
export function parseScene(raw: unknown): SceneFile {
  if (typeof raw !== 'object' || raw === null) fail('not an object');
  const o = raw as Record<string, any>;
  if (!(o.spacetime in catalog)) fail(`unknown spacetime '${o.spacetime}'`);
  const spacetime = o.spacetime as CatalogName;
  const { metric } = loadSpacetime(spacetime);
  const params: Record<string, number> = {};
  for (const [k, v] of Object.entries(o.params ?? {})) {
    if (!metric.params.some(p => p.name === k)) fail(`${spacetime} has no parameter '${k}'`);
    params[k] = num(v, `params.${k}`);
  }
  const res = o.resolution;
  if (!Array.isArray(res) || res.length !== 2 || !res.every(n => Number.isInteger(n) && n > 0)) fail('resolution must be [width, height]');
  if (!['stars', 'grid', 'none'].includes(o.sky)) fail(`sky must be stars, grid or none`);
  if (o.integrator?.method !== 'rk4') fail(`integrator.method must be 'rk4'`);
  return {
    name: typeof o.name === 'string' ? o.name : fail('name must be a string'),
    spacetime,
    params,
    camera: {
      radius: num(o.camera?.radius, 'camera.radius'),
      yaw: num(o.camera?.yaw, 'camera.yaw'),
      pitch: num(o.camera?.pitch, 'camera.pitch'),
      fovDegrees: num(o.camera?.fovDegrees, 'camera.fovDegrees'),
    },
    resolution: [res[0], res[1]],
    sky: o.sky,
    disk: Boolean(o.disk),
    redshift: Boolean(o.redshift),
    exposure: num(o.exposure, 'exposure'),
    diskTemp: num(o.diskTemp, 'diskTemp'),
    time: num(o.time ?? 0, 'time'),
    integrator: { method: 'rk4', maxSteps: num(o.integrator.maxSteps, 'integrator.maxSteps'), stepFraction: num(o.integrator.stepFraction, 'integrator.stepFraction') },
  };
}

/** Parameter values in the metric's declaration order. */
export function sceneParams(scene: SceneFile): number[] {
  return loadSpacetime(scene.spacetime).metric.params.map(p => scene.params[p.name] ?? p.value);
}

/** Viewer settings for a scene (render scale aside, which only the page layout decides). */
export function sceneToSettings(scene: SceneFile, base: Settings = defaultSettings()): Settings {
  return {
    ...base,
    spacetime: scene.spacetime,
    params: { ...base.params, [scene.spacetime]: sceneParams(scene) },
    maxSteps: scene.integrator.maxSteps,
    stepFraction: scene.integrator.stepFraction,
    disk: scene.disk,
    background: scene.sky,
    redshift: scene.redshift,
    exposure: scene.exposure,
    diskTemp: scene.diskTemp,
    camRadius: scene.camera.radius,
    camYaw: scene.camera.yaw,
    camPitch: scene.camera.pitch,
    fovDegrees: scene.camera.fovDegrees,
  };
}

/** The scene the viewer currently shows, at the given image size. */
export function settingsToScene(settings: Settings, params: number[], width: number, height: number, time: number, name = 'exported'): SceneFile {
  const spacetime = settings.spacetime;
  if (spacetime === 'custom') throw new Error('Scene files only reference catalog spacetimes; share a custom metric with its link instead');
  const { metric } = loadSpacetime(spacetime);
  return {
    name,
    spacetime,
    params: Object.fromEntries(metric.params.map((p, i) => [p.name, params[i]])),
    camera: { radius: settings.camRadius, yaw: settings.camYaw, pitch: settings.camPitch, fovDegrees: settings.fovDegrees },
    resolution: [width, height],
    sky: settings.background,
    disk: settings.disk,
    redshift: settings.redshift,
    exposure: settings.exposure,
    diskTemp: settings.diskTemp,
    time,
    integrator: { method: 'rk4', maxSteps: settings.maxSteps, stepFraction: settings.stepFraction },
  };
}
