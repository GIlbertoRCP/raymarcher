import {
  compileMetric,
  compileToJs,
  compileToWgsl,
  lower,
  Spacetime,
  type CompiledMetric,
  type JsMetric,
  type Vec4,
} from '../../compiler/src/index.ts';
import { catalog, type CatalogName } from '../../metrics/index.ts';
import { orbitCamera } from './camera.ts';
import type { Settings } from './settings.ts';

export interface LoadedSpacetime {
  name: CatalogName;
  metric: CompiledMetric;
  js: JsMetric;
  wgsl: string;
}

const cache = new Map<CatalogName, LoadedSpacetime>();

/** Compiles a catalog spacetime once for both the CPU (camera) and GPU (tracer). */
export function loadSpacetime(name: CatalogName): LoadedSpacetime {
  let loaded = cache.get(name);
  if (!loaded) {
    const metric = compileMetric(catalog[name]);
    loaded = { name, metric, js: compileToJs(metric), wgsl: compileToWgsl(metric) };
    cache.set(name, loaded);
  }
  return loaded;
}

export function paramsFor(loaded: LoadedSpacetime, settings: Settings): number[] {
  return settings.params[loaded.name] ?? loaded.metric.params.map(p => p.value);
}

export interface CameraState {
  camPos: Vec4;
  /** Covariant frame g e_a of a static observer: 4-velocity, right, up, forward. */
  frame: Vec4[];
  lengthScale: number;
  escapeRadius: number;
}

/**
 * Places a static observer on an orbit around the origin and builds its
 * orthonormal frame from the local metric, in double precision.
 */
export function cameraState(loaded: LoadedSpacetime, params: number[], settings: Settings): CameraState {
  const lengthScale = loaded.js.scale([0, 0, 0, 0], params);
  const cam = orbitCamera(settings.camRadius * lengthScale, settings.camYaw, settings.camPitch);
  const camPos: Vec4 = [0, ...cam.pos];
  const st = new Spacetime(loaded.js, params);
  const g = st.metricAt(camPos);
  const frame = st.staticFrame(camPos, cam.dir, [0, 0, 1]).map(e => lower(g, e));
  return {
    camPos,
    frame,
    lengthScale,
    escapeRadius: Math.max(2 * settings.camRadius, 60) * lengthScale,
  };
}
