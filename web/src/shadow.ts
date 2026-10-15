// Traces the black hole's shadow edge on the CPU, in double precision, with
// the same compiled metric the GPU uses. Drawn over the render, it is a live
// cross-check between the two backends.

import { Spacetime } from '../../compiler/src/index.ts';
import type { CameraState, LoadedSpacetime } from './scene.ts';

export interface ShadowEdge {
  /** Edge points in camera tangent-plane coordinates (x right, y up), tan(angle) units. */
  points: { x: number; y: number }[];
  /** Mean angular radius about the image centre, degrees. */
  meanRadiusDeg: number;
}

/** Initial ray state for a point (x, y) on the camera's tangent plane. */
function rayState(cam: CameraState, x: number, y: number): Float64Array {
  // cam.frame holds covariant vectors (g e_a); p = -f0 + (x f1 + y f2 + f3) / |d|.
  const len = Math.hypot(x, y, 1);
  const p = [0, 1, 2, 3].map(mu => -cam.frame[0][mu] + (x * cam.frame[1][mu] + y * cam.frame[2][mu] + cam.frame[3][mu]) / len);
  return Float64Array.from([...cam.camPos, ...p]);
}

/**
 * Finds where rays switch from captured to escaped along `directions` spokes
 * from the image centre. Yields between spokes so the page stays responsive;
 * returns null when the centre ray is not captured (no shadow in view).
 */
export async function traceShadowEdge(
  loaded: LoadedSpacetime,
  params: number[],
  cam: CameraState,
  options: { directions?: number; maxTan?: number; iterations?: number; signal?: AbortSignal; onProgress?: (done: number, total: number) => void } = {},
): Promise<ShadowEdge | null> {
  const { directions = 32, maxTan = 1.2, iterations = 11, signal, onProgress } = options;
  if (!loaded.js.horizon) return null;
  const st = new Spacetime(loaded.js, params);
  const traceOpts = { stepFraction: 0.01, maxSteps: 20000, escapeRadius: cam.escapeRadius, disk: false };
  const captured = (x: number, y: number) => st.trace(rayState(cam, x, y), traceOpts).fate === 'captured';

  if (!captured(0, 0)) return null;

  const points: { x: number; y: number }[] = [];
  for (let i = 0; i < directions; i++) {
    if (signal?.aborted) return null;
    const angle = (2 * Math.PI * i) / directions;
    const cx = Math.cos(angle), cy = Math.sin(angle);
    let inside = 0, outside = maxTan;
    if (captured(cx * outside, cy * outside)) continue; // edge beyond the field of view
    for (let k = 0; k < iterations; k++) {
      const mid = (inside + outside) / 2;
      if (captured(cx * mid, cy * mid)) inside = mid;
      else outside = mid;
    }
    const s = (inside + outside) / 2;
    points.push({ x: cx * s, y: cy * s });
    onProgress?.(i + 1, directions);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  if (points.length < directions / 2) return null;
  const meanRadiusDeg = (points.reduce((sum, p) => sum + Math.atan(Math.hypot(p.x, p.y)), 0) / points.length) * (180 / Math.PI);
  return { points, meanRadiusDeg };
}

export interface DiskEdgeMark {
  /** Tangent-plane position of the disk's inner edge as seen straight above the hole. */
  x: number;
  y: number;
  /** Inner-edge radius from the metric (usually the ISCO), in coordinate units. */
  radius: number;
}

/**
 * Finds the lensed image of the disk's inner edge along the upward spoke:
 * the first point above the shadow where rays start landing on the disk.
 */
export function traceDiskInnerEdge(
  loaded: LoadedSpacetime,
  params: number[],
  cam: CameraState,
  edge: ShadowEdge,
  maxTan: number,
): DiskEdgeMark | null {
  const disk = loaded.js.disk;
  if (!disk) return null;
  const st = new Spacetime(loaded.js, params);
  const opts = { stepFraction: 0.01, maxSteps: 20000, escapeRadius: cam.escapeRadius, disk: true };
  const hitsDisk = (y: number) => st.trace(rayState(cam, 0, y), opts).fate === 'disk';
  const top = Math.max(...edge.points.filter(p => Math.abs(p.x) < 0.2 * Math.abs(p.y) + 1e-9).map(p => p.y), 0);
  if (top <= 0) return null;
  // Scan upward from the shadow edge for the first disk hit, then bisect.
  const samples = 48;
  let below = top, above = -1;
  for (let i = 1; i <= samples; i++) {
    const y = top + ((maxTan - top) * i) / samples;
    if (hitsDisk(y)) {
      above = y;
      break;
    }
    below = y;
  }
  if (above < 0) return null;
  for (let k = 0; k < 10; k++) {
    const mid = (below + above) / 2;
    if (hitsDisk(mid)) above = mid;
    else below = mid;
  }
  return { x: 0, y: above, radius: disk.inner([0, 0, 0, 0], params) };
}
