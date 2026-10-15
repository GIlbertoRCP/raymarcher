export type Vec3 = [number, number, number];

export interface CameraBasis {
  pos: Vec3;
  dir: Vec3;
  right: Vec3;
  up: Vec3;
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * Orbit camera looking at the origin, with +z as the world up axis.
 * Pitch must stay inside (-pi/2, pi/2) or the right vector degenerates.
 */
export function orbitCamera(radius: number, yaw: number, pitch: number): CameraBasis {
  const pos: Vec3 = [
    radius * Math.cos(pitch) * Math.sin(yaw),
    radius * Math.cos(pitch) * Math.cos(yaw),
    radius * Math.sin(pitch),
  ];
  const dir = normalize([-pos[0], -pos[1], -pos[2]]);
  const right = normalize([dir[1], -dir[0], 0]);
  const up = cross(right, dir);
  return { pos, dir, right, up };
}
