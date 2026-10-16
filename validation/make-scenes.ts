// Writes the cross-backend scene set (docs/phase-2-plan.md, step 4) to scenes/:
//   Schwarzschild and Kerr (a = 0.6, 0.95) at inclinations 17, 45 and 85 degrees,
//   disk on and off; flat space at the same inclinations.
// Everything else is the web app's default view.
//   npx vite-node validation/make-scenes.ts
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { SceneFile } from '../web/src/scene-file.ts';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(root, 'scenes');

/** Observer inclination from the spin (z) axis; the camera pitch is measured from the disk plane. */
const INCLINATIONS = [17, 45, 85];

const spacetimes: { id: string; spacetime: SceneFile['spacetime']; params: Record<string, number>; disks: boolean[] }[] = [
  { id: 'schwarzschild', spacetime: 'schwarzschild', params: { M: 1 }, disks: [true, false] },
  { id: 'kerr-a060', spacetime: 'kerr', params: { M: 1, a: 0.6 }, disks: [true, false] },
  { id: 'kerr-a095', spacetime: 'kerr', params: { M: 1, a: 0.95 }, disks: [true, false] },
  { id: 'minkowski', spacetime: 'minkowski', params: { M: 1 }, disks: [true] },
];

mkdirSync(outDir, { recursive: true });
let count = 0;
for (const st of spacetimes) {
  for (const inc of INCLINATIONS) {
    for (const disk of st.disks) {
      const name = `${st.id}-i${inc}-${disk ? 'disk' : 'nodisk'}`;
      const scene: SceneFile = {
        name,
        spacetime: st.spacetime,
        params: st.params,
        camera: { radius: 26, yaw: 0, pitch: Number((((90 - inc) * Math.PI) / 180).toFixed(6)), fovDegrees: 45 },
        resolution: [960, 540],
        sky: 'stars',
        disk,
        redshift: false,
        exposure: 0.55,
        diskTemp: 4500,
        time: 0,
        integrator: { method: 'rk4', maxSteps: 300, stepFraction: 0.03 },
      };
      writeFileSync(join(outDir, `${name}.json`), JSON.stringify(scene, null, 2) + '\n');
      count++;
    }
  }
}
console.log(`wrote ${count} scenes to scenes/`);
