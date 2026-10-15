// Runs the CPU shadow trace off the main thread so rendering stays smooth.
import type { CatalogName } from '../../metrics/index.ts';
import { loadSpacetime, type CameraState } from './scene.ts';
import { traceDiskInnerEdge, traceShadowEdge } from './shadow.ts';

export interface TraceRequest {
  id: number;
  spacetime: CatalogName;
  params: number[];
  camera: CameraState;
  maxTan: number;
  disk: boolean;
}

let current: AbortController | null = null;

self.onmessage = async (e: MessageEvent<TraceRequest>) => {
  current?.abort();
  const controller = new AbortController();
  current = controller;
  const { id, spacetime, params, camera, maxTan, disk } = e.data;
  const loaded = loadSpacetime(spacetime);
  const edge = await traceShadowEdge(loaded, params, camera, {
    maxTan,
    signal: controller.signal,
    onProgress: (done, total) => self.postMessage({ id, type: 'progress', done, total }),
  });
  if (controller.signal.aborted) return;
  const diskEdge = edge && disk ? traceDiskInnerEdge(loaded, params, camera, edge, maxTan) : null;
  if (!controller.signal.aborted) self.postMessage({ id, type: 'done', edge, diskEdge });
};
