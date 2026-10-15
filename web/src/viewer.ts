import type { CatalogName } from '../../metrics/index.ts';
import { Renderer } from './renderer.ts';
import { cameraState, loadSpacetime, paramsFor, type CameraState, type LoadedSpacetime } from './scene.ts';
import type { Settings } from './settings.ts';
import { packUniforms } from './uniforms.ts';

export interface ViewerEvents {
  onFrame?: (fps: number) => void;
  onError?: (message: string) => void;
}

/**
 * A ray-traced view of one spacetime on a canvas, with orbit controls.
 * Independent of the page UI so several can be embedded in one document.
 */
export class Viewer {
  readonly renderer: Renderer;
  private loaded: LoadedSpacetime | null = null;
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private running = false;
  private frameTimes: number[] = [];
  private lastInteraction = 0;
  autoRotate = 0; // radians per second
  /** When false the disk texture stops moving (prefers-reduced-motion). */
  animate = true;
  /** True while frames render at preview resolution. */
  previewing = false;
  private frozenTime = 0;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    readonly settings: Settings,
    private readonly events: ViewerEvents = {},
  ) {
    this.renderer = new Renderer(canvas);
    this.bindControls();
  }

  async init(): Promise<boolean> {
    if (!(await this.renderer.init())) return false;
    await this.setSpacetime(this.settings.spacetime);
    return true;
  }

  get spacetime(): LoadedSpacetime {
    if (!this.loaded) throw new Error('Viewer not initialised');
    return this.loaded;
  }

  async setSpacetime(name: CatalogName): Promise<void> {
    const loaded = loadSpacetime(name);
    await this.renderer.setMetric(loaded.wgsl, loaded.metric.name);
    this.touch();
    this.settings.spacetime = name;
    this.loaded = loaded;
  }

  params(): number[] {
    return paramsFor(this.spacetime, this.settings);
  }

  /** Call when a setting changes, so the next frames render at preview resolution. */
  touch() {
    this.lastInteraction = performance.now();
  }

  setParam(index: number, value: number) {
    this.touch();
    const params = [...this.params()];
    params[index] = value;
    this.settings.params[this.spacetime.name] = params;
  }

  /** Camera state for the current settings (camera position and frame). */
  camera(): CameraState {
    return cameraState(this.spacetime, this.params(), this.settings);
  }

  /** Milliseconds since the camera or a parameter last changed. */
  idleFor(): number {
    return this.dragging ? 0 : performance.now() - this.lastInteraction;
  }

  start() {
    if (this.running) return;
    this.running = true;
    let last = performance.now();
    const frame = (now: number) => {
      if (!this.running) return;
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      if (!this.dragging) this.settings.camYaw += this.autoRotate * dt;
      if (this.animate) this.frozenTime = now / 1000;
      this.draw(this.frozenTime);
      this.frameTimes.push(now);
      while (this.frameTimes.length && now - this.frameTimes[0] > 1000) this.frameTimes.shift();
      this.events.onFrame?.(this.frameTimes.length);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
  }

  stop() {
    this.running = false;
  }

  private draw(timeSeconds: number) {
    if (!this.loaded) return;
    // Half resolution while the camera is moving, full resolution when it settles.
    const moving = this.dragging || performance.now() - this.lastInteraction < 250;
    this.previewing = moving;
    const { width, height } = this.renderer.resize(this.settings.renderScale * (moving ? 0.5 : 1));
    const params = this.params();
    try {
      const cam = cameraState(this.loaded, params, this.settings);
      this.renderer.render(packUniforms(this.settings, { width, height, timeSeconds, params, ...cam }));
    } catch (e) {
      this.events.onError?.(e instanceof Error ? e.message : String(e));
    }
  }

  private bindControls() {
    const c = this.canvas;
    c.addEventListener('pointerdown', e => {
      this.dragging = true;
      this.lastX = e.clientX;
      this.lastY = e.clientY;
      c.setPointerCapture(e.pointerId);
    });
    c.addEventListener('pointermove', e => {
      if (!this.dragging) return;
      this.lastInteraction = performance.now();
      this.settings.camYaw -= (e.clientX - this.lastX) * 0.005;
      this.settings.camPitch = Math.max(-1.45, Math.min(1.45, this.settings.camPitch + (e.clientY - this.lastY) * 0.005));
      this.lastX = e.clientX;
      this.lastY = e.clientY;
    });
    const end = () => (this.dragging = false);
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener(
      'wheel',
      e => {
        e.preventDefault();
        this.lastInteraction = performance.now();
        this.settings.camRadius = Math.max(4, Math.min(80, this.settings.camRadius * Math.exp(e.deltaY * 0.001)));
      },
      { passive: false },
    );
  }
}
