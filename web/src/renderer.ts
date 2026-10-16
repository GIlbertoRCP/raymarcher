import { buildTracerShader } from './shaders/tracer.wgsl.ts';
import { UNIFORM_BUFFER_SIZE } from './uniforms.ts';

export class ShaderCompileError extends Error {}

/** IEEE 754 half-precision bits to a number. */
function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0) return sign * frac * 2 ** -24;
  if (exp === 31) return frac ? NaN : sign * Infinity;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

/** WebGPU host: a compute pass that traces rays, then a composite pass to the canvas. */
export class Renderer {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format!: GPUTextureFormat;
  private uniformBuffer!: GPUBuffer;
  private sampler!: GPUSampler;

  private computePipeline: GPUComputePipeline | null = null;
  private compositePipeline: GPURenderPipeline | null = null;
  private computeBindGroup: GPUBindGroup | null = null;
  private compositeUniforms: GPUBindGroup | null = null;
  private compositeScene: GPUBindGroup | null = null;
  private target: GPUTexture | null = null;

  constructor(private readonly canvas: HTMLCanvasElement) {}

  async init(): Promise<boolean> {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return false;
    // Timestamp queries, where available, let timeTrace measure GPU time.
    const timestamps = adapter.features.has('timestamp-query');
    this.device = await adapter.requestDevice({ requiredFeatures: timestamps ? ['timestamp-query'] : [] });
    const context = this.canvas.getContext('webgpu');
    if (!context) return false;
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'opaque' });
    this.uniformBuffer = this.device.createBuffer({
      label: 'Tracer uniforms',
      size: UNIFORM_BUFFER_SIZE,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.sampler = this.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });
    return true;
  }

  /** Compiles the tracer for a metric module from the WGSL backend. */
  async setMetric(metricWgsl: string, label: string): Promise<void> {
    const module = this.device.createShaderModule({ label: `Tracer (${label})`, code: buildTracerShader(metricWgsl) });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter(m => m.type === 'error');
    if (errors.length) {
      throw new ShaderCompileError(errors.map(e => `${e.lineNum}:${e.linePos} ${e.message}`).join('\n'));
    }
    const [computePipeline, compositePipeline] = await Promise.all([
      this.device.createComputePipelineAsync({
        label: `Trace rays (${label})`,
        layout: 'auto',
        compute: { module, entryPoint: 'cs_main' },
      }),
      this.device.createRenderPipelineAsync({
        label: 'Composite',
        layout: 'auto',
        vertex: { module, entryPoint: 'vs_composite' },
        fragment: { module, entryPoint: 'fs_composite', targets: [{ format: this.format }] },
        primitive: { topology: 'triangle-list' },
      }),
    ]);
    this.computePipeline = computePipeline;
    this.compositePipeline = compositePipeline;
    this.compositeUniforms = this.device.createBindGroup({
      layout: compositePipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
    });
    this.dropTarget(); // rebuild bind groups against the new pipelines
  }

  /** Internal render size for a given resolution scale. */
  resize(scale: number): { width: number; height: number } {
    // CSS pixels, not device pixels: each pixel costs a full geodesic.
    const width = Math.max(64, Math.floor(this.canvas.clientWidth * scale));
    const height = Math.max(64, Math.floor(this.canvas.clientHeight * scale));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.dropTarget();
    }
    return { width, height };
  }

  private dropTarget() {
    this.target?.destroy();
    this.target = null;
  }

  /**
   * Traces one frame offscreen at an exact size and reads the HDR texture back:
   * linear RGB plus the ray fate in alpha (0 escaped, 1 captured, 2 disk,
   * 3 unfinished). For cross-backend validation; the canvas is untouched.
   */
  async capture(uniforms: ArrayBuffer, width: number, height: number): Promise<{ rgba: Float32Array; gpuMs: number }> {
    if (!this.computePipeline) throw new Error('No metric loaded');
    const texture = this.device.createTexture({
      label: 'Capture target',
      size: [width, height],
      format: 'rgba16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
    const readback = this.device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const bindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: texture.createView() },
      ],
    });
    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass({ label: 'Trace rays (capture)' });
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    pass.end();
    encoder.copyTextureToBuffer({ texture }, { buffer: readback, bytesPerRow }, [width, height]);
    const t0 = performance.now();
    this.device.queue.submit([encoder.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    const gpuMs = performance.now() - t0; // wall time including the copy; not a benchmark
    await readback.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(readback.getMappedRange());
    const rgba = new Float32Array(width * height * 4);
    const rowHalves = bytesPerRow / 2;
    for (let y = 0; y < height; y++) {
      for (let i = 0; i < width * 4; i++) rgba[y * width * 4 + i] = halfToFloat(halves[y * rowHalves + i]);
    }
    readback.unmap();
    readback.destroy();
    texture.destroy();
    return { rgba, gpuMs };
  }

  /**
   * GPU time of the trace pass alone, per frame, for benchmarks: timestamp
   * queries around each compute pass when the device has them (Chrome rounds
   * them to 0.1 ms), otherwise wall time around submit + onSubmittedWorkDone.
   */
  async timeTrace(uniforms: ArrayBuffer, width: number, height: number, frames: number): Promise<{ ms: number[]; method: string }> {
    if (!this.computePipeline) throw new Error('No metric loaded');
    const texture = this.device.createTexture({ size: [width, height], format: 'rgba16float', usage: GPUTextureUsage.STORAGE_BINDING });
    const bindGroup = this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: texture.createView() },
      ],
    });
    const hasTimestamps = this.device.features.has('timestamp-query');
    const ms: number[] = [];
    for (let i = 0; i <= frames; i++) {
      const querySet = hasTimestamps ? this.device.createQuerySet({ type: 'timestamp', count: 2 }) : null;
      const resolve = querySet ? this.device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
      const read = querySet ? this.device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : null;
      const encoder = this.device.createCommandEncoder();
      const pass = encoder.beginComputePass(
        querySet ? { timestampWrites: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : {},
      );
      pass.setPipeline(this.computePipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
      pass.end();
      if (querySet) {
        encoder.resolveQuerySet(querySet, 0, 2, resolve!, 0);
        encoder.copyBufferToBuffer(resolve!, 0, read!, 0, 16);
      }
      // Written right before each submit: a frame of the viewer's own loop could
      // otherwise land in between and leave its (smaller) image size in the buffer.
      this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
      const t0 = performance.now();
      this.device.queue.submit([encoder.finish()]);
      await this.device.queue.onSubmittedWorkDone();
      const wall = performance.now() - t0;
      let frameMs = wall;
      if (querySet) {
        await read!.mapAsync(GPUMapMode.READ);
        const t = new BigInt64Array(read!.getMappedRange());
        frameMs = Number(t[1] - t[0]) / 1e6;
        read!.unmap();
        for (const b of [resolve!, read!]) b.destroy();
        querySet.destroy();
      }
      if (i > 0) ms.push(frameMs); // the first frame is a warm-up
    }
    texture.destroy();
    return { ms, method: hasTimestamps ? 'timestamp-query' : 'wall time (submit to onSubmittedWorkDone)' };
  }

  render(uniforms: ArrayBuffer): void {
    if (!this.computePipeline || !this.compositePipeline || !this.compositeUniforms) return;
    const { width, height } = this.canvas;

    if (!this.target) {
      this.target = this.device.createTexture({
        label: 'HDR ray-traced image',
        size: [width, height],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      const view = this.target.createView();
      this.computeBindGroup = this.device.createBindGroup({
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: view },
        ],
      });
      this.compositeScene = this.device.createBindGroup({
        layout: this.compositePipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: view },
          { binding: 1, resource: this.sampler },
        ],
      });
    }

    this.device.queue.writeBuffer(this.uniformBuffer, 0, uniforms);
    const encoder = this.device.createCommandEncoder();

    const trace = encoder.beginComputePass({ label: 'Trace rays' });
    trace.setPipeline(this.computePipeline);
    trace.setBindGroup(0, this.computeBindGroup!);
    trace.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
    trace.end();

    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 1] }],
    });
    pass.setPipeline(this.compositePipeline);
    pass.setBindGroup(0, this.compositeUniforms);
    pass.setBindGroup(1, this.compositeScene!);
    pass.draw(3);
    pass.end();

    this.device.queue.submit([encoder.finish()]);
  }
}
