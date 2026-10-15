import { buildTracerShader } from './shaders/tracer.wgsl.ts';
import { UNIFORM_BUFFER_SIZE } from './uniforms.ts';

export class ShaderCompileError extends Error {}

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
    this.device = await adapter.requestDevice();
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
