import { Settings } from './main.ts';
import { raymarchWGSL } from './shaders/raymarch.wgsl.ts';

export class WebGPURaymarcher {
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  
  // Pipeline 1: Compute Shader Raymarching Physics
  private computePipeline!: GPUComputePipeline;
  private uniformBuffer!: GPUBuffer;
  private noiseTexture!: GPUTexture;
  private noiseSampler!: GPUSampler;
  private bindGroup!: GPUBindGroup;
  
  // Pipeline 2: Screen Compositing & Multi-Tap Bloom
  private compositePipeline!: GPURenderPipeline;
  private sceneSampler!: GPUSampler;
  private hdrTexture: GPUTexture | null = null;
  private hdrTextureView!: GPUTextureView;
  private compositeBindGroup!: GPUBindGroup;
  private uniformBindGroup!: GPUBindGroup;

  private presentationFormat!: GPUTextureFormat;
  private isInitialized = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
  }

  public async init(): Promise<boolean> {
    try {
      if (!navigator.gpu) {
        console.warn('WebGPU: navigator.gpu is not available.');
        return false;
      }
      
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        console.warn('WebGPU: No GPU adapter found.');
        return false;
      }

      const device = await adapter.requestDevice();
      if (!device) {
        console.warn('WebGPU: Could not request GPU device.');
        return false;
      }
      this.device = device;

      const context = this.canvas.getContext('webgpu');
      if (!context) {
        console.warn('WebGPU: Could not get WebGPU context from canvas.');
        return false;
      }
      this.context = context;

      this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({
        device: this.device,
        format: this.presentationFormat,
        alphaMode: 'opaque',
      });

      // Compile compute and compositing WGSL shader module
      const shaderModule = this.device.createShaderModule({
        label: 'Compute Raymarching & Compositing WGSL Shaders',
        code: raymarchWGSL,
      });

      // Create uniform buffer (192 bytes total aligned)
      this.uniformBuffer = this.device.createBuffer({
        label: 'Raymarcher Uniform Buffer',
        size: 192,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // Create noise texture for accretion disk turbulence (256x256 RGBA)
      const size = 256;
      this.noiseTexture = this.device.createTexture({
        label: 'Accretion Disk Noise Texture',
        size: [size, size, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });

      const noiseData = new Uint8Array(size * size * 4);
      for (let i = 0; i < size * size * 4; i += 4) {
        noiseData[i]     = Math.floor(Math.random() * 255);
        noiseData[i + 1] = Math.floor(Math.random() * 255);
        noiseData[i + 2] = Math.floor(Math.random() * 255);
        noiseData[i + 3] = 255;
      }

      this.device.queue.writeTexture(
        { texture: this.noiseTexture },
        noiseData,
        { bytesPerRow: size * 4, rowsPerImage: size },
        [size, size, 1]
      );

      this.noiseSampler = this.device.createSampler({
        label: 'Noise Sampler',
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
      });

      this.sceneSampler = this.device.createSampler({
        label: 'Scene Sampler',
        magFilter: 'linear',
        minFilter: 'linear',
      });

      // Create Compute Pipeline for Raymarching Physics
      this.computePipeline = this.device.createComputePipeline({
        label: 'WebGPU Raymarching Compute Pipeline',
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint: 'cs_main',
        },
      });

      // Create Composite Render Pipeline for Screen Presentation & Bloom
      this.compositePipeline = this.device.createRenderPipeline({
        label: 'Screen Compositing Pipeline',
        layout: 'auto',
        vertex: {
          module: shaderModule,
          entryPoint: 'vs_composite',
        },
        fragment: {
          module: shaderModule,
          entryPoint: 'fs_composite',
          targets: [
            {
              format: this.presentationFormat,
            },
          ],
        },
        primitive: {
          topology: 'triangle-list',
        },
      });

      this.uniformBindGroup = this.device.createBindGroup({
        label: 'Uniform Only Bind Group',
        layout: this.compositePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.uniformBuffer,
            },
          },
        ],
      });

      this.isInitialized = true;
      console.log('Event Horizon: WebGPU Compute Pipeline & Storage Texture setup completed successfully.');
      return true;
    } catch (err) {
      console.error('WebGPU: Initialization error:', err);
      return false;
    }
  }

  public render(settings: Settings) {
    if (!this.isInitialized) return;

    const scale = Math.max(0.25, Math.min(1.0, settings.renderScale || 0.5));
    const displayWidth = Math.max(256, Math.floor(this.canvas.clientWidth * scale));
    const displayHeight = Math.max(256, Math.floor(this.canvas.clientHeight * scale));
    let resized = false;
    if (this.canvas.width !== displayWidth || this.canvas.height !== displayHeight || !this.hdrTexture) {
      this.canvas.width = displayWidth;
      this.canvas.height = displayHeight;
      this.context.configure({
        device: this.device,
        format: this.presentationFormat,
        alphaMode: 'opaque',
      });
      resized = true;
    }

    if (resized || !this.hdrTexture) {
      if (this.hdrTexture) {
        this.hdrTexture.destroy();
      }
      // Storage texture for compute shader raymarching output
      this.hdrTexture = this.device.createTexture({
        label: 'HDR Storage Raymarch Texture Target',
        size: [this.canvas.width, this.canvas.height, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.hdrTextureView = this.hdrTexture.createView();

      // Compute pipeline bind group
      this.bindGroup = this.device.createBindGroup({
        label: 'Raymarcher Compute Bind Group',
        layout: this.computePipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: { buffer: this.uniformBuffer },
          },
          {
            binding: 1,
            resource: this.noiseTexture.createView(),
          },
          {
            binding: 2,
            resource: this.noiseSampler,
          },
          {
            binding: 3,
            resource: this.hdrTextureView,
          },
        ],
      });

      // Composite pass bind group
      this.compositeBindGroup = this.device.createBindGroup({
        label: 'Composite Pass Bind Group',
        layout: this.compositePipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: this.hdrTextureView,
          },
          {
            binding: 1,
            resource: this.sceneSampler,
          },
        ],
      });
    }

    // Orbit Camera Math
    const cosPitch = Math.cos(settings.camPitch);
    const sinPitch = Math.sin(settings.camPitch);
    const cosYaw = Math.cos(settings.camYaw);
    const sinYaw = Math.sin(settings.camYaw);

    const camPos = [
      settings.camRadius * cosPitch * sinYaw,
      settings.camRadius * cosPitch * cosYaw,
      settings.camRadius * sinPitch
    ];

    const rawDir = [-camPos[0], -camPos[1], -camPos[2]];
    const dirLen = Math.sqrt(rawDir[0]*rawDir[0] + rawDir[1]*rawDir[1] + rawDir[2]*rawDir[2]);
    const camDir = [rawDir[0] / dirLen, rawDir[1] / dirLen, rawDir[2] / dirLen];

    const rawRight = [camDir[1], -camDir[0], 0];
    const rightLen = Math.sqrt(rawRight[0]*rawRight[0] + rawRight[1]*rawRight[1] + rawRight[2]*rawRight[2]);
    const camRight = [rawRight[0] / rightLen, rawRight[1] / rightLen, rawRight[2] / rightLen];

    const camUp = [
      camRight[1] * camDir[2] - camRight[2] * camDir[1],
      camRight[2] * camDir[0] - camRight[0] * camDir[2],
      camRight[0] * camDir[1] - camRight[1] * camDir[0]
    ];

    const fovScale = 0.577;

    // Pack Uniform Buffer (192 bytes)
    const arrayBuffer = new ArrayBuffer(192);
    const floatView = new Float32Array(arrayBuffer);
    const uintView = new Uint32Array(arrayBuffer);

    floatView[0] = this.canvas.width;
    floatView[1] = this.canvas.height;
    floatView[2] = performance.now() * 0.001;
    floatView[3] = settings.mass;

    floatView[4] = settings.spin;
    uintView[5] = settings.relativity ? 1 : 0;
    uintView[6] = settings.disk ? 1 : 0;
    uintView[7] = settings.grid ? 1 : 0;

    uintView[8] = settings.stars ? 1 : 0;
    floatView[9] = settings.diskTemp;
    uintView[10] = settings.steps;
    floatView[11] = fovScale;

    floatView[12] = settings.diskThickness;
    floatView[13] = settings.diskDensity;
    floatView[14] = settings.bloomIntensity;
    floatView[15] = settings.diskSpeed;

    uintView[16] = settings.showPhotonSphere ? 1 : 0;
    uintView[17] = settings.shiftVisualizer ? 1 : 0;
    uintView[18] = settings.splitActive ? 1 : 0;
    floatView[19] = settings.splitX;

    // camPos + spectrumMode
    floatView[20] = camPos[0];
    floatView[21] = camPos[1];
    floatView[22] = camPos[2];
    floatView[23] = settings.spectrumMode;

    // camDir + csgMode
    floatView[24] = camDir[0];
    floatView[25] = camDir[1];
    floatView[26] = camDir[2];
    uintView[27] = settings.csgMode;

    // camUp + csgBlend
    floatView[28] = camUp[0];
    floatView[29] = camUp[1];
    floatView[30] = camUp[2];
    floatView[31] = settings.csgBlend;

    // camRight + softShadows
    floatView[32] = camRight[0];
    floatView[33] = camRight[1];
    floatView[34] = camRight[2];
    uintView[35] = settings.softShadows ? 1 : 0;

    // shadowK, aoEnabled, aoIntensity, padding
    floatView[36] = settings.shadowK;
    uintView[37] = settings.aoEnabled ? 1 : 0;
    floatView[38] = settings.aoIntensity;
    floatView[39] = 0.0;

    this.device.queue.writeBuffer(this.uniformBuffer, 0, arrayBuffer);

    const commandEncoder = this.device.createCommandEncoder();

    // PASS 1: Dispatch Compute Shader Raymarching
    const computePass = commandEncoder.beginComputePass({ label: 'Raymarching Compute Pass' });
    computePass.setPipeline(this.computePipeline);
    computePass.setBindGroup(0, this.bindGroup);
    computePass.dispatchWorkgroups(
      Math.ceil(this.canvas.width / 16),
      Math.ceil(this.canvas.height / 16),
      1
    );
    computePass.end();

    // PASS 2: Composite & Bloom to screen swapchain
    const canvasTextureView = this.context.getCurrentTexture().createView();
    const compositePassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: canvasTextureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const compEncoder = commandEncoder.beginRenderPass(compositePassDescriptor);
    compEncoder.setPipeline(this.compositePipeline);
    compEncoder.setBindGroup(0, this.uniformBindGroup);
    compEncoder.setBindGroup(1, this.compositeBindGroup);
    compEncoder.draw(3);
    compEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }
}
