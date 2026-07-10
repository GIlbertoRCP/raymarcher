import { Settings } from './main.ts';
import { raymarchWGSL } from './shaders/raymarch.wgsl.ts';

export class WebGPURaymarcher {
  private canvas: HTMLCanvasElement;
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  
  // Pipeline 1: Raymarching Physics (HDR target)
  private pipeline!: GPURenderPipeline;
  private uniformBuffer!: GPUBuffer;
  private noiseTexture!: GPUTexture;
  private noiseSampler!: GPUSampler;
  private bindGroup!: GPUBindGroup;
  
  // Pipeline 2: Bloom Post-Processing & Compositing
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

      // Compile shader module (contains physics and composite passes)
      const shaderModule = this.device.createShaderModule({
        label: 'Relativistic Raymarching Shaders',
        code: raymarchWGSL,
      });

      // Create uniform buffer (144 bytes total)
      this.uniformBuffer = this.device.createBuffer({
        label: 'Raymarcher Uniform Buffer',
        size: 144,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // Create noise texture for accretion disk structures (256x256 RGBA)
      const size = 256;
      this.noiseTexture = this.device.createTexture({
        label: 'Accretion Disk Noise Texture',
        size: [size, size, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });

      // Generate random high frequency noise values
      const noiseData = new Uint8Array(size * size * 4);
      for (let i = 0; i < size * size * 4; i += 4) {
        noiseData[i] = Math.floor(Math.random() * 255);     // Red
        noiseData[i + 1] = Math.floor(Math.random() * 255); // Green
        noiseData[i + 2] = Math.floor(Math.random() * 255); // Blue
        noiseData[i + 3] = 255;
      }

      this.device.queue.writeTexture(
        { texture: this.noiseTexture },
        noiseData,
        { bytesPerRow: size * 4, rowsPerImage: size },
        [size, size, 1]
      );

      // Create linear wrap noise sampler
      this.noiseSampler = this.device.createSampler({
        label: 'Noise Sampler',
        addressModeU: 'repeat',
        addressModeV: 'repeat',
        magFilter: 'linear',
        minFilter: 'linear',
      });

      // Create scene linear sampler for bloom sampling
      this.sceneSampler = this.device.createSampler({
        label: 'Scene Sampler',
        magFilter: 'linear',
        minFilter: 'linear',
      });

      // Build Render Pipeline 1 (Raymarching)
      this.pipeline = this.device.createRenderPipeline({
        label: 'Raymarching HDR Pipeline',
        layout: 'auto',
        vertex: {
          module: shaderModule,
          entryPoint: 'vs_main',
        },
        fragment: {
          module: shaderModule,
          entryPoint: 'fs_main',
          targets: [
            {
              format: 'rgba16float', // HDR target format
            },
          ],
        },
        primitive: {
          topology: 'triangle-list',
        },
      });

      // Build Render Pipeline 2 (Compositing Bloom)
      this.compositePipeline = this.device.createRenderPipeline({
        label: 'Compositing Bloom Pipeline',
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

      // Create Bind Group for physics pass matching the pipeline layout (Group 0)
      this.bindGroup = this.device.createBindGroup({
        label: 'Raymarcher Bind Group',
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: this.uniformBuffer,
            },
          },
          {
            binding: 1,
            resource: this.noiseTexture.createView(),
          },
          {
            binding: 2,
            resource: this.noiseSampler,
          },
        ],
      });

      // Create Bind Group for composite pass matching composite pipeline layout (Group 0)
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
      console.log('Event Horizon: WebGPU context created and dual pipelines built successfully.');
      return true;
    } catch (err) {
      console.error('WebGPU: Initialization error:', err);
      return false;
    }
  }

  public render(settings: Settings) {
    if (!this.isInitialized) return;

    // Handle canvas resizing & offscreen texture allocation
    const displayWidth = this.canvas.clientWidth;
    const displayHeight = this.canvas.clientHeight;
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
      this.hdrTexture = this.device.createTexture({
        label: 'HDR Raymarch Texture Target',
        size: [this.canvas.width, this.canvas.height, 1],
        format: 'rgba16float',
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      });
      this.hdrTextureView = this.hdrTexture.createView();

      // Update the composite bind group with the new texture view
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

    // Camera Vector Calculations
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

    // Pack Uniform Buffer (144 bytes total)
    const arrayBuffer = new ArrayBuffer(144);
    const floatView = new Float32Array(arrayBuffer);
    const uintView = new Uint32Array(arrayBuffer);

    floatView[0] = this.canvas.width;
    floatView[1] = this.canvas.height;
    floatView[2] = performance.now() * 0.001; // u_time
    floatView[3] = settings.mass;

    floatView[4] = settings.spin;
    uintView[5] = settings.relativity ? 1 : 0;
    uintView[6] = settings.disk ? 1 : 0;
    uintView[7] = settings.grid ? 1 : 0;

    uintView[8] = settings.stars ? 1 : 0;
    floatView[9] = settings.diskTemp;
    uintView[10] = settings.steps;
    floatView[11] = fovScale;

    // Sliders
    floatView[12] = settings.diskThickness;
    floatView[13] = settings.diskDensity;
    floatView[14] = settings.bloomIntensity;
    floatView[15] = settings.diskSpeed;

    // Lab options
    uintView[16] = settings.showPhotonSphere ? 1 : 0;
    uintView[17] = settings.shiftVisualizer ? 1 : 0;
    uintView[18] = settings.splitActive ? 1 : 0;
    floatView[19] = settings.splitX;

    // u_cam_pos (offset 80 bytes -> index 20)
    floatView[20] = camPos[0];
    floatView[21] = camPos[1];
    floatView[22] = camPos[2];
    floatView[23] = settings.spectrumMode; // Replaces padding1

    // u_cam_dir (offset 96 bytes -> index 24)
    floatView[24] = camDir[0];
    floatView[25] = camDir[1];
    floatView[26] = camDir[2];
    floatView[27] = 0.0; // padding2

    // u_cam_up (offset 112 bytes -> index 28)
    floatView[28] = camUp[0];
    floatView[29] = camUp[1];
    floatView[30] = camUp[2];
    floatView[31] = 0.0; // padding3

    // u_cam_right (offset 128 bytes -> index 32)
    floatView[32] = camRight[0];
    floatView[33] = camRight[1];
    floatView[34] = camRight[2];
    floatView[35] = 0.0; // padding4

    // Copy packed variables to uniform buffer
    this.device.queue.writeBuffer(this.uniformBuffer, 0, arrayBuffer);

    // Draw using Command Encoder
    const commandEncoder = this.device.createCommandEncoder();

    // PASS 1: Raymarch Physics onto the offscreen HDR texture
    const raymarchPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: this.hdrTextureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    };

    const passEncoder = commandEncoder.beginRenderPass(raymarchPassDescriptor);
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, this.bindGroup);
    passEncoder.draw(3); // Fullscreen triangle covering viewport
    passEncoder.end();

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
    compEncoder.draw(3); // Composites with screen space multi-tap bloom
    compEncoder.end();

    this.device.queue.submit([commandEncoder.finish()]);
  }
}
