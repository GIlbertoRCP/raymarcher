import { Settings } from './main.ts';
import { vertexShaderSource } from './shaders/raymarch.vert.ts';
import { fragmentShaderSource } from './shaders/raymarch.frag.ts';
import { compositeShaderSource } from './shaders/composite.frag.ts';
import { WebGPURaymarcher } from './raymarcher_webgpu.ts';

export class Raymarcher {
  private canvas: HTMLCanvasElement;
  private webgpu: WebGPURaymarcher | null = null;
  private isWebGPUReady = false;
  private isWebGLFallback = false;

  // WebGL engine fields (initialized only if fallback occurs)
  private gl!: WebGL2RenderingContext | WebGLRenderingContext;
  private program!: WebGLProgram;
  private compositeProgram!: WebGLProgram;
  private positionBuffer!: WebGLBuffer;
  private noiseTexture!: WebGLTexture;
  private uniforms: { [name: string]: WebGLUniformLocation | null } = {};
  private compositeUniforms: { [name: string]: WebGLUniformLocation | null } = {};

  // Framebuffer elements for WebGL HDR bloom (WebGL2 only)
  private fbo: WebGLFramebuffer | null = null;
  private fboTexture: WebGLTexture | null = null;
  private fboWidth = 0;
  private fboHeight = 0;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    
    // Feature detect WebGPU support
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      this.tryInitWebGPU();
    } else {
      console.warn('Event Horizon: WebGPU not supported in this browser. Initializing WebGL synchronously...');
      this.initWebGL();
      this.isWebGLFallback = true;
      // Delay UI update slightly to let DOM initialize
      setTimeout(() => this.updateUIRendererStatus('WebGL1'), 0);
    }
  }

  private async tryInitWebGPU() {
    this.webgpu = new WebGPURaymarcher(this.canvas);
    const success = await this.webgpu.init();
    if (success) {
      this.isWebGPUReady = true;
      this.updateUIRendererStatus('WebGPU (High Performance)');
    } else {
      console.warn('Event Horizon: WebGPU initialization failed. Falling back to WebGL...');
      this.webgpu = null;
      this.initWebGL();
      this.isWebGLFallback = true;
      
      const isWebGL2 = typeof WebGL2RenderingContext !== 'undefined' && this.gl instanceof WebGL2RenderingContext;
      this.updateUIRendererStatus(isWebGL2 ? 'WebGL2 (Fallback)' : 'WebGL1 (Fallback)');
    }
  }

  private updateUIRendererStatus(statusText: string) {
    const label = document.getElementById('stat-renderer');
    if (!label) {
      const statsGrid = document.querySelector('.stats-grid');
      if (statsGrid) {
        const box = document.createElement('div');
        box.className = 'stat-box';
        box.innerHTML = `
          <span class="stat-label">Pipeline</span>
          <span class="stat-val" id="stat-renderer" style="color: #00ffcc; font-weight: 600;">${statusText}</span>
        `;
        statsGrid.appendChild(box);
      }
    } else {
      label.textContent = statusText;
    }
  }

  private initWebGL() {
    try {
      console.log('Event Horizon WebGL: Initializing context...');
      let gl = this.canvas.getContext('webgl2', { antialias: false, depth: false }) as WebGL2RenderingContext | WebGLRenderingContext | null;
      let isWebGL2 = true;

      if (!gl) {
        console.warn('Event Horizon WebGL: WebGL2 not supported, falling back to WebGL1...');
        gl = (this.canvas.getContext('webgl', { antialias: false, depth: false }) ||
              this.canvas.getContext('experimental-webgl', { antialias: false, depth: false })) as WebGLRenderingContext | null;
        isWebGL2 = false;
      }

      if (!gl) {
        throw new Error('WebGL is not supported by your browser or hardware acceleration is disabled.');
      }
      this.gl = gl;
      console.log(`Event Horizon WebGL: Context created successfully (WebGL2: ${isWebGL2}).`);

      // Enable WebGL2 float texture render extension
      if (isWebGL2) {
        gl.getExtension('EXT_color_buffer_float');
      }

      // Create shaders and programs for Physics pass
      console.log('Event Horizon WebGL: Compiling shaders...');
      const vsSource = isWebGL2 ? vertexShaderSource : this.translateShaderToWebGL1(vertexShaderSource, true);
      const fsSource = isWebGL2 ? fragmentShaderSource : this.translateShaderToWebGL1(fragmentShaderSource, false);

      const vertexShader = this.compileShader(gl.VERTEX_SHADER, vsSource);
      const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fsSource);
      this.program = this.createProgram(vertexShader, fragmentShader);

      if (isWebGL2) {
        // Create shaders and programs for Composite pass (only if WebGL2 is active)
        const compFsSource = compositeShaderSource;
        const compFragmentShader = this.compileShader(gl.FRAGMENT_SHADER, compFsSource);
        this.compositeProgram = this.createProgram(vertexShader, compFragmentShader);
        console.log('Event Horizon WebGL: Shader programs linked successfully for WebGL2.');
      } else {
        console.log('Event Horizon WebGL: Shader programs linked successfully for WebGL1.');
      }

      // Setup fullscreen quad positions
      this.positionBuffer = gl.createBuffer()!;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      const positions = new Float32Array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
        -1.0,  1.0,
         1.0, -1.0,
         1.0,  1.0,
      ]);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

      // Create noise texture
      this.noiseTexture = this.createNoiseTexture();

      // Cache uniform locations for Physics pass
      const uniformNames = [
        'u_resolution', 'u_time', 'u_mass', 'u_spin', 'u_relativity',
        'u_disk', 'u_grid', 'u_stars', 'u_disk_temp', 'u_max_steps',
        'u_noise_tex', 'u_cam_pos', 'u_cam_dir', 'u_cam_up', 'u_cam_right', 'u_fov_scale',
        'u_disk_thickness', 'u_disk_density', 'u_bloom_intensity',
        'u_disk_speed', 'u_show_photon_sphere', 'u_shift_visualizer',
        'u_split_active', 'u_split_x', 'u_spectrum_mode',
        'u_csg_mode', 'u_csg_blend', 'u_soft_shadows', 'u_shadow_k', 'u_ao_enabled', 'u_ao_intensity'
      ];
      for (const name of uniformNames) {
        this.uniforms[name] = gl.getUniformLocation(this.program, name);
      }

      if (isWebGL2) {
        // Cache uniform locations for Composite pass
        const compUniformNames = [
          'u_scene_tex', 'u_resolution', 'u_bloom_intensity'
        ];
        for (const name of compUniformNames) {
          this.compositeUniforms[name] = gl.getUniformLocation(this.compositeProgram, name);
        }
      }
    } catch (e: any) {
      this.showShaderError(this.canvas, e.message || String(e));
      throw e;
    }
  }

  private setupFramebuffer(width: number, height: number) {
    const gl = this.gl;
    if (this.fbo && this.fboWidth === width && this.fboHeight === height) {
      return;
    }

    if (this.fbo) {
      gl.deleteFramebuffer(this.fbo);
    }
    if (this.fboTexture) {
      gl.deleteTexture(this.fboTexture);
    }

    this.fboWidth = width;
    this.fboHeight = height;

    this.fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);

    this.fboTexture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, this.fboTexture);

    const isWebGL2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;
    let internalFormat: number = gl.RGBA;
    let format: number = gl.RGBA;
    let type: number = gl.UNSIGNED_BYTE;

    if (isWebGL2) {
      const gl2 = gl as WebGL2RenderingContext;
      internalFormat = gl2.RGBA16F;
      type = gl2.FLOAT;
    }

    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fboTexture, 0);

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      console.warn('WebGL FBO: Framebuffer is incomplete, rendering direct fallback:', status);
      gl.deleteFramebuffer(this.fbo);
      gl.deleteTexture(this.fboTexture);
      this.fbo = null;
      this.fboTexture = null;
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  private compileShader(type: number, source: string): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compilation failed: ${info}`);
    }
    return shader;
  }

  private createProgram(vs: WebGLShader, fs: WebGLShader): WebGLProgram {
    const gl = this.gl;
    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program);
      gl.deleteProgram(program);
      throw new Error(`Shader program linking failed: ${info}`);
    }
    return program;
  }

  private createNoiseTexture(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);

    const size = 256;
    const data = new Uint8Array(size * size * 4);
    for (let i = 0; i < size * size * 4; i += 4) {
      data[i] = Math.floor(Math.random() * 255);
      data[i + 1] = Math.floor(Math.random() * 255);
      data[i + 2] = Math.floor(Math.random() * 255);
      data[i + 3] = 255;
    }

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, size, size, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    return texture;
  }

  public render(settings: Settings) {
    if (this.isWebGPUReady && this.webgpu) {
      this.webgpu.render(settings);
    } else if (this.isWebGLFallback) {
      this.renderWebGL(settings);
    }
  }

  private renderWebGL(settings: Settings) {
    const gl = this.gl;
    const canvas = gl.canvas as HTMLCanvasElement;

    const scale = Math.max(0.25, Math.min(1.0, settings.renderScale || 0.5));
    const displayWidth = Math.max(256, Math.floor(canvas.clientWidth * scale));
    const displayHeight = Math.max(256, Math.floor(canvas.clientHeight * scale));
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }

    const isWebGL2 = typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext;

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

    if (isWebGL2) {
      // Allocate / resize offscreen HDR Framebuffer
      this.setupFramebuffer(canvas.width, canvas.height);

      if (this.fbo) {
        // PASS 1: Raymarch Physics onto offscreen HDR texture FBO
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
        gl.viewport(0, 0, canvas.width, canvas.height);

        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.program);

        gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
        const positionLoc = gl.getAttribLocation(this.program, 'position');
        gl.enableVertexAttribArray(positionLoc);
        gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.noiseTexture);
        gl.uniform1i(this.uniforms['u_noise_tex'], 0);

        gl.uniform2f(this.uniforms['u_resolution'], canvas.width, canvas.height);
        gl.uniform1f(this.uniforms['u_time'], performance.now() * 0.001);
        gl.uniform1f(this.uniforms['u_mass'], settings.mass);
        gl.uniform1f(this.uniforms['u_spin'], settings.spin);
        gl.uniform1i(this.uniforms['u_relativity'], settings.relativity ? 1 : 0);
        gl.uniform1i(this.uniforms['u_disk'], settings.disk ? 1 : 0);
        gl.uniform1i(this.uniforms['u_grid'], settings.grid ? 1 : 0);
        gl.uniform1i(this.uniforms['u_stars'], settings.stars ? 1 : 0);
        gl.uniform1f(this.uniforms['u_disk_temp'], settings.diskTemp);
        gl.uniform1i(this.uniforms['u_max_steps'], settings.steps);
        gl.uniform1f(this.uniforms['u_disk_thickness'], settings.diskThickness);
        gl.uniform1f(this.uniforms['u_disk_density'], settings.diskDensity);
        gl.uniform1f(this.uniforms['u_bloom_intensity'], settings.bloomIntensity);
        gl.uniform1f(this.uniforms['u_disk_speed'], settings.diskSpeed);
        gl.uniform1i(this.uniforms['u_show_photon_sphere'], settings.showPhotonSphere ? 1 : 0);
        gl.uniform1i(this.uniforms['u_shift_visualizer'], settings.shiftVisualizer ? 1 : 0);
        gl.uniform1i(this.uniforms['u_split_active'], settings.splitActive ? 1 : 0);
        gl.uniform1f(this.uniforms['u_split_x'], settings.splitX);
        gl.uniform1i(this.uniforms['u_spectrum_mode'], settings.spectrumMode);
        gl.uniform1i(this.uniforms['u_csg_mode'], settings.csgMode);
        gl.uniform1f(this.uniforms['u_csg_blend'], settings.csgBlend);
        gl.uniform1i(this.uniforms['u_soft_shadows'], settings.softShadows ? 1 : 0);
        gl.uniform1f(this.uniforms['u_shadow_k'], settings.shadowK);
        gl.uniform1i(this.uniforms['u_ao_enabled'], settings.aoEnabled ? 1 : 0);
        gl.uniform1f(this.uniforms['u_ao_intensity'], settings.aoIntensity);

        gl.uniform3f(this.uniforms['u_cam_pos'], camPos[0], camPos[1], camPos[2]);
        gl.uniform3f(this.uniforms['u_cam_dir'], camDir[0], camDir[1], camDir[2]);
        gl.uniform3f(this.uniforms['u_cam_right'], camRight[0], camRight[1], camRight[2]);
        gl.uniform3f(this.uniforms['u_cam_up'], camUp[0], camUp[1], camUp[2]);
        gl.uniform1f(this.uniforms['u_fov_scale'], fovScale);

        gl.drawArrays(gl.TRIANGLES, 0, 6);

        // PASS 2: Composite FBO HDR texture with Bloom to screen swapchain
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, canvas.width, canvas.height);

        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.compositeProgram);

        const compPositionLoc = gl.getAttribLocation(this.compositeProgram, 'position');
        gl.enableVertexAttribArray(compPositionLoc);
        gl.vertexAttribPointer(compPositionLoc, 2, gl.FLOAT, false, 0, 0);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, this.fboTexture!);
        gl.uniform1i(this.compositeUniforms['u_scene_tex'], 0);

        gl.uniform2f(this.compositeUniforms['u_resolution'], canvas.width, canvas.height);
        gl.uniform1f(this.compositeUniforms['u_bloom_intensity'], settings.bloomIntensity);

        gl.drawArrays(gl.TRIANGLES, 0, 6);
        return;
      }
    }

    // Single-pass direct screen rendering for WebGL1 fallback (or incomplete WebGL2 framebuffers)
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvas.width, canvas.height);
    
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    const positionLoc = gl.getAttribLocation(this.program, 'position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.noiseTexture);
    gl.uniform1i(this.uniforms['u_noise_tex'], 0);

    gl.uniform2f(this.uniforms['u_resolution'], canvas.width, canvas.height);
    gl.uniform1f(this.uniforms['u_time'], performance.now() * 0.001);
    gl.uniform1f(this.uniforms['u_mass'], settings.mass);
    gl.uniform1f(this.uniforms['u_spin'], settings.spin);
    gl.uniform1i(this.uniforms['u_relativity'], settings.relativity ? 1 : 0);
    gl.uniform1i(this.uniforms['u_disk'], settings.disk ? 1 : 0);
    gl.uniform1i(this.uniforms['u_grid'], settings.grid ? 1 : 0);
    gl.uniform1i(this.uniforms['u_stars'], settings.stars ? 1 : 0);
    gl.uniform1f(this.uniforms['u_disk_temp'], settings.diskTemp);
    gl.uniform1i(this.uniforms['u_max_steps'], settings.steps);
    gl.uniform1f(this.uniforms['u_disk_thickness'], settings.diskThickness);
    gl.uniform1f(this.uniforms['u_disk_density'], settings.diskDensity);
    gl.uniform1f(this.uniforms['u_bloom_intensity'], settings.bloomIntensity);
    gl.uniform1f(this.uniforms['u_disk_speed'], settings.diskSpeed);
    gl.uniform1i(this.uniforms['u_show_photon_sphere'], settings.showPhotonSphere ? 1 : 0);
    gl.uniform1i(this.uniforms['u_shift_visualizer'], settings.shiftVisualizer ? 1 : 0);
    gl.uniform1i(this.uniforms['u_split_active'], settings.splitActive ? 1 : 0);
    gl.uniform1f(this.uniforms['u_split_x'], settings.splitX);
    gl.uniform1i(this.uniforms['u_spectrum_mode'], settings.spectrumMode);

    gl.uniform3f(this.uniforms['u_cam_pos'], camPos[0], camPos[1], camPos[2]);
    gl.uniform3f(this.uniforms['u_cam_dir'], camDir[0], camDir[1], camDir[2]);
    gl.uniform3f(this.uniforms['u_cam_right'], camRight[0], camRight[1], camRight[2]);
    gl.uniform3f(this.uniforms['u_cam_up'], camUp[0], camUp[1], camUp[2]);
    gl.uniform1f(this.uniforms['u_fov_scale'], fovScale);

    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  private showShaderError(canvas: HTMLCanvasElement, message: string) {
    const errorDiv = document.createElement('div');
    errorDiv.style.position = 'absolute';
    errorDiv.style.top = '20px';
    errorDiv.style.right = '20px';
    errorDiv.style.left = '420px';
    errorDiv.style.bottom = '20px';
    errorDiv.style.background = 'rgba(255, 0, 50, 0.15)';
    errorDiv.style.border = '1px solid #ff3366';
    errorDiv.style.backdropFilter = 'blur(10px)';
    errorDiv.style.borderRadius = '16px';
    errorDiv.style.padding = '24px';
    errorDiv.style.color = '#ffccd7';
    errorDiv.style.fontFamily = 'monospace';
    errorDiv.style.fontSize = '12px';
    errorDiv.style.overflowY = 'auto';
    errorDiv.style.zIndex = '100';
    errorDiv.innerHTML = `<h3>WebGL Compilation Error</h3><pre style="white-space: pre-wrap; margin-top: 10px;">${message}</pre>`;
    canvas.parentElement?.appendChild(errorDiv);
  }

  private translateShaderToWebGL1(source: string, isVertex: boolean): string {
    let translated = source;
    translated = translated.replace(/#version\s+300\s+es/, '');
    
    if (isVertex) {
      translated = translated.replace(/\bin\s+vec2\s+position;/, 'attribute vec2 position;');
      translated = translated.replace(/\bout\s+vec2\s+v_uv;/, 'varying vec2 v_uv;');
    } else {
      translated = translated.replace(/\bin\s+vec2\s+v_uv;/, 'varying vec2 v_uv;');
      translated = translated.replace(/\bout\s+vec4\s+fragColor;/, '');
      translated = translated.replace(/\btexture\s*\(/g, 'texture2D(');
      
      // Translate raw output to Reinhard + Gamma for WebGL1 screen render
      translated = translated.replace('fragColor = vec4(color_out, 1.0);', `
        vec3 color_exposed = color_out * u_bloom_intensity;
        vec3 tone_mapped = color_exposed / (color_exposed + vec3(1.0));
        gl_FragColor = vec4(pow(tone_mapped, vec3(1.0 / 2.2)), 1.0);
      `);
      
      // Translate all other assignments to gl_FragColor
      translated = translated.replace(/\bfragColor\s*=/g, 'gl_FragColor =');
      
      translated = 'precision mediump float;\n' + translated.replace(/precision\s+highp\s+float;/, `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
      `);
    }
    return translated;
  }
}
