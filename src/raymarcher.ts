import { Settings } from './main.ts';
import { vertexShaderSource } from './shaders/raymarch.vert.ts';
import { fragmentShaderSource } from './shaders/raymarch.frag.ts';

export class Raymarcher {
  private gl: WebGL2RenderingContext;
  private program: WebGLProgram;
  private positionBuffer: WebGLBuffer;
  private noiseTexture: WebGLTexture;

  // Uniform locations cache
  private uniforms: { [name: string]: WebGLUniformLocation | null } = {};

  constructor(canvas: HTMLCanvasElement) {
    // Initialize WebGL2 context
    const gl = canvas.getContext('webgl2', { antialias: false, depth: false });
    if (!gl) {
      throw new Error('WebGL2 is not supported by your browser.');
    }
    this.gl = gl;

    // Create shaders and programs
    const vertexShader = this.compileShader(gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = this.compileShader(gl.FRAGMENT_SHADER, fragmentShaderSource);
    this.program = this.createProgram(vertexShader, fragmentShader);

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

    // Create noise texture for accretion disk structures
    this.noiseTexture = this.createNoiseTexture();

    // Cache uniform locations
    const uniformNames = [
      'u_resolution', 'u_time', 'u_mass', 'u_spin', 'u_relativity',
      'u_disk', 'u_grid', 'u_stars', 'u_disk_temp', 'u_max_steps',
      'u_noise_tex', 'u_cam_pos', 'u_cam_dir', 'u_cam_up', 'u_cam_right', 'u_fov_scale'
    ];
    for (const name of uniformNames) {
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }
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

    // Generate high frequency noise values
    const size = 256;
    const data = new Uint8Array(size * size * 4);
    for (let i = 0; i < size * size * 4; i += 4) {
      data[i] = Math.floor(Math.random() * 255);     // Red
      data[i + 1] = Math.floor(Math.random() * 255); // Green
      data[i + 2] = Math.floor(Math.random() * 255); // Blue
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
    const gl = this.gl;
    const canvas = gl.canvas as HTMLCanvasElement;

    // Resize viewport to match display dimensions
    const displayWidth = canvas.clientWidth;
    const displayHeight = canvas.clientHeight;
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth;
      canvas.height = displayHeight;
    }
    gl.viewport(0, 0, canvas.width, canvas.height);

    // Clear buffer
    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Use shader program
    gl.useProgram(this.program);

    // Bind full screen quad coordinates
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    const positionLoc = gl.getAttribLocation(this.program, 'position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 0, 0);

    // Bind noise texture unit
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.noiseTexture);
    gl.uniform1i(this.uniforms['u_noise_tex'], 0);

    // Set uniforms
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

    // Camera Vector Calculations
    // 3D position based on pitch/yaw angles and radius
    const cosPitch = Math.cos(settings.camPitch);
    const sinPitch = Math.sin(settings.camPitch);
    const cosYaw = Math.cos(settings.camYaw);
    const sinYaw = Math.sin(settings.camYaw);

    const camPos = [
      settings.camRadius * cosPitch * sinYaw,
      settings.camRadius * cosPitch * cosYaw,
      settings.camRadius * sinPitch
    ];

    // Camera look direction vector (points from camera directly to the black hole center)
    const rawDir = [-camPos[0], -camPos[1], -camPos[2]];
    const dirLen = Math.sqrt(rawDir[0]*rawDir[0] + rawDir[1]*rawDir[1] + rawDir[2]*rawDir[2]);
    const camDir = [rawDir[0] / dirLen, rawDir[1] / dirLen, rawDir[2] / dirLen];

    // Right vector (cross product of camDir and world up [0, 0, 1])
    const rawRight = [
      camDir[1] * 1.0 - camDir[2] * 0.0,
      camDir[2] * 0.0 - camDir[0] * 1.0,
      camDir[0] * 0.0 - camDir[1] * 0.0
    ];
    // Simplifying: rawRight = [camDir[1], -camDir[0], 0]
    const rightLen = Math.sqrt(rawRight[0]*rawRight[0] + rawRight[1]*rawRight[1] + rawRight[2]*rawRight[2]);
    const camRight = [rawRight[0] / rightLen, rawRight[1] / rightLen, rawRight[2] / rightLen];

    // Up vector (cross product of camRight and camDir)
    const camUp = [
      camRight[1] * camDir[2] - camRight[2] * camDir[1],
      camRight[2] * camDir[0] - camRight[0] * camDir[2],
      camRight[0] * camDir[1] - camRight[1] * camDir[0]
    ];

    // Scale uniform representing field of view (FOV angle of 60 degrees -> tan(30) = 0.577)
    const fovScale = 0.577;

    gl.uniform3f(this.uniforms['u_cam_pos'], camPos[0], camPos[1], camPos[2]);
    gl.uniform3f(this.uniforms['u_cam_dir'], camDir[0], camDir[1], camDir[2]);
    gl.uniform3f(this.uniforms['u_cam_right'], camRight[0], camRight[1], camRight[2]);
    gl.uniform3f(this.uniforms['u_cam_up'], camUp[0], camUp[1], camUp[2]);
    gl.uniform1f(this.uniforms['u_fov_scale'], fovScale);

    // Draw full screen quad
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}
