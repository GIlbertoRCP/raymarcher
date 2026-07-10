#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <cstring>
#include <string>
#include <algorithm>

// Include GLFW and CUDA interop headers
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "raymarch.h"


#pragma pack(push, 1)
struct BMPHeader {
    unsigned short bfType;
    unsigned int bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int bfOffBits;
};
struct BMPInfoHeader {
    unsigned int biSize;
    int biWidth;
    int biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int biCompression;
    unsigned int biSizeImage;
    int biXPelsPerMeter;
    int biYPelsPerMeter;
    unsigned int biClrUsed;
    unsigned int biClrImportant;
};
#pragma pack(pop)

// Saves the frame buffer to a top-down Windows BMP file
void save_bmp(const char* filename, float4* host_buffer, int width, int height) {
    BMPHeader header;
    BMPInfoHeader infoHeader;
    
    int rowSize = (width * 3 + 3) & ~3; // 4-byte row alignment
    int dataSize = rowSize * height;
    
    header.bfType = 0x4D42; // "BM"
    header.bfSize = sizeof(BMPHeader) + sizeof(BMPInfoHeader) + dataSize;
    header.bfReserved1 = 0;
    header.bfReserved2 = 0;
    header.bfOffBits = sizeof(BMPHeader) + sizeof(BMPInfoHeader);
    
    infoHeader.biSize = sizeof(BMPInfoHeader);
    infoHeader.biWidth = width;
    infoHeader.biHeight = -height; // negative height means top-down row layout
    infoHeader.biPlanes = 1;
    infoHeader.biBitCount = 24;
    infoHeader.biCompression = 0; // Uncompressed RGB
    infoHeader.biSizeImage = dataSize;
    infoHeader.biXPelsPerMeter = 2835; // 72 DPI
    infoHeader.biYPelsPerMeter = 2835;
    infoHeader.biClrUsed = 0;
    infoHeader.biClrImportant = 0;
    
    FILE* f = fopen(filename, "wb");
    if (!f) {
        std::cerr << "Error: Could not open BMP file " << filename << " for writing.\n";
        return;
    }
    
    fwrite(&header, sizeof(header), 1, f);
    fwrite(&infoHeader, sizeof(infoHeader), 1, f);
    
    unsigned char* rowBuffer = (unsigned char*)malloc(rowSize);
    for (int y = 0; y < height; ++y) {
        memset(rowBuffer, 0, rowSize);
        for (int x = 0; x < width; ++x) {
            float4 p = host_buffer[y * width + x];
            // BMP uses BGR color order, clamping components in [0.0, 1.0]
            rowBuffer[x * 3 + 0] = (unsigned char)(fmaxf(0.0f, fminf(p.z * 255.0f, 255.0f))); // Blue
            rowBuffer[x * 3 + 1] = (unsigned char)(fmaxf(0.0f, fminf(p.y * 255.0f, 255.0f))); // Green
            rowBuffer[x * 3 + 2] = (unsigned char)(fmaxf(0.0f, fminf(p.x * 255.0f, 255.0f))); // Red
        }
        fwrite(rowBuffer, rowSize, 1, f);
    }
    free(rowBuffer);
    fclose(f);
    std::cout << "Snapshot saved to " << filename << "\n";
}

// Define OpenGL 1.5/3.0 types and macros if not defined (e.g. standard Windows gl.h is OpenGL 1.1)
#ifndef GL_VERSION_1_5
typedef ptrdiff_t GLsizeiptr;
typedef ptrdiff_t GLintptr;
#endif

#ifndef GL_RGBA32F
#define GL_RGBA32F 0x8814
#endif

#define GL_PIXEL_UNPACK_BUFFER 0x88EC
#define GL_DYNAMIC_DRAW 0x88E8

// OpenGL Extension function prototypes
typedef void (APIENTRY * PFNGLGENBUFFERSPROC) (GLsizei n, GLuint *buffers);
typedef void (APIENTRY * PFNGLBINDBUFFERPROC) (GLenum target, GLuint buffer);
typedef void (APIENTRY * PFNGLBUFFERDATAPROC) (GLenum target, GLsizeiptr size, const void *data, GLenum usage);
typedef void (APIENTRY * PFNGLGETBUFFERSUBDATAPROC) (GLenum target, GLintptr offset, GLsizeiptr size, void *data);

static PFNGLGENBUFFERSPROC my_glGenBuffers = nullptr;
static PFNGLBINDBUFFERPROC my_glBindBuffer = nullptr;
static PFNGLBUFFERDATAPROC my_glBufferData = nullptr;
static PFNGLGETBUFFERSUBDATAPROC my_glGetBufferSubData = nullptr;

inline void loadGLExtensions() {
    my_glGenBuffers = (PFNGLGENBUFFERSPROC)glfwGetProcAddress("glGenBuffers");
    my_glBindBuffer = (PFNGLBINDBUFFERPROC)glfwGetProcAddress("glBindBuffer");
    my_glBufferData = (PFNGLBUFFERDATAPROC)glfwGetProcAddress("glBufferData");
    my_glGetBufferSubData = (PFNGLGETBUFFERSUBDATAPROC)glfwGetProcAddress("glGetBufferSubData");
}


// Global simulation states
static SimulationSettings settings;
static float camRadius = 12.0f;
static float camPitch = 0.15f; // slightly tilted down
static float camYaw = 0.0f;

static bool isDragging = false;
static double lastMouseX = 0.0;
static double lastMouseY = 0.0;
static bool isPaused = false;
static bool takeSnapshot = false;

// Compute camera vectors based on yaw, pitch, radius
void updateCamera() {
    float cosPitch = cosf(camPitch);
    float sinPitch = sinf(camPitch);
    float cosYaw = cosf(camYaw);
    float sinYaw = sinf(camYaw);
    
    settings.cam_pos = make_float3(
        camRadius * cosPitch * sinYaw,
        camRadius * cosPitch * cosYaw,
        camRadius * sinPitch
    );
    
    float3 rawDir = make_float3(-settings.cam_pos.x, -settings.cam_pos.y, -settings.cam_pos.z);
    float dirLen = sqrtf(rawDir.x*rawDir.x + rawDir.y*rawDir.y + rawDir.z*rawDir.z);
    settings.cam_dir = make_float3(rawDir.x / dirLen, rawDir.y / dirLen, rawDir.z / dirLen);
    
    float3 rawRight = make_float3(settings.cam_dir.y, -settings.cam_dir.x, 0.0f);
    float rightLen = sqrtf(rawRight.x*rawRight.x + rawRight.y*rawRight.y + rawRight.z*rawRight.z);
    settings.cam_right = make_float3(rawRight.x / rightLen, rawRight.y / rightLen, rawRight.z / rightLen);
    
    settings.cam_up = make_float3(
        settings.cam_right.y * settings.cam_dir.z - settings.cam_right.z * settings.cam_dir.y,
        settings.cam_right.z * settings.cam_dir.x - settings.cam_right.x * settings.cam_dir.z,
        settings.cam_right.x * settings.cam_dir.y - settings.cam_right.y * settings.cam_dir.x
    );
}

// GLFW mouse button callback
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        } else if (action == GLFW_RELEASE) {
            isDragging = false;
        }
    }
}

// GLFW cursor position callback
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;
    if (isDragging) {
        double dx = xpos - lastMouseX;
        double dy = ypos - lastMouseY;
        camYaw -= (float)dx * 0.005f;
        camPitch += (float)dy * 0.005f;
        
        // Clamp pitch to avoid singular gimbal lock flips
        if (camPitch > 1.4f) camPitch = 1.4f;
        if (camPitch < -1.4f) camPitch = -1.4f;
    }
    lastMouseX = xpos;
    lastMouseY = ypos;
}

// GLFW scroll callback (zooming)
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)window;
    (void)xoffset;
    camRadius -= (float)yoffset * 0.5f;
    if (camRadius < 4.0f) camRadius = 4.0f;
    if (camRadius > 30.0f) camRadius = 30.0f;
}

// GLFW keyboard callback
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)window;
    (void)scancode;
    (void)mods;
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (key == GLFW_KEY_R) {
            settings.relativity = !settings.relativity;
            std::cout << "Relativity: " << (settings.relativity ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_D) {
            settings.disk = !settings.disk;
            std::cout << "Accretion Disk: " << (settings.disk ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_G) {
            settings.grid = !settings.grid;
            if (settings.grid) settings.stars = 0;
            std::cout << "Calibration Grid: " << (settings.grid ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_S) {
            settings.stars = !settings.stars;
            if (settings.stars) settings.grid = 0;
            std::cout << "Starfield Background: " << (settings.stars ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_SPACE) {
            isPaused = !isPaused;
            std::cout << "Animation: " << (isPaused ? "PAUSED" : "PLAYING") << "\n";
        } else if (key == GLFW_KEY_P) {
            takeSnapshot = true;
        } else if (key == GLFW_KEY_UP) {
            settings.mass += 0.1f;
            std::cout << "Mass: " << settings.mass << "\n";
        } else if (key == GLFW_KEY_DOWN) {
            settings.mass = fmaxf(0.0f, settings.mass - 0.1f);
            std::cout << "Mass: " << settings.mass << "\n";
        } else if (key == GLFW_KEY_RIGHT) {
            settings.spin = fminf(0.99f, settings.spin + 0.05f);
            std::cout << "Spin parameter a: " << settings.spin << "\n";
        } else if (key == GLFW_KEY_LEFT) {
            settings.spin = fmaxf(0.0f, settings.spin - 0.05f);
            std::cout << "Spin parameter a: " << settings.spin << "\n";
        } else if (key == GLFW_KEY_PAGE_UP) {
            settings.max_steps = std::min(300, settings.max_steps + 10);
            std::cout << "Max Steps: " << settings.max_steps << "\n";
        } else if (key == GLFW_KEY_PAGE_DOWN) {
            settings.max_steps = std::max(10, settings.max_steps - 10);
            std::cout << "Max Steps: " << settings.max_steps << "\n";
        }
    }
}

int main(int argc, char** argv) {
    // 1. Initial configuration setup (Default Gargantua)
    settings.resolution = make_float2(1280.0f, 720.0f); // Render at virtual HD resolution
    settings.time = 0.0f;
    settings.mass = 1.80f;
    settings.spin = 0.60f;
    settings.relativity = 1;
    settings.disk = 1;
    settings.grid = 0;
    settings.stars = 1;
    settings.disk_temp = 6500.0f;
    settings.max_steps = 160;
    settings.fov_scale = 0.577f;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--resolution") == 0 && i + 2 < argc) {
            settings.resolution.x = (float)atof(argv[++i]);
            settings.resolution.y = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--mass") == 0 && i + 1 < argc) {
            settings.mass = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin") == 0 && i + 1 < argc) {
            settings.spin = (float)atof(argv[++i]);
        }
    }
    
    int width = (int)settings.resolution.x;
    int height = (int)settings.resolution.y;
    
    // 2. Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Error: Failed to initialize GLFW.\n";
        return -1;
    }
    
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    
    // Open window of width 1280x720 (resizable)
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Relativistic Black Hole CUDA Live Viewer", NULL, NULL);
    if (!window) {
        std::cerr << "Error: Failed to create GLFW window.\n";
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync
    
    // Register event callbacks
    glfwSetKeyCallback(window, key_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetScrollCallback(window, scroll_callback);
    
    // Load OpenGL entry points manually to avoid Glad/GLEW package requirements
    loadGLExtensions();
    if (!my_glGenBuffers || !my_glBindBuffer || !my_glBufferData) {
        std::cerr << "Error: Required OpenGL PBO extension functions could not be loaded.\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    
    // Print keyboard instructions
    std::cout << "=========================================================\n";
    std::cout << " Relativistic Black Hole Raymarcher - Live CUDA Desktop\n";
    std::cout << "=========================================================\n";
    std::cout << " Interactive Controls:\n";
    std::cout << " - Mouse Left Drag : Orbit camera angle rotation\n";
    std::cout << " - Mouse Scroll    : Zoom camera distance in / out\n";
    std::cout << " - [Space]         : Pause / Resume accretion disk spin animation\n";
    std::cout << " - [R]             : Toggle Relativistic Lensing vs Newtonian space\n";
    std::cout << " - [D]             : Toggle Accretion Disk visualization\n";
    std::cout << " - [G]             : Toggle Calibration grid background\n";
    std::cout << " - [S]             : Toggle Starfield background\n";
    std::cout << " - [UP]/[DOWN]     : Increase / Decrease Gravity Mass parameter\n";
    std::cout << " - [RIGHT]/[LEFT]  : Increase / Decrease Black hole spin parameter a\n";
    std::cout << " - [PGUP]/[PGDOWN] : Adjust Max step resolution quality\n";
    std::cout << " - [P]             : Take snapshot BMP image\n";
    std::cout << " - [ESC]           : Exit application\n";
    std::cout << "=========================================================\n";
    
    // 3. Set up OpenGL Texture to map rendering buffer
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindTexture(GL_TEXTURE_2D, 0);
    
    // 4. Set up OpenGL Pixel Buffer Object (PBO)
    GLuint pbo;
    my_glGenBuffers(1, &pbo);
    my_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    my_glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float4), NULL, GL_DYNAMIC_DRAW);
    my_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    // 5. Register PBO with CUDA Graphics Interop
    cudaGraphicsResource* cuda_pbo_resource = nullptr;
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: Failed to register PBO with CUDA: " << cudaGetErrorString(err) << "\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }
    
    // 6. Generate noise texture on CPU and copy to GPU CUDA Array
    const int noise_size = 256;
    unsigned char* noise_data = new unsigned char[noise_size * noise_size * 4];
    srand((unsigned int)std::time(nullptr));
    for (int i = 0; i < noise_size * noise_size * 4; i += 4) {
        noise_data[i]     = rand() % 256; // Red
        noise_data[i + 1] = rand() % 256; // Green
        noise_data[i + 2] = rand() % 256; // Blue
        noise_data[i + 3] = 255;
    }
    
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, noise_size, noise_size);
    cudaMemcpy2DToArray(cuArray, 0, 0, noise_data, noise_size * 4 * sizeof(unsigned char),
                              noise_size * 4 * sizeof(unsigned char), noise_size, cudaMemcpyHostToDevice);
    
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    
    cudaTextureObject_t noiseTex = 0;
    cudaCreateTextureObject(&noiseTex, &resDesc, &texDesc, NULL);
    
    // FPS accounting variables
    double lastFPSTime = glfwGetTime();
    int framesCounted = 0;
    double lastFrameTime = glfwGetTime();
    
    // 7. Interactive Render Loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        
        // Advance time if animation is not paused
        double currentFrameTime = glfwGetTime();
        double dt = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;
        
        if (!isPaused) {
            settings.time += (float)dt * 1.5f; // match speed in web simulator
        }
        
        // Update camera position vectors from orbit angles
        updateCamera();
        
        // Map OpenGL Pixel Buffer Object to CUDA memory space
        float4* d_output = nullptr;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cuda_pbo_resource);
        
        // Launch parallel CUDA raymarch kernel
        run_raymarch_kernel(d_output, settings, noiseTex);
        
        // Unmap the interop resource
        cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
        
        // Stream PBO content into the OpenGL texture
        my_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_FLOAT, NULL);
        my_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        
        // Handle snapshot key press
        if (takeSnapshot) {
            takeSnapshot = false;
            float4* hostBuffer = new float4[width * height];
            my_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            // Copy pixel data directly from device PBO memory back to CPU
            my_glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, width * height * sizeof(float4), hostBuffer);
            my_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            
            std::string snapName = "blackhole_snap_" + std::to_string((unsigned int)std::time(nullptr)) + ".bmp";
            save_bmp(snapName.c_str(), hostBuffer, width, height);
            delete[] hostBuffer;
        }
        
        // Get window dimensions to dynamically fit the viewport
        int winWidth, winHeight;
        glfwGetWindowSize(window, &winWidth, &winHeight);
        glViewport(0, 0, winWidth, winHeight);
        
        // Draw fullscreen textured quad representing the raymarch frame
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, tex);
        glBegin(GL_QUADS);
          glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f, -1.0f);
          glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f, -1.0f);
          glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f,  1.0f);
          glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f,  1.0f);
        glEnd();
        glDisable(GL_TEXTURE_2D);
        
        glfwSwapBuffers(window);
        
        // Measure real-time FPS
        framesCounted++;
        if (currentFrameTime - lastFPSTime >= 1.0) {
            double fpsVal = (double)framesCounted / (currentFrameTime - lastFPSTime);
            std::string title = "Relativistic Black Hole CUDA Live Viewer (" + std::to_string((int)fpsVal) + " FPS)";
            glfwSetWindowTitle(window, title.c_str());
            framesCounted = 0;
            lastFPSTime = currentFrameTime;
        }
    }
    
    // 8. Resource cleanup
    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    cudaDestroyTextureObject(noiseTex);
    cudaFreeArray(cuArray);
    delete[] noise_data;
    
    // Free OpenGL objects
    glDeleteTextures(1, &tex);
    // Since glDeleteBuffers is OpenGL 1.5, we can fetch it to delete the PBO
    typedef void (APIENTRY * PFNGLDELETEBUFFERSPROC) (GLsizei n, const GLuint *buffers);
    PFNGLDELETEBUFFERSPROC my_glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glfwGetProcAddress("glDeleteBuffers");
    if (my_glDeleteBuffers) {
        my_glDeleteBuffers(1, &pbo);
    }
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
