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

// Include Dear ImGui headers
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

// Custom Glassmorphic Dark Theme for Dear ImGui
void setup_imgui_style() {
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 10.0f;
    style.FrameRounding = 6.0f;
    style.GrabRounding = 6.0f;
    style.ScrollbarRounding = 6.0f;
    style.ChildRounding = 8.0f;
    style.PopupRounding = 8.0f;
    style.WindowPadding = ImVec2(14.0f, 14.0f);
    style.ItemSpacing = ImVec2(8.0f, 8.0f);

    ImVec4* colors = style.Colors;
    colors[ImGuiCol_WindowBg]           = ImVec4(0.04f, 0.05f, 0.09f, 0.82f); // Glass translucent
    colors[ImGuiCol_Header]             = ImVec4(0.12f, 0.20f, 0.32f, 0.80f);
    colors[ImGuiCol_HeaderHovered]      = ImVec4(0.18f, 0.35f, 0.55f, 0.90f);
    colors[ImGuiCol_HeaderActive]       = ImVec4(0.00f, 0.70f, 0.90f, 1.00f);
    colors[ImGuiCol_Button]             = ImVec4(0.10f, 0.16f, 0.26f, 0.85f);
    colors[ImGuiCol_ButtonHovered]      = ImVec4(0.00f, 0.65f, 0.85f, 0.90f);
    colors[ImGuiCol_ButtonActive]       = ImVec4(0.00f, 0.85f, 1.00f, 1.00f);
    colors[ImGuiCol_FrameBg]            = ImVec4(0.07f, 0.09f, 0.15f, 0.80f);
    colors[ImGuiCol_FrameBgHovered]     = ImVec4(0.14f, 0.18f, 0.28f, 0.85f);
    colors[ImGuiCol_FrameBgActive]      = ImVec4(0.20f, 0.26f, 0.40f, 0.90f);
    colors[ImGuiCol_SliderGrab]         = ImVec4(0.00f, 0.85f, 1.00f, 1.00f);
    colors[ImGuiCol_SliderGrabActive]   = ImVec4(0.30f, 0.92f, 1.00f, 1.00f);
    colors[ImGuiCol_CheckMark]          = ImVec4(0.00f, 0.90f, 1.00f, 1.00f);
    colors[ImGuiCol_Text]               = ImVec4(0.95f, 0.96f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled]       = ImVec4(0.50f, 0.55f, 0.65f, 1.00f);
    colors[ImGuiCol_TitleBg]            = ImVec4(0.04f, 0.04f, 0.07f, 0.95f);
    colors[ImGuiCol_TitleBgActive]      = ImVec4(0.08f, 0.12f, 0.20f, 0.95f);
    colors[ImGuiCol_Separator]          = ImVec4(0.20f, 0.25f, 0.35f, 0.60f);
}


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

// CUDA Real-Time Profiler & Metrics Overlay State
static float latencyHistory[120] = { 0.0f };
static int latencyIndex = 0;
static unsigned long long* d_live_step_counter = nullptr;
static cudaEvent_t liveStart, liveStop;

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
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) return;
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
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) return;
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
    if (ImGui::GetCurrentContext() && ImGui::GetIO().WantCaptureMouse) return;
    camRadius -= (float)yoffset * 0.5f;
    if (camRadius < 4.0f) camRadius = 4.0f;
    if (camRadius > 30.0f) camRadius = 30.0f;
}

static bool triggerBenchmark = false;

// Automated Benchmark Suite across 1080p, 1440p, 4K and Workgroup Block Configurations
void run_benchmark_suite(SimulationSettings currentSettings, cudaTextureObject_t noiseTex) {
    std::cout << "\n=========================================================================================================\n";
    std::cout << "                   CUDA RAYMARCHER AUTOMATED PARALLEL COMPUTING BENCHMARK SUITE                          \n";
    std::cout << "=========================================================================================================\n";
    std::cout << " Config: Max Steps = " << currentSettings.max_steps << " | Mass = " << currentSettings.mass
              << " | Spin = " << currentSettings.spin << " | Relativity = " << (currentSettings.relativity ? "ON" : "OFF") << "\n";
    std::cout << "---------------------------------------------------------------------------------------------------------\n";
    std::cout << " Res   | Block Dim | Spatial Accel | Frame Time (ms) |    FPS    | Ray Steps / Px | Throughput (GB/s)\n";
    std::cout << "-------+-----------+---------------+-----------------+-----------+----------------+------------------\n";

    struct ResConfig { const char* name; int w; int h; };
    ResConfig configs[] = {
        { "1080p", 1920, 1080 },
        { "1440p", 2560, 1440 },
        { "4K",    3840, 2160 }
    };

    struct BlockConfig { int x; int y; const char* name; };
    BlockConfig blocks[] = {
        { 16, 16, "16x16" },
        { 32,  8, "32x8"  },
        { 32, 32, "32x32" },
        {  8,  8, "8x8"   }
    };

    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    unsigned long long* d_step_counter = nullptr;
    cudaMalloc(&d_step_counter, sizeof(unsigned long long));

    FILE* jsonFile = fopen("benchmark_report.json", "w");
    if (jsonFile) fprintf(jsonFile, "[\n");

    int testIndex = 0;
    int totalTests = 3 * 4;

    for (int i = 0; i < 3; ++i) {
        int w = configs[i].w;
        int h = configs[i].h;
        size_t bufferSize = (size_t)w * h * sizeof(float4);

        float4* d_output = nullptr;
        cudaMalloc(&d_output, bufferSize);

        for (int b = 0; b < 4; ++b) {
            testIndex++;
            SimulationSettings bSettings = currentSettings;
            bSettings.resolution = make_float2((float)w, (float)h);
            bSettings.block_dim_x = blocks[b].x;
            bSettings.block_dim_y = blocks[b].y;
            bSettings.spatial_accel = 1;

            // Warmup passes
            for (int warmup = 0; warmup < 3; ++warmup) {
                run_raymarch_kernel(d_output, nullptr, bSettings, noiseTex);
            }

            cudaMemset(d_step_counter, 0, sizeof(unsigned long long));

            const int num_frames = 30;
            cudaEventRecord(startEvent);
            for (int f = 0; f < num_frames; ++f) {
                bSettings.time += 0.016f;
                run_raymarch_kernel(d_output, d_step_counter, bSettings, noiseTex);
            }
            cudaEventRecord(stopEvent);
            cudaEventSynchronize(stopEvent);

            float total_ms = 0.0f;
            cudaEventElapsedTime(&total_ms, startEvent, stopEvent);

            unsigned long long h_step_counter = 0;
            cudaMemcpy(&h_step_counter, d_step_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

            float avg_frame_time = total_ms / (float)num_frames;
            float fps = 1000.0f / fmaxf(avg_frame_time, 0.001f);
            double total_pixels = (double)w * (double)h;
            double avg_steps_per_px = (double)h_step_counter / (total_pixels * (double)num_frames);

            double bytes_per_frame = (double)bufferSize + (double)h_step_counter / (double)num_frames * 16.0;
            double throughput_gbps = (bytes_per_frame / (avg_frame_time * 1e-3)) / 1e9;

            printf(" %-5s | %-9s | %-13s | %15.3f | %9.1f | %14.2f | %16.2f\n",
                   configs[i].name, blocks[b].name, "ENABLED", avg_frame_time, fps, avg_steps_per_px, throughput_gbps);

            if (jsonFile) {
                fprintf(jsonFile, "  {\n");
                fprintf(jsonFile, "    \"resolution\": \"%s\",\n", configs[i].name);
                fprintf(jsonFile, "    \"width\": %d,\n", w);
                fprintf(jsonFile, "    \"height\": %d,\n", h);
                fprintf(jsonFile, "    \"block_dim_x\": %d,\n", blocks[b].x);
                fprintf(jsonFile, "    \"block_dim_y\": %d,\n", blocks[b].y);
                fprintf(jsonFile, "    \"spatial_accel\": 1,\n");
                fprintf(jsonFile, "    \"avg_frame_time_ms\": %.3f,\n", avg_frame_time);
                fprintf(jsonFile, "    \"fps\": %.2f,\n", fps);
                fprintf(jsonFile, "    \"avg_steps_per_pixel\": %.2f,\n", avg_steps_per_px);
                fprintf(jsonFile, "    \"memory_throughput_gbps\": %.2f\n", throughput_gbps);
                fprintf(jsonFile, "  }%s\n", (testIndex == totalTests ? "" : ","));
            }
        }

        cudaFree(d_output);
    }

    if (jsonFile) {
        fprintf(jsonFile, "]\n");
        fclose(jsonFile);
        std::cout << "\n[Benchmark] Performance report exported cleanly to benchmark_report.json\n";
    }

    cudaFree(d_step_counter);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    std::cout << "=========================================================================================================\n\n";
}

// GLFW keyboard callback
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)window;
    (void)scancode;
    (void)mods;
    if (action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        } else if (key == GLFW_KEY_B) {
            triggerBenchmark = true;
        } else if (key == GLFW_KEY_R) {
            settings.relativity = !settings.relativity;
            std::cout << "Relativity: " << (settings.relativity ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_D) {
            settings.disk = !settings.disk;
            std::cout << "Accretion Disk: " << (settings.disk ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_M) {
            settings.csg_mode = (settings.csg_mode + 1) % 5;
            const char* csg_names[] = { "Disabled", "Union", "Smooth Min", "Subtraction", "Intersection" };
            std::cout << "CSG Mode: " << csg_names[settings.csg_mode] << "\n";
        } else if (key == GLFW_KEY_N) {
            settings.soft_shadows = !settings.soft_shadows;
            std::cout << "Soft Shadows: " << (settings.soft_shadows ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_A) {
            settings.ao_enabled = !settings.ao_enabled;
            std::cout << "SDF Ambient Occlusion: " << (settings.ao_enabled ? "ENABLED" : "DISABLED") << "\n";
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
        } else if (key == GLFW_KEY_C) {
            settings.spectrum_mode = (settings.spectrum_mode + 1) % 4;
            std::cout << "Spectrum Mode: " << settings.spectrum_mode << "\n";
        } else if (key == GLFW_KEY_V) {
            settings.shift_visualizer = !settings.shift_visualizer;
            std::cout << "Redshift Visualizer: " << (settings.shift_visualizer ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_X) {
            settings.split_active = !settings.split_active;
            std::cout << "Split Screen Comparison: " << (settings.split_active ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_O) {
            settings.show_photon_sphere = !settings.show_photon_sphere;
            std::cout << "Photon Sphere Shell: " << (settings.show_photon_sphere ? "ENABLED" : "DISABLED") << "\n";
        } else if (key == GLFW_KEY_I) {
            settings.disk_density = fminf(5.00f, settings.disk_density + 0.05f);
            std::cout << "Disk Density: " << settings.disk_density << "\n";
        } else if (key == GLFW_KEY_K) {
            settings.disk_density = fmaxf(0.10f, settings.disk_density - 0.05f);
            std::cout << "Disk Density: " << settings.disk_density << "\n";
        } else if (key == GLFW_KEY_LEFT_BRACKET) {
            settings.split_x = fmaxf(0.0f, settings.split_x - 0.02f);
            std::cout << "Split Line X: " << settings.split_x << "\n";
        } else if (key == GLFW_KEY_RIGHT_BRACKET) {
            settings.split_x = fminf(1.0f, settings.split_x + 0.02f);
            std::cout << "Split Line X: " << settings.split_x << "\n";
        }
    }
}

int main(int argc, char** argv) {
    // 1. Initial configuration setup (Safe minimal mode)
    settings.resolution = make_float2(960.0f, 540.0f); // Render at safe minimal 540p resolution
    settings.time = 0.0f;
    settings.mass = 1.80f;
    settings.spin = 0.60f;
    settings.relativity = 1;
    settings.disk = 1;
    settings.grid = 0;
    settings.stars = 1;
    settings.disk_temp = 6500.0f;
    settings.max_steps = 60; // Minimal step count to prevent GPU strain
    settings.disk_thickness = 0.08f;
    settings.disk_density = 1.00f;
    settings.bloom_intensity = 1.00f;
    settings.disk_speed = 1.00f;
    settings.show_photon_sphere = 0;
    settings.shift_visualizer = 0;
    settings.split_active = 0;
    settings.split_x = 0.5f;
    settings.spectrum_mode = 0;
    settings.fov_scale = 0.577f;
    
    // CSG & Lighting Defaults
    settings.csg_mode = 0;
    settings.csg_blend = 0.5f;
    settings.soft_shadows = 0;
    settings.shadow_k = 16.0f;
    settings.ao_enabled = 0;
    settings.ao_intensity = 1.0f;

    // GW, TAA & Polarization Defaults
    settings.gw_enabled = 0;
    settings.gw_amplitude = 0.05f;
    settings.gw_frequency = 1.20f;
    settings.polarization_mode = 0;
    settings.taa_enabled = 0;
    settings.taa_blend = 0.15f;

    // CUDA Optimization & Diagnostics Defaults
    settings.block_dim_x = 32;
    settings.block_dim_y = 8;
    settings.spatial_accel = 1;
    settings.heatmap_mode = 0;

    bool runBenchmarkOnly = false;
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "--benchmark") == 0 || strcmp(argv[i], "-b") == 0)) {
            runBenchmarkOnly = true;
        } else if (strcmp(argv[i], "--resolution") == 0 && i + 2 < argc) {
            settings.resolution.x = (float)atof(argv[++i]);
            settings.resolution.y = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--mass") == 0 && i + 1 < argc) {
            settings.mass = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin") == 0 && i + 1 < argc) {
            settings.spin = (float)atof(argv[++i]);
        }
    }

    // Helper setup noise texture for headless or interactive mode
    const int noise_size = 256;
    unsigned char* noise_data = new unsigned char[noise_size * noise_size * 4];
    srand((unsigned int)std::time(nullptr));
    for (int i = 0; i < noise_size * noise_size * 4; i += 4) {
        noise_data[i]     = rand() % 256;
        noise_data[i + 1] = rand() % 256;
        noise_data[i + 2] = rand() % 256;
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

    if (runBenchmarkOnly) {
        run_benchmark_suite(settings, noiseTex);
        cudaDestroyTextureObject(noiseTex);
        cudaFreeArray(cuArray);
        delete[] noise_data;
        return 0;
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

    // Initialize Dear ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    setup_imgui_style();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Initialize CUDA events & step counters for real-time profiling overlay
    cudaEventCreate(&liveStart);
    cudaEventCreate(&liveStop);
    cudaMalloc((void**)&d_live_step_counter, sizeof(unsigned long long));

    // Allocate VRAM history buffer for Temporal Anti-Aliasing (TAA)
    float4* d_history_buffer = nullptr;
    cudaMalloc((void**)&d_history_buffer, width * height * sizeof(float4));
    cudaMemset(d_history_buffer, 0, width * height * sizeof(float4));

    // Multi-GPU Device & Topology Detection
    int deviceCount = 0;
    std::string gpuDeviceName = "NVIDIA GPU Target";
    int gpuComputeCapability = 75;
    size_t gpuVRAMTotalMB = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        gpuDeviceName = prop.name;
        gpuComputeCapability = prop.major * 10 + prop.minor;
        gpuVRAMTotalMB = prop.totalGlobalMem / (1024 * 1024);
        std::cout << "[CUDA Hardware Engine] Detected " << deviceCount << " GPU Device(s). Primary: "
                  << gpuDeviceName << " (sm_" << gpuComputeCapability << ", " << gpuVRAMTotalMB << " MB VRAM)\n";
    }
    
    // Print keyboard instructions
    std::cout << "=========================================================\n";
    std::cout << " Relativistic Black Hole Raymarcher - Live CUDA Desktop\n";
    std::cout << "=========================================================\n";
    std::cout << " Interactive Controls (GUI Overlay + Shortcuts):\n";
    std::cout << " - Mouse Left Drag : Orbit camera angle rotation\n";
    std::cout << " - Mouse Scroll    : Zoom camera distance in / out\n";
    std::cout << " - [Space]         : Pause / Resume accretion disk spin animation\n";
    std::cout << " - [B]             : Run Automated Performance Benchmark Suite (1080p, 1440p, 4K)\n";
    std::cout << " - [M]             : Cycle CSG Mode (None, Union, Smooth Min, Subtraction, Intersection)\n";
    std::cout << " - [N]             : Toggle Soft Shadows (Cone Tracing)\n";
    std::cout << " - [A]             : Toggle SDF Ambient Occlusion\n";
    std::cout << " - [R]             : Toggle Relativistic Lensing vs Newtonian space\n";
    std::cout << " - [D]             : Toggle Accretion Disk visualization\n";
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
    
    // FPS accounting variables
    double lastFPSTime = glfwGetTime();
    int framesCounted = 0;
    double lastFrameTime = glfwGetTime();
    
    // 7. Interactive Render Loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        if (triggerBenchmark) {
            triggerBenchmark = false;
            run_benchmark_suite(settings, noiseTex);
        }
        
        // Advance time if animation is not paused
        double currentFrameTime = glfwGetTime();
        double dt = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;
        
        if (!isPaused) {
            settings.time += (float)dt * 1.5f; // match speed in web simulator
        }
        
        // Update camera position vectors from orbit angles
        updateCamera();

        // Check for window size changes and dynamically update render buffers
        int winWidth, winHeight;
        glfwGetFramebufferSize(window, &winWidth, &winHeight);
        winWidth = std::max(256, winWidth);
        winHeight = std::max(256, winHeight);

        if (winWidth != width || winHeight != height) {
            width = winWidth;
            height = winHeight;
            settings.resolution = make_float2((float)width, (float)height);

            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
            glBindTexture(GL_TEXTURE_2D, 0);

            if (cuda_pbo_resource) {
                cudaGraphicsUnregisterResource(cuda_pbo_resource);
                cuda_pbo_resource = nullptr;
            }

            my_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
            my_glBufferData(GL_PIXEL_UNPACK_BUFFER, width * height * sizeof(float4), NULL, GL_DYNAMIC_DRAW);
            my_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsRegisterFlagsWriteDiscard);

            if (d_history_buffer) {
                cudaFree(d_history_buffer);
                cudaMalloc((void**)&d_history_buffer, width * height * sizeof(float4));
                cudaMemset(d_history_buffer, 0, width * height * sizeof(float4));
            }
        }
        
        // Map OpenGL Pixel Buffer Object to CUDA memory space
        float4* d_output = nullptr;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_output, &num_bytes, cuda_pbo_resource);
        
        // Launch parallel CUDA raymarch kernel with precise execution timing
        cudaMemset(d_live_step_counter, 0, sizeof(unsigned long long));
        cudaEventRecord(liveStart);
        run_raymarch_kernel(d_output, d_live_step_counter, settings, noiseTex);

        if (settings.taa_enabled && d_history_buffer) {
            apply_temporal_reprojection(d_output, d_history_buffer, width, height, settings.taa_blend);
        }

        cudaEventRecord(liveStop);
        cudaEventSynchronize(liveStop);

        float kernelTimeMs = 0.0f;
        cudaEventElapsedTime(&kernelTimeMs, liveStart, liveStop);

        unsigned long long hostRaySteps = 0;
        cudaMemcpy(&hostRaySteps, d_live_step_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

        size_t freeMem = 0, totalMem = 0;
        cudaMemGetInfo(&freeMem, &totalMem);
        double usedVRAM_MB = (double)(totalMem - freeMem) / (1024.0 * 1024.0);
        double totalVRAM_MB = (double)totalMem / (1024.0 * 1024.0);

        double totalPixels = (double)width * (double)height;
        double mRaysPerSec = (kernelTimeMs > 0.0001f) ? (totalPixels * 1000.0 / kernelTimeMs) / 1e6 : 0.0;
        double mStepsPerSec = (kernelTimeMs > 0.0001f) ? ((double)hostRaySteps * 1000.0 / kernelTimeMs) / 1e6 : 0.0;

        // Push kernel latency to rolling history buffer
        latencyHistory[latencyIndex] = kernelTimeMs;
        latencyIndex = (latencyIndex + 1) % 120;
        
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
            my_glGetBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, width * height * sizeof(float4), hostBuffer);
            my_glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
            
            std::string snapName = "blackhole_snap_" + std::to_string((unsigned int)std::time(nullptr)) + ".bmp";
            save_bmp(snapName.c_str(), hostBuffer, width, height);
            delete[] hostBuffer;
        }
        
        // Set glViewport to fit current window size
        glViewport(0, 0, width, height);
        
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

        // Render ImGui Control Panel Overlay
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(410, 720), ImGuiCond_FirstUseEver);

        if (ImGui::Begin("Event Horizon - CUDA Controls", nullptr, ImGuiWindowFlags_AlwaysUseWindowPadding)) {
            ImGui::TextColored(ImVec4(0.0f, 0.9f, 1.0f, 1.0f), "NVIDIA CUDA Raymarcher v2.0");
            ImGui::Separator();

            // Real-Time CUDA Profiler & Telemetry Panel
            if (ImGui::CollapsingHeader("Real-Time CUDA Profiler & Metrics", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("GPU Device: %s (sm_%d)", gpuDeviceName.c_str(), gpuComputeCapability);
                float fpsVal = (framesCounted > 0) ? (float)framesCounted / (float)(currentFrameTime - lastFPSTime + 1e-4) : (1000.0f / fmaxf(kernelTimeMs, 0.001f));
                ImGui::Text("Viewport FPS: %.1f FPS", fpsVal);
                ImGui::Text("CUDA Kernel Latency: %.3f ms", kernelTimeMs);
                ImGui::PlotLines("Latency Graph", latencyHistory, 120, latencyIndex, "0..5ms", 0.0f, 5.0f, ImVec2(0, 40));

                ImGui::Text("GPU VRAM Usage: %.1f MB / %.1f MB", usedVRAM_MB, totalVRAM_MB);
                float vramFraction = (float)(usedVRAM_MB / fmax(totalVRAM_MB, 1.0));
                ImGui::ProgressBar(vramFraction, ImVec2(-1, 14));

                ImGui::Text("Rays Throughput: %.2f Million Rays/sec", mRaysPerSec);
                ImGui::Text("Step Throughput: %.2f Million Steps/sec", mStepsPerSec);
                ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.4f, 1.0f), "[NVIDIA CUDA Cores Active]");
            }
            ImGui::Separator();

            // Thread Block & Hardware Optimization Tuning Panel
            if (ImGui::CollapsingHeader("CUDA Thread Block & Acceleration Tuning", ImGuiTreeNodeFlags_DefaultOpen)) {
                int current_block_cfg = 1;
                if (settings.block_dim_x == 16 && settings.block_dim_y == 16) current_block_cfg = 0;
                else if (settings.block_dim_x == 32 && settings.block_dim_y == 8) current_block_cfg = 1;
                else if (settings.block_dim_x == 32 && settings.block_dim_y == 32) current_block_cfg = 2;
                else if (settings.block_dim_x == 8 && settings.block_dim_y == 8) current_block_cfg = 3;

                const char* block_names[] = { "16x16 (256 threads)", "32x8 (256 threads - Warp Aligned)", "32x32 (1024 threads - Max)", "8x8 (64 threads)" };
                if (ImGui::Combo("Thread Block Size", &current_block_cfg, block_names, IM_ARRAYSIZE(block_names))) {
                    if (current_block_cfg == 0) { settings.block_dim_x = 16; settings.block_dim_y = 16; }
                    else if (current_block_cfg == 1) { settings.block_dim_x = 32; settings.block_dim_y = 8; }
                    else if (current_block_cfg == 2) { settings.block_dim_x = 32; settings.block_dim_y = 32; }
                    else if (current_block_cfg == 3) { settings.block_dim_x = 8; settings.block_dim_y = 8; }
                }

                bool spatial_accel_bool = settings.spatial_accel != 0;
                if (ImGui::Checkbox("Spatial Bounding Acceleration", &spatial_accel_bool)) {
                    settings.spatial_accel = spatial_accel_bool ? 1 : 0;
                }

                bool heatmap_bool = settings.heatmap_mode != 0;
                if (ImGui::Checkbox("Ray-Step Heatmap Visualizer", &heatmap_bool)) {
                    settings.heatmap_mode = heatmap_bool ? 1 : 0;
                }
            }
            ImGui::Separator();

            // Presets
            ImGui::Text("Presets:");
            if (ImGui::Button("Gargantua")) {
                settings.mass = 1.80f; settings.spin = 0.60f; settings.relativity = 1; settings.disk = 1;
                settings.disk_temp = 6500.0f; settings.disk_thickness = 0.08f; settings.disk_density = 1.0f; settings.spectrum_mode = 0;
            }
            ImGui::SameLine();
            if (ImGui::Button("Micro-BH")) {
                settings.mass = 3.50f; settings.spin = 0.00f; settings.relativity = 1; settings.disk = 1;
                settings.disk_temp = 11000.0f; settings.disk_thickness = 0.04f; settings.disk_density = 3.0f; settings.spectrum_mode = 1;
            }
            ImGui::SameLine();
            if (ImGui::Button("Kerr")) {
                settings.mass = 1.50f; settings.spin = 0.98f; settings.relativity = 1; settings.disk = 1;
                settings.disk_temp = 9500.0f; settings.disk_thickness = 0.12f; settings.disk_density = 1.2f; settings.spectrum_mode = 2;
            }
            ImGui::SameLine();
            if (ImGui::Button("Newtonian")) {
                settings.mass = 0.0f; settings.spin = 0.0f; settings.relativity = 0; settings.disk = 1;
                settings.disk_temp = 6000.0f; settings.spectrum_mode = 0;
            }

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Spacetime & Render Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::SliderFloat("Mass (M)", &settings.mass, 0.1f, 4.0f, "%.2f");
                ImGui::SliderFloat("Spin (a/M)", &settings.spin, 0.0f, 0.99f, "%.2f");
                ImGui::SliderInt("Quality (Max Steps)", &settings.max_steps, 30, 300);
                ImGui::SliderFloat("Disk Thickness", &settings.disk_thickness, 0.01f, 0.50f, "%.2f");
                ImGui::SliderFloat("Disk Density", &settings.disk_density, 0.10f, 5.00f, "%.2f");
                ImGui::SliderFloat("Disk Temp (K)", &settings.disk_temp, 1000.0f, 15000.0f, "%.0f");
                ImGui::SliderFloat("Disk Speed", &settings.disk_speed, 0.00f, 2.00f, "%.2f");
                ImGui::SliderFloat("Glow Intensity", &settings.bloom_intensity, 0.00f, 2.00f, "%.2f");

                const char* spectrumItems[] = { "Planckian (Thermal)", "Nebula Glow (Violet)", "Plasma Fire (Orange)", "Quantum Shift (Cyan)" };
                ImGui::Combo("Color Spectrum", &settings.spectrum_mode, spectrumItems, IM_ARRAYSIZE(spectrumItems));
            }

            if (ImGui::CollapsingHeader("Gravitational Wave Perturbations (h_ij)")) {
                bool gw = (settings.gw_enabled != 0);
                if (ImGui::Checkbox("Gravitational Waves", &gw)) settings.gw_enabled = gw ? 1 : 0;
                ImGui::SliderFloat("Wave Strain (Amplitude)", &settings.gw_amplitude, 0.00f, 0.20f, "%.3f");
                ImGui::SliderFloat("Wave Frequency (Hz)", &settings.gw_frequency, 0.10f, 5.00f, "%.2f");
            }

            if (ImGui::CollapsingHeader("Adaptive Temporal Sampling (TAA)")) {
                bool taa = (settings.taa_enabled != 0);
                if (ImGui::Checkbox("Temporal Anti-Aliasing", &taa)) settings.taa_enabled = taa ? 1 : 0;
                ImGui::SliderFloat("Blend Alpha", &settings.taa_blend, 0.05f, 0.50f, "%.2f");
            }

            if (ImGui::CollapsingHeader("CSG Geometry & Mechanics")) {
                const char* csgItems[] = { "Disabled", "Union", "Smooth Min", "Subtraction", "Intersection" };
                ImGui::Combo("CSG Operation", &settings.csg_mode, csgItems, IM_ARRAYSIZE(csgItems));
                ImGui::SliderFloat("Smooth Blend (k)", &settings.csg_blend, 0.05f, 1.50f, "%.2f");
            }

            if (ImGui::CollapsingHeader("Lighting & Shadows")) {
                bool shadows = (settings.soft_shadows != 0);
                if (ImGui::Checkbox("Soft Shadows (Cone Tracing)", &shadows)) settings.soft_shadows = shadows ? 1 : 0;
                ImGui::SliderFloat("Penumbra Softness", &settings.shadow_k, 4.0f, 32.0f, "%.1f");

                bool ao = (settings.ao_enabled != 0);
                if (ImGui::Checkbox("SDF Ambient Occlusion", &ao)) settings.ao_enabled = ao ? 1 : 0;
                ImGui::SliderFloat("AO Intensity", &settings.ao_intensity, 0.20f, 2.50f, "%.2f");
            }

            if (ImGui::CollapsingHeader("Visual Elements")) {
                bool rel = (settings.relativity != 0);
                if (ImGui::Checkbox("Relativistic Lensing", &rel)) settings.relativity = rel ? 1 : 0;

                bool dsk = (settings.disk != 0);
                if (ImGui::Checkbox("Accretion Disk", &dsk)) settings.disk = dsk ? 1 : 0;

                bool pol = (settings.polarization_mode != 0);
                if (ImGui::Checkbox("Synchrotron Polarization Vector Overlay", &pol)) settings.polarization_mode = pol ? 1 : 0;

                bool str = (settings.stars != 0);
                if (ImGui::Checkbox("Starfield Background", &str)) { settings.stars = str ? 1 : 0; if (str) settings.grid = 0; }

                bool grd = (settings.grid != 0);
                if (ImGui::Checkbox("Calibration Grid", &grd)) { settings.grid = grd ? 1 : 0; if (grd) settings.stars = 0; }

                bool ps = (settings.show_photon_sphere != 0);
                if (ImGui::Checkbox("Photon Sphere Shell", &ps)) settings.show_photon_sphere = ps ? 1 : 0;

                bool shift = (settings.shift_visualizer != 0);
                if (ImGui::Checkbox("Redshift Visualizer", &shift)) settings.shift_visualizer = shift ? 1 : 0;

                bool split = (settings.split_active != 0);
                if (ImGui::Checkbox("Split Screen Comparison", &split)) settings.split_active = split ? 1 : 0;
                if (split) {
                    ImGui::SliderFloat("Split Line X", &settings.split_x, 0.0f, 1.0f, "%.2f");
                }
            }

            ImGui::Separator();
            if (ImGui::Button("Run Benchmark Suite (1080p, 1440p, 4K)", ImVec2(-1, 32))) {
                triggerBenchmark = true;
            }
        }
        ImGui::End();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        
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
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    cudaEventDestroy(liveStart);
    cudaEventDestroy(liveStop);
    if (d_live_step_counter) cudaFree(d_live_step_counter);

    cudaGraphicsUnregisterResource(cuda_pbo_resource);
    cudaDestroyTextureObject(noiseTex);
    cudaFreeArray(cuArray);
    delete[] noise_data;
    
    // Free OpenGL objects
    glDeleteTextures(1, &tex);
    typedef void (APIENTRY * PFNGLDELETEBUFFERSPROC) (GLsizei n, const GLuint *buffers);
    PFNGLDELETEBUFFERSPROC my_glDeleteBuffers = (PFNGLDELETEBUFFERSPROC)glfwGetProcAddress("glDeleteBuffers");
    if (my_glDeleteBuffers) {
        my_glDeleteBuffers(1, &pbo);
    }
    
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
