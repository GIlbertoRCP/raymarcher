cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(raymarcher_native LANGUAGES CXX)

# The CUDA viewer needs the CUDA toolkit and an NVIDIA GPU. The CPU tests
# need neither, so CI configures with -DRAYMARCHER_BUILD_VIEWER=OFF.
option(RAYMARCHER_BUILD_VIEWER "Build the CUDA viewer (requires the CUDA toolkit)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CPU tests for the generated metrics and shared geodesic code ----------
# native/generated/ is written by `npm run export-cuda` and committed.
set(RM_GENERATED_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated)
enable_testing()
add_executable(raymarcher_tests tests/test_main.cpp)
target_include_directories(raymarcher_tests PRIVATE src ${RM_GENERATED_DIR})
target_compile_definitions(raymarcher_tests PRIVATE RM_GENERATED_DIR="${RM_GENERATED_DIR}")
if(MSVC)
    target_compile_options(raymarcher_tests PRIVATE /W4)
else()
    target_compile_options(raymarcher_tests PRIVATE -Wall -Wextra)
endif()
add_test(NAME physics COMMAND raymarcher_tests)

if(NOT RAYMARCHER_BUILD_VIEWER)
    return()
endif()

# --- CUDA viewer -----------------------------------------------------------
# The tracer kernel (kernels/tracer.cu) is compiled at runtime with NVRTC for
# the chosen metric and precision, so the build itself compiles no device code.
find_package(CUDAToolkit 11.2 REQUIRED)

include(FetchContent)
FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.3.8
)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.8
)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw imgui)

find_package(OpenGL REQUIRED)

add_executable(raymarcher
    src/main.cpp
    src/nvrtc_tracer.cpp
    src/validate.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(raymarcher PRIVATE
    src
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${glfw_SOURCE_DIR}/deps  # stb_image_write.h
)
target_link_libraries(raymarcher PRIVATE glfw OpenGL::GL CUDA::cudart CUDA::cuda_driver CUDA::nvrtc CUDA::nvml)
# Runtime sources are read from the source tree, so kernel edits and
# `npm run export-cuda` take effect without a rebuild. A copy next to the
# executable is the fallback when the source tree is not there.
target_compile_definitions(raymarcher PRIVATE RM_NATIVE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_custom_command(TARGET raymarcher POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/kernels $<TARGET_FILE_DIR:raymarcher>/native/kernels
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/generated $<TARGET_FILE_DIR:raymarcher>/native/generated
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/rm $<TARGET_FILE_DIR:raymarcher>/native/src/rm
)

if(MSVC)
    target_compile_options(raymarcher PRIVATE /W4)
    target_compile_definitions(raymarcher PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(raymarcher PRIVATE -Wall -Wextra)
endif()
