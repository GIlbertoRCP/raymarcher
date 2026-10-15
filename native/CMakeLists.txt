cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(raymarcher_native LANGUAGES CXX)

# The CUDA viewer needs the CUDA toolkit and an NVIDIA GPU. The CPU tests
# need neither, so CI configures with -DRAYMARCHER_BUILD_VIEWER=OFF.
option(RAYMARCHER_BUILD_VIEWER "Build the CUDA viewer (requires the CUDA toolkit)" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- CPU tests for the shared physics -------------------------------------
enable_testing()
add_executable(raymarcher_tests tests/test_main.cpp)
target_include_directories(raymarcher_tests PRIVATE src)
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
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Turing (7.5) through Ada (8.9). An RTX 3070 is 8.6.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
endif()

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
    src/raymarch.cu
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
)
target_link_libraries(raymarcher PRIVATE glfw OpenGL::GL)
target_compile_options(raymarcher PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)

if(MSVC)
    target_compile_options(raymarcher PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    target_compile_definitions(raymarcher PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(raymarcher PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()
