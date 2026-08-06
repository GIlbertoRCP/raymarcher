cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cuda_raymarcher LANGUAGES CXX CUDA)

# Force standard C++17 and CUDA C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Define CUDA architecture targets (modern architectures from Turing 7.5 to Ada Lovelace 8.9)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
endif()

# Enable CUDA Hardware Fast Math Intrinsic Optimizations (-O3 --use_fast_math)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")

# Fetch GLFW from GitHub automatically
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

# Disable building GLFW documentation, tests, and examples
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(glfw imgui)

# Find system OpenGL
find_package(OpenGL REQUIRED)

# ImGui sources
set(IMGUI_SOURCES
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)

# Setup executable target
add_executable(cuda_raymarcher
    main.cpp
    raymarch.cu
    raymarch.h
    ${IMGUI_SOURCES}
)

# Add source directory & ImGui to includes
target_include_directories(cuda_raymarcher PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)

# Link against GLFW and OpenGL
target_link_libraries(cuda_raymarcher PRIVATE
    glfw
    OpenGL::GL
)

# Target compiler flags and definitions
if(MSVC)
    target_compile_options(cuda_raymarcher PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
    target_compile_definitions(cuda_raymarcher PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(cuda_raymarcher PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

# Automated CTest Unit Testing Target
enable_testing()
add_executable(cuda_raymarcher_tests
    test_main.cpp
    raymarch.h
)
target_include_directories(cuda_raymarcher_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME CUDARaymarcherUnitTests COMMAND cuda_raymarcher_tests)
