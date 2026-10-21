# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Jolt CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
set(PROTO_SOURCES
    proto/sim.cpp
    proto/kernels.cu
)

add_library(proto ${PROTO_SOURCES})

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
target_include_directories(proto
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# ---------------------------------------------------------------------------
# Linked libraries
# ---------------------------------------------------------------------------
target_link_libraries(proto
    PUBLIC
        Jolt::Jolt
        CUDA::cudart
)

# ---------------------------------------------------------------------------
# CUDA settings
# ---------------------------------------------------------------------------
set_target_properties(proto PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES native          # auto-detect local GPU at configure time
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# ---------------------------------------------------------------------------
# Compiler warnings
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(proto PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/diagnostics:caret>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/diagnostics:caret>
    )
else()
    target_compile_options(proto PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# ---------------------------------------------------------------------------
# viz – realtime Jolt debug-draw viewer (GLFW + legacy OpenGL)
# ---------------------------------------------------------------------------
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_library(viz viz/viewer.cpp)

target_include_directories(viz
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(viz
    PUBLIC
        Jolt::Jolt
        glfw
        OpenGL::GL
)

if(MSVC)
    target_compile_options(viz PRIVATE /W3 /MP)
else()
    target_compile_options(viz PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# poisson – GPU Poisson disk sampling (grid-based parallel dart throwing)
# ---------------------------------------------------------------------------
add_library(poisson
    poisson/poisson_disk.cu
    poisson/poisson_disk_cpu.cpp
)

target_include_directories(poisson
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(poisson PUBLIC CUDA::cudart)

set_target_properties(poisson PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

if(MSVC)
    target_compile_options(poisson PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()

# ---------------------------------------------------------------------------
# voromesh – 2D VoroMesh: canvas (CPU) + CUDA optimiser
# ---------------------------------------------------------------------------
add_library(voromesh
    voromesh/shape2d.cpp
    voromesh/canvas.cpp
    voromesh/voro_optimizer.cu
    voromesh/cpu_voro_optimizer.cpp
)

target_include_directories(voromesh
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(voromesh
    PUBLIC
        CUDA::cudart
)

set_target_properties(voromesh PROPERTIES
    CUDA_ARCHITECTURES native
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

if(MSVC)
    target_compile_options(voromesh PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()
