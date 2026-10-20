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
