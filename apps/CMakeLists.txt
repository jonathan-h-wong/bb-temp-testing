add_executable(main main.cpp)

target_link_libraries(main PRIVATE proto)

target_include_directories(main PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Jolt sphere-complex + dynamic add/remove test with realtime visualization
add_executable(sphere_complex_demo sphere_complex_demo.cpp)

target_link_libraries(sphere_complex_demo PRIVATE viz)

# Interactive GPU Poisson disk sampling: click chunks to clear + resample them
find_package(glfw3 CONFIG REQUIRED)
find_package(OpenGL REQUIRED)

add_executable(poisson_demo poisson_demo.cpp)

target_link_libraries(poisson_demo PRIVATE poisson glfw OpenGL::GL)

# 2D VoroMesh of a heart: sparse-grid boundary target + CUDA VoroLoss optimisation
find_path(STB_INCLUDE_DIRS "stb_image_write.h")

add_executable(voromesh_demo voromesh_demo.cpp)

target_include_directories(voromesh_demo PRIVATE ${STB_INCLUDE_DIRS})

target_link_libraries(voromesh_demo PRIVATE voromesh poisson glfw OpenGL::GL)
