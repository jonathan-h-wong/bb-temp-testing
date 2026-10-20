cmake_minimum_required(VERSION 3.20)
project(temporary_testing CUDA CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Library (shared algorithm/simulation code) and entry-point executables
add_subdirectory(src)
add_subdirectory(apps)
