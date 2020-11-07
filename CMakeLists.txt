# CuBool library Cmake config file
# Add this file as sub-directory to your project to use library functionality
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(CUBOOL LANGUAGES CXX CUDA)

# Exposed to the user build options
option(CUBOOL_DEBUG "Build library in debug mode" ON)
option(CUBOOL_RELEASE "Build library in release mode" OFF)
option(CUBOOL_WITH_CUB "Build project with locally stored CUB library" ON)
option(CUBOOL_WITH_NSPARSE "Build library with nsparse crs matrix multiplication backend" ON)
option(CUBOOL_WITH_NAIVE "Build library with naive and naive-shared dense matrix multiplication" ON)
option(CUBOOL_BUILD_TESTS "Build project unit-tests with gtest" ON)

SET(CUBOOL_WITH_CUB ON CACHE BOOL "" FORCE)
SET(CUBOOL_WITH_NSPARSE ON CACHE BOOL "" FORCE)
SET(CUBOOL_WITH_NAIVE ON CACHE BOOL "" FORCE)
SET(CUBOOL_BUILD_TESTS ON CACHE BOOL "" FORCE)

# Library version setup
set(CUBOOL_VERSION_MAJOR 0)
set(CUBOOL_VERSION_MINOR 1)

configure_file(
    ${CMAKE_CURRENT_LIST_DIR}/src/cubool/cubool_build.h.in
    ${CMAKE_BINARY_DIR}/src/cubool/cubool_build.h)

# Library sources
set(CUBOOL_SOURCES
    # Public sources
    include/cubool/cubool.h
    # Private sources
    src/cubool/cubool.cpp
    src/cubool/version.hpp
    src/cubool/instance.cu
    src/cubool/instance.hpp
    src/cubool/utils/gpu_buffer.cu
    src/cubool/utils/gpu_buffer.hpp
    src/cubool/utils/cpu_buffer.cpp
    src/cubool/utils/cpu_buffer.hpp
    src/cubool/matrix_dense.cpp
    src/cubool/matrix_dense.hpp
    src/cubool/kernels/matrix_dense_multiply_add.cu
    src/cubool/kernels/matrix_dense_frontend.hpp
    src/cubool/kernels/matrix_dense_shared.cuh src/cubool/instance.cpp)

# Shared library object config
add_library(cubool SHARED ${CUBOOL_SOURCES})

target_include_directories(cubool PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(cubool PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
target_include_directories(cubool PRIVATE ${CMAKE_BINARY_DIR}/src)

target_compile_features(cubool PUBLIC cxx_std_14)

set_target_properties(cubool PROPERTIES CXX_STANDARD 11)
set_target_properties(cubool PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(cubool PROPERTIES CUDA_STANDARD 11)
set_target_properties(cubool PROPERTIES CUDA_STANDARD_REQUIRED ON)
set_target_properties(cubool PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Static library config
add_library(cubool_static SHARED ${CUBOOL_SOURCES})

target_include_directories(cubool_static PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(cubool_static PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
target_include_directories(cubool_static PRIVATE ${CMAKE_BINARY_DIR}/src)

target_compile_features(cubool_static PUBLIC cxx_std_11)

set_target_properties(cubool_static PROPERTIES CXX_STANDARD 11)
set_target_properties(cubool_static PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(cubool_static PROPERTIES CUDA_STANDARD 11)
set_target_properties(cubool_static PROPERTIES CUDA_STANDARD_REQUIRED ON)
set_target_properties(cubool_static PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

if (CUBOOL_WITH_CUB)
    message(STATUS "Add cub as cuda utility")
    set(CUB_ENABLE_HEADER_TESTING OFF CACHE BOOL "" FORCE)
    set(CUB_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(CUB_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(thirdparty/cub)
    add_library(cub INTERFACE IMPORTED)
    target_link_libraries(cub INTERFACE CUB::CUB)
endif()

if (CUBOOL_WITH_NSPARSE)
    message(STATUS "Add nsparse library as crs matrix multiplication backend")
    add_subdirectory(thirdparty/nsparse)
    add_subdirectory(thirdparty/nsparse-um)
endif()

if (CUBOOL_WITH_NAIVE)
    message(STATUS "Add naive library as dense matrix multiplication implementation for benchmarks")
    add_subdirectory(thirdparty/naive)
endif()

# If tests enabled, add tests sources to the build
if (CUBOOL_BUILD_TESTS)
    message(STATUS "Add googletest as unit-testing library")
    add_subdirectory(thirdparty/googletest)
    add_subdirectory(tests)
endif()