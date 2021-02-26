# CuBool library Cmake config file
# Add this file as sub-directory to your project to use library functionality

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(cubool LANGUAGES CXX)

# Exposed to the user build options
option(CUBOOL_WITH_CUDA       "Build library with cuda backend (default)" ON)
option(CUBOOL_WITH_SEQUENTIAL "Build library with cpu sequential backend (fallback)" ON)
option(CUBOOL_WITH_NAIVE      "Build library with naive and naive-shared dense matrix multiplication" ON)
option(CUBOOL_BUILD_TESTS     "Build project unit-tests with gtest" ON)

set(CUBOOL_VERSION_MAJOR 1)
set(CUBOOL_VERSION_MINOR 0)
set(CUBOOL_VERSION_SUB 0)

# Configure cuda dependencies
if (CUBOOL_WITH_CUDA)
    message(STATUS "Add cub as cuda utility")
    set(CUB_ENABLE_HEADER_TESTING OFF CACHE BOOL "" FORCE)
    set(CUB_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(CUB_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(deps/cub)
    add_library(cub INTERFACE IMPORTED)
    target_link_libraries(cub INTERFACE CUB::CUB)

    message(STATUS "Add nsparse library as crs matrix multiplication backend")
    add_subdirectory(deps/nsparse-um)
endif()

if (CUBOOL_WITH_NAIVE)
    message(STATUS "Add naive library as dense matrix multiplication implementation for benchmarks")
    add_subdirectory(deps/naive)
endif()

if (CUBOOL_BUILD_TESTS)
    message(STATUS "Add googletest as unit-testing library")
    add_subdirectory(deps/gtest)
endif()

# Actual cxx implementation
add_subdirectory(cubool)

# Copy scripts into binary directory
file(COPY scripts DESTINATION ${CMAKE_BINARY_DIR}/)

# Copy python related stuff
file(COPY python DESTINATION ${CMAKE_BINARY_DIR}/)