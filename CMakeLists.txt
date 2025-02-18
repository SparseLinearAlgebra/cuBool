# CuBool library Cmake config file
# Add this file as sub-directory to your project to use library functionality

cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(cuBool LANGUAGES CXX)

# Exposed to the user build options
option(CUBOOL_WITH_CUDA          "Build library with cuda backend (default)" ON)
option(CUBOOL_WITH_SEQUENTIAL    "Build library with cpu sequential backend (fallback)" ON)
option(CUBOOL_WITH_NAIVE         "Build library with naive and naive-shared dense matrix multiplication" OFF)
option(CUBOOL_BUILD_TESTS        "Build project unit-tests with gtest" ON)
option(CUBOOL_COPY_TO_PY_PACKAGE "Copy compiled shared library into python package folder (for package use purposes)" ON)

option(CUBOOL_USE_NSPARSE_MERGE_FUNCTOR "Use nsparse optimiztion for matrix addition function" ON)

set(CUBOOL_VERSION_MAJOR 1)
set(CUBOOL_VERSION_MINOR 0)
set(CUBOOL_VERSION_SUB 0)

set(CUBOOL_DEBUG OFF)
set(CUBOOL_RELEASE OFF)

if (${CMAKE_BUILD_TYPE} MATCHES Release)
    message(STATUS "Build cubool in release mode")
    set(CUBOOL_RELEASE ON)
elseif (${CMAKE_BUILD_TYPE} MATCHES Debug)
    message(STATUS "Build cubool in debug mode")
    set(CUBOOL_DEBUG ON)
else()
    message(STATUS "Build cubool in release mode (default: was not specified)")
    set(CUBOOL_RELEASE ON)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Configure cuda dependencies
if (CUBOOL_WITH_CUDA)
    message(STATUS "Add cub as cuda utility")
    # set(CUB_ENABLE_HEADER_TESTING OFF CACHE BOOL "" FORCE)
    # set(CUB_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    # set(CUB_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    # add_subdirectory(deps/cub)
    # add_library(cub INTERFACE IMPORTED)
    # target_link_libraries(cub INTERFACE CUB::CUB)

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
# file(COPY scripts DESTINATION ${CMAKE_BINARY_DIR}/)

# Copy python related stuff
# file(COPY python DESTINATION ${CMAKE_BINARY_DIR}/)
