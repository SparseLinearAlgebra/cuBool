# CuBool library Cmake config file
# Add this file as sub-directory to your project to use library functionality
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(CUBOOL LANGUAGES CXX CUDA)

# Exposed to the user build options
option(CUBOOL_DEBUG "Build library in debug mode" TRUE)
option(CUBOOL_RELEASE "Build library in release mode" FALSE)
option(CUBOOL_BUILD_TESTS "Build project unit-tests with gtest" YES)

# Library version setup
set(CUBOOL_VERSION_MAJOR 0)
set(CUBOOL_VERSION_MINOR 1)

configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/include/cubool/cubool_build.h.in
        ${CMAKE_BINARY_DIR}/include/cubool/cubool_build.h)

# Library sources
set(CUBOOL_SOURCES
        # Public sources
        include/cubool/cubool.h
        include/cubool/cubool_types.h
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
        )

# Shared library object config
add_library(cubool SHARED ${CUBOOL_SOURCES})

target_include_directories(cubool PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(cubool PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src)
target_include_directories(cubool PRIVATE ${CMAKE_BINARY_DIR}/include)

target_compile_features(cubool PUBLIC cxx_std_11)

set_target_properties(cubool PROPERTIES CXX_STANDARD 11)
set_target_properties(cubool PROPERTIES CXX_STANDARD_REQUIRED YES)
set_target_properties(cubool PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# If tests enabled, add tests sources to the build
if (CUBOOL_BUILD_TESTS)
    message(STATUS "Add googletest as unit-testing library")
    add_subdirectory(thirdparty/googletest)
    add_subdirectory(tests)
endif()