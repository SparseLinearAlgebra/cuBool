cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(CUBOOL LANGUAGES CXX CUDA)

option(CUBOOL_BUILD_TESTS "Build project unit-tests with gtest" YES)

if (CUBOOL_BUILD_TESTS)
    add_subdirectory(thirdparty/googletest)
    add_subdirectory(tests)
endif()