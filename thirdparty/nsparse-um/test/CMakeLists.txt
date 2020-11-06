cmake_minimum_required(VERSION 3.10)
project(nsparse_um_test CXX CUDA)

add_executable(${PROJECT_NAME} src/nsparse_test.cu src/utils.cpp)
target_link_libraries(${PROJECT_NAME} gtest_main nsparse_um cub)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_14)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(
    NAME ${PROJECT_NAME}
    COMMAND ${PROJECT_BINARY_DIR}/${PROJECT_NAME}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
