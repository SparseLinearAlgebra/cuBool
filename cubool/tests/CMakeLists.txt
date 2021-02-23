add_executable(test_basic_cuda test_basic_cuda.cu)
target_link_libraries(test_basic_cuda PUBLIC testing)
target_compile_features(test_basic_cuda PUBLIC cxx_std_11)
set_target_properties(test_basic_cuda PROPERTIES CXX_STANDARD 11)
set_target_properties(test_basic_cuda PROPERTIES CXX_STANDARD_REQUIRED YES)
set_target_properties(test_basic_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(test_library_api test_library_api.cpp)
target_link_libraries(test_library_api PUBLIC testing)

add_executable(test_matrix_setup test_matrix_setup.cpp)
target_link_libraries(test_matrix_setup PUBLIC testing)

add_executable(test_matrix_misc test_matrix_misc.cpp)
target_link_libraries(test_matrix_misc PUBLIC testing)

add_executable(test_matrix_transpose test_matrix_transpose.cpp)
target_link_libraries(test_matrix_transpose PUBLIC testing)

add_executable(test_matrix_mxm test_matrix_mxm.cpp)
target_link_libraries(test_matrix_mxm PUBLIC testing)

add_executable(test_matrix_kronecker test_matrix_kronecker.cpp)
target_link_libraries(test_matrix_kronecker PUBLIC testing)

add_executable(test_matrix_ewiseadd test_matrix_ewiseadd.cpp)
target_link_libraries(test_matrix_ewiseadd PUBLIC testing)

if (CUBOOL_WITH_NAIVE)
    add_executable(test_matrix_dense_vs_naive test_matrix_dense_vs_naive.cpp)
    target_link_libraries(test_matrix_dense_vs_naive PUBLIC testing)
    target_link_libraries(test_matrix_dense_vs_naive PUBLIC naive_gpu)
    target_link_libraries(test_matrix_dense_vs_naive PUBLIC naive_gpu_shared)
endif()