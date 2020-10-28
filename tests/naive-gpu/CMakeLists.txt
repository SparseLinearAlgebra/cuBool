set(SOURCES
        GpuMatrix.cpp
        GpuMatrix.h
        Matrix.h
        Multiplication.cu
        Multiplication.h
        Parameters.h)

add_library(naive_gpu STATIC ${SOURCES})
target_compile_features(naive_gpu PUBLIC cxx_std_11)
target_include_directories(naive_gpu PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(naive_gpu PROPERTIES CXX_STANDARD 11)
set_target_properties(naive_gpu PROPERTIES CXX_STANDARD_REQUIRED YES)
set_target_properties(naive_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)