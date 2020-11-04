set(NAIVE_SOURCES
        naive-gpu/GpuMatrix.cpp
        naive-gpu/GpuMatrix.h
        naive-gpu/Matrix.h
        naive-gpu/Multiplication.cu
        naive-gpu/Multiplication.h
        naive-gpu/Parameters.h)

add_library(naive_gpu STATIC ${NAIVE_SOURCES})
target_compile_features(naive_gpu PUBLIC cxx_std_11)
target_include_directories(naive_gpu PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(naive_gpu PROPERTIES CXX_STANDARD 11)
set_target_properties(naive_gpu PROPERTIES CXX_STANDARD_REQUIRED YES)
set_target_properties(naive_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set(NAIVE_SHARED_SOURCES
        naive-gpu-shared/GpuMatrix.cpp
        naive-gpu-shared/GpuMatrix.h
        naive-gpu-shared/Matrix.h
        naive-gpu-shared/Multiplication.cu
        naive-gpu-shared/Multiplication.h
        naive-gpu-shared/Parameters.h)

add_library(naive_gpu_shared STATIC ${NAIVE_SHARED_SOURCES})
target_compile_features(naive_gpu_shared PUBLIC cxx_std_11)
target_include_directories(naive_gpu_shared PUBLIC ${CMAKE_CURRENT_LIST_DIR})
set_target_properties(naive_gpu_shared PROPERTIES CXX_STANDARD 11)
set_target_properties(naive_gpu_shared PROPERTIES CXX_STANDARD_REQUIRED YES)
set_target_properties(naive_gpu_shared PROPERTIES CUDA_SEPARABLE_COMPILATION ON)