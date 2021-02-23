if (CUBOOL_WITH_CUDA)
    # If Cuda backend is compiled, we must tell cmake, that we will use Cuda
    project(cubool-cpp LANGUAGES CXX CUDA)
else()
    # Otherwise only C++
    project(cubool-cpp LANGUAGES CXX)
endif()

# Notify user about selected backend options
if (CUBOOL_WITH_CPU)
    message(STATUS "Add SuiteSparse based CPU backend as fallback")
endif()
if (CUBOOL_WITH_CUDA)
    message(STATUS "Add CUDA backend for GPGPU computations")
endif()

set(CUBOOL_SOURCES)

# Library sources
set(CUBOOL_SOURCES
    sources/core/config.hpp
    sources/core/version.hpp
    sources/core/error.hpp
        sources/core/library.cpp
        sources/core/library.hpp
    sources/core/matrix.cpp
    sources/core/matrix.hpp)

set(CUBOOL_C_API_SOURCES
    include/cubool/cubool.h
    sources/cuBool_About_Get.cpp
    sources/cuBool_LicenseInfo_Get.cpp
    sources/cuBool_Initialize.cpp
    sources/cuBool_Finalize.cpp
    sources/cuBool_Matrix_New.cpp
    sources/cuBool_Matrix_Build.cpp
    sources/cuBool_Matrix_ExtractPairs.cpp
    sources/cuBool_Matrix_Duplicate.cpp
    sources/cuBool_Matrix_Transpose.cpp
    sources/cuBool_Matrix_Nvals.cpp
    sources/cuBool_Matrix_Nrows.cpp
    sources/cuBool_Matrix_Ncols.cpp
    sources/cuBool_Matrix_Free.cpp
    sources/cuBool_Matrix_EWiseAdd.cpp
    sources/cuBool_MxM.cpp
    sources/cuBool_Kronecker.cpp)

set(CUBOOL_BACKEND_SOURCES
    sources/backend/backend_base.hpp
    sources/backend/matrix_base.hpp)

set(CUBOOL_CUDA_SOURCES)
set(CUBOOL_CPU_SOURCES)

# Cuda backend sources
if (CUBOOL_WITH_CUDA)
    set(CUBOOL_CUDA_SOURCES
        sources/cuda/cuda_backend.hpp
        sources/cuda/cuda_backend.cu
        sources/cuda/instance.hpp
        sources/cuda/instance.cu
        sources/cuda/instance.cpp
        sources/cuda/matrix_csr.hpp
        sources/cuda/matrix_csr.cu
        sources/cuda/matrix_csr_ewiseadd.cu
        sources/cuda/matrix_csr_kronecker.cu
        sources/cuda/matrix_csr_multiply_add.cu
        sources/cuda/matrix_csr_multiply_sum.cu
        sources/cuda/matrix_csr_transpose.cu)
endif()

# Shared library object config
add_library(cubool SHARED
    ${CUBOOL_SOURCES}
    ${CUBOOL_C_API_SOURCES}
    ${CUBOOL_CUDA_SOURCES}
    ${CUBOOL_CPU_SOURCES}
    ${CUBOOL_BACKEND_SOURCES})

target_include_directories(cubool PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(cubool PRIVATE ${CMAKE_CURRENT_LIST_DIR}/sources)

target_compile_definitions(cubool PRIVATE CUBOOL_VERSION_MAJOR=${CUBOOL_VERSION_MAJOR})
target_compile_definitions(cubool PRIVATE CUBOOL_VERSION_MINOR=${CUBOOL_VERSION_MINOR})

target_compile_features(cubool PUBLIC cxx_std_14)

set_target_properties(cubool PROPERTIES CXX_STANDARD 17)
set_target_properties(cubool PROPERTIES CXX_STANDARD_REQUIRED ON)

# Cuda specifics
if (CUBOOL_WITH_CUDA)
    set_target_properties(cubool PROPERTIES CUDA_STANDARD 14)
    set_target_properties(cubool PROPERTIES CUDA_STANDARD_REQUIRED ON)
    set_target_properties(cubool PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_compile_definitions(cubool PRIVATE CUBOOL_WITH_CUDA)
    target_link_libraries(cubool PRIVATE nsparse_um)
endif()

# SuiteSparse based backend
if (CUBOOL_WITH_CPU)
    target_compile_definitions(cubool PRIVATE CUBOOL_WITH_CPU)
endif()

# If tests enabled, add tests sources to the build
if (CUBOOL_BUILD_TESTS)
    message(STATUS "Add unit tests directory to the project")
    add_subdirectory(tests)
endif()