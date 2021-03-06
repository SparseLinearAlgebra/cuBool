project(cubool-cpp LANGUAGES CXX)

if (CUBOOL_WITH_CUDA)
    # If Cuda backend is compiled, we must tell cmake, that we will use Cuda
    enable_language(CUDA)
endif()

# Notify user about selected backend options
if (CUBOOL_WITH_SEQUENTIAL)
    message(STATUS "Add CPU sequential fallback backend")
endif()
if (CUBOOL_WITH_CUDA)
    message(STATUS "Add CUDA backend for GPGPU computations")
endif()

# Library sources
set(CUBOOL_SOURCES
    sources/core/config.hpp
    sources/core/version.hpp
    sources/core/error.hpp
    sources/core/library.cpp
    sources/core/library.hpp
    sources/core/matrix.cpp
    sources/core/matrix.hpp
    sources/io/logger.cpp
    sources/io/logger.hpp
    sources/utils/exclusive_scan.hpp
    sources/utils/timer.hpp
        )

set(CUBOOL_C_API_SOURCES
    include/cubool/cubool.h
    sources/cuBool_GetAbout.cpp
    sources/cuBool_GetVersion.cpp
    sources/cuBool_GetLicenseInfo.cpp
    sources/cuBool_GetDeviceCaps.cpp
    sources/cuBool_Initialize.cpp
    sources/cuBool_Finalize.cpp
    sources/cuBool_SetupLogger.cpp
    sources/cuBool_Matrix_New.cpp
    sources/cuBool_Matrix_Build.cpp
    sources/cuBool_Matrix_SetElement.cpp
    sources/cuBool_Matrix_SetMarker.cpp
    sources/cuBool_Matrix_Marker.cpp
    sources/cuBool_Matrix_ExtractPairs.cpp
    sources/cuBool_Matrix_ExtractSubMatrix.cpp
    sources/cuBool_Matrix_Duplicate.cpp
    sources/cuBool_Matrix_Transpose.cpp
    sources/cuBool_Matrix_Nvals.cpp
    sources/cuBool_Matrix_Nrows.cpp
    sources/cuBool_Matrix_Ncols.cpp
    sources/cuBool_Matrix_Free.cpp
    sources/cuBool_Matrix_Reduce.cpp
    sources/cuBool_Matrix_EWiseAdd.cpp
    sources/cuBool_MxM.cpp
    sources/cuBool_Kronecker.cpp)

set(CUBOOL_BACKEND_SOURCES
    sources/backend/backend_base.hpp
    sources/backend/matrix_base.hpp)

set(CUBOOL_CUDA_SOURCES)
set(CUBOOL_SEQUENTIAL_SOURCES)

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
        sources/cuda/matrix_csr_multiply.cu
        sources/cuda/matrix_csr_transpose.cu
        sources/cuda/matrix_csr_reduce.cu
        sources/cuda/matrix_csr_extract_sub_matrix.cu
        sources/cuda/kernels/slow_sort.cuh
        sources/cuda/kernels/bin_search.cuh
        sources/cuda/kernels/sptranspose.cuh
        sources/cuda/kernels/sptranspose2.cuh
        sources/cuda/kernels/spkron.cuh
        sources/cuda/kernels/spmerge.cuh
        sources/cuda/kernels/spreduce.cuh
        sources/cuda/kernels/spsubmatrix.cuh)
endif()

# Cpu sequential backend sources
if (CUBOOL_WITH_SEQUENTIAL)
    set(CUBOOL_SEQUENTIAL_SOURCES
        sources/sequential/sq_backend.cpp
        sources/sequential/sq_backend.hpp
        sources/sequential/sq_matrix.cpp
        sources/sequential/sq_matrix.hpp
        sources/sequential/sq_csr_data.hpp
        sources/sequential/sq_transpose.cpp
        sources/sequential/sq_transpose.hpp
        sources/sequential/sq_kronecker.cpp
        sources/sequential/sq_kronecker.hpp
        sources/sequential/sq_ewiseadd.cpp
        sources/sequential/sq_ewiseadd.hpp
        sources/sequential/sq_spgemm.cpp
        sources/sequential/sq_spgemm.hpp
        sources/sequential/sq_reduce.cpp
        sources/sequential/sq_reduce.hpp
        sources/sequential/sq_submatrix.cpp
        sources/sequential/sq_submatrix.hpp)
endif()

# Shared library object config
add_library(cubool SHARED
    ${CUBOOL_SOURCES}
    ${CUBOOL_C_API_SOURCES}
    ${CUBOOL_BACKEND_SOURCES}
    ${CUBOOL_CUDA_SOURCES}
    ${CUBOOL_SEQUENTIAL_SOURCES})

target_include_directories(cubool PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(cubool PRIVATE ${CMAKE_CURRENT_LIST_DIR}/sources)

target_compile_definitions(cubool PRIVATE CUBOOL_EXPORTS)
target_compile_definitions(cubool PRIVATE CUBOOL_VERSION_MAJOR=${CUBOOL_VERSION_MAJOR})
target_compile_definitions(cubool PRIVATE CUBOOL_VERSION_MINOR=${CUBOOL_VERSION_MINOR})
target_compile_definitions(cubool PRIVATE CUBOOL_VERSION_SUB=${CUBOOL_VERSION_SUB})

target_compile_features(cubool PUBLIC cxx_std_14)

target_compile_options(cubool PRIVATE $<$<COMPILE_LANGUAGE:CXX>: -Wall>)
target_compile_options(cubool PRIVATE $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>: -O2>)
target_compile_options(cubool PRIVATE $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>: -O0>)

set_target_properties(cubool PROPERTIES CXX_STANDARD 17)
set_target_properties(cubool PROPERTIES CXX_STANDARD_REQUIRED ON)

# Cuda specifics
if (CUBOOL_WITH_CUDA)
    set_target_properties(cubool PROPERTIES CUDA_STANDARD 14)
    set_target_properties(cubool PROPERTIES CUDA_STANDARD_REQUIRED ON)
    set_target_properties(cubool PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Settings: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
    target_compile_options(cubool PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -arch=sm_30
        -gencode=arch=compute_30,code=sm_30
        -gencode=arch=compute_35,code=sm_35
        -gencode=arch=compute_50,code=sm_50
        -gencode=arch=compute_52,code=sm_52
        -gencode=arch=compute_52,code=compute_52>)

    target_compile_options(cubool PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -use_fast_math -Xptxas -O2>)

    target_compile_definitions(cubool PRIVATE CUBOOL_WITH_CUDA)
    target_link_libraries(cubool PRIVATE nsparse_um)
endif()

# Sequential Cpu based backend
if (CUBOOL_WITH_SEQUENTIAL)
    target_compile_definitions(cubool PRIVATE CUBOOL_WITH_SEQUENTIAL)
endif()

# If tests enabled, add tests sources to the build
if (CUBOOL_BUILD_TESTS)
    add_library(testing INTERFACE)
    target_include_directories(testing INTERFACE ${CMAKE_CURRENT_LIST_DIR}/utils/)
    target_link_libraries(testing INTERFACE cubool)
    target_link_libraries(testing INTERFACE gtest)

    message(STATUS "Add unit tests directory to the project")
    add_subdirectory(tests)
endif()

# todo: Different platforms has specific naming conventions
set(LIBRARY_FILE_NAME "libcubool.so")

# Copy cubool library after build
add_custom_command(
    TARGET cubool POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E
    copy
    "${CMAKE_BINARY_DIR}/cubool/${LIBRARY_FILE_NAME}"
    "${CMAKE_BINARY_DIR}/python/pycubool"
    COMMENT "Copy cubool lib into python folder")