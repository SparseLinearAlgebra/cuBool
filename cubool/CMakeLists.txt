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

set(TARGET_NAME cubool)
set(TARGET_FILE_NAME)
set(DEFINES_LIST)

# Mode
if (CUBOOL_DEBUG)
    list(APPEND DEFINES_LIST CUBOOL_DEBUG)
endif()
if (CUBOOL_RELEASE)
    list(APPEND DEFINES_LIST CUBOOL_RELEASE)
endif()

# Platform checks
if(APPLE)
    list(APPEND DEFINES_LIST CUBOOL_PLATFORM_MACOS)
    set(TARGET_FILE_NAME "lib${TARGET_NAME}.dylib")
elseif(UNIX)
    list(APPEND DEFINES_LIST CUBOOL_PLATFORM_LINUX)
    set(TARGET_FILE_NAME "lib${TARGET_NAME}.so")
elseif(WIN32)
    list(APPEND DEFINES_LIST CUBOOL_PLATFORM_WIN)
    set(TARGET_FILE_NAME "${TARGET_NAME}.dll")
else()
    message(FATAL_ERROR "Platform not specified")
endif()

# Library sources
set(CUBOOL_SOURCES
    sources/core/config.hpp
    sources/core/version.hpp
    sources/core/error.hpp
    sources/core/library.cpp
    sources/core/library.hpp
    sources/core/object.cpp
    sources/core/object.hpp
    sources/core/matrix.cpp
    sources/core/matrix.hpp
    sources/core/vector.cpp
    sources/core/vector.hpp
    sources/io/logger.cpp
    sources/io/logger.hpp
    sources/utils/algo_utils.hpp
    sources/utils/timer.hpp
    sources/utils/data_utils.cpp
    sources/utils/data_utils.hpp)

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
    sources/cuBool_Matrix_ExtractRow.cpp
    sources/cuBool_Matrix_ExtractCol.cpp
    sources/cuBool_Matrix_Duplicate.cpp
    sources/cuBool_Matrix_Transpose.cpp
    sources/cuBool_Matrix_Nvals.cpp
    sources/cuBool_Matrix_Nrows.cpp
    sources/cuBool_Matrix_Ncols.cpp
    sources/cuBool_Matrix_Free.cpp
    sources/cuBool_Matrix_Reduce.cpp
    sources/cuBool_Matrix_Reduce2.cpp
    sources/cuBool_Matrix_EWiseAdd.cpp
    sources/cuBool_Matrix_EWiseMult.cpp
    sources/cuBool_Matrix_EWiseMultInverted.cpp
    sources/cuBool_Vector_New.cpp
    sources/cuBool_Vector_Build.cpp
    sources/cuBool_Vector_SetElement.cpp
    sources/cuBool_Vector_SetMarker.cpp
    sources/cuBool_Vector_Marker.cpp
    sources/cuBool_Vector_ExtractValues.cpp
    sources/cuBool_Vector_ExtractSubVector.cpp
    sources/cuBool_Vector_Duplicate.cpp
    sources/cuBool_Vector_Nvals.cpp
    sources/cuBool_Vector_Nrows.cpp
    sources/cuBool_Vector_Free.cpp
    sources/cuBool_Vector_Reduce.cpp
    sources/cuBool_Vector_EWiseAdd.cpp
    sources/cuBool_Vector_EWiseMult.cpp
    sources/cuBool_MxM.cpp
    sources/cuBool_MxV.cpp
    sources/cuBool_VxM.cpp
    sources/cuBool_Kronecker.cpp)

set(CUBOOL_BACKEND_SOURCES
    sources/backend/backend_base.hpp
    sources/backend/matrix_base.hpp
    sources/backend/vector_base.hpp)

set(CUBOOL_CUDA_SOURCES)
set(CUBOOL_SEQUENTIAL_SOURCES)

# Cuda backend sources
if (CUBOOL_WITH_CUDA)
    set(CUBOOL_CUDA_SOURCES
        sources/cuda/cuda_backend.hpp
        sources/cuda/cuda_backend.cu
        sources/cuda/cuda_instance.hpp
        sources/cuda/cuda_instance.cu
        sources/cuda/cuda_matrix.hpp
        sources/cuda/cuda_matrix.cu
        sources/cuda/cuda_matrix_ewiseadd.cu
        sources/cuda/cuda_matrix_ewisemult.cu
        sources/cuda/cuda_matrix_ewisemult_inverted.cu
        sources/cuda/cuda_matrix_kronecker.cu
        sources/cuda/cuda_matrix_multiply.cu
        sources/cuda/cuda_matrix_transpose.cu
        sources/cuda/cuda_matrix_reduce.cu
        sources/cuda/cuda_matrix_extract_sub_matrix.cu
        sources/cuda/cuda_vector.hpp
        sources/cuda/cuda_vector.cu
        sources/cuda/cuda_vector_mxv.cu
        sources/cuda/cuda_vector_vxm.cu
        sources/cuda/cuda_vector_ewiseadd.cu
        sources/cuda/cuda_vector_ewisemult.cu
        sources/cuda/cuda_vector_reduce.cu
        sources/cuda/details/meta.hpp
        sources/cuda/details/sp_vector.hpp
        sources/cuda/details/host_allocator.hpp
        sources/cuda/details/device_allocator.cuh
        sources/cuda/kernels/slow_sort.cuh
        sources/cuda/kernels/bin_search.cuh
        sources/cuda/kernels/spgemv.cuh
        sources/cuda/kernels/spgemv_t.cuh
        sources/cuda/kernels/spewiseadd.cuh
        sources/cuda/kernels/spewisemult.cuh
        sources/cuda/kernels/spewisemultinverted.cuh
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
        sources/sequential/sq_vector.cpp
        sources/sequential/sq_vector.hpp
        sources/sequential/sq_data.hpp
        sources/sequential/sq_transpose.cpp
        sources/sequential/sq_transpose.hpp
        sources/sequential/sq_kronecker.cpp
        sources/sequential/sq_kronecker.hpp
        sources/sequential/sq_ewiseadd.cpp
        sources/sequential/sq_ewiseadd.hpp
        sources/sequential/sq_ewisemult.cpp
        sources/sequential/sq_ewisemult.hpp
        sources/sequential/sq_ewisemultinverted.cpp
        sources/sequential/sq_ewisemultinverted.hpp
        sources/sequential/sq_spgemm.cpp
        sources/sequential/sq_spgemm.hpp
        sources/sequential/sq_spgemv.cpp
        sources/sequential/sq_spgemv.hpp
        sources/sequential/sq_reduce.cpp
        sources/sequential/sq_reduce.hpp
        sources/sequential/sq_submatrix.cpp
        sources/sequential/sq_submatrix.hpp
        sources/sequential/sq_subvector.cpp
        sources/sequential/sq_subvector.hpp)
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

# Platform defines
foreach(DEFINE ${DEFINES_LIST})
    target_compile_definitions(cubool PUBLIC ${DEFINE})
endforeach()

# Cuda specifics
if (CUBOOL_WITH_CUDA)
    set_target_properties(cubool PROPERTIES CUDA_STANDARD 14)
    set_target_properties(cubool PROPERTIES CUDA_STANDARD_REQUIRED ON)
    # set_target_properties(cubool PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # Settings: https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/

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

# Copy cubool library after build if allowed
# if (CUBOOL_COPY_TO_PY_PACKAGE)
#     set(LIBRARY_FILE_NAME ${TARGET_FILE_NAME})
# 
#     add_custom_command(
#         TARGET cubool POST_BUILD
#         COMMAND "${CMAKE_COMMAND}" -E
#         copy
#         "${CMAKE_BINARY_DIR}/cubool/${LIBRARY_FILE_NAME}"
#         "${CMAKE_BINARY_DIR}/python/pycubool"
#         COMMENT "Copy ${LIBRARY_FILE_NAME} compiled lib into python folder")
# endif()
