/**********************************************************************************/
/*                                                                                */
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020 JetBrains-Research                                          */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/*                                                                                */
/**********************************************************************************/

#include <cubool/cubool.h>
#include <cubool/version.hpp>
#include <cubool/instance.hpp>
#include <cubool/matrix_dense.cuh>
#include <cubool/matrix_csr.cuh>
#include <cubool/details/error.hpp>
#include <cstdlib>
#include <string>

#define CUBOOL_CHECK_INSTANCE(instance)                                                                 \
    if (!instance) {                                                                                    \
        return CUBOOL_STATUS_INVALID_ARGUMENT;                                                          \
    }

#define CUBOOL_CHECK_ARG_NOT_NULL(arg)                                                                  \
    if (!arg) {                                                                                         \
        std::string message = "Passed null ptr to arg: ";                                               \
        message += #arg ;                                                                               \
        instanceImpl->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, message.c_str());                     \
        return CUBOOL_STATUS_INVALID_ARGUMENT;                                                          \
    }

#define CUBOOL_BEGIN_BODY \
        try {

#define CUBOOL_END_BODY } \
        catch (const cubool::details::Error& err) { \
             instanceImpl->sendMessage(err.status(), err.what()); \
             return err.status(); \
        }                      \
        return CuBoolStatus::CUBOOL_STATUS_SUCCESS; \

const char* CuBool_About_Get() {
    static const char about[] =
            "CuBool is a linear boolean algebra library primitives and operations for \n"
            "work with dense and sparse matrices written on the NVIDIA CUDA platform. The primary \n"
            "goal of the library is implementation, testing and profiling algorithms for\n"
            "solving *formal-language-constrained problems*, such as *context-free* \n"
            "and *recursive* path queries with various semantics for graph databases."
            ""
            "Contributors:"
            "- Semyon Grigorev (Github: https://github.com/gsvgit)\n"
            "- Egor Orachyov (Github: https://github.com/EgorOrachyov)";

    return about;
}

const char* CuBool_LicenseInfo_Get() {
    static const char license[] =
            "MIT License\n"
            "\n"
            "Copyright (c) 2020 JetBrains-Research\n"
            "\n"
            "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
            "of this software and associated documentation files (the \"Software\"), to deal\n"
            "in the Software without restriction, including without limitation the rights\n"
            "to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
            "copies of the Software, and to permit persons to whom the Software is\n"
            "furnished to do so, subject to the following conditions:\n"
            "\n"
            "The above copyright notice and this permission notice shall be included in all\n"
            "copies or substantial portions of the Software.\n"
            "\n"
            "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
            "IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
            "FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
            "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
            "LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
            "OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
            "SOFTWARE.";

    return license;
}

CuBoolStatus CuBool_Version_Get(int* major, int* minor, int* version) {
    if (major) {
        *major = CUBOOL_VERSION_MAJOR;
    }

    if (minor) {
        *minor = CUBOOL_VERSION_MINOR;
    }

    if (version) {
        *version = CUBOOL_VERSION;
    }

    return CUBOOL_STATUS_SUCCESS;
}

CuBoolStatus CuBool_DeviceCaps_Get(CuBoolDeviceCaps* deviceCaps) {
    if (!deviceCaps) {
        // Null caps structure
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    if (!cubool::Instance::isCudaDeviceSupported()) {
        // No cuda support
        return CUBOOL_STATUS_DEVICE_NOT_PRESENT;
    }

    cubool::Instance::queryDeviceCapabilities(*deviceCaps);

    return CUBOOL_STATUS_SUCCESS;
}

CuBoolStatus CuBool_Instance_New(const CuBoolInstanceDesc* instanceDesc, CuBoolInstance* instance) {
    if (!instanceDesc) {
        // Instance descriptor could not be null
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    if (!instance) {
        // Null to safe instance reference
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    if (!cubool::Instance::isCudaDeviceSupported()) {
        // No device for cuda computations
        return CUBOOL_STATUS_DEVICE_ERROR;
    }

    auto& desc = *instanceDesc;
    auto& allocator = desc.allocationCallback;

    CuBoolSize_t instanceSize = sizeof(cubool::Instance);
    CuBoolCpuPtr_t instanceMem = nullptr;

    if (allocator.allocateFun && allocator.deallocateFun) {
        instanceMem = allocator.allocateFun(instanceSize, allocator.userData);
    }
    else {
        instanceMem = malloc(instanceSize);
    }

    if (!instanceMem) {
        // Failed to allocate instance
        return CUBOOL_STATUS_MEM_OP_FAILED;
    }

    auto instanceImpl = new(instanceMem) cubool::Instance(desc);

    // Raw cast. We expose to the user non-existing language structure
    *instance = (CuBoolInstance) instanceImpl;

    return CUBOOL_STATUS_SUCCESS;
}

CuBoolStatus CuBool_Instance_Free(CuBoolInstance instance) {
    auto instanceImpl = (cubool::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);

    if (instanceImpl->hasUserDefinedAllocator()) {
        auto allocator = instanceImpl->getUserDefinedAllocator();
        instanceImpl->~Instance();
        allocator.deallocateFun(instanceImpl, allocator.userData);
    }
    else {
        instanceImpl->~Instance();
        free(instanceImpl);
    }

    return CUBOOL_STATUS_SUCCESS;
}

CuBoolStatus CuBool_HostDevice_Sync(CuBoolInstance instance) {
    auto instanceImpl = (cubool::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);

    CUBOOL_BEGIN_BODY
        instanceImpl->syncHostDevice();
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_New(CuBoolInstance instance, CuBoolMatrixDense *matrix, CuBoolSize_t nrows, CuBoolSize_t ncols) {
    auto instanceImpl = (cubool::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);

    CUBOOL_BEGIN_BODY
        cubool::MatrixDense* matrixImpl = nullptr;
        instanceImpl->createMatrixDense(matrixImpl);
        matrixImpl->resize(nrows, ncols);
        *matrix = (CuBoolMatrixDense) matrixImpl;
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_Free(CuBoolInstance instance, CuBoolMatrixDense matrix) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    CUBOOL_BEGIN_BODY
        instanceImpl->destroyMatrix(matrixImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_Resize(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t nrows, CuBoolSize_t ncols) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        matrixImpl->resize(nrows, ncols);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_Build(CuBoolInstance instance, CuBoolMatrixDense matrix, const CuBoolIndex_t *rows, const CuBoolIndex_t *cols, CuBoolSize_t nvals) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(rows);
    CUBOOL_CHECK_ARG_NOT_NULL(cols);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        matrixImpl->build(rows, cols, nvals);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_ExtractPairs(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolIndex_t **rows, CuBoolIndex_t **cols, CuBoolSize_t *nvals) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(rows);
    CUBOOL_CHECK_ARG_NOT_NULL(cols);
    CUBOOL_CHECK_ARG_NOT_NULL(nvals);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        matrixImpl->extract(rows, cols, nvals);
    CUBOOL_END_BODY
}


CuBoolStatus CuBool_Vals_Free(CuBoolInstance instance, CuBoolIndex_t* vals) {
    auto instanceImpl = (cubool::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(vals);

    CUBOOL_BEGIN_BODY
        instanceImpl->deallocate(vals);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_MxM(CuBoolInstance instance, CuBoolMatrixDense r, CuBoolMatrixDense a, CuBoolMatrixDense b) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto rImpl = (cubool::MatrixDense*) r;
    auto aImpl = (cubool::MatrixDense*) a;
    auto bImpl = (cubool::MatrixDense*) b;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(r);
    CUBOOL_CHECK_ARG_NOT_NULL(a);
    CUBOOL_CHECK_ARG_NOT_NULL(b);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(aImpl);
        instanceImpl->validateMatrix(bImpl);
        instanceImpl->validateMatrix(rImpl);
        rImpl->multiplyAdd(*aImpl, *bImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MatrixDense_MultSum(CuBoolInstance instance, CuBoolMatrixDense r, CuBoolMatrixDense a, CuBoolMatrixDense b, CuBoolMatrixDense c) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto rImpl = (cubool::MatrixDense*) r;
    auto aImpl = (cubool::MatrixDense*) a;
    auto bImpl = (cubool::MatrixDense*) b;
    auto cImpl = (cubool::MatrixDense*) c;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(r);
    CUBOOL_CHECK_ARG_NOT_NULL(a);
    CUBOOL_CHECK_ARG_NOT_NULL(b);
    CUBOOL_CHECK_ARG_NOT_NULL(c);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(aImpl);
        instanceImpl->validateMatrix(bImpl);
        instanceImpl->validateMatrix(cImpl);
        instanceImpl->validateMatrix(rImpl);
        rImpl->multiplySum(*aImpl, *bImpl, *cImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_New(CuBoolInstance instance, CuBoolMatrix *matrix, CuBoolSize_t nrows, CuBoolSize_t ncols) {
    auto instanceImpl = (cubool::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);

    CUBOOL_BEGIN_BODY
        cubool::MatrixCsr* matrixImpl = nullptr;
        instanceImpl->createMatrixCsr(matrixImpl);
        matrixImpl->resize(nrows, ncols);
        *matrix = (CuBoolMatrix) matrixImpl;
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Free(CuBoolInstance instance, CuBoolMatrix matrix) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixCsr*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    CUBOOL_BEGIN_BODY
        instanceImpl->destroyMatrix(matrixImpl);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Resize(CuBoolInstance instance, CuBoolMatrix matrix, CuBoolSize_t nrows, CuBoolSize_t ncols) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixCsr*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        matrixImpl->resize(nrows, ncols);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_Build(CuBoolInstance instance, CuBoolMatrix matrix, const CuBoolIndex_t *rows, const CuBoolIndex_t *cols, CuBoolSize_t nvals) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixCsr*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(rows);
    CUBOOL_CHECK_ARG_NOT_NULL(cols);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        matrixImpl->build(rows, cols, nvals);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_Matrix_ExtractPairs(CuBoolInstance instance, CuBoolMatrix matrix, CuBoolIndex_t **rows, CuBoolIndex_t **cols, CuBoolSize_t *nvals) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixCsr*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(rows);
    CUBOOL_CHECK_ARG_NOT_NULL(cols);
    CUBOOL_CHECK_ARG_NOT_NULL(nvals);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(matrixImpl);
        matrixImpl->extract(rows, cols, nvals);
    CUBOOL_END_BODY
}

CuBoolStatus CuBool_MxM(CuBoolInstance instance, CuBoolMatrix r, CuBoolMatrix a, CuBoolMatrix b) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto rImpl = (cubool::MatrixCsr*) r;
    auto aImpl = (cubool::MatrixCsr*) a;
    auto bImpl = (cubool::MatrixCsr*) b;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(r);
    CUBOOL_CHECK_ARG_NOT_NULL(a);
    CUBOOL_CHECK_ARG_NOT_NULL(b);

    CUBOOL_BEGIN_BODY
        instanceImpl->validateMatrix(aImpl);
        instanceImpl->validateMatrix(bImpl);
        instanceImpl->validateMatrix(rImpl);
        rImpl->multiplyAdd(*aImpl, *bImpl);
    CUBOOL_END_BODY
}