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

#include <cubool/kernels/matrix_dense_frontend.hpp>

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

#define CUBOOL_VALIDATE_MATRIX(matrix)                                                                  \
    {                                                                                                   \
        CuBoolStatus status = instanceImpl->validateMatrix(matrix);                                     \
        if (status != CUBOOL_STATUS_SUCCESS) {                                                          \
            return status;                                                                              \
        }                                                                                               \
    }

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

CuBoolStatus CuBool_Instance_Delete(CuBoolInstance instance) {
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

CuBoolStatus CuBool_SyncHostDevice(CuBoolInstance instance) {
    auto instanceImpl = (cubool::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);

    return instanceImpl->syncHostDevice();
}

CuBoolStatus CuBool_MatrixDense_New(CuBoolInstance instance, CuBoolMatrixDense* matrix) {
    auto instanceImpl = (cubool::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    cubool::MatrixDense* matrixImpl = nullptr;
    CuBoolStatus status = instanceImpl->createMatrixDense(matrixImpl);
    *matrix = (CuBoolMatrixDense) matrixImpl;

    return status;
}

CuBoolStatus CuBool_MatrixDense_Delete(CuBoolInstance instance, CuBoolMatrixDense matrix) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    return instanceImpl->destroyMatrixDense(matrixImpl);
}

CuBoolStatus CuBool_MatrixDense_Resize(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t rows, CuBoolSize_t columns) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);

    CUBOOL_VALIDATE_MATRIX(matrixImpl);

    return matrixImpl->resize(rows, columns);
}

CuBoolStatus CuBool_MatrixDense_Build(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t count, const CuBoolPair* values) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);
    CUBOOL_CHECK_ARG_NOT_NULL(values);

    CUBOOL_VALIDATE_MATRIX(matrixImpl);

    return matrixImpl->writeValues(count, values);
}

CuBoolStatus CuBool_MatrixDense_ExtractPairs(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t* count, CuBoolPair** values) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(matrix);
    CUBOOL_CHECK_ARG_NOT_NULL(count);
    CUBOOL_CHECK_ARG_NOT_NULL(values);

    CUBOOL_VALIDATE_MATRIX(matrixImpl);

    return matrixImpl->readValues(count, values);
}

CuBoolStatus CuBool_Vals_Delete(CuBoolInstance instance, CuBoolPair* values) {
    auto instanceImpl = (cubool::Instance*) instance;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(values);

    return instanceImpl->deallocate(values);
}

CuBoolStatus CuBool_MatrixDense_MultAdd(CuBoolInstance instance, CuBoolMatrixDense res, CuBoolMatrixDense a, CuBoolMatrixDense b, CuBoolMatrixDense c) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto resImpl = (cubool::MatrixDense*) res;
    auto aImpl = (cubool::MatrixDense*) a;
    auto bImpl = (cubool::MatrixDense*) b;
    auto cImpl = (cubool::MatrixDense*) c;

    CUBOOL_CHECK_INSTANCE(instance);
    CUBOOL_CHECK_ARG_NOT_NULL(res);
    CUBOOL_CHECK_ARG_NOT_NULL(a);
    CUBOOL_CHECK_ARG_NOT_NULL(b);
    CUBOOL_CHECK_ARG_NOT_NULL(c);

    CUBOOL_VALIDATE_MATRIX(resImpl);
    CUBOOL_VALIDATE_MATRIX(aImpl);
    CUBOOL_VALIDATE_MATRIX(bImpl);
    CUBOOL_VALIDATE_MATRIX(cImpl);

    return cubool::MatrixDenseKernels::invokeMultiplyAdd(*instanceImpl, *resImpl, *aImpl, *bImpl, *cImpl);
}