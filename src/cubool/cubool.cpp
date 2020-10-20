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

#include <cstdlib>

CuBoolStatus CuBoolGetLibraryVersion(int* major, int* minor, int* version) {
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

CuBoolStatus CuBoolGetDeviceCapabilities(CuBoolDeviceCaps* deviceCaps) {
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

CuBoolStatus CuBoolCreateInstance(const CuBoolInstanceDesc* instanceDesc, CuBoolInstance* instance) {
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

CuBoolStatus CuBoolDestroyInstance(CuBoolInstance instance) {
    if (!instance) {
        // Null instance passed
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    auto instanceImpl = (cubool::Instance*) instance;

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

CuBoolStatus CuBoolCreateMatrixDense(CuBoolInstance instance, CuBoolMatrixDense* matrix) {
    if (!instance) {
        // Null instance passed
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    auto instanceImpl = (cubool::Instance*) instance;

    if (!matrix) {
        instanceImpl->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null ptr to place to save dense matrix handle");
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    cubool::MatrixDense* matrixImpl = nullptr;
    CuBoolStatus status = instanceImpl->createMatrixDense(matrixImpl);
    *matrix = (CuBoolMatrixDense) matrixImpl;

    return status;
}

CuBoolStatus CuBoolDestroyMatrixDense(CuBoolInstance instance, CuBoolMatrixDense matrix) {
    if (!instance) {
        // Null instance passed
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    return instanceImpl->destroyMatrixDense(matrixImpl);
}

CuBoolStatus CuBoolMatrixDenseResize(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t rows, CuBoolSize_t columns) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    if (!instance) {
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    if (!matrix) {
        instanceImpl->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null ptr to place to save dense matrix handle");
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    CuBoolStatus status = instanceImpl->validateMatrixDense(matrixImpl);
    if (status != CUBOOL_STATUS_SUCCESS) {
        return status;
    }

    return matrixImpl->resize(rows, columns);
}

CuBoolStatus CuBoolDenseMatrixWriteData(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t count, const CuBoolPair* values) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    if (!instance) {
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    if (!matrix) {
        instanceImpl->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null ptr to place to save dense matrix handle");
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    CuBoolStatus status = instanceImpl->validateMatrixDense(matrixImpl);
    if (status != CUBOOL_STATUS_SUCCESS) {
        return status;
    }

    if (values == nullptr) {
        instanceImpl->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null values array to write");
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    return matrixImpl->writeValues(count, values);
}

CuBoolStatus CuBoolDenseMatrixReadData(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t* count, CuBoolPair** values) {
    auto instanceImpl = (cubool::Instance*) instance;
    auto matrixImpl = (cubool::MatrixDense*) matrix;

    if (!instance) {
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    if (!matrix) {
        instanceImpl->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null ptr to place to save dense matrix handle");
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    CuBoolStatus status = instanceImpl->validateMatrixDense(matrixImpl);

    if (status != CUBOOL_STATUS_SUCCESS) {
        return status;
    }

    if (!count || !values) {
        instanceImpl->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null ptr to operation results");
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    return matrixImpl->readValues(count, values);
}

CuBoolStatus CuBoolReleaseValuesArray(CuBoolInstance instance, CuBoolPair* values) {
    auto instanceImpl = (cubool::Instance*) instance;

    if (!instance) {
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    if (!values) {
        instanceImpl->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null ptr to release values array");
        return CUBOOL_STATUS_INVALID_ARGUMENT;
    }

    return instanceImpl->deallocate(values);
}

CuBoolStatus CuBoolMultiplyAdd(CuBoolInstance instance, CuBoolMatrixDense result, CuBoolMatrixDense a, CuBoolMatrixDense b, CuBoolMatrixDense c) {
    return CUBOOL_STATUS_NOT_IMPLEMENTED;
}