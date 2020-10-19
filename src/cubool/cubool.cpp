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
#include <cubool/instance.hpp>

CuBoolError CuBoolCreateInstance(const CuBoolInstanceDesc* instanceDesc, CuBoolInstance* instance) {
    if (!instanceDesc) {
        // Instance descriptor could not be null
        return CUBOOL_ERROR_INVALID_ARGUMENT;
    }

    if (!instance) {
        // Null to safe instance reference
        return CUBOOL_ERROR_INVALID_ARGUMENT;
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
        return CUBOOL_ERROR_MEM_OP_FAILED;
    }

    auto instanceImpl = new(instanceMem) cubool::Instance(desc);

    // Raw cast. We expose to the user non-existing language structure
    *instance = (CuBoolInstance) instanceImpl;

    return CUBOOL_ERROR_SUCCESS;
}

CuBoolError CuBoolDestroyInstance(CuBoolInstance instance) {
    if (!instance) {
        // Null instance passed
        return CUBOOL_ERROR_INVALID_ARGUMENT;
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

    return CUBOOL_ERROR_SUCCESS;
}

CuBoolError CuBoolCreateMatrixDense(CuBoolInstance instance, CuBoolMatrixDense* matrix) {
    return CUBOOL_ERROR_NOT_IMPLEMENTED;
}

CuBoolError CuBoolDestroyMatrixDense(CuBoolInstance instance, CuBoolMatrixDense matrix) {
    return CUBOOL_ERROR_NOT_IMPLEMENTED;
}

CuBoolError CuBoolMultiplyAdd(CuBoolInstance instance, CuBoolMatrixDense result, CuBoolMatrixDense a, CuBoolMatrixDense b, CuBoolMatrixDense c) {
    return CUBOOL_ERROR_NOT_IMPLEMENTED;
}