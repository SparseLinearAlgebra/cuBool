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

#include <cubool/instance.hpp>
#include <cubool/cubool_build.h>
#include <cstdlib>

namespace cubool {

    Instance::Instance(const CuBoolInstanceDesc &desc) {
        mAllocCallback = desc.allocationCallback;
        mErrorCallback = desc.errorCallback;
        mMemoryType = desc.memoryType;

        bool onlyOneFunctionProvided =
            !mAllocCallback.allocateFun && mAllocCallback.deallocateFun ||
            mAllocCallback.allocateFun && !mAllocCallback.deallocateFun;

        if (onlyOneFunctionProvided) {
            errorMessage(CUBOOL_ERROR_INVALID_ARGUMENT, "User must provide both allocate and deallocate functions");
            mAllocCallback.allocateFun = nullptr;
            mAllocCallback.deallocateFun = nullptr;
            mAllocCallback.userData = nullptr;
        }

#ifdef CUBOOL_DEBUG
        errorMessage(CUBOOL_ERROR_SUCCESS, "Initial library message");
#endif // CUBOOL_DEBUG
    }

    CuBoolError Instance::allocate(CuBoolCpuPtr_t* ptr, CuBoolSize_t size) const {
        if (!ptr) {
            errorMessage(CUBOOL_ERROR_INVALID_ARGUMENT, "Passed null pointer to store allocated memory ptr");
            return CUBOOL_ERROR_INVALID_ARGUMENT;
        }

        *ptr = hasUserDefinedAllocator() ? mAllocCallback.allocateFun(size, mAllocCallback.userData) : malloc(size);

        if (!(*ptr)) {
            errorMessage(CUBOOL_ERROR_MEM_OP_FAILED, "Failed to allocate memory on the CPU");
            return CUBOOL_ERROR_SUCCESS;
        }

        return CUBOOL_ERROR_SUCCESS;
    }

    CuBoolError Instance::allocateOnGpu(CuBoolGpuPtr_t* ptr, CuBoolSize_t size) const {
        if (!ptr) {
            errorMessage(CUBOOL_ERROR_INVALID_ARGUMENT, "Passed null pointer to store allocated memory ptr");
            return CUBOOL_ERROR_INVALID_ARGUMENT;
        }

        cudaError error;

        switch (mMemoryType) {
            case CUBOOL_GPU_MEMORY_TYPE_GENERIC:
                error = cudaMalloc(ptr, size);
                break;
            case CUBOOL_GPU_MEMORY_TYPE_MANAGED:
                error = cudaMallocManaged(ptr, size);
                break;
            default:
                errorMessage(CUBOOL_ERROR_INVALID_STATE, "Failed to fined suitable allocator");
                return CUBOOL_ERROR_INVALID_STATE;
        }

        if (error != cudaSuccess) {
            std::string message = "Failed to allocate Gpu memory: ";
            message += cudaGetErrorString(error);

            errorMessage(CUBOOL_ERROR_MEM_OP_FAILED, message.c_str());
            return CUBOOL_ERROR_MEM_OP_FAILED;
        }

        return CUBOOL_ERROR_SUCCESS;
    }

    CuBoolError Instance::deallocate(CuBoolCpuPtr_t ptr) const {
        if (!ptr) {
            errorMessage(CUBOOL_ERROR_INVALID_ARGUMENT, "Passed null pointer to free");
            return CUBOOL_ERROR_INVALID_ARGUMENT;
        }

        if (hasUserDefinedAllocator()) {
            mAllocCallback.deallocateFun(ptr, mAllocCallback.userData);
        }
        else {
            free(ptr);
        }

        return CUBOOL_ERROR_SUCCESS;
    }

    CuBoolError Instance::deallocateOnGpu(CuBoolGpuPtr_t ptr) const {
        if (!ptr) {
            errorMessage(CUBOOL_ERROR_INVALID_ARGUMENT, "Passed null pointer to free");
            return CUBOOL_ERROR_INVALID_ARGUMENT;
        }

        cudaError error = cudaFree(ptr);

        if (error != cudaSuccess) {
            std::string message = "Failed to deallocate Gpu memory: ";
            message += cudaGetErrorString(error);

            errorMessage(CUBOOL_ERROR_MEM_OP_FAILED, message.c_str());
            return CUBOOL_ERROR_MEM_OP_FAILED;
        }

        return CUBOOL_ERROR_SUCCESS;
    }

    void Instance::errorMessage(CuBoolError status, const char *message) const {
        if (hasUserDefinedErrorCallback())
            mErrorCallback.errorMsgFun(status, message, mErrorCallback.userData);
    }

    bool Instance::hasUserDefinedAllocator() const {
        return mAllocCallback.allocateFun != nullptr;
    }

    bool Instance::hasUserDefinedErrorCallback() const {
        return mErrorCallback.errorMsgFun != nullptr;
    }

}
