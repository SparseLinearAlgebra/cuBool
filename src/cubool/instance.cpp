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

#include <string>
#include <cstdlib>

namespace cubool {

    Instance::Instance(const CuBoolInstanceDesc &desc) {
        mAllocCallback = desc.allocationCallback;
        mMessageCallback = desc.errorCallback;
        mMemoryType = desc.memoryType;

        bool onlyOneFunctionProvided =
                !mAllocCallback.allocateFun && mAllocCallback.deallocateFun ||
                mAllocCallback.allocateFun && !mAllocCallback.deallocateFun;

        if (onlyOneFunctionProvided) {
            sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "User must provide both allocate and deallocate functions");
            mAllocCallback.allocateFun = nullptr;
            mAllocCallback.deallocateFun = nullptr;
            mAllocCallback.userData = nullptr;
        }

#ifdef CUBOOL_DEBUG
        sendMessage(CUBOOL_STATUS_SUCCESS, "Initialize CuBool instance");
        printDeviceCapabilities();
#endif // CUBOOL_DEBUG
    }

    CuBoolStatus Instance::createMatrixDense(MatrixDense *&matrixDense) {
        CuBoolCpuPtr_t mem = nullptr;
        CuBoolStatus status = allocate(&mem, sizeof(MatrixDense));

        if (status != CUBOOL_STATUS_SUCCESS) {
            return status;
        }

        auto matrix = new (mem) MatrixDense(*this);
        matrixDense = matrix;

        mMatrixDense.emplace(matrix);
        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus Instance::validateMatrix(MatrixDense* matrixDense) {
        bool contains = mMatrixDense.find(matrixDense) != mMatrixDense.end();

        if (!contains) {
            sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "No matrix dense for provided handler");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus Instance::destroyMatrixDense(MatrixDense* matrixDense) {
        auto itr = mMatrixDense.find(matrixDense);

        if (itr == mMatrixDense.end()) {
            sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "No matrix dense to destroy for provided handler");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        mMatrixDense.erase(itr);

        matrixDense->~MatrixDense();
        return deallocate(matrixDense);
    }

    CuBoolStatus Instance::allocate(CuBoolCpuPtr_t* ptr, CuBoolSize_t size) const {
        if (!ptr) {
            sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null pointer to store allocated memory ptr");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        *ptr = hasUserDefinedAllocator() ? mAllocCallback.allocateFun(size, mAllocCallback.userData) : malloc(size);

        if (!(*ptr)) {
            sendMessage(CUBOOL_STATUS_MEM_OP_FAILED, "Failed to allocate memory on the CPU");
            return CUBOOL_STATUS_SUCCESS;
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus Instance::deallocate(CuBoolCpuPtr_t ptr) const {
        if (!ptr) {
            sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null pointer to free");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        if (hasUserDefinedAllocator()) {
            mAllocCallback.deallocateFun(ptr, mAllocCallback.userData);
        }
        else {
            free(ptr);
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    void Instance::sendMessage(CuBoolStatus status, const char *message) const {
        if (hasUserDefinedErrorCallback())
            mMessageCallback.msgFun(status, message, mMessageCallback.userData);
    }

    void Instance::printDeviceCapabilities() const {
        static const CuBoolSize_t BUFFER_SIZE = 1024;

        CuBoolDeviceCaps deviceCaps;
        queryDeviceCapabilities(deviceCaps);

        char deviceInfo[BUFFER_SIZE];
        snprintf(deviceInfo, BUFFER_SIZE, "Device name: %s version: %i.%i",
                 deviceCaps.name, deviceCaps.major, deviceCaps.major);

        char memoryInfo[BUFFER_SIZE];
        snprintf(memoryInfo, BUFFER_SIZE, "Global memory: %llu KiB",
                 (unsigned long long) deviceCaps.globalMemoryKiBs);

        char sharedMemoryInfo[BUFFER_SIZE];
        snprintf(sharedMemoryInfo, BUFFER_SIZE, "Shared memory: multi-proc %llu KiB block %llu KiB",
                 (unsigned long long) deviceCaps.sharedMemoryPerMultiProcKiBs, (unsigned long long) deviceCaps.sharedMemoryPerBlockKiBs);

        char structInfo[BUFFER_SIZE];
        snprintf(structInfo, BUFFER_SIZE, "Kernel: warp %llu", (unsigned long long) deviceCaps.warp);

        sendMessage(CUBOOL_STATUS_SUCCESS, deviceInfo);
        sendMessage(CUBOOL_STATUS_SUCCESS, memoryInfo);
        sendMessage(CUBOOL_STATUS_SUCCESS, sharedMemoryInfo);
        sendMessage(CUBOOL_STATUS_SUCCESS, structInfo);
    }

    bool Instance::hasUserDefinedAllocator() const {
        return mAllocCallback.allocateFun != nullptr;
    }

    bool Instance::hasUserDefinedErrorCallback() const {
        return mMessageCallback.msgFun != nullptr;
    }

}