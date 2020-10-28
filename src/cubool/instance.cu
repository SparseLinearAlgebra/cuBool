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

#include <string>
#include <cstring>

namespace cubool {

    CuBoolStatus Instance::allocateOnGpu(CuBoolGpuPtr_t* ptr, CuBoolSize_t size) const {
        if (!ptr) {
            sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null pointer to store allocated memory ptr");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
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
                sendMessage(CUBOOL_STATUS_INVALID_STATE, "Failed to fined suitable allocator");
                return CUBOOL_STATUS_INVALID_STATE;
        }

        if (error != cudaSuccess) {
            std::string message = "Failed to allocate Gpu memory: ";
            message += cudaGetErrorString(error);

            sendMessage(CUBOOL_STATUS_MEM_OP_FAILED, message.c_str());
            return CUBOOL_STATUS_MEM_OP_FAILED;
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus Instance::deallocateOnGpu(CuBoolGpuPtr_t ptr) const {
        if (!ptr) {
            sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Passed null pointer to free");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        cudaError error = cudaFree(ptr);

        if (error != cudaSuccess) {
            std::string message = "Failed to deallocate Gpu memory: ";
            message += cudaGetErrorString(error);

            sendMessage(CUBOOL_STATUS_MEM_OP_FAILED, message.c_str());
            return CUBOOL_STATUS_MEM_OP_FAILED;
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus Instance::syncHostDevice() const {
        cudaError error = cudaDeviceSynchronize();

        if (error != cudaSuccess) {
            sendMessage(CUBOOL_STATUS_DEVICE_ERROR, "Failed to synchronize host and device");
            return CUBOOL_STATUS_DEVICE_ERROR;
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    bool Instance::isCudaDeviceSupported() {
        int device;
        cudaError error = cudaGetDevice(&device);
        return error == cudaSuccess;
    }

    void Instance::queryDeviceCapabilities(CuBoolDeviceCaps &deviceCaps) {
        static const CuBoolSize_t KiB = 1024;

        int device;
        cudaError error = cudaGetDevice(&device);

        if (error == cudaSuccess) {
            cudaDeviceProp deviceProp{};
            error = cudaGetDeviceProperties(&deviceProp, device);

            if (error == cudaSuccess) {
                strcpy(deviceCaps.name, deviceProp.name);
                deviceCaps.cudaSupported = true;
                deviceCaps.minor = deviceProp.minor;
                deviceCaps.major = deviceProp.major;
                deviceCaps.warp = deviceProp.warpSize;
                deviceCaps.globalMemoryKiBs = deviceProp.totalGlobalMem / KiB;
                deviceCaps.sharedMemoryPerMultiProcKiBs = deviceProp.sharedMemPerMultiprocessor / KiB;
                deviceCaps.sharedMemoryPerBlockKiBs = deviceProp.sharedMemPerBlock / KiB;
            }
        }
    }

}
