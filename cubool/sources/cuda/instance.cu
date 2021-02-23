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

#include <cuda/instance.hpp>
#include <cuda/matrix_dense.hpp>
#include <core/error.hpp>
#include <string>
#include <cstring>

namespace cubool {

    Instance::~Instance() {
        gInstance = nullptr;
    }

    void Instance::allocateOnGpu(void* &ptr, size_t size) const {
        cudaError error;

        switch (mMemoryType) {
            case MemType::Default:
                error = cudaMalloc(&ptr, size);
                break;
            case MemType::Managed:
                error = cudaMallocManaged(&ptr, size);
                break;
            default:
                RAISE_ERROR(MemOpFailed, "Failed to fined suitable allocator");
        }

        if (error != cudaSuccess) {
            std::string message = std::string{"Failed to allocate Gpu memory: "} + cudaGetErrorString(error);
            RAISE_ERROR(MemOpFailed, message);
        }

#if 0
        {
            char buffer[2000];
            snprintf(buffer, 2000, "============> allocate on gpu %llu", (long long unsigned) size);
            this->sendMessage(CUBOOL_STATUS_SUCCESS, buffer);
        }
#endif
    }

    void Instance::deallocateOnGpu(void* ptr) const {
        cudaError error = cudaFree(ptr);

        if (error != cudaSuccess) {
            std::string message = std::string{"Failed to deallocate Gpu memory: "} + cudaGetErrorString(error);
            RAISE_ERROR(MemOpFailed, message);
        }
    }

    void Instance::syncHostDevice() const {
        cudaError error = cudaDeviceSynchronize();

        if (error != cudaSuccess) {
            std::string message = std::string{"Failed to synchronize host and device: "} + cudaGetErrorString(error);
            RAISE_ERROR(DeviceError, message);
        }
    }

    bool Instance::isCudaDeviceSupported() {
        int device;
        cudaError error = cudaGetDevice(&device);
        return error == cudaSuccess;
    }

    void Instance::queryDeviceCapabilities(CuBoolDeviceCaps &deviceCaps) {
        static const size_t KiB = 1024;

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
