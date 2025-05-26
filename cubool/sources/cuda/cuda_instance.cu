/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020, 2021 JetBrains-Research                                    */
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
/**********************************************************************************/

#include <cassert>
#include <core/error.hpp>
#include <cstring>
#include <cuda/cuda_instance.hpp>
#include <string>

namespace cubool {

    volatile CudaInstance* CudaInstance::gInstance = nullptr;

    CudaInstance::CudaInstance(bool useManagedMemory) {
        gInstance   = this;
        mMemoryType = useManagedMemory ? Managed : Default;
    }

    CudaInstance::~CudaInstance() {
        assert(mHostAllocCount == 0);
        assert(mDeviceAllocCount == 0);

        gInstance = nullptr;
    }

    void CudaInstance::allocateOnGpu(void*& ptr, size_t size) const {
        cudaError error;

        switch (mMemoryType) {
            case MemType::Default:
                error = cudaMalloc(&ptr, size);
                break;
            case MemType::Managed:
                error = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
                break;
            default:
                RAISE_ERROR(MemOpFailed, "Failed to fined suitable allocator");
        }

        if (error != cudaSuccess) {
            std::string message = std::string{"Failed to allocate Gpu memory: "} + cudaGetErrorString(error);
            RAISE_ERROR(MemOpFailed, message);
        }

        mDeviceAllocCount++;
    }

    void CudaInstance::deallocateOnGpu(void* ptr) const {
        cudaError error = cudaFree(ptr);

        if (error != cudaSuccess) {
            std::string message = std::string{"Failed to deallocate Gpu memory: "} + cudaGetErrorString(error);
            RAISE_ERROR(MemOpFailed, message);
        }

        mDeviceAllocCount--;
    }

    void CudaInstance::syncHostDevice() const {
        cudaError error = cudaDeviceSynchronize();

        if (error != cudaSuccess) {
            std::string message = std::string{"Failed to synchronize host and device: "} + cudaGetErrorString(error);
            RAISE_ERROR(DeviceError, message);
        }
    }

    void CudaInstance::queryDeviceCapabilities(cuBool_DeviceCaps& deviceCaps) const {
        const unsigned long long KiB = 1024;

        int       device;
        cudaError error = cudaGetDevice(&device);

        if (error == cudaSuccess) {
            cudaDeviceProp deviceProp{};
            error = cudaGetDeviceProperties(&deviceProp, device);

            if (error == cudaSuccess) {
                strcpy(deviceCaps.name, deviceProp.name);
                deviceCaps.cudaSupported                = true;
                deviceCaps.managedMem                   = mMemoryType == MemType::Managed;
                deviceCaps.minor                        = deviceProp.minor;
                deviceCaps.major                        = deviceProp.major;
                deviceCaps.warp                         = deviceProp.warpSize;
                deviceCaps.globalMemoryKiBs             = deviceProp.totalGlobalMem / KiB;
                deviceCaps.sharedMemoryPerMultiProcKiBs = deviceProp.sharedMemPerMultiprocessor / KiB;
                deviceCaps.sharedMemoryPerBlockKiBs     = deviceProp.sharedMemPerBlock / KiB;
            }
        }
    }

    CudaInstance::MemType CudaInstance::getMemoryType() const {
        return mMemoryType;
    }

    bool CudaInstance::isCudaDeviceSupported() {
        int       device;
        cudaError error = cudaGetDevice(&device);
        return error == cudaSuccess;
    }

    void CudaInstance::allocate(void*& ptr, size_t size) const {
        ptr = malloc(size);
        CHECK_RAISE_ERROR(ptr != nullptr, MemOpFailed, "Failed to allocate memory on the CPU");
        mHostAllocCount++;
    }

    void CudaInstance::deallocate(void* ptr) const {
        CHECK_RAISE_ERROR(ptr != nullptr, InvalidArgument, "Passed null ptr to free");
        free(ptr);
        mHostAllocCount--;
    }

    CudaInstance& CudaInstance::getInstanceRef() {
        CHECK_RAISE_ERROR(gInstance != nullptr, InvalidState, "No instance in the system");
        return (CudaInstance&) *gInstance;
    }

    CudaInstance* CudaInstance::getInstancePtr() {
        return (CudaInstance*) gInstance;
    }

    bool CudaInstance::isInstancePresent() {
        return gInstance != nullptr;
    }

}// namespace cubool
