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
#include <cubool/matrix_dense.cuh>
#include <cubool/matrix_csr.cuh>
#include <cubool/details/error.hpp>
#include <string>
#include <cstring>

namespace cubool {

    Instance::~Instance() {
#ifdef CUBOOL_DEBUG
        // Notify user if he lost or forgot to release some library objects

        if (!mMatrixDenseSet.empty()) {
            char message[2000];

            snprintf(message, 2000, "Some MatrixDense objects (%llu) were not properly deallocated by user",
                     (unsigned long long) mMatrixDenseSet.size());

            sendMessage(CUBOOL_STATUS_INVALID_STATE, message);
        }

        if (!mMatrixCsrSet.empty()) {
            char message[2000];

            snprintf(message, 2000, "Some MatrixCsr objects (%llu) were not properly deallocated by user",
                     (unsigned long long) mMatrixCsrSet.size());

            sendMessage(CUBOOL_STATUS_INVALID_STATE, message);
        }

#endif //CUBOOL_DEBUG

        for (auto matrix: mMatrixDenseSet) {
            matrix->~MatrixDense();
            deallocate(matrix);
        }

        for (auto matrix: mMatrixCsrSet) {
            matrix->~MatrixCsr();
            deallocate(matrix);
        }
    }

    void Instance::createMatrixDense(MatrixDense *&matrix) {
        CuBoolCpuPtr_t mem = nullptr;
        allocate(&mem, sizeof(MatrixDense));

        matrix = new (mem) MatrixDense(*this);
        mMatrixDenseSet.emplace(matrix);
    }

    void Instance::validateMatrixDense(MatrixDense* matrix) {
        bool contains = mMatrixDenseSet.find(matrix) != mMatrixDenseSet.end();

        if (!contains)
            throw details::InvalidArgument("No such matrix for provided handler");
    }

    void Instance::destroyMatrixDense(MatrixDense* matrix) {
        auto itr = mMatrixDenseSet.find(matrix);

        if (itr == mMatrixDenseSet.end())
            throw details::InvalidArgument("No such matrix for provided handler");

        mMatrixDenseSet.erase(itr);
        matrix->~MatrixDense();
        deallocate(matrix);
    }

    void Instance::createMatrixCsr(MatrixCsr *&matrix) {
        CuBoolCpuPtr_t mem = nullptr;
        allocate(&mem, sizeof(MatrixCsr));

        matrix = new (mem) MatrixCsr(*this);
        mMatrixCsrSet.emplace(matrix);
    }

    void Instance::validateMatrixCsr(MatrixCsr* matrix) {
        bool contains = mMatrixCsrSet.find(matrix) != mMatrixCsrSet.end();

        if (!contains)
            throw details::InvalidArgument("No such matrix for provided handler");
    }

    void Instance::destroyMatrixCsr(MatrixCsr* matrix) {
        auto itr = mMatrixCsrSet.find(matrix);

        if (itr == mMatrixCsrSet.end())
            throw details::InvalidArgument("No such matrix for provided handler");

        mMatrixCsrSet.erase(itr);
        matrix->~MatrixCsr();
        deallocate(matrix);
    }

    void Instance::allocateOnGpu(CuBoolGpuPtr_t* ptr, CuBoolSize_t size) const {
        if (!ptr)
            throw details::InvalidArgument("Passed null pointer to store allocated memory ptr");

        cudaError error;

        switch (mMemoryType) {
            case CUBOOL_GPU_MEMORY_TYPE_GENERIC:
                error = cudaMalloc(ptr, size);
                break;
            case CUBOOL_GPU_MEMORY_TYPE_MANAGED:
                error = cudaMallocManaged(ptr, size);
                break;
            default:
                throw details::MemOpFailed("Failed to fined suitable allocator");
        }

        if (error != cudaSuccess)
            throw details::MemOpFailed(std::string{"Failed to allocate Gpu memory: "} + cudaGetErrorString(error));
    }

    void Instance::deallocateOnGpu(CuBoolGpuPtr_t ptr) const {
        if (!ptr)
            throw details::InvalidArgument("Passed null pointer to free");

        cudaError error = cudaFree(ptr);

        if (error != cudaSuccess)
            throw details::MemOpFailed(std::string{"Failed to deallocate Gpu memory: "} + cudaGetErrorString(error));
    }

    void Instance::syncHostDevice() const {
        cudaError error = cudaDeviceSynchronize();

        if (error != cudaSuccess)
            throw details::MemOpFailed(std::string{"Failed to synchronize host and device: "} + cudaGetErrorString(error));
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
