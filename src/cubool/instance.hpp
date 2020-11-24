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

#ifndef CUBOOL_INSTANCE_HPP
#define CUBOOL_INSTANCE_HPP

#include <cubool/cubool.h>
#include <unordered_set>

namespace cubool {

    class Instance {
    public:
        explicit Instance(const CuBoolInstanceDesc& desc);
        Instance(const Instance& other) = delete;
        Instance(Instance&& other) noexcept = delete;
        ~Instance();

        void createMatrixDense(class MatrixDense* &matrix);
        void createMatrixCsr(class MatrixCsr* &matrix);
        void validateMatrix(class MatrixBase* matrix);
        void destroyMatrix(class MatrixBase* matrix);

        void allocate(CuBoolCpuPtr_t* ptr, CuBoolSize_t size) const;
        void allocateOnGpu(CuBoolGpuPtr_t* ptr, CuBoolSize_t size) const;

        void deallocate(CuBoolCpuPtr_t ptr) const;
        void deallocateOnGpu(CuBoolGpuPtr_t ptr) const;

        void syncHostDevice() const;

        void sendMessage(CuBoolStatus status, const char* message) const;
        void printDeviceCapabilities() const;

        bool hasUserDefinedAllocator() const;
        bool hasUserDefinedErrorCallback() const;

        const CuBoolAllocationCallback& getUserDefinedAllocator() const { return mAllocCallback; }

        static bool isCudaDeviceSupported();
        static void queryDeviceCapabilities(CuBoolDeviceCaps& deviceCaps);

    private:
        std::unordered_set<class MatrixBase*> mMatrixSet;

        CuBoolAllocationCallback mAllocCallback{};
        CuBoolMessageCallback mMessageCallback{};
        CuBoolGpuMemoryType mMemoryType{};
    };

}

#endif //CUBOOL_INSTANCE_HPP