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

#ifndef CUBOOL_DEVICE_ALLOCATOR_CUH
#define CUBOOL_DEVICE_ALLOCATOR_CUH

#include <cubool/instance.hpp>
#include <cubool/details/error.hpp>
#include <thrust/system/cuda/memory.h>

namespace cubool {
    namespace details {

        template<typename T>
        struct DeviceAllocator : thrust::cuda::allocator<T> {
        public:
            typedef thrust::cuda::allocator<T> super;
            typedef typename super::pointer pointer;
            typedef typename super::size_type size_type;

            __host__
            explicit DeviceAllocator(Instance &instance) : super(), mInstanceRef(instance) {}

            __host__
            DeviceAllocator(const DeviceAllocator<T> &other) : super(other), mInstanceRef(other.mInstanceRef) {}

            template <typename U>
            __host__
            explicit DeviceAllocator(DeviceAllocator<U> &&other) noexcept: super(std::move(other)), mInstanceRef(other.mInstance) {}

            DeviceAllocator& operator=(const DeviceAllocator<T> &other) {
                this->~DeviceAllocator();
                new (this) DeviceAllocator<T>(other);
                return *this;
            }

            __host__
            pointer allocate(size_type n) {
                CuBoolGpuPtr_t ptr = nullptr;
                mInstanceRef.allocateOnGpu(&ptr, n * sizeof(T));
                return pointer((T*)ptr);
            }

            __host__
            void deallocate(pointer p, size_type n) {
                (void)n;
                mInstanceRef.deallocateOnGpu(p.get());
            }

        private:
            Instance &mInstanceRef;
        };
    }
}

#endif //CUBOOL_DEVICE_ALLOCATOR_CUH
