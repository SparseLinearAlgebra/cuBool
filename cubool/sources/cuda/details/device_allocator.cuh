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

#ifndef CUBOOL_DEVICE_ALLOCATOR_CUH
#define CUBOOL_DEVICE_ALLOCATOR_CUH

#include <cuda/cuda_instance.hpp>
#include <core/error.hpp>
#include <thrust/system/cuda/memory.h>
#include <thrust/device_malloc_allocator.h>

namespace cubool {
    namespace details {

        /**
         * Device allocator, which uses instance reference for unified memory
         * resources access. The instance reference is required to modify in the
         * runtime the type of the used memory: device or managed.
         *
         * @tparam T Type of the elements allocated by this allocator
         */
        template<typename T>
        struct DeviceAllocator {
        public:
            typedef thrust::device_malloc_allocator<T> super;
            typedef typename super::value_type value_type;
            typedef typename super::size_type size_type;
            typedef typename super::pointer pointer;
            typedef typename super::const_pointer const_pointer;
            typedef typename super::reference reference;
            typedef typename super::const_reference const_reference;

            template <class U>
            struct rebind { typedef DeviceAllocator<U> other; };

            __host__ explicit DeviceAllocator() : mInstanceRef(CudaInstance::getInstanceRef()) { }

            __host__ DeviceAllocator(const DeviceAllocator<T> &other) : mInstanceRef(other.mInstanceRef) { }

            template <typename U>
            __host__ explicit DeviceAllocator(const DeviceAllocator<U> &other) : mInstanceRef(other.mInstance) { }

            ~DeviceAllocator() = default;

            __host__ bool operator!=(const DeviceAllocator<T> &other) const {
                return &mInstanceRef != &other.mInstanceRef;
            }

            __host__ DeviceAllocator& operator=(const DeviceAllocator<T> &other) {
                this->~DeviceAllocator();
                new (this) DeviceAllocator<T>(other);
                return *this;
            }

            __host__ pointer allocate(size_type n) {
                void* ptr = nullptr;
                mInstanceRef.allocateOnGpu(ptr, n * sizeof(T));
                return pointer((T*)ptr);
            }

            __host__ void deallocate(pointer p, size_type n) {
                (void)n;
                mInstanceRef.deallocateOnGpu(p.get());
            }

        private:
            CudaInstance &mInstanceRef;
        };
    }
}

#endif //CUBOOL_DEVICE_ALLOCATOR_CUH
