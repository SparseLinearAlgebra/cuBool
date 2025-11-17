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

#ifndef CUBOOL_HOST_ALLOCATOR_HPP
#define CUBOOL_HOST_ALLOCATOR_HPP

#include <core/error.hpp>
#include <cuda/cuda_instance.hpp>

namespace cubool {
    namespace details {

        template<class T>
        class HostAllocator {
        public:
            typedef size_t   size_type;
            typedef T        value_type;
            typedef T*       pointer;
            typedef const T* const_pointer;
            typedef T&       reference;
            typedef const T& const_reference;

            template<class U>
            struct rebind {
                typedef HostAllocator<U> other;
            };

            explicit HostAllocator() : mInstanceRef(CudaInstance::getInstanceRef()) {
            }

            HostAllocator(const HostAllocator<T>& other) : mInstanceRef(other.mInstanceRef) {
            }

            HostAllocator(HostAllocator<T>&& other) noexcept : mInstanceRef(other.mInstanceRef) {
            }

            ~HostAllocator() = default;

            bool operator!=(const HostAllocator<T>& other) const {
                return &mInstanceRef != &other.mInstanceRef;
            }

            HostAllocator& operator=(const HostAllocator<T>& other) {
                this->~HostAllocator();
                new (this) HostAllocator<T>(other);
                return *this;
            }

            pointer allocate(size_type n) {
                void* ptr = nullptr;
                mInstanceRef.allocate(ptr, n * sizeof(T));
                return (pointer) ptr;
            }

            void deallocate(pointer p, size_type n) {
                (void) n;
                mInstanceRef.deallocate(p);
            }

        private:
            CudaInstance& mInstanceRef;
        };

    }// namespace details
}// namespace cubool

#endif//CUBOOL_HOST_ALLOCATOR_HPP
