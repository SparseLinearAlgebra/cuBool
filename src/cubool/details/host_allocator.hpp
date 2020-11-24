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

#ifndef CUBOOL_HOST_ALLOCATOR_HPP
#define CUBOOL_HOST_ALLOCATOR_HPP

#include <cubool/instance.hpp>
#include <cubool/details/error.hpp>

namespace cubool {
    namespace details {

        template <class T>
        class HostAllocator {
        public:
            typedef std::allocator<T> super;
            typedef typename super::pointer pointer;
            typedef typename super::const_pointer const_pointer;
            typedef typename super::size_type size_type;
            typedef typename super::value_type value_type;
            typedef typename super::reference reference;
            typedef typename super::const_reference const_reference;

            template <class U>
            struct rebind { typedef HostAllocator<U> other; };

            explicit HostAllocator(Instance& instance): mInstanceRef(instance) {

            }

            HostAllocator(const HostAllocator<T> &other): mInstanceRef(other.mInstanceRef) {

            }

            HostAllocator(HostAllocator<T> &&other) noexcept: mInstanceRef(other.mInstanceRef) {

            }

            ~HostAllocator() = default;

            bool operator!=(const HostAllocator<T> &other) const {
                return &mInstanceRef != &other.mInstanceRef;
            }

            HostAllocator& operator=(const HostAllocator<T> &other) {
                this->~HostAllocator();
                new (this) HostAllocator<T>(other);
                return *this;
            }

            pointer allocate(size_type n) {
                CuBoolCpuPtr_t ptr = nullptr;
                mInstanceRef.allocate(&ptr, n * sizeof(T));
                return (pointer) ptr;
            }

            void deallocate(pointer p, size_type n) {
                (void)n;
                mInstanceRef.deallocate(p);
            }

        private:
            Instance& mInstanceRef;
        };

    }
}

#endif //CUBOOL_HOST_ALLOCATOR_HPP
