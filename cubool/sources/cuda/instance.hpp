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

#ifndef CUBOOL_INSTANCE_HPP
#define CUBOOL_INSTANCE_HPP

#include <core/config.hpp>
#include <unordered_set>

namespace cubool {

    /**
     * Manages global state for various internal operations.
     */
    class Instance {
    public:
        enum MemType {
            Default,
            Managed
        };

        explicit Instance(bool useManagedMemory);
        Instance(const Instance& other) = delete;
        Instance(Instance&& other) noexcept = delete;
        ~Instance();

        // For custom host & device allocators
        void allocate(void* &ptr, size_t s) const;
        void allocateOnGpu(void* &ptr, size_t s) const;
        void deallocate(void* ptr) const;
        void deallocateOnGpu(void* ptr) const;

        void syncHostDevice() const;
        void printDeviceCapabilities() const;

        static bool isCudaDeviceSupported();
        static void queryDeviceCapabilities(cuBool_DeviceCaps& deviceCaps);
        static Instance& getInstanceRef();
        static Instance* getInstancePtr();
        static bool isInstancePresent();

    private:
        MemType mMemoryType = Default;

        static volatile Instance* gInstance;
    };

}

#endif //CUBOOL_INSTANCE_HPP