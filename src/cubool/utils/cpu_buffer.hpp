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

#ifndef CUBOOL_CPU_BUFFER_HPP
#define CUBOOL_CPU_BUFFER_HPP

#include <cubool/cubool_types.h>

namespace cubool {

    /** Cpu buffer wrapper to provide safe destruction logic */
    class CpuBuffer {
    public:
        explicit CpuBuffer(class Instance& instance);
        CpuBuffer(const CpuBuffer& other);
        CpuBuffer(CpuBuffer&& other) noexcept;
        ~CpuBuffer();

        CuBoolStatus resizeNoContentKeep(CuBoolSize_t size);
        CuBoolStatus copy(CuBoolGpuConstPtr_t source, CuBoolSize_t size, CuBoolSize_t writeOffset);

        bool isEmpty() const { return mSize == 0; }
        bool isNotEmpty() const { return mSize > 0; }

        CuBoolCpuConstPtr_t getMemory() const { return mMemory; }
        CuBoolCpuPtr_t getMemory() { return mMemory; }
        CuBoolSize_t getSize() const { return mSize; }

    private:
        CuBoolSize_t mSize = 0;
        CuBoolCpuPtr_t mMemory = nullptr;
        class Instance* mInstancePtr = nullptr;
    };

}

#endif //CUBOOL_CPU_BUFFER_HPP