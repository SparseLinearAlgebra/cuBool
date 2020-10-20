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

#include <cubool/utils/cpu_buffer.hpp>
#include <cubool/instance.hpp>

#include <memory>

namespace cubool {

    CpuBuffer::CpuBuffer(class Instance &instance) {
        mInstancePtr = &instance;
    }

    CpuBuffer::CpuBuffer(const CpuBuffer &other) {
        mInstancePtr = other.mInstancePtr;
        resizeNoContentKeep(other.mSize);
        copy(other.mMemory, other.mSize, 0);
    }

    CpuBuffer::CpuBuffer(CpuBuffer &&other) noexcept {
        mInstancePtr = other.mInstancePtr;
        mMemory = other.mMemory;
        mSize = other.mSize;

        other.mMemory = nullptr;
        other.mSize = 0;
    }

    CpuBuffer::~CpuBuffer() {
        resizeNoContentKeep(0);
    }

    CuBoolStatus CpuBuffer::resizeNoContentKeep(CuBoolSize_t size) {
        if (isNotEmpty()) {
            CuBoolStatus error = mInstancePtr->deallocate(mMemory);
            mMemory = nullptr;
            mSize = 0;

            if (error != CUBOOL_STATUS_SUCCESS) {
                return error;
            }
        }

        if (size > 0) {
            mSize = size;
            return mInstancePtr->allocate(&mMemory, mSize);
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus CpuBuffer::copy(CuBoolGpuConstPtr_t source, CuBoolSize_t size, CuBoolSize_t writeOffset) {
        if (size == 0) {
            return CUBOOL_STATUS_SUCCESS;
        }

        if (!source) {
            mInstancePtr->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Null pointer to copy passed");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        if (size + writeOffset > mSize) {
            mInstancePtr->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Write region out of buffer bounds");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        CuBoolCpuPtr_t destination = ((uint8_t*)mMemory) + writeOffset;
        memcpy(destination, source, size);

        return CUBOOL_STATUS_SUCCESS;
    }

}
