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

#include <cubool/utils/gpu_buffer.hpp>
#include <cubool/instance.hpp>

#include <string>

namespace cubool {

    GpuBuffer::GpuBuffer(class Instance &instance) {
        mInstancePtr = &instance;
    }

    GpuBuffer::GpuBuffer(const GpuBuffer &other) {
        mInstancePtr = other.mInstancePtr;
        resizeNoContentKeep(other.mSize);
        copy(other.mMemory, other.mSize, 0);
    }

    GpuBuffer::GpuBuffer(GpuBuffer &&other) noexcept {
        mInstancePtr = other.mInstancePtr;
        mMemory = other.mMemory;
        mSize = other.mSize;

        other.mMemory = nullptr;
        other.mSize = 0;
    }

    GpuBuffer::~GpuBuffer() {
        resizeNoContentKeep(0);
    }

    CuBoolStatus GpuBuffer::resizeNoContentKeep(CuBoolSize_t size) {
        if (isNotEmpty()) {
            CuBoolStatus error = mInstancePtr->deallocateOnGpu(mMemory);
            mSize = 0;
            mMemory = nullptr;

            if (error != CUBOOL_STATUS_SUCCESS) {
                return error;
            }
        }

        if (size > 0) {
            mSize = size;
            return mInstancePtr->allocateOnGpu(&mMemory, mSize);
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus GpuBuffer::copy(CuBoolGpuConstPtr_t source, CuBoolSize_t size, CuBoolSize_t writeOffset) {
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

        CuBoolGpuPtr_t destination = ((uint8_t*)mMemory) + writeOffset;
        cudaError error = cudaMemcpy(destination, source, size, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

        if (error != cudaSuccess) {
            std::string message = "Failed to copy gpu buffers data: ";
            message += cudaGetErrorString(error);

            mInstancePtr->sendMessage(CUBOOL_STATUS_MEM_OP_FAILED, message.c_str());
            return CUBOOL_STATUS_MEM_OP_FAILED;
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus GpuBuffer::transferFromCpu(CuBoolCpuConstPtr_t source, CuBoolSize_t size, CuBoolSize_t writeOffset) {
        if (size == 0) {
            return CUBOOL_STATUS_SUCCESS;
        }

        if (!source) {
            mInstancePtr->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Null pointer to transfer passed");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        if (size + writeOffset > mSize) {
            mInstancePtr->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Write region out of buffer bounds");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        CuBoolGpuPtr_t destination = ((uint8_t*)mMemory) + writeOffset;
        cudaError error = cudaMemcpy(destination, source, size, cudaMemcpyKind::cudaMemcpyHostToDevice);

        if (error != cudaSuccess) {
            std::string message = "Failed to transfer data from cpu to gpu buffer: ";
            message += cudaGetErrorString(error);

            mInstancePtr->sendMessage(CUBOOL_STATUS_MEM_OP_FAILED, message.c_str());
            return CUBOOL_STATUS_MEM_OP_FAILED;
        }

        return CUBOOL_STATUS_SUCCESS;
    }

    CuBoolStatus GpuBuffer::transferToCpu(CuBoolCpuPtr_t destination, CuBoolSize_t size, CuBoolSize_t readOffset) const {
        if (size == 0) {
            return CUBOOL_STATUS_SUCCESS;
        }

        if (!destination) {
            mInstancePtr->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Null pointer to transfer to passed");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        if (readOffset + size > mSize) {
            mInstancePtr->sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Copy region out of buffer bounds");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        CuBoolGpuConstPtr_t source = ((const uint8_t*)mMemory) + readOffset;
        cudaError error = cudaMemcpy(destination, source, size, cudaMemcpyKind::cudaMemcpyDeviceToHost);

        if (error != cudaSuccess) {
            std::string message = "Failed to transfer data from gpu buffer to cpu: ";
            message += cudaGetErrorString(error);

            mInstancePtr->sendMessage(CUBOOL_STATUS_MEM_OP_FAILED, message.c_str());
            return CUBOOL_STATUS_MEM_OP_FAILED;
        }

        return CUBOOL_STATUS_SUCCESS;
    }

}
