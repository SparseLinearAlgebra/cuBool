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

#ifndef CUBOOL_CUBOOL_TYPES_H
#define CUBOOL_CUBOOL_TYPES_H

#include <memory.h>

/** Possible status codes that can be returned from cubool api */
typedef enum CuBoolStatus {
    CUBOOL_STATUS_SUCCESS,
    CUBOOL_STATUS_ERROR,
    CUBOOL_STATUS_DEVICE_NOT_PRESENT,
    CUBOOL_STATUS_DEVICE_ERROR,
    CUBOOL_STATUS_MEM_OP_FAILED,
    CUBOOL_STATUS_INVALID_ARGUMENT,
    CUBOOL_STATUS_INVALID_STATE,
    CUBOOL_STATUS_NOT_IMPLEMENTED
} CuBoolStatus;

/** Type of the GPU memory used to allocated gpu resources */
typedef enum CuBoolGpuMemoryType {
    CUBOOL_GPU_MEMORY_TYPE_MANAGED,
    CUBOOL_GPU_MEMORY_TYPE_GENERIC
} CuBoolGpuMemoryType;

typedef enum CuBoolMajorOrder {
    CUBOOL_MAJOR_ORDER_ROW,
    CUBOOL_MAJOR_ORDER_COLUMN,
} CuBoolMajorOrder;

/** Alias size type for memory and indexing types */
typedef size_t CuBoolSize_t;

/** Alias cpu (ram) memory pointer */
typedef void* CuBoolCpuPtr_t;

/** Alias cpu (ram) const memory pointer */
typedef const void* CuBoolCpuConstPtr_t;

/** Alias gpu (vram or managed) memory pointer */
typedef void* CuBoolGpuPtr_t;

/** Alias gpu (vram or managed) const memory pointer */
typedef const void* CuBoolGpuConstPtr_t;

/** Cubool library instance handler */
typedef struct CuBoolInstance_t*    CuBoolInstance;

/** Cubool dense boolean matrix handler */
typedef struct CuBoolMatrixDense_t* CuBoolMatrixDense;

/**
 * @brief Memory allocate callback
 * Signature for user-provided function pointer, used to allocate CPU memory for library resources
 */
typedef CuBoolCpuPtr_t (*CuBoolCpuMemAllocateFun)(CuBoolSize_t size, void* userData);

/**
 * @brief Memory deallocate callback
 * Signature for user-provided function pointer, used to deallocate CPU memory, previously allocated with CuBoolCpuMemAllocateFun
 */
typedef void (*CuBoolCpuMemDeallocateFun)(CuBoolCpuPtr_t ptr, void* userData);

/**
 * @brief Message callback
 * User provided message callback to observe library messages and errors
 */
typedef void (*CuBoolMsgFun)(CuBoolStatus status, const char* message, void* userData);

/** Pair of the indices used to represent non-empty matrices values */
typedef struct CuBoolPair {
    CuBoolSize_t i;
    CuBoolSize_t j;
} CuBoolPair;

typedef struct CuBoolDeviceCaps {
    char name[256];
    int major;
    int minor;
    int warp;
    bool cudaSupported;
    CuBoolSize_t globalMemoryKiBs;
    CuBoolSize_t sharedMemoryPerMultiProcKiBs;
    CuBoolSize_t sharedMemoryPerBlockKiBs;
} CuBoolDeviceCaps;

typedef struct CuBoolGpuAllocation {
    CuBoolGpuMemoryType memoryType;
    CuBoolGpuPtr_t memoryPtr;
    CuBoolSize_t size;
} CuBoolGpuAllocation;

typedef struct CuBoolAllocationCallback {
    void* userData;
    CuBoolCpuMemAllocateFun allocateFun;
    CuBoolCpuMemDeallocateFun deallocateFun;
} CuAllocationCallback;

typedef struct CuBoolMessageCallback {
    void* userData;
    CuBoolMsgFun msgFun;
} CuBoolMessageCallback;

typedef struct CuBoolInstanceDesc {
    CuBoolGpuMemoryType memoryType;
    CuBoolMessageCallback errorCallback;
    CuAllocationCallback allocationCallback;
} CuBoolInstanceDesc;

#endif //CUBOOL_CUBOOL_TYPES_H
