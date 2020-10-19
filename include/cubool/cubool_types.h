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

typedef size_t      CuBoolSize_t;
typedef void*       CuBoolCpuPtr_t;
typedef const void* CuBoolCpuConstPtr_t;
typedef void*       CuBoolGpuPtr_t;
typedef const void* CuBoolGpuConstPtr_t;

typedef enum CuBoolError {
    CUBOOL_ERROR_SUCCESS,
    CUBOOL_ERROR_ERROR,
    CUBOOL_ERROR_MEM_OP_FAILED,
    CUBOOL_ERROR_INVALID_ARGUMENT,
    CUBOOL_ERROR_INVALID_STATE,
    CUBOOL_ERROR_NOT_IMPLEMENTED
} CuBoolStatus;

typedef enum CuBoolGpuMemoryType {
    CUBOOL_GPU_MEMORY_TYPE_MANAGED,
    CUBOOL_GPU_MEMORY_TYPE_GENERIC
} CuBoolGpuMemoryType;

typedef enum CuBoolMajorOrder {
    CUBOOL_MAJOR_ORDER_ROW,
    CUBOOL_MAJOR_ORDER_COLUMN,
} CuBoolMajorOrder;

typedef struct CuBoolGpuAllocation {
    CuBoolGpuMemoryType memoryType;
    CuBoolGpuPtr_t memoryPtr;
    CuBoolSize_t size;
} CuBoolGpuAllocation;

typedef CuBoolCpuPtr_t  (*CuBoolCpuMemAllocateFun)(CuBoolSize_t size, void* userData);
typedef void            (*CuBoolCpuMemDeallocateFun)(CuBoolCpuPtr_t ptr, void* userData);
typedef void            (*CuBoolErrorMsgFun)(CuBoolError status, const char* message, void* userData);

typedef struct CuBoolAllocationCallback {
    void* userData;
    CuBoolCpuMemAllocateFun allocateFun;
    CuBoolCpuMemDeallocateFun deallocateFun;
} CuAllocationCallback;

typedef struct CuBoolErrorCallback {
    void* userData;
    CuBoolErrorMsgFun errorMsgFun;
} CuBoolErrorCallback;

typedef struct CuBoolInstanceDesc {
    CuBoolGpuMemoryType memoryType;
    CuBoolErrorCallback errorCallback;
    CuAllocationCallback allocationCallback;
} CuBoolInstanceDesc;

typedef struct CuBoolInstance_t*    CuBoolInstance;
typedef struct CuBoolMatrixDense_t* CuBoolMatrixDense;

#endif //CUBOOL_CUBOOL_TYPES_H
