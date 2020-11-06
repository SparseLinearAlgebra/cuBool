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

#ifndef CUBOOL_CUBOOL_H
#define CUBOOL_CUBOOL_H

#include <memory.h>

/** Export library function (does not handles msvc or intel compilers) */
#define CuBoolAPI extern "C"

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
typedef size_t                      CuBoolSize_t;

/** Alias cpu (ram) memory pointer */
typedef void*                       CuBoolCpuPtr_t;

/** Alias cpu (ram) const memory pointer */
typedef const void*                 CuBoolCpuConstPtr_t;

/** Alias gpu (vram or managed) memory pointer */
typedef void*                       CuBoolGpuPtr_t;

/** Alias gpu (vram or managed) const memory pointer */
typedef const void*                 CuBoolGpuConstPtr_t;

/** Cubool library instance handler */
typedef struct CuBoolInstance_t*    CuBoolInstance;

/** Cubool dense boolean matrix handler */
typedef struct CuBoolMatrixDense_t* CuBoolMatrixDense;

/**
 * @brief Memory allocate callback
 * Signature for user-provided function pointer, used to allocate CPU memory for library resources
 */
typedef CuBoolCpuPtr_t (*CuBoolCpuMemAllocateFun)(
    CuBoolSize_t                size,
    void*                       userData
);

/**
 * @brief Memory deallocate callback
 * Signature for user-provided function pointer, used to deallocate CPU memory, previously allocated with CuBoolCpuMemAllocateFun
 */
typedef void (*CuBoolCpuMemDeallocateFun)(
    CuBoolCpuPtr_t              ptr,
    void*                       userData
);

/**
 * @brief Message callback
 * User provided message callback to observe library messages and errors
 */
typedef void (*CuBoolMsgFun)(
    CuBoolStatus                status,
    const char*                 message,
    void*                       userData
);

/** Pair of the indices used to represent non-empty matrices values */
typedef struct CuBoolPair {
    CuBoolSize_t                i;
    CuBoolSize_t                j;
} CuBoolPair;

typedef struct CuBoolDeviceCaps {
    char                        name[256];
    int                         major;
    int                         minor;
    int                         warp;
    bool                        cudaSupported;
    CuBoolSize_t                globalMemoryKiBs;
    CuBoolSize_t                sharedMemoryPerMultiProcKiBs;
    CuBoolSize_t                sharedMemoryPerBlockKiBs;
} CuBoolDeviceCaps;

typedef struct CuBoolGpuAllocation {
    CuBoolGpuMemoryType         memoryType;
    CuBoolGpuPtr_t              memoryPtr;
    CuBoolSize_t                size;
} CuBoolGpuAllocation;

typedef struct CuBoolAllocationCallback {
    void*                       userData;
    CuBoolCpuMemAllocateFun     allocateFun;
    CuBoolCpuMemDeallocateFun   deallocateFun;
} CuBoolAllocationCallback;

typedef struct CuBoolMessageCallback {
    void*                       userData;
    CuBoolMsgFun                msgFun;
} CuBoolMessageCallback;

typedef struct CuBoolInstanceDesc {
    CuBoolGpuMemoryType         memoryType;
    CuBoolMessageCallback       errorCallback;
    CuBoolAllocationCallback    allocationCallback;
} CuBoolInstanceDesc;

/**
 * Query human-readable text info about the project implementation
 * @return Read-only library about info
 */
CuBoolAPI const char* CuBool_About_Get();

/**
 * Query human-readable text info about the project implementation
 * @return Read-only library license info
 */
CuBoolAPI const char* CuBool_LicenseInfo_Get();

/**
 * Query library version number in form MAJOR.MINOR
 *
 * @param major Major version number part
 * @param minor Minor version number part
 * @param version Composite integer version
 *
 * @return Error if failed to query version info
 */
CuBoolAPI CuBoolStatus CuBool_Version_Get(
    int*                        major,
    int*                        minor,
    int*                        version
);

/**
 * Query device capabilities/properties if cuda compatible device is present
 *
 * @param deviceCaps Pointer to device caps structure to store result
 *
 * @return Error if cuda device not present or if failed to query capabilities
 */
CuBoolAPI CuBoolStatus CuBool_DeviceCaps_Get(
    CuBoolDeviceCaps*           deviceCaps
);

/**
 * Initialize library instance object, which provides context to all library operations and objects
 *
 * @param instanceDesc User provided instance configuration for memory operations and error handling
 * @param instance Pointer to the place where to store instance handler
 *
 * @return Error code on this operations
 */
CuBoolAPI CuBoolStatus CuBool_Instance_New(
    const CuBoolInstanceDesc*   instanceDesc,
    CuBoolInstance*             instance
);

/**
 * Destroy library instance and all objects, which were created on this library context.
 *
 * @note Invalidates all handler to the resources, created within this library instance
 * @param instance An instance object reference to perform this operation
 *
 * @return Error code on this operations
 */
CuBoolAPI CuBoolStatus CuBool_Instance_Delete(
    CuBoolInstance              instance
);

/**
 *
 * @param instance
 * @return
 */
CuBoolAPI CuBoolStatus CuBool_SyncHostDevice(
    CuBoolInstance              instance
);

/**
 *
 * @param instance
 * @param matrix
 * @param nrows
 * @param ncols
 * @return
 */
CuBoolAPI CuBoolStatus CuBool_MatrixDense_New(
    CuBoolInstance              instance,
    CuBoolMatrixDense*          matrix,
    CuBoolSize_t                nrows,
    CuBoolSize_t                ncols
);

/**
 *
 * @param instance
 * @param matrix
 * @return
 */
CuBoolAPI CuBoolStatus CuBool_MatrixDense_New_(
    CuBoolInstance              instance,
    CuBoolMatrixDense*          matrix
);

/**
 *
 * @param instance
 * @param matrix
 * @return
 */
CuBoolAPI CuBoolStatus CuBool_MatrixDense_Delete(
    CuBoolInstance              instance,
    CuBoolMatrixDense           matrix
);

/**
 *
 * @param instance
 * @param matrix
 * @param nrows
 * @param ncols
 * @return
 */
CuBoolAPI CuBoolStatus CuBool_MatrixDense_Resize (
    CuBoolInstance              instance,
    CuBoolMatrixDense           matrix,
    CuBoolSize_t                nrows,
    CuBoolSize_t                ncols
);

/**
 *
 * @param instance
 * @param matrix
 * @param rows
 * @param cols
 * @param nvals
 * @return
 */
CuBoolAPI CuBoolStatus CuBool_MatrixDense_Build (
    CuBoolInstance              instance,
    CuBoolMatrixDense           matrix,
    const CuBoolSize_t*         rows,
    const CuBoolSize_t*         cols,
    CuBoolSize_t                nvals
);

/**
 *
 * @param instance
 * @param matrix
 * @param count
 * @param values
 * @return
 */
CuBoolAPI CuBoolStatus CuBool_MatrixDense_Build_ (
    CuBoolInstance              instance,
    CuBoolMatrixDense           matrix,
    CuBoolSize_t                count,
    const CuBoolPair*           values
);

/**
 * Reads matrix data to the host visible CPU buffer as an array of values pair
 *
 * @note Returned pointer to the allocated values buffers must be explicitly freed by the user
 *       via function call CuBool_Vals_Delete
 *
 * @param instance An instance object reference to perform this operation
 * @param matrix Matrix handler to perform operation on
 * @param[out] count Number of pairs in the returned array
 * @param[out] values Allocated buffer to the pairs indices. Buffer must be freed after usage
 *
 * @return Error code on this operations
 */
CuBoolAPI CuBoolStatus CuBool_MatrixDense_ExtractPairs (
    CuBoolInstance              instance,
    CuBoolMatrixDense           matrix,
    CuBoolSize_t*               count,
    CuBoolPair**                values
);

/**
 * Performs result = a x b + c evaluation, where '+' and 'x' are boolean semiring operations
 *
 * @note to perform this operation matrices must be compatible
 *       dim(a) = M x T
 *       dim(b) = T x N
 *       dim(c) = M x N
 *
 * @note result matrix will be automatically properly resized to store operation result
 *
 * @param instance An instance object reference to perform this operation
 * @param result Matrix handler where to store operation result
 * @param a Input a matrix
 * @param b Input a matrix
 * @param c Input a matrix
 *
 * @return Error code on this operations
 */
CuBoolAPI CuBoolStatus CuBool_MatrixDense_MultAdd (
    CuBoolInstance              instance,
    CuBoolMatrixDense           res,
    CuBoolMatrixDense           a,
    CuBoolMatrixDense           b,
    CuBoolMatrixDense           c
);

/**
 * Release values array buffer, allocated by one of *ReadData operations
 *
 * @param instance An instance object reference to perform this operation
 * @param values Valid pointer to returned arrays buffer from *ReadData method
 *
 * @return Error code on this operations
 */
CuBoolAPI CuBoolStatus CuBool_Vals_Delete (
    CuBoolInstance              instance,
    CuBoolPair*                 values
);

#endif //CUBOOL_CUBOOL_H
