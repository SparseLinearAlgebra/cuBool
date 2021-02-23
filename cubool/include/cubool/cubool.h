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

#ifdef __cplusplus
    #include <cinttypes>
#else
    #include <inttypes.h>
#endif

// Preserve C names in shared library
#define CUBOOL_EXPORT extern "C"

// Exporting/importing symbols for Microsoft Visual Studio
#if (_MSC_VER && !__INTEL_COMPILER)
    #ifdef CUBOOL_EXPORTS
        // Compile the library source code itself
        #define CUBOOL_API __declspec(dllexport)
    #else
        // Import symbols from library into user space
        #define CUBOOL_API __declspec(dllimport)
    #endif
#else
    // Default case
    #define CUBOOL_API
#endif

/** Possible status codes that can be returned from cubool api */
typedef enum CuBoolStatus {
    /** Successful execution of the function */
    CUBOOL_STATUS_SUCCESS = 0,
    /** Generic error code */
    CUBOOL_STATUS_ERROR = 1,
    /** No cuda compatible device in the system */
    CUBOOL_STATUS_DEVICE_NOT_PRESENT = 2,
    /** Device side error */
    CUBOOL_STATUS_DEVICE_ERROR = 3,
    /** Failed to allocate memory on cpy or gpu side */
    CUBOOL_STATUS_MEM_OP_FAILED = 4,
    /** Passed invalid argument to some function */
    CUBOOL_STATUS_INVALID_ARGUMENT = 5,
    /** Call of the function is not possible for some context */
    CUBOOL_STATUS_INVALID_STATE = 6,
    /** Some library feature is not implemented */
    CUBOOL_STATUS_NOT_IMPLEMENTED = 7
} CuBoolStatus;

/** Type of the GPU memory used to allocated gpu resources */
typedef enum CuBoolGpuMemoryType {
    /** Unified memory space */
    CUBOOL_GPU_MEMORY_TYPE_MANAGED = 0,
    /** Device only allocations */
    CUBOOL_GPU_MEMORY_TYPE_GENERIC = 1
} CuBoolGpuMemoryType;

/** Generic lib hits for matrix processing */
typedef enum CuBoolHint {
    /** No hints passed */
    CUBOOL_HINT_NO = 0x0,
    /** Mark input data as row-col sorted */
    CUBOOL_HINT_VALUES_SORTED = 0x1
} CuBoolHint;

/** Hit mask */
typedef uint32_t                    CuBoolHints;

/** Alias integer type for indexing operations */
typedef uint32_t                    CuBoolIndex;

/** Cubool dense boolean matrix handle */
typedef struct CuBoolMatrixDense_t* CuBoolMatrixDense;

/** Cubool sparse boolean matrix handle */
typedef struct CuBoolMatrix_t*      CuBoolMatrix;

/**
 * @brief Memory allocate callback
 * Signature for user-provided function pointer, used to allocate CPU memory for library resources
 */
typedef void* (*CuBoolCpuMemAllocateFun)(
    CuBoolIndex                 size,
    void*                       userData
);

/**
 * @brief Memory deallocate callback
 * Signature for user-provided function pointer, used to deallocate CPU memory, previously allocated with CuBoolCpuMemAllocateFun
 */
typedef void (*CuBoolCpuMemDeallocateFun)(
    void*                       ptr,
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

typedef struct CuBoolDeviceCaps {
    char                        name[256];
    int                         major;
    int                         minor;
    int                         warp;
    bool                        cudaSupported;
    CuBoolIndex                 globalMemoryKiBs;
    CuBoolIndex                 sharedMemoryPerMultiProcKiBs;
    CuBoolIndex                 sharedMemoryPerBlockKiBs;
} CuBoolDeviceCaps;

typedef struct CuBoolAllocationCallback {
    void*                       userData;
    CuBoolCpuMemAllocateFun     allocateFun;
    CuBoolCpuMemDeallocateFun   deallocateFun;
} CuBoolAllocationCallback;

typedef struct CuBoolMessageCallback {
    void*                       userData;
    CuBoolMsgFun                msgFun;
} CuBoolMessageCallback;

/**
 * Extension descriptor used for library setup within python wrapper.
 */
typedef struct CuBoolInstanceDescExt {
    CuBoolGpuMemoryType         memoryType;
} CuBoolInstanceDescExt;

/**
 * Query human-readable text info about the project implementation
 * @return Read-only library about info
 */
CUBOOL_EXPORT CUBOOL_API const char* cuBool_About_Get(
);

/**
 * Query human-readable text info about the project implementation
 * @return Read-only library license info
 */
CUBOOL_EXPORT CUBOOL_API const char* cuBool_LicenseInfo_Get(
);

/**
 * Query library version number in form MAJOR.MINOR
 *
 * @param major Major version number part
 * @param minor Minor version number part
 * @param version Composite integer version
 *
 * @return Error if failed to query version info
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Version_Get(
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
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_DeviceCaps_Get(
    CuBoolDeviceCaps*           deviceCaps
);

/**
 * Initialize library instance object, which provides context to all library operations and primitives.
 *
 * @param memoryType Type of the Cuda side memory
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Initialize(
    CuBoolGpuMemoryType         memoryType
);

/**
 * Destroy library instance and all objects, which were created on this library context.
 *
 * @note Invalidates all handle to the resources, created within this library instance
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Finalize(
);

/**
 * Synchronize host and associated to the instance device execution flows.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_HostDevice_Sync(
);

/**
 * Creates new sparse matrix with specified size.
 *
 * @param matrix Pointer where to store created matrix handle
 * @param nrows Matrix rows count
 * @param ncols Matrix columns count
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_New(
    CuBoolMatrix*               matrix,
    CuBoolIndex                 nrows,
    CuBoolIndex                 ncols
);

/**
 * Resize the sparse matrix. All previous values will be lost.
 *
 * @param matrix Matrix handle to perform operation on
 * @param nrows Matrix new rows count
 * @param ncols Matrix new columns count
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_Resize(
    CuBoolMatrix                matrix,
    CuBoolIndex                 nrows,
    CuBoolIndex                 ncols
);

/**
 * Build sparse matrix from provided pairs array. Pairs are supposed to be stored
 * as (rows[i],cols[i]) for pair with i-th index.
 *
 * @param matrix Matrix handle to perform operation on
 * @param rows Array of pairs row indices
 * @param cols Array of pairs column indices
 * @param nvals Number of the pairs passed
 * @param hints Hits flags for processing. Pass VALUES_SORTED if values already in the proper order.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_Build(
    CuBoolMatrix                matrix,
    const CuBoolIndex*          rows,
    const CuBoolIndex*          cols,
    CuBoolIndex                 nvals,
    CuBoolHints                 hints
);

/**
 * Reads matrix data to the host visible CPU buffer as an array of values pair.
 *
 * The arrays must be provided by the user and the size of this arrays must
 * be greater or equal the values count of the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param[in,out] rows Buffer to store row indices
 * @param[in,out] cols Buffer to store column indices
 * @param[in,out] nvals Total number of the pairs
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_ExtractPairs(
    CuBoolMatrix                matrix,
    CuBoolIndex*                rows,
    CuBoolIndex*                cols,
    CuBoolIndex*                nvals
);

/**
 * Creates new sparse matrix, duplicates content and stores handle in the provided pointer.
 * 
 * @param matrix Matrix handle to perform operation on
 * @param duplicated[out] Pointer to the matrix handle where to store created matrix
 * 
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_Duplicate(
    CuBoolMatrix                matrix,
    CuBoolMatrix*               duplicated
);

/**
 * Transpose source matrix and store result of this operation in result matrix.
 * Formally: result = matrix ^ T
 *
 * @param result Matrix handle to store result of the operation
 * @param matrix The source matrix
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_Transpose(
    CuBoolMatrix                result,
    CuBoolMatrix                matrix
);

/**
 *
 * @param matrix Matrix handle to perform operation on
 * @param nvals[out] Pointer to  the place where to store number of the non-zero elements of the matrix
 * 
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_Nvals(
    CuBoolMatrix                matrix,
    CuBoolIndex*                nvals
);

/**
 * 
 * @param matrix Matrix handle to perform operation on
 * @param nrows[out] Pointer to the place where to store number of matrix rows
 * 
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_Nrows(
    CuBoolMatrix                matrix,
    CuBoolIndex*                nrows
);

/**
 * 
 * @param matrix Matrix handle to perform operation on
 * @param ncols[out] Pointer to the place where to store number of matrix columns
 * 
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_Ncols(
    CuBoolMatrix                matrix,
    CuBoolIndex*                ncols
);

/**
 * Deletes sparse matrix object.
 *
 * @param matrix Matrix handle to delete the matrix
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_Free(
    CuBoolMatrix                matrix
);

/**
 * Performs result += left, where '+' is boolean semiring operation.
 *
 * @note Matrices must be compatible
 *      dim(result) = M x N
 *      dim(left) = M x N
 *
 * @param result Destination matrix for add-and-assign operation
 * @param left Source matrix to be added
 * @param right Source matrix to be added
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Matrix_EWiseAdd(
    CuBoolMatrix                result,
    CuBoolMatrix                left,
    CuBoolMatrix                right
);

/**
 * Performs result = left x right evaluation, where source '+' and 'x' are boolean semiring operations.
 *
 * @note to perform this operation matrices must be compatible
 *       dim(left) = M x T
 *       dim(right) = T x N
 *       dim(result) = M x N
 *
 * @param result Matrix handle where to store operation result
 * @param left Input left matrix
 * @param right Input right matrix
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_MxM(
    CuBoolMatrix                result,
    CuBoolMatrix                left,
    CuBoolMatrix                right
);

/**
 * Performs result = left `kron` right, where `kron` is a Kronecker product for boolean semiring.
 *
 * @note when the operation is performed, the result matrix has the following dimension
 *      dim(left) = M x N
 *      dim(right) = K x T
 *      dim(result) = MK x NT
 *
 * @param result Matrix handle where to store operation result
 * @param left Input left matrix
 * @param right Input right matrix
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Kronecker(
    CuBoolMatrix                result,
    CuBoolMatrix                left,
    CuBoolMatrix                right
);

/**
 * Release values array buffer, allocated by one of *ReadData operations.
 *
 * @param vals Valid pointer to returned arrays buffer from *ReadData method
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API CuBoolStatus cuBool_Vals_Free(
    CuBoolIndex*                vals
);

#endif //CUBOOL_CUBOOL_H
