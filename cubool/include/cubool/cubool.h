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
typedef enum cuBool_Status {
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
    /** Failed to select supported backend for computations */
    CUBOOL_STATUS_BACKEND_ERROR = 7,
    /** Some library feature is not implemented */
    CUBOOL_STATUS_NOT_IMPLEMENTED = 8
} cuBool_Status;

/** Generic lib hits for matrix processing */
typedef enum cuBool_Hint {
    /** No hints passed */
    CUBOOL_HINT_NO = 0x0,
    /** Force Cpu based backend usage */
    CUBOOL_HINT_CPU_BACKEND = 0x1,
    /** Use managed gpu memory type instead of default (device) memory */
    CUBOOL_HINT_GPU_MEM_MANAGED = 0x2,
    /** Mark input data as row-col sorted */
    CUBOOL_HINT_VALUES_SORTED = 0x4,
    /** Accumulate result of the operation in the result matrix */
    CUBOOL_HINT_ACCUMULATE = 0x8,
    /** Finalize library state, even if not all resources were explicitly released */
    CUBOOL_HINT_RELAXED_FINALIZE = 0x16
} cuBool_Hint;

/** Hit mask */
typedef uint32_t cuBool_Hints;

/** Alias integer type for indexing operations */
typedef uint32_t cuBool_Index;

/** Cubool sparse boolean matrix handle */
typedef struct cuBoolMatrix_t* cuBool_Matrix;

/** Cuda device capabilities */
typedef struct CuBool_DeviceCaps {
    char name[256];
    int major;
    int minor;
    int warp;
    bool cudaSupported;
    cuBool_Index globalMemoryKiBs;
    cuBool_Index sharedMemoryPerMultiProcKiBs;
    cuBool_Index sharedMemoryPerBlockKiBs;
} CuBool_DeviceCaps;

/**
 * Query human-readable text info about the project implementation
 * @note It is safe to call this function before the library is initialized.
 *
 * @return Read-only library about info
 */
CUBOOL_EXPORT CUBOOL_API const char* cuBool_About_Get(
);

/**
 * Query human-readable text info about the project implementation
 * @note It is safe to call this function before the library is initialized.

 * @return Read-only library license info
 */
CUBOOL_EXPORT CUBOOL_API const char* cuBool_LicenseInfo_Get(
);

/**
 * Query library version number in form MAJOR.MINOR
 * @note It is safe to call this function before the library is initialized.
 *
 * @param major Major version number part
 * @param minor Minor version number part
 * @param version Composite integer version
 *
 * @return Error if failed to query version info
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Version_Get(
    int* major,
    int* minor,
    int* version
);

/**
 * Initialize library instance object, which provides context to all library operations and primitives.
 * This function must be called before any other library function is called,
 * except first get-info functions.
 *
 * @note Pass CUBOOL_HINT_RELAXED_FINALIZE for library setup within python.
 *
 * @param hints Init hints.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Initialize(
    cuBool_Hints hints
);

/**
 * Finalize library state and all objects, which were created on this library context.
 * This function always must be called as the last library function in the application.
 *
 * @note Pass CUBOOL_HINT_RELAXED_FINALIZE for library init call, if relaxed finalize is required.
 * @note Invalidates all handle to the resources, created within this library instance
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Finalize(
);

/**
 * Query device capabilities/properties if cuda compatible device is present.
 *
 * @note This function must be called only for cuda backend.
 * @param deviceCaps Pointer to device caps structure to store result
 *
 * @return Error if cuda device not present or if failed to query capabilities
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_DeviceCaps_Get(
        CuBool_DeviceCaps* deviceCaps
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
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_New(
    cuBool_Matrix* matrix,
    cuBool_Index nrows,
    cuBool_Index ncols
);

/**
 * Build sparse matrix from provided pairs array. Pairs are supposed to be stored
 * as (rows[i],cols[i]) for pair with i-th index.
 *
 * @note Pass CUBOOL_HINT_VALUES_SORTED if values already in the row-col order.
 *
 * @param matrix Matrix handle to perform operation on
 * @param rows Array of pairs row indices
 * @param cols Array of pairs column indices
 * @param nvals Number of the pairs passed
 * @param hints Hits flags for processing.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Build(
    cuBool_Matrix matrix,
    const cuBool_Index* rows,
    const cuBool_Index* cols,
    cuBool_Index nvals,
    cuBool_Hints hints
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
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_ExtractPairs(
    cuBool_Matrix matrix,
    cuBool_Index* rows,
    cuBool_Index* cols,
    cuBool_Index* nvals
);

/**
 * Creates new sparse matrix, duplicates content and stores handle in the provided pointer.
 * 
 * @param matrix Matrix handle to perform operation on
 * @param duplicated[out] Pointer to the matrix handle where to create and store created matrix
 * 
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Duplicate(
    cuBool_Matrix matrix,
    cuBool_Matrix* duplicated
);

/**
 * Transpose source matrix and store result of this operation in result matrix.
 * Formally: result = matrix ^ T.
 *
 * @param result Matrix handle to store result of the operation
 * @param matrix The source matrix
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Transpose(
    cuBool_Matrix result,
    cuBool_Matrix matrix
);

/**
 * Query number of non-zero values of the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param nvals[out] Pointer to  the place where to store number of the non-zero elements of the matrix
 * 
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Nvals(
    cuBool_Matrix matrix,
    cuBool_Index* nvals
);

/**
 * Query number of rows in the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param nrows[out] Pointer to the place where to store number of matrix rows
 * 
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Nrows(
    cuBool_Matrix matrix,
    cuBool_Index* nrows
);

/**
 * Query number of columns in the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param ncols[out] Pointer to the place where to store number of matrix columns
 * 
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Ncols(
    cuBool_Matrix matrix,
    cuBool_Index* ncols
);

/**
 * Deletes sparse matrix object.
 *
 * @param matrix Matrix handle to delete the matrix
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Free(
    cuBool_Matrix matrix
);

/**
 * Reduce the source matrix to the column matrix result (column vector).
 * Formally: result = sum(cols of matrix).

 * @note Matrices must be compatible
 *          dim(matrix) = M x N
 *          dim(result) = M x 1
 *
 * @param result Matrix hnd where to store result
 * @param matrix Source matrix to reduce
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Reduce(
    cuBool_Matrix result,
    cuBool_Matrix matrix
);

/**
 * Performs result = left + right, where '+' is boolean semiring operation.
 *
 * @note Matrices must be compatible
 *          dim(result) = M x N
 *          dim(left) = M x N
 *          dim(right) = M x N
 *
 * @param result Destination matrix for add-and-assign operation
 * @param left Source matrix to be added
 * @param right Source matrix to be added
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_EWiseAdd(
    cuBool_Matrix result,
    cuBool_Matrix left,
    cuBool_Matrix right
);

/**
 * Performs result (accum)= left x right evaluation, where source '+' and 'x' are boolean semiring operations.
 * If accum hint passed, the the result of the multiplication is added to the result matrix.
 *
 * @note To perform this operation matrices must be compatible
 *          dim(left) = M x T
 *          dim(right) = T x N
 *          dim(result) = M x N
 *
 * @note Pass CUBOOL_HINT_ACCUMULATE hint to add result of the left x right operation.
 *
 * @param result Matrix handle where to store operation result
 * @param left Input left matrix
 * @param right Input right matrix
 * @param hints Hints for the operation.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_MxM(
    cuBool_Matrix result,
    cuBool_Matrix left,
    cuBool_Matrix right,
    cuBool_Hints hints
);

/**
 * Performs result = left `kron` right, where `kron` is a Kronecker product for boolean semiring.
 *
 * @note When the operation is performed, the result matrix has the following dimension
 *          dim(left) = M x N
 *          dim(right) = K x T
 *          dim(result) = MK x NT
 *
 * @param result Matrix handle where to store operation result
 * @param left Input left matrix
 * @param right Input right matrix
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Kronecker(
    cuBool_Matrix result,
    cuBool_Matrix left,
    cuBool_Matrix right
);

#endif //CUBOOL_CUBOOL_H
