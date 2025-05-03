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

#ifndef CUBOOL_CUBOOL_H
#define CUBOOL_CUBOOL_H

#ifdef __cplusplus
    #include <cinttypes>
#else
    #include <inttypes.h>
#endif

// Preserve C names in shared library
#ifdef __cplusplus
    #define CUBOOL_EXPORT extern "C"
#else
    #define CUBOOL_EXPORT
#endif

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
    CUBOOL_HINT_NO = 0,
    /** Force Cpu based backend usage */
    CUBOOL_HINT_CPU_BACKEND = 1,
    /** Use managed gpu memory type instead of default (device) memory */
    CUBOOL_HINT_GPU_MEM_MANAGED = 2,
    /** Mark input data as row-col sorted */
    CUBOOL_HINT_VALUES_SORTED = 4,
    /** Accumulate result of the operation in the result matrix */
    CUBOOL_HINT_ACCUMULATE = 8,
    /** Finalize library state, even if not all resources were explicitly released */
    CUBOOL_HINT_RELAXED_FINALIZE = 16,
    /** Logging hint: log includes error message */
    CUBOOL_HINT_LOG_ERROR = 32,
    /** Logging hint: log includes warning message */
    CUBOOL_HINT_LOG_WARNING = 64,
    /** Logging hint: log includes all types of messages */
    CUBOOL_HINT_LOG_ALL = 128,
    /** No duplicates in the build data */
    CUBOOL_HINT_NO_DUPLICATES = 256,
    /** Performs time measurement and logs elapsed operation time */
    CUBOOL_HINT_TIME_CHECK = 512,
    /** Transpose matrix before operation */
    CUBOOL_HINT_TRANSPOSE = 1024
} cuBool_Hint;

/** Hit mask */
typedef uint32_t cuBool_Hints;

/** Alias integer type for indexing operations */
typedef uint32_t cuBool_Index;

/** cuBool sparse boolean matrix handle */
typedef struct cuBool_Matrix_t* cuBool_Matrix;

/** cuBool sparse boolean vector handle */
typedef struct cuBool_Vector_t* cuBool_Vector;

/** Cuda device capabilities */
typedef struct cuBool_DeviceCaps {
    char name[256];
    bool cudaSupported;
    bool managedMem;
    int major;
    int minor;
    int warp;
    int globalMemoryKiBs;
    int sharedMemoryPerMultiProcKiBs;
    int sharedMemoryPerBlockKiBs;
} cuBool_DeviceCaps;

/**
 * Query human-readable text info about the project implementation
 * @note It is safe to call this function before the library is initialized.
 *
 * @return Read-only library about info
 */
CUBOOL_EXPORT CUBOOL_API const char* cuBool_GetAbout(
);

/**
 * Query human-readable text info about the project implementation
 * @note It is safe to call this function before the library is initialized.

 * @return Read-only library license info
 */
CUBOOL_EXPORT CUBOOL_API const char* cuBool_GetLicenseInfo(
);

/**
 * Query library version number in form MAJOR.MINOR
 * @note It is safe to call this function before the library is initialized.
 *
 * @param major Major version number part
 * @param minor Minor version number part
 * @param sub Sub version number part
 *
 * @return Error if failed to query version info
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_GetVersion(
    int* major,
    int* minor,
    int* sub
);

/**
 * Allows to setup logging file for all operations, invoked after this function call.
 * Empty hints field is interpreted as `CUBOOL_HINT_LOG_ALL` by default.
 *
 * @note It is safe to call this function before the library is initialized.
 *
 * @note Pass `CUBOOL_HINT_LOG_ERROR` to include error messages into log
 * @note Pass `CUBOOL_HINT_LOG_WARNING` to include warning messages into log
 * @note Pass `CUBOOL_HINT_LOG_ALL` to include all messages into log
 *
 * @param logFileName UTF-8 encoded null-terminated file name and path string.
 * @param hints Logging hints to filter messages.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_SetupLogging(
    const char* logFileName,
    cuBool_Hints hints
);

/**
 * Initialize library instance object, which provides context to all library operations and primitives.
 * This function must be called before any other library function is called,
 * except first get-info functions.
 *
 * @note Pass `CUBOOL_HINT_RELAXED_FINALIZE` for library setup within python.
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
 * @note Pass `CUBOOL_HINT_RELAXED_FINALIZE` for library init call, if relaxed finalize is required.
 * @note Invalidates all handle to the resources, created within this library instance
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Finalize(
);

/**
 * Query device capabilities/properties if cuda compatible device is present.
 *
 * @note This function returns no actual info if cuda backend is not presented.
 * @param deviceCaps Pointer to device caps structure to store result
 *
 * @return Error if cuda device not present or if failed to query capabilities
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_GetDeviceCaps(
    cuBool_DeviceCaps* deviceCaps
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
 * By default automatically sorts values and reduces duplicates in the input arrays.
 *
 * @note Pass `CUBOOL_HINT_VALUES_SORTED` if values already in the row-col order.
 * @note Pass `CUBOOL_HINT_NO_DUPLICATES` if values has no duplicates
 *
 * @param matrix Matrix handle to perform operation on
 * @param rows Array of pairs row indices
 * @param cols Array of pairs column indices
 * @param nvals Number of the pairs passed
 * @param hints Hits flags for processing
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
 * Sets specified (i, j) value of the matrix to True.
 *
 * @note This function automatically reduces duplicates
 *
 * @param matrix Matrix handle to perform operation on
 * @param i Row index
 * @param j Column Index
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_SetElement(
    cuBool_Matrix matrix,
    cuBool_Index i,
    cuBool_Index j
);

/**
 * Sets to the matrix specific debug string marker.
 * This marker will appear in the log messages as string identifier of the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param marker UTF-8 encoded null-terminated string marker name.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_SetMarker(
    cuBool_Matrix matrix,
    const char* marker
);

/**
 * Allows to get matrix debug string marker.
 *
 * @note Pass null marker if you want to retrieve only the required marker buffer size.
 * @note After the function call the actual size of the marker is stored in the size variable.
 *
 * @note size is set to the actual marker length plus null terminator symbol.
 *       For marker "matrix" size variable will be set to 7.
 *
 * @param matrix Matrix handle to perform operation on
 * @param[in,out] marker Where to store null-terminated UTF-8 encoded marker string.
 * @param[in,out] size Size of the provided buffer in bytes to save marker string.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Marker(
    cuBool_Matrix matrix,
    char* marker,
    cuBool_Index* size
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
 * Extracts sub-matrix of the input matrix and stores it into result matrix.
 *
 * @note The result sub-matrix have nrows x ncols dimension,
 *       and includes [i;i+nrow) x [j;j+ncols) cells of the input matrix.
 *
 * @note Result matrix must have compatible size
 *          dim(result) = nrows x ncols
 *
 * @note Provided sub-matrix region must be within the input matrix.
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Matrix handle where to store result of the operation
 * @param matrix Input matrix to extract values from
 * @param i First row id to extract
 * @param j First column id to extract
 * @param nrows Number of rows to extract
 * @param ncols Number of columns to extract
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_ExtractSubMatrix(
    cuBool_Matrix result,
    cuBool_Matrix matrix,
    cuBool_Index i,
    cuBool_Index j,
    cuBool_Index nrows,
    cuBool_Index ncols,
    cuBool_Hints hints
);

/**
 * Extract specified matrix row as vector.
 *
 * @note Vector and matrix must have compatible size.
 *       dim(matrix) = M x N
 *       dim(vector) = N
 *
 * @param result Vector handle where to store extracted row
 * @param matrix Source matrix
 * @param i Index of the matrix row to extract
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_ExtractRow(
    cuBool_Vector result,
    cuBool_Matrix matrix,
    cuBool_Index i,
    cuBool_Hints hints
);

/**
 * Extract specified matrix col as vector.
 *
 * @note Vector and matrix must have compatible size.
 *       dim(matrix) = M x N
 *       dim(vector) = M
 *
 * @param result Vector handle where to store extracted column
 * @param matrix Source matrix
 * @param j Index of the matrix column to extract
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_ExtractCol(
    cuBool_Vector result,
    cuBool_Matrix matrix,
    cuBool_Index j,
    cuBool_Hints hints
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
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Matrix handle to store result of the operation
 * @param matrix The source matrix
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Transpose(
    cuBool_Matrix result,
    cuBool_Matrix matrix,
    cuBool_Hints hints
);

/**
 * Query number of non-zero values of the matrix.
 *
 * @param matrix Matrix handle to perform operation on
 * @param nvals[out] Pointer to the place where to store number of the non-zero elements of the matrix
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
 * @param matrix Matrix handle to delete
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Free(
    cuBool_Matrix matrix
);

/**
 * Reduce the source matrix to the column vector.
 * Pass optionally transpose hint to transpose matrix before the reduce.
 * Formally: result = sum(cols of matrix M), where
 *   M = matrix, or M = matrix^T (if passed transpose hint)
 *
 * @note Pass `CUBOOL_HINT_TRANSPOSE` hint to reduce transposed matrix
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param[out] result Vector handle where to store result
 * @param matrix Source matrix to reduce
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Reduce(
    cuBool_Vector result,
    cuBool_Matrix matrix,
    cuBool_Hints hints
);

/**
 * Reduce the source matrix to the column matrix result (column vector).
 * Formally: result = sum(cols of matrix).

 * @note Matrices must be compatible
 *          dim(matrix) = M x N
 *          dim(result) = M x 1
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param[out] result Matrix handle where to store result
 * @param matrix Source matrix to reduce
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_Reduce2(
    cuBool_Matrix result,
    cuBool_Matrix matrix,
    cuBool_Hints hints
);

/**
 * Performs result = left + right, where '+' is boolean semiring 'or' operation.
 *
 * @note Matrices must be compatible
 *          dim(result) = M x N
 *          dim(left) = M x N
 *          dim(right) = M x N
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Destination matrix to store result
 * @param left Source matrix to be added
 * @param right Source matrix to be added
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_EWiseAdd(
    cuBool_Matrix result,
    cuBool_Matrix left,
    cuBool_Matrix right,
    cuBool_Hints hints
);

/**
 * Performs result = left * right, where '*' is boolean semiring 'and' operation.
 *
 * @note Matrices must be compatible
 *          dim(result) = M x N
 *          dim(left) = M x N
 *          dim(right) = M x N
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Destination matrix to store result
 * @param left Source matrix to be multiplied
 * @param right Source matrix to be multiplied
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_EWiseMult(
    cuBool_Matrix result,
    cuBool_Matrix left,
    cuBool_Matrix right,
    cuBool_Hints hints
);

/**
 * Creates new sparse vector with specified size.
 *
 * @param vector Pointer where to store created vector
 * @param nrows Vector rows count
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_New(
    cuBool_Vector* vector,
    cuBool_Index nrows
);

/**
 * Build sparse vector from provided indices array.
 * By default automatically sorts values and reduces duplicates in the input array.
 *
 * @note Pass `CUBOOL_HINT_VALUES_SORTED` if values already in the row-col order.
 * @note Pass `CUBOOL_HINT_NO_DUPLICATES` if values has no duplicates
 *
 * @param vector Vector handle to perform operation on
 * @param rows Array of row indices
 * @param nvals Number of the indices passed
 * @param hints Hits flags for processing
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_Build(
    cuBool_Vector vector,
    const cuBool_Index* rows,
    cuBool_Index nvals,
    cuBool_Hints hints
);

/**
 * Sets specified (j) value of the vector to True.
 *
 * @note This function automatically reduces duplicates
 *
 * @param vector Vector handle to perform operation on
 * @param i Row index
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_SetElement(
    cuBool_Vector vector,
    cuBool_Index i
);

/**
 * Sets to the vector specific debug string marker.
 * This marker will appear in the log messages as string identifier of the vector.
 *
 * @param vector Vector handle to perform operation on
 * @param marker UTF-8 encoded null-terminated string marker name.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_SetMarker(
    cuBool_Vector vector,
    const char* marker
);

/**
 * Allows to get vector debug string marker.
 *
 * @note Pass null marker if you want to retrieve only the required marker buffer size.
 * @note After the function call the actual size of the marker is stored in the size variable.
 *
 * @note size is set to the actual marker length plus null terminator symbol.
 *       For marker "vector" size variable will be set to 7.
 *
 * @param vector Vector handle to perform operation on
 * @param[in,out] marker Where to store null-terminated UTF-8 encoded marker string.
 * @param[in,out] size Size of the provided buffer in bytes to save marker string.
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_Marker(
    cuBool_Vector vector,
    char* marker,
    cuBool_Index* size
);

/**
 * Reads vector data to the host visible CPU buffer as an array of indices.
 *
 * The array must be provided by the user and the size of this array must
 * be greater or equal the values count of the vector.
 *
 * @param vector Matrix handle to perform operation on
 * @param[in,out] rows Buffer to store row indices
 * @param[in,out] nvals Total number of the indices
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_ExtractValues(
    cuBool_Vector vector,
    cuBool_Index* rows,
    cuBool_Index* nvals
);

/**
 * Extracts sub-vector of the input vector and stores it into result vector.
 *
 * @note The result sub-vector have nrows dimension,
 *       and includes [i;i+nrow) rows of the input vector.
 *
 * @note Result vector must have compatible size
 *          dim(result) = nrows
 *
 * @note Provided sub-vector region must be within the input vector.
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Vector handle where to store result of the operation
 * @param vector Input vector to extract values from
 * @param i First row id to extract
 * @param nrows Number of rows to extract
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_ExtractSubVector(
    cuBool_Vector result,
    cuBool_Vector vector,
    cuBool_Index i,
    cuBool_Index nrows,
    cuBool_Hints hints
);

/**
 * Creates new sparse vector, duplicates content and stores handle in the provided pointer.
 *
 * @param vector Vector handle to perform operation on
 * @param duplicated[out] Pointer to the vector handle where to create and store created vector
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_Duplicate(
    cuBool_Vector vector,
    cuBool_Vector* duplicated
);

/**
 * Query number of non-zero values of the vector.
 *
 * @param vector Vector handle to perform operation on
 * @param nvals[out] Pointer to the place where to store number of the non-zero elements of the vector
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_Nvals(
    cuBool_Vector vector,
    cuBool_Index* nvals
);

/**
 * Query number of rows in the vector.
 *
 * @param vector Vector handle to perform operation on
 * @param nrows[out] Pointer to the place where to store number of vector rows
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_Nrows(
    cuBool_Vector vector,
    cuBool_Index* nrows
);

/**
 * Deletes sparse vector object.
 *
 * @param vector Vector handle to delete
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_Free(
    cuBool_Vector vector
);


/**
 * Reduces vector to single value (equals nnz of the vector for boolean case).
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result Pointer to index value where to store result
 * @param vector Vector handle to perform operation on
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_Reduce(
    cuBool_Index* result,
    cuBool_Vector vector,
    cuBool_Hints hints
);

/**
 * Performs result = left + right, where '+' is boolean semiring 'or' operation.
 *
 * @note Matrices must be compatible
 *          dim(result) = M
 *          dim(left) = M
 *          dim(right) = M
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out]Destination vector to store result
 * @param left Source vector to be added
 * @param right Source vector to be added
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_EWiseAdd(
    cuBool_Vector result,
    cuBool_Vector left,
    cuBool_Vector right,
    cuBool_Hints hints
);

/**
 * Performs result = left * right, where '*' is boolean semiring 'and' operation.
 *
 * @note Matrices must be compatible
 *          dim(result) = M
 *          dim(left) = M
 *          dim(right) = M
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out]Destination vector to store result
 * @param left Source vector to be multiplied
 * @param right Source vector to be multiplied
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Vector_EWiseMult(
    cuBool_Vector result,
    cuBool_Vector left,
    cuBool_Vector right,
    cuBool_Hints hints
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
 * @note Pass `CUBOOL_HINT_ACCUMULATE` hint to add result of the left x right operation.
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Matrix handle where to store operation result
 * @param left Input left matrix
 * @param right Input right matrix
 * @param hints Hints for the operation
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
 * Performs result = left x right evaluation, where source '+' and 'x' are boolean semiring operations.
 * Formally: column vector `right` multiplied to the matrix `left`. The result is column vector.
 *
 * @note To perform this operation matrix and vector must be compatible
 *          dim(left) = M x N
 *          dim(right) = N
 *          dim(result) = M
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Vector handle where to store operation result
 * @param left Input left matrix
 * @param right Input right vector
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_MxV(
    cuBool_Vector result,
    cuBool_Matrix left,
    cuBool_Vector right,
    cuBool_Hints hints
);

/**
 * Performs result = left x right evaluation, where source '+' and 'x' are boolean semiring operations.
 * Formally: row vector `left` multiplied to the matrix `right`. The result is row vector.
 *
 * @note To perform this operation matrix and vector must be compatible
 *          dim(left) = M
 *          dim(right) = M x N
 *          dim(result) = N
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Vector handle where to store operation result
 * @param left Input left vector
 * @param right Input right matrix
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_VxM(
    cuBool_Vector result,
    cuBool_Vector left,
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
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Matrix handle where to store operation result
 * @param left Input left matrix
 * @param right Input right matrix
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Kronecker(
    cuBool_Matrix result,
    cuBool_Matrix left,
    cuBool_Matrix right,
    cuBool_Hints hints
);

/**
 * Performs result = left * ~right, where
 *     '*' is boolean semiring 'and' operation
 *     '~' is operation for invert matrix (0 swaps to 1 and 1 to 0)
 *
 * @note To perform this operation matrices must be compatible
 *          dim(left) = M x T
 *          dim(right) = T x N
 *          dim(result) = M x N
 *
 * @note Pass `CUBOOL_HINT_TIME_CHECK` hint to measure operation time
 *
 * @param result[out] Destination matrix to store result
 * @param left Source matrix to be multiplied
 * @param right Source matrix to be inverted and multiplied
 * @param hints Hints for the operation
 *
 * @return Error code on this operation
 */
CUBOOL_EXPORT CUBOOL_API cuBool_Status cuBool_Matrix_EWiseMulInverted(
    cuBool_Matrix result,
    cuBool_Matrix matrix,
    cuBool_Matrix mask,
    cuBool_Hints hints
);

#endif //CUBOOL_CUBOOL_H
