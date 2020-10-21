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

#include <cubool/cubool_types.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

    /**
     * Query library version number in form MAJOR.MINOR
     *
     * @param major Major version number part
     * @param minor Minor version number part
     * @param version Composite integer version
     *
     * @return Error if failed to query version info
     */
    CuBoolStatus CuBoolLibraryVersionGet(int* major, int* minor, int* version);

    /**
     * Query device capabilities/properties if cuda compatible device is present
     *
     * @param deviceCaps Pointer to device caps structure to store result
     *
     * @return Error if cuda device not present or if failed to query capabilities
     */
    CuBoolStatus CuBoolDeviceCapsGet(CuBoolDeviceCaps* deviceCaps);

    /**
     * Initialize library instance object, which provides context to all library operations and objects
     *
     * @param instanceDesc User provided instance configuration for memory operations and error handling
     * @param instance Pointer to the place where to store instance handler
     *
     * @return Error code on this operations
     */
    CuBoolStatus CuBoolInstanceCreate(const CuBoolInstanceDesc* instanceDesc, CuBoolInstance* instance);

    /**
     * Destroy library instance and all objects, which were created on this library context.
     *
     * @note Invalidates all handler to the resources, created within this library instance
     * @param instance An instance object reference to perform this operation
     *
     * @return Error code on this operations
     */
    CuBoolStatus CuBoolInstanceDestroy(CuBoolInstance instance);

    /**
     *
     * @param instance
     * @param matrix
     * @return
     */
    CuBoolStatus CuBoolMatrixDenseCreate(CuBoolInstance instance, CuBoolMatrixDense* matrix);

    /**
     *
     * @param instance
     * @param matrix
     * @return
     */
    CuBoolStatus CuBoolMatrixDenseDestroy(CuBoolInstance instance, CuBoolMatrixDense matrix);

    /**
     *
     * @param instance
     * @param matrix
     * @param rows
     * @param columns
     * @return
     */
    CuBoolStatus CuBoolMatrixDenseResize(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t rows, CuBoolSize_t columns);

    /**
     *
     * @param instance
     * @param matrix
     * @param count
     * @param values
     * @return
     */
    CuBoolStatus CuBoolMatrixDenseWriteData(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t count, const CuBoolPair* values);

    /**
     * Reads matrix data to the host visible CPU buffer as an array of values pair
     *
     * @note Returned pointer to the allocated values buffers must be explicitly freed by the user
     *       via function call CuBoolValuesArrayFree
     *
     * @param instance An instance object reference to perform this operation
     * @param matrix Matrix handler to perform operation on
     * @param[out] count Number of pairs in the returned array
     * @param[out] values Allocated buffer to the pairs indices. Buffer must be freed after usage
     *
     * @return Error code on this operations
     */
    CuBoolStatus CuBoolMatrixDenseReadData(CuBoolInstance instance, CuBoolMatrixDense matrix, CuBoolSize_t* count, CuBoolPair** values);

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
    CuBoolStatus CuBoolMatrixDenseMultiplyAdd(CuBoolInstance instance, CuBoolMatrixDense result, CuBoolMatrixDense a, CuBoolMatrixDense b, CuBoolMatrixDense c);

    /**
     * Release values array buffer, allocated by one of *ReadData operations
     *
     * @param instance An instance object reference to perform this operation
     * @param values Valid pointer to returned arrays buffer from *ReadData method
     *
     * @return Error code on this operations
     */
    CuBoolStatus CuBoolValuesArrayFree(CuBoolInstance instance, CuBoolPair* values);

#ifdef __cplusplus
};
#endif // __cplusplus

#endif //CUBOOL_CUBOOL_H
