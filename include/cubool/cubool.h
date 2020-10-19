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
     * @return Error code on this operations
     */
    CuBoolError CuBoolGetLibraryVersion(int* major, int* minor, int* version);

    /**
     * Initialize library instance object, which provides context to all library operations and objects
     *
     * @param instanceDesc User provided instance configuration for memory operations and error handling
     * @param instance Pointer to the place where to store instance handler
     * @return Error code on this operations
     */
    CuBoolError CuBoolCreateInstance(const CuBoolInstanceDesc* instanceDesc, CuBoolInstance* instance);

    /**
     * Destroy library instance and all objects, which were created on this library context.
     *
     * @note Invalidates all handler to the resources, created within this library instance
     * @param instance An instance object reference to perform this operation
     * @return Error code on this operations
     */
    CuBoolError CuBoolDestroyInstance(CuBoolInstance instance);

    /**
     *
     * @param instance
     * @param matrix
     * @return
     */
    CuBoolError CuBoolCreateMatrixDense(CuBoolInstance instance, CuBoolMatrixDense* matrix);

    /**
     *
     * @param instance
     * @param matrix
     * @return
     */
    CuBoolError CuBoolDestroyMatrixDense(CuBoolInstance instance, CuBoolMatrixDense matrix);

    /**
     *
     * @param instance
     * @param result
     * @param a
     * @param b
     * @param c
     * @return
     */
    CuBoolError CuBoolMultiplyAdd(CuBoolInstance instance, CuBoolMatrixDense result, CuBoolMatrixDense a, CuBoolMatrixDense b, CuBoolMatrixDense c);

#ifdef __cplusplus
};
#endif // __cplusplus

#endif //CUBOOL_CUBOOL_H
