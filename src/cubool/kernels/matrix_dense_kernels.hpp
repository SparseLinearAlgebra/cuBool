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

#ifndef CUBOOL_MATRIX_DENSE_KERNELS_HPP
#define CUBOOL_MATRIX_DENSE_KERNELS_HPP

#include <cubool/matrix_dense.hpp>

namespace cubool {

    class MatrixDenseKernels {
    public:

        /**
         * Run kernel r = a x b + c
         * Validates arguments and matrices size compatibility
         *
         * @param result Matrix to store result
         * @param a Input a matrix
         * @param b Input b matrix
         * @param c Input c matrix
         * @return Operation status
         */
        static CuBoolStatus invoke_MatrixDenseMultiplyAdd(
            class Instance &instance,
            MatrixDense &result,
            const MatrixDense &a,
            const MatrixDense &b,
            const MatrixDense &c
        );
    };
}

#endif //CUBOOL_MATRIX_DENSE_KERNELS_HPP
