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

#ifndef CUBOOL_MATRIX_DENSE_SHARED_CUH
#define CUBOOL_MATRIX_DENSE_SHARED_CUH

#include <cubool/matrix_dense.hpp>

namespace cubool {

    using PackType_t = MatrixDense::PackType_t;

    struct Matrix {
        CuBoolSize_t rows;
        CuBoolSize_t columns;
        CuBoolSize_t stride;
        PackType_t* buffer;
    };

    __host__ Matrix getMatrixFromDenseMatrixClass(const MatrixDense& m) {
        Matrix r{};
        r.rows = m.getRowsPackedCount();
        r.columns = m.getColumnsPaddedCount();
        r.stride = r.columns;
        r.buffer = (PackType_t*) m.getBuffer().getMemory();
        return r;
    }

    __device__ Matrix getMatrixBlock(const Matrix& m, CuBoolSize_t row, CuBoolSize_t column) {
        Matrix subMatrix{};
        subMatrix.rows = 1;
        subMatrix.columns = MatrixDense::PACK_TYPE_SIZE_BITS;
        subMatrix.stride = m.stride;
        subMatrix.buffer = &m.buffer[m.stride * row + MatrixDense::PACK_TYPE_SIZE_BITS * column];
        return subMatrix;
    }

    __device__ PackType_t getMatrixElementPacked(const Matrix& m, CuBoolSize_t row, CuBoolSize_t column) {
        return m.buffer[m.stride * row + column];
    }

    __device__ void setMatrixElementPacked(const Matrix& m, CuBoolSize_t row, CuBoolSize_t column, PackType_t value) {
        if (value) {
            m.buffer[m.stride * row + column] = value;
        }
    }

}

#endif //CUBOOL_MATRIX_DENSE_SHARED_CUH
