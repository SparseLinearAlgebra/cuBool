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

#ifndef CUBOOL_MATRIX_DENSE_COMMON_CUH
#define CUBOOL_MATRIX_DENSE_COMMON_CUH

#include <config.hpp>
#include <matrix_dense.hpp>

namespace cubool {
    namespace kernels {

        // Base type within kernels to pack boolean values
        using PackType_t = MatrixDense::PackType_t;

        // Pack type size in bits
        static const size PACT_TYPE_SIZE = MatrixDense::PACK_TYPE_SIZE_BITS;

        // Factor, used to scale the base shared block (depends on device capabilities)
        static const size BLOCK_FACTOR = 4;

        // Shared packed block size in x dimension (columns)
        static const size BLOCK_SIZE_X = PACT_TYPE_SIZE * BLOCK_FACTOR;

        // Shared packed block size in y dimension (rows)
        static const size BLOCK_SIZE_Y = 1 * BLOCK_FACTOR;

        // Base block schema
        //               columns
        //           0      ... 31
        //
        //       0   a_0,0  ... a_0,31
        // rows  .
        //       .
        //       31  a_31,0 ... a_31,31
        //
        // packed as PackType_t block[PACT_TYPE_SIZE];
        // value  a_i,j = (block[j] & 1u << i) != 0

        struct Matrix {
            size rows;
            size columns;
            size stride;
            PackType_t* buffer;
        };

        __host__ void getGridConfig(size_t rows, size_t columns, dim3& block, dim3& grid) {
            block = dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            grid = dim3(columns / BLOCK_SIZE_X + (columns % BLOCK_SIZE_X? 1: 0), rows / BLOCK_SIZE_Y + (rows % BLOCK_SIZE_Y? 1: 0));
        }

        __host__ Matrix getMatrixFromDenseMatrixClass(const MatrixDense& m) {
            Matrix r{};
            r.rows = m.getNumRowsPacked();
            r.columns = m.getNumColsPadded();
            r.stride = r.columns;
            r.buffer = (PackType_t*) m.getBuffer().data().get();
            return r;
        }

        __host__ Matrix getMatrixFromDenseMatrixClass(MatrixDense& m) {
            Matrix r{};
            r.rows = m.getNumRowsPacked();
            r.columns = m.getNumColsPadded();
            r.stride = r.columns;
            r.buffer = (PackType_t*) m.getBuffer().data().get();
            return r;
        }

        __device__ Matrix getMatrixBlock(const Matrix& m, size row, size column) {
            Matrix subMatrix{};
            subMatrix.rows = BLOCK_SIZE_Y;
            subMatrix.columns = BLOCK_SIZE_X;
            subMatrix.stride = m.stride;
            subMatrix.buffer = &m.buffer[m.stride * BLOCK_SIZE_Y * row + BLOCK_SIZE_X * column];
            return subMatrix;
        }

        __device__ bool isBlockValueWithinMatrix(const Matrix& parent, size blockRow, size blockColumn, size row, size column) {
            return (blockRow * BLOCK_SIZE_Y + row) < parent.rows && (blockColumn * BLOCK_SIZE_X + column) < parent.columns;
        }

        __device__ PackType_t getMatrixElementPacked(const Matrix& m, size row, size column) {
            return m.buffer[m.stride * row + column];
        }

        __device__ void setMatrixElementPacked(const Matrix& m, size row, size column, PackType_t value) {
            m.buffer[m.stride * row + column] = value;
        }

        __device__ void addMatrixElementPacked(const Matrix& m, size row, size column, PackType_t value) {
            m.buffer[m.stride * row + column] |= value;
        }

    }
}

#endif //CUBOOL_MATRIX_DENSE_COMMON_CUH
