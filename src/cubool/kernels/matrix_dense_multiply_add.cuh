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

#ifndef CUBOOL_MATRIX_DENSE_MULTIPLY_ADD_CUH
#define CUBOOL_MATRIX_DENSE_MULTIPLY_ADD_CUH

#include <cubool/kernels/matrix_dense_common.cuh>
#include <cubool/matrix_dense.hpp>
#include <cubool/instance.hpp>

namespace cubool {
    namespace kernels {

        __global__ void matrixDenseMultiplyAdd(Matrix r, Matrix a, Matrix b) {
            size blockRow = blockIdx.y;
            size blockColumn = blockIdx.x;

            size row = threadIdx.y;
            size column = threadIdx.x;

            Matrix rBlock = getMatrixBlock(r, blockRow, blockColumn);

            // Result matrix block packed column to eval
            PackType_t rBlockColumn = 0x0;

            // Assumed to be aligned by the invoker of this kernel
            size blocks = a.columns / BLOCK_SIZE_X + (a.columns % BLOCK_SIZE_X? 1: 0);

            for (size k = 0; k < blocks; k++) {
                Matrix aBlock = getMatrixBlock(a, blockRow, k);
                Matrix bBlock = getMatrixBlock(b, k, blockColumn);

                __shared__ PackType_t aBlockShared[BLOCK_SIZE_Y][BLOCK_SIZE_X];
                __shared__ PackType_t bBlockShared[BLOCK_SIZE_Y][BLOCK_SIZE_X];
                __shared__ PackType_t atBlockShared[BLOCK_SIZE_X][BLOCK_SIZE_Y];

                aBlockShared[row][column] = isBlockValueWithinMatrix(a, blockRow, k, row, column) ? getMatrixElementPacked(aBlock, row, column) : 0;
                bBlockShared[row][column] = isBlockValueWithinMatrix(b, k, blockColumn, row, column) ? getMatrixElementPacked(bBlock, row, column) : 0;

                // Wait to finish copy
                __syncthreads();

                PackType_t atPackedRow = 0;
                size rowPart = row * PACT_TYPE_SIZE;
                size rowMajor = column / PACT_TYPE_SIZE;
                size rowMinor = column % PACT_TYPE_SIZE;

                #pragma unroll
                for (size l = 0; l < PACT_TYPE_SIZE; l++) {
                    atPackedRow |= (aBlockShared[rowMajor][rowPart + l] & (1u << rowMinor) ? (1u << l) : 0u);
                }

                atBlockShared[column][row] = atPackedRow;

                // Wait to finish a-block transpose op
                __syncthreads();

                size offset = row * PACT_TYPE_SIZE;

                #pragma unroll
                for (size l = 0; l < PACT_TYPE_SIZE; l++) {
                    PackType_t accum = 0;

                    size rowNum = offset + l;

                    #pragma unroll
                    for (size part = 0; part < BLOCK_SIZE_Y; part++) {
                        accum |= atBlockShared[rowNum][part] & bBlockShared[part][column];
                    }

                    rBlockColumn |= accum != 0 ? 1u << l: 0x0;
                }

                // Wait to finis multiplication eval on this iteration
                __syncthreads();
            }

            // Ensure, that we do not read and write out of bounds values of the result blocks
            if (isBlockValueWithinMatrix(r, blockRow, blockColumn, row, column)) {
                addMatrixElementPacked(rBlock, row, column, rBlockColumn);
            }
        }

    }
}

#endif //CUBOOL_MATRIX_DENSE_MULTIPLY_ADD_CUH
