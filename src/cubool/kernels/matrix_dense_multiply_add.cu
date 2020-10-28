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

#include <cubool/kernels/matrix_dense_frontend.hpp>
#include <cubool/kernels/matrix_dense_shared.cuh>
#include <cubool/matrix_dense.hpp>
#include <cubool/instance.hpp>

namespace cubool {

    __global__ void kernelMatrixDenseMultiplyAdd(Matrix result, Matrix a, Matrix b, Matrix c) {
        CuBoolSize_t blockRow = blockIdx.y;
        CuBoolSize_t blockColumn = blockIdx.x;

        Matrix rBlock = getMatrixBlock(result, blockRow, blockColumn);
        Matrix cBlock = getMatrixBlock(c, blockRow, blockColumn);

        // Result matrix block packed column to eval
        PackType_t rBlockColumn = 0x0;
        CuBoolSize_t column = threadIdx.x;

        // Assumed to be aligned by the invoker of this kernel
        CuBoolSize_t blocks = a.columns / MatrixDense::PACK_TYPE_SIZE_BITS;

        for (CuBoolSize_t k = 0; k < blocks; k++) {
            Matrix aBlock = getMatrixBlock(a, blockRow, k);
            Matrix bBlock = getMatrixBlock(b, k, blockColumn);

            __shared__ PackType_t aBlockShared[MatrixDense::PACK_TYPE_SIZE_BITS];
            __shared__ PackType_t atBlockShared[MatrixDense::PACK_TYPE_SIZE_BITS];
            __shared__ PackType_t bBlockShared[MatrixDense::PACK_TYPE_SIZE_BITS];

            aBlockShared[column] = getMatrixElementPacked(aBlock, 0, column);
            bBlockShared[column] = getMatrixElementPacked(bBlock, 0, column);

            // Wait to finish copy
            __syncthreads();

            PackType_t aBlockRow = 0x0;
            CuBoolSize_t rowIdx = column;

            #pragma unroll
            for (CuBoolSize_t l = 0; l < MatrixDense::PACK_TYPE_SIZE_BITS; l++) {
                aBlockRow |= (aBlockShared[l] & (1u << rowIdx) ? (1u << l) : 0u);
            }

            atBlockShared[rowIdx] = aBlockRow;

            // Wait to finish a-block transpose op
            __syncthreads();

            #pragma unroll
            for (CuBoolSize_t l = 0; l < MatrixDense::PACK_TYPE_SIZE_BITS; l++) {
                rBlockColumn |= (atBlockShared[l] & bBlockShared[column]) != 0x0 ? 1u << l : 0x0;
            }

            // Wait to finis multiplication eval on this iteration
            __syncthreads();
        }

        PackType_t cBlockColumn = getMatrixElementPacked(cBlock, 0, column);
        rBlockColumn |= cBlockColumn;

        setMatrixElementPacked(rBlock, 0, column, rBlockColumn);
    }

    CuBoolStatus MatrixDenseKernels::invoke_MatrixDenseMultiplyAdd(
            Instance &instance,
            MatrixDense &result,
            const MatrixDense &a,
            const MatrixDense &b,
            const MatrixDense &c) {

        if (a.isZeroDim() || b.isZeroDim() || c.isZeroDim()) {
            instance.sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "An attempt to operate on 0-dim matrices");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        if (a.getColumnsCount() != b.getRowsCount()) {
            instance.sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Incompatible matrix size to multiply");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        CuBoolSize_t M = a.getRowsCount();
        CuBoolSize_t N = b.getColumnsCount();

        if (c.getRowsCount() != M || c.getColumnsCount() != N) {
            instance.sendMessage(CUBOOL_STATUS_INVALID_ARGUMENT, "Incompatible matrix size to add");
            return CUBOOL_STATUS_INVALID_ARGUMENT;
        }

        // Will be automatically padded
        CuBoolStatus status = result.resize(M, N);
        if (status != CUBOOL_STATUS_SUCCESS) {
            return status;
        }

        Matrix aGlobal = getMatrixFromDenseMatrixClass(a);
        Matrix bGlobal = getMatrixFromDenseMatrixClass(b);
        Matrix cGlobal = getMatrixFromDenseMatrixClass(c);
        Matrix rGlobal = getMatrixFromDenseMatrixClass(result);

        dim3 dimBLock(MatrixDense::PACK_TYPE_SIZE_BITS, 1);
        dim3 dimGrid(bGlobal.columns / MatrixDense::PACK_TYPE_SIZE_BITS, aGlobal.rows);

        kernelMatrixDenseMultiplyAdd<<<dimGrid,dimBLock>>>(rGlobal, aGlobal, bGlobal, cGlobal);

        return CUBOOL_STATUS_SUCCESS;
    }

}

