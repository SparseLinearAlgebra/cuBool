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

#include <cubool/matrix_dense.cuh>
#include <cubool/instance.hpp>
#include <cubool/kernels/matrix_dense_multiply_sum.cuh>

namespace cubool {

    MatrixDense::MatrixDense(class Instance &instance) : Super(instance), mBuffer(GpuAllocator(mInstanceRef)) {

    }

    void MatrixDense::resize(CuBoolSize_t nrows, CuBoolSize_t ncols) {
        if (nrows == mNumRows && ncols == mNumCols)
            return;

        mNumRows = nrows;
        mNumCols = ncols;
        mNumRowsPacked = getNumRowsPackedFromRows(mNumRows);
        mNumColsPadded = getNumColsPaddedFromCols(mNumCols);

        return mBuffer.resize(mNumColsPadded * mNumRowsPacked, 0x0);
    }

    void MatrixDense::build(const CuBoolIndex_t *rows, const CuBoolIndex_t *cols, CuBoolSize_t nvals) {
        if (nvals == 0)
            return;

        if (mBuffer.empty())
            throw details::InvalidState("An attempt to write to empty matrix");

        CpuBuffer cpuBuffer((CpuAllocator(mInstanceRef)));
        cpuBuffer.resize(mBuffer.size(), 0x0);

        for (CuBoolSize_t idx = 0; idx < nvals; idx++) {
            CuBoolSize_t rowIdx = rows[idx];
            CuBoolSize_t columnIdx = cols[idx];

            if (rowIdx >= getNumRows() || columnIdx >= getNumCols())
                throw details::InvalidArgument(std::string{"Out of matrix bounds value"});

            CuBoolSize_t i;
            CuBoolSize_t k;
            CuBoolSize_t j = columnIdx;

            getRowPackedIndex(rowIdx, i, k);

            cpuBuffer[i * mNumColsPadded + j] |= (1u << k);
        }

        mBuffer = cpuBuffer;
    }

    void MatrixDense::extract(CuBoolIndex_t **rows, CuBoolIndex_t **cols, CuBoolSize_t *nvals) const {
        std::vector<CuBoolPair> vals((CpuAllocator(mInstanceRef)));
        extractVector(vals);

        *nvals = vals.size();
        mInstanceRef.allocate((CuBoolCpuPtr_t*)rows, sizeof(CuBoolIndex_t) * vals.size());
        mInstanceRef.allocate((CuBoolCpuPtr_t*)cols, sizeof(CuBoolIndex_t) * vals.size());

        CuBoolIndex_t* rowsp = *rows;
        CuBoolIndex_t* colsp = *cols;

        for (CuBoolSize_t idx = 0; idx < vals.size(); idx++) {
            rowsp[idx] = vals[idx].i;
            colsp[idx] = vals[idx].j;
        }
    }

    void MatrixDense::multiplySum(const MatrixBase &aBase, const MatrixBase &bBase, const MatrixBase &cBase) {
        auto a = dynamic_cast<const MatrixDense*>(&aBase);
        auto b = dynamic_cast<const MatrixDense*>(&bBase);
        auto c = dynamic_cast<const MatrixDense*>(&cBase);

        if (!a || !b || !c)
            throw details::InvalidArgument("Passed matrices do not belong to dense matrix class");

        if (a->isZeroDim() || b->isZeroDim() || c->isZeroDim())
            throw details::InvalidArgument("An attempt to operate on 0-dim matrices");

        if (a->getNumCols() != b->getNumRows())
            throw details::InvalidArgument("Incompatible matrix size to multiply");

        CuBoolSize_t M = a->getNumRows();
        CuBoolSize_t N = b->getNumCols();

        if (c->getNumRows() != M || c->getNumCols() != N)
            throw details::InvalidArgument("Incompatible matrix size to add");

        this->resize(M, N);

        Matrix aGlobal = getMatrixFromDenseMatrixClass(*a);
        Matrix bGlobal = getMatrixFromDenseMatrixClass(*b);
        Matrix cGlobal = getMatrixFromDenseMatrixClass(*c);
        Matrix rGlobal = getMatrixFromDenseMatrixClass(*this);

        dim3 dimBLock;
        dim3 dimGrid;
        getGridConfig(aGlobal.rows, bGlobal.columns, dimBLock, dimGrid);

        kernelMatrixDenseMultiplyAdd<<<dimGrid,dimBLock>>>(rGlobal, aGlobal, bGlobal, cGlobal);
    }

    void MatrixDense::getRowPackedIndex(CuBoolSize_t rowIndex, CuBoolSize_t &rowPackIdxMajor, CuBoolSize_t &rowPackIdxMinor) {
        rowPackIdxMajor = rowIndex / PACK_TYPE_SIZE_BITS;
        rowPackIdxMinor = rowIndex % PACK_TYPE_SIZE_BITS;
    }

    CuBoolSize_t MatrixDense::getNumRowsPackedFromRows(CuBoolSize_t rows) {
        return rows / PACK_TYPE_SIZE_BITS + (rows % PACK_TYPE_SIZE_BITS? 1: 0);
    }

    CuBoolSize_t MatrixDense::getNumColsPaddedFromCols(CuBoolSize_t cols) {
        return cols + (cols % PACK_TYPE_SIZE_BITS ? PACK_TYPE_SIZE_BITS - cols % PACK_TYPE_SIZE_BITS : 0);
    }

    void MatrixDense::extractVector(std::vector<CuBoolPair> &vals) const {
        if (mBuffer.empty())
            return;

        CpuBuffer cpuBuffer((CpuAllocator(mInstanceRef)));
        cpuBuffer = mBuffer;

        for (CuBoolSize_t i = 0; i < mNumRowsPacked; i++) {
            for (CuBoolSize_t j = 0; j < getNumCols(); j++) {
                PackType_t pack = cpuBuffer[i * mNumColsPadded + j];

                for (CuBoolSize_t k = 0; k < PACK_TYPE_SIZE_BITS; k++) {
                    CuBoolSize_t rowIdx = i * PACK_TYPE_SIZE_BITS + k;
                    CuBoolSize_t columnIdx = j;

                    if (rowIdx < getNumRows() && ((pack & 0x1u) != 0x0u)) {
                        vals.push_back(CuBoolPair{ rowIdx, columnIdx });
                    }

                    pack = pack >> 1u;
                }
            }
        }
    }

}