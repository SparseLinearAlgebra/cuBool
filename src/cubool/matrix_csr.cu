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

#include <cubool/matrix_csr.hpp>
#include <cubool/details/error.hpp>
#include <cubool/kernels/matrix_csr_spkron.cuh>
#include <nsparse/spgemm.h>
#include <algorithm>

namespace cubool {

    MatrixCsr::MatrixCsr(Instance &instance): Super(instance) {

    }

    void MatrixCsr::resize(CuBoolSize_t nrows, CuBoolSize_t ncols) {
        if ((mNumRows == nrows) && (mNumCols == ncols))
            return;

        mNumRows = nrows;
        mNumCols = ncols;
        mMatrixImpl.zero_dim();  // no content, empty matrix
    }

    void MatrixCsr::build(const CuBoolIndex_t *rows, const CuBoolIndex_t *cols, CuBoolSize_t nvals) {
        // Create tmp buffer and pack indices into pairs
        using CuBoolPairAlloc = details::HostAllocator<CuBoolPair>;
        std::vector<CuBoolPair, CuBoolPairAlloc> values;
        values.reserve(nvals);

        for (CuBoolSize_t idx = 0; idx < nvals; idx++) {
            CuBoolIndex_t i = rows[idx];
            CuBoolIndex_t j = cols[idx];

            if ((i >= getNumRows()) || (j >= getNumCols()))
                throw details::InvalidArgument(std::string{"Out of matrix bounds value"});

            values.push_back({i, j});
        }

        // Sort pairs to ensure that data are in the proper format
        std::sort(values.begin(), values.end(), [](const CuBoolPair& a, const CuBoolPair& b) {
            return a.i < b.i || (a.i == b.i && a.j < b.j);
        });

        thrust::host_vector<IndexType, HostAlloc<IndexType>> rowsVec;
        rowsVec.resize(getNumRows() + 1);

        thrust::host_vector<IndexType, HostAlloc<IndexType>> colsVec;
        colsVec.reserve(nvals);

        {
            // Start from the first (indexed as 0) row
            CuBoolSize_t currentRow = 0;
            rowsVec[currentRow] = 0;

            for (CuBoolSize_t idx = 0; idx < nvals; idx++) {
                CuBoolIndex_t i = values[idx].i;
                CuBoolIndex_t j = values[idx].j;

                // When go to the new line commit previous offsets
                if (currentRow < i) {
                    do {
                        // Since we can skip some lines and explicitly offsets here
                        currentRow += 1;
                        rowsVec[currentRow] = colsVec.size();
                    } while (currentRow != i);
                }

                // Add column index
                colsVec.push_back(j);
            }

            // Before finish commit the rest of the rows (note: rows_vec is from [0 .. nrows])
            do {
                currentRow += 1;
                rowsVec[currentRow] = colsVec.size();
            } while (currentRow != getNumRows());
        }

        // Create device buffers and copy data from the cpu side
        thrust::device_vector<IndexType, DeviceAlloc<IndexType>> rowsDeviceVec = rowsVec;
        thrust::device_vector<IndexType, DeviceAlloc<IndexType>> colsDeviceVec = colsVec;

        // Move actual data to the matrix implementation
        mMatrixImpl = std::move(MatrixImplType(std::move(colsDeviceVec), std::move(rowsDeviceVec), getNumRows(), getNumCols(), nvals));
    }

    void MatrixCsr::extract(CuBoolIndex_t **rows, CuBoolIndex_t **cols, CuBoolSize_t *nvals) const {
        // Allocate host memory vectors and copy data from the device side
        auto& rowsDeviceVec = mMatrixImpl.m_row_index;
        auto& colsDeviceVec = mMatrixImpl.m_col_index;

        // Ensure that we allocated area for the copy operations
        thrust::host_vector<IndexType, HostAlloc<IndexType>> rowsVec = rowsDeviceVec;
        thrust::host_vector<IndexType, HostAlloc<IndexType>> colsVec = colsDeviceVec;

        auto valuesCount = mMatrixImpl.m_vals;

        // Set values count and allocate buffers for rows/cols indices, returned to the caller
        *nvals = valuesCount;
        mInstanceRef.allocate((CuBoolCpuPtr_t*)rows, sizeof(CuBoolIndex_t) * valuesCount);
        mInstanceRef.allocate((CuBoolCpuPtr_t*)cols, sizeof(CuBoolIndex_t) * valuesCount);

        CuBoolIndex_t* rowsp = *rows;
        CuBoolIndex_t* colsp = *cols;

        // Iterate over csr formatted data
        CuBoolSize_t idx = 0;
        for (CuBoolSize_t i = 0; i < getNumRows(); i++) {
            for (CuBoolSize_t j = rowsVec[i]; j < rowsVec[i + 1]; j++) {
                rowsp[idx] = i;
                colsp[idx] = colsVec[j];

                idx += 1;
            }
        }
    }

    void MatrixCsr::clone(const MatrixBase &otherBase) {
        auto other = dynamic_cast<const MatrixCsr*>(&otherBase);

        if (!other)
            throw details::InvalidArgument("Passed to be cloned matrix does not belong to csr matrix class");

        CuBoolSize_t M = other->getNumRows();
        CuBoolSize_t N = other->getNumCols();

        this->resize(M, N);
        this->mMatrixImpl = other->mMatrixImpl;
    }

    void MatrixCsr::multiplySum(const MatrixBase &aBase, const MatrixBase &bBase, const MatrixBase &cBase) {
        auto a = dynamic_cast<const MatrixCsr*>(&aBase);
        auto b = dynamic_cast<const MatrixCsr*>(&bBase);
        auto c = dynamic_cast<const MatrixCsr*>(&cBase);

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

        if (a->isMatrixEmpty() || b->isMatrixEmpty()) {
            // A or B emtpy, therefore result equals C matrix data
            this->resize(M, N);
            this->mMatrixImpl = c->mMatrixImpl;
            return;
        }

        // Call backend r = c + a * b implementation
        nsparse::spgemm_functor_t<bool, IndexType, DeviceAlloc<IndexType>> spgemmFunctor;
        auto result = std::move(spgemmFunctor(c->mMatrixImpl, a->mMatrixImpl, b->mMatrixImpl));

        // Assign result r to this matrix
        this->resize(M, N);
        this->mMatrixImpl = std::move(result);
    }

    void MatrixCsr::multiplyAdd(const MatrixBase &aBase, const MatrixBase &bBase) {
        auto a = dynamic_cast<const MatrixCsr*>(&aBase);
        auto b = dynamic_cast<const MatrixCsr*>(&bBase);

        if (!a || !b)
            throw details::InvalidArgument("Passed matrices do not belong to csr matrix class");

        if (a->isZeroDim() || b->isZeroDim())
            throw details::InvalidArgument("An attempt to operate on 0-dim matrices");

        if (a->getNumCols() != b->getNumRows())
            throw details::InvalidArgument("Incompatible matrix size to multiply");

        CuBoolSize_t M = a->getNumRows();
        CuBoolSize_t N = b->getNumCols();

        if (this->getNumRows() != M || this->getNumCols() != N)
            throw details::InvalidArgument("Incompatible matrix size to add");

        if (a->isMatrixEmpty() || b->isMatrixEmpty()) {
            // A or B has no values
            return;
        }

        if (this->isStorageEmpty()) {
            // If this was resized but actual data was not allocated
            this->resizeStorageToDim();
        }

        // Call backend r = c + a * b implementation, as C this is passed
        nsparse::spgemm_functor_t<bool, IndexType, DeviceAlloc<IndexType>> spgemmFunctor;
        auto result = std::move(spgemmFunctor(mMatrixImpl, a->mMatrixImpl, b->mMatrixImpl));

        // Assign result to this
        this->resize(M, N);
        this->mMatrixImpl = std::move(result);
    }

    void MatrixCsr::kron(const MatrixBase &aBase, const MatrixBase &bBase) {
        auto a = dynamic_cast<const MatrixCsr*>(&aBase);
        auto b = dynamic_cast<const MatrixCsr*>(&bBase);

        if (!a || !b)
            throw details::InvalidArgument("Passed matrices do not belong to csr matrix class");

        kernels::SpKronFunctor<IndexType, DeviceAlloc<IndexType>> spKronFunctor;
        auto result = std::move(spKronFunctor(a->mMatrixImpl, b->mMatrixImpl));
    }

    void MatrixCsr::resizeStorageToDim() {
        if (mMatrixImpl.is_zero_dim()) {
            // If actual storage was not allocated, allocate one for an empty matrix
            mMatrixImpl = std::move(MatrixImplType(mNumRows, mNumCols));
        }
    }

    bool MatrixCsr::isStorageEmpty() const {
        return mMatrixImpl.is_zero_dim();
    }

    bool MatrixCsr::isMatrixEmpty() const {
        return mMatrixImpl.m_vals == 0;
    }

}