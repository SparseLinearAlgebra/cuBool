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

#include <algorithm>
#include <core/error.hpp>
#include <cuda/cuda_matrix.hpp>
#include <utils/data_utils.hpp>
#include <utils/timer.hpp>

namespace cubool {

    CudaMatrix::CudaMatrix(size_t nrows, size_t ncols, CudaInstance& instance) : mInstance(instance) {
        mNrows = nrows;
        mNcols = ncols;
    }

    void CudaMatrix::setElement(index i, index j) {
        RAISE_ERROR(NotImplemented, "This function is not supported for this matrix class");
    }

    void CudaMatrix::build(const index* rows, const index* cols, size_t nvals, bool isSorted, bool noDuplicates) {
        if (nvals == 0) {
            mMatrixImpl.zero_dim();// no content, empty matrix
            return;
        }

        // Build csr structure and store on cpu side
        std::vector<index> rowOffsets;
        std::vector<index> colIndices;

        DataUtils::buildFromData(getNrows(), getNcols(), rows, cols, nvals, rowOffsets, colIndices, isSorted, noDuplicates);

        // Move actual data to the matrix implementation
        this->transferToDevice(rowOffsets, colIndices);
    }

    void CudaMatrix::extract(index* rows, index* cols, size_t& nvals) {
        assert(nvals >= getNvals());

        // Set nvals to the exact number of nnz values
        nvals = getNvals();

        if (nvals > 0) {
            // Copy data to the host
            std::vector<index> rowOffsets;
            std::vector<index> colIndices;

            this->transferFromDevice(rowOffsets, colIndices);

            DataUtils::extractData(getNrows(), getNcols(), rows, cols, nvals, rowOffsets, colIndices);
        }
    }

    void CudaMatrix::clone(const MatrixBase& otherBase) {
        auto other = dynamic_cast<const CudaMatrix*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed matrix does not belong to csr matrix class");
        CHECK_RAISE_ERROR(other != this, InvalidArgument, "Matrices must differ");

        size_t M = other->getNrows();
        size_t N = other->getNcols();

        assert(this->getNrows() == M);
        assert(this->getNcols() == N);

        if (other->isMatrixEmpty()) {
            mMatrixImpl.zero_dim();
            return;
        }

        this->mMatrixImpl = other->mMatrixImpl;
    }

    void CudaMatrix::resizeStorageToDim() const {
        if (mMatrixImpl.is_zero_dim()) {
            // If actual storage was not allocated, allocate one for an empty matrix
            mMatrixImpl = std::move(MatrixImplType(mNrows, mNcols));
        }
    }

    void CudaMatrix::clearAndResizeStorageToDim() const {
        if (mMatrixImpl.m_vals > 0) {
            // Release only if have some nnz values
            mMatrixImpl.zero_dim();
        }

        // Normally resize if no storage is actually allocated
        this->resizeStorageToDim();
    }

    index CudaMatrix::getNrows() const {
        return mNrows;
    }

    index CudaMatrix::getNcols() const {
        return mNcols;
    }

    index CudaMatrix::getNvals() const {
        return mMatrixImpl.m_vals;
    }

    bool CudaMatrix::isMatrixEmpty() const {
        return mMatrixImpl.m_vals == 0;
    }

    void CudaMatrix::transferToDevice(const std::vector<index>& rowOffsets, const std::vector<index>& colIndices) const {
        // Create device buffers and copy data from the cpu side
        thrust::device_vector<index, DeviceAlloc<index>> rowsDeviceVec(rowOffsets.size());
        thrust::device_vector<index, DeviceAlloc<index>> colsDeviceVec(colIndices.size());

        thrust::copy(rowOffsets.begin(), rowOffsets.end(), rowsDeviceVec.begin());
        thrust::copy(colIndices.begin(), colIndices.end(), colsDeviceVec.begin());

        // Move actual data to the matrix implementation
        mMatrixImpl = std::move(MatrixImplType(std::move(colsDeviceVec), std::move(rowsDeviceVec), getNrows(), getNcols(), colIndices.size()));
    }

    void CudaMatrix::transferFromDevice(std::vector<index>& rowOffsets, std::vector<index>& colIndices) const {
        rowOffsets.resize(mMatrixImpl.m_row_index.size());
        colIndices.resize(mMatrixImpl.m_col_index.size());

        thrust::copy(mMatrixImpl.m_row_index.begin(), mMatrixImpl.m_row_index.end(), rowOffsets.begin());
        thrust::copy(mMatrixImpl.m_col_index.begin(), mMatrixImpl.m_col_index.end(), colIndices.begin());
    }

}// namespace cubool