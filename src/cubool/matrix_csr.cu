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
#include <algorithm>

namespace cubool {

    MatrixCsr::MatrixCsr(Instance &instance): Super(instance) {

    }

    void MatrixCsr::resize(index nrows, index ncols) {
        if ((mNumRows == nrows) && (mNumCols == ncols))
            return;

        mNumRows = nrows;
        mNumCols = ncols;
        mMatrixImpl.zero_dim();  // no content, empty matrix
    }

    void MatrixCsr::build(const index *rows, const index *cols, size nvals, bool isSorted) {
        if (nvals == 0) {
            mMatrixImpl.zero_dim();  // no content, empty matrix
            return;
        }

        thrust::host_vector<index, HostAlloc<index>> rowsVec;
        rowsVec.resize(getNumRows() + 1, 0);

        thrust::host_vector<index, HostAlloc<index>> colsVec;
        colsVec.reserve(nvals);

        if (isSorted) {
            // Assign cols indices and count num nnz per each row
            for (size_t idx = 0; idx < nvals; idx++) {
                index i = rows[idx];
                index j = cols[idx];

                if ((i >= getNumRows()) || (j >= getNumCols()))
                    throw details::InvalidArgument(std::string{"Out of matrix bounds value"});

                rowsVec[i] += 1;
                colsVec.push_back(j);
            }
        }
        else {
            // Create tmp buffer and pack indices into pairs
            using CuBoolPairAlloc = details::HostAllocator<Pair>;

            std::vector<Pair, CuBoolPairAlloc> values;
            values.reserve(nvals);

            for (size idx = 0; idx < nvals; idx++) {
                index i = rows[idx];
                index j = cols[idx];

                if ((i >= getNumRows()) || (j >= getNumCols()))
                    throw details::InvalidArgument(std::string{"Out of matrix bounds value"});

                values.push_back({i, j});
            }

            // Sort pairs to ensure that data are in the proper format
            std::sort(values.begin(), values.end(), [](const Pair& a, const Pair& b) {
                return a.i < b.i || (a.i == b.i && a.j < b.j);
            });

            // Assign cols indices and count num nnz per each row
            for (const auto& p: values) {
                rowsVec[p.i] += 1;
                colsVec.push_back(p.j);
            }
        }

        // Exclusive scan to eval rows offsets
        index sum = 0;
        for (size_t i = 0; i < rowsVec.size(); i++) {
            index prevSum = sum;
            sum += rowsVec[i];
            rowsVec[i] = prevSum;
        }

        // Create device buffers and copy data from the cpu side
        thrust::device_vector<index, DeviceAlloc<index>> rowsDeviceVec = rowsVec;
        thrust::device_vector<index, DeviceAlloc<index>> colsDeviceVec = colsVec;

        // Move actual data to the matrix implementation
        mMatrixImpl = std::move(MatrixImplType(std::move(colsDeviceVec), std::move(rowsDeviceVec), getNumRows(), getNumCols(), nvals));
    }

    void MatrixCsr::extract(index *rows, index *cols, size_t &nvals) {
        if (getNumVals() > nvals)
            throw details::InvalidArgument("Provided buffers sizes must be greater or equal the number of values in the matrix");

        // Set nvals to the exact number of nnz values
        nvals = getNumVals();

        auto& rowsDeviceVec = mMatrixImpl.m_row_index;
        auto& colsDeviceVec = mMatrixImpl.m_col_index;

        // Copy data to the host
        thrust::host_vector<index, HostAlloc<index>> rowsVec = rowsDeviceVec;
        thrust::host_vector<index, HostAlloc<index>> colsVec = colsDeviceVec;

        if (nvals > 0) {
            // Iterate over csr formatted data
            size idx = 0;
            for (index i = 0; i < getNumRows(); i++) {
                for (index j = rowsVec[i]; j < rowsVec[i + 1]; j++) {
                    rows[idx] = i;
                    cols[idx] = colsVec[j];

                    idx += 1;
                }
            }
        }
    }

    void MatrixCsr::extractExt(index* &rows, index* &cols, size_t &nvals) const {
        // Allocate host memory vectors and copy data from the device side
        auto& rowsDeviceVec = mMatrixImpl.m_row_index;
        auto& colsDeviceVec = mMatrixImpl.m_col_index;

        // Ensure that we allocated area for the copy operations
        thrust::host_vector<index, HostAlloc<index>> rowsVec = rowsDeviceVec;
        thrust::host_vector<index, HostAlloc<index>> colsVec = colsDeviceVec;

        auto valuesCount = mMatrixImpl.m_vals;

        // Set values count and allocate buffers for rows/cols indices, returned to the caller
        nvals = valuesCount;

        if (nvals > 0) {
            mInstanceRef.allocate((void* &) rows, sizeof(index) * valuesCount);
            mInstanceRef.allocate((void* &) cols, sizeof(index) * valuesCount);

            // Iterate over csr formatted data
            size idx = 0;
            for (index i = 0; i < getNumRows(); i++) {
                for (index j = rowsVec[i]; j < rowsVec[i + 1]; j++) {
                    rows[idx] = i;
                    cols[idx] = colsVec[j];

                    idx += 1;
                }
            }
        }
    }

    void MatrixCsr::clone(const MatrixBase &otherBase) {
        auto other = dynamic_cast<const MatrixCsr*>(&otherBase);

        if (!other)
            throw details::InvalidArgument("Passed to be cloned matrix does not belong to csr matrix class");

        size M = other->getNumRows();
        size N = other->getNumCols();

        this->resize(M, N);
        this->mMatrixImpl = other->mMatrixImpl;
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