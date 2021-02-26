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

#include <cuda/matrix_csr.hpp>
#include <core/error.hpp>
#include <algorithm>

namespace cubool {

    MatrixCsr::MatrixCsr(size_t nrows, size_t ncols, Instance &instance) : mInstance(instance) {
        mNrows = nrows;
        mNcols = ncols;
    }

    void MatrixCsr::build(const index *rows, const index *cols, size_t nvals, bool isSorted) {
        if (nvals == 0) {
            mMatrixImpl.zero_dim();  // no content, empty matrix
            return;
        }

        thrust::host_vector<index, HostAlloc<index>> rowsVec;
        rowsVec.resize(getNrows() + 1, 0);

        thrust::host_vector<index, HostAlloc<index>> colsVec;
        colsVec.reserve(nvals);

        if (isSorted) {
            // Assign cols indices and count num nnz per each row
            for (size_t idx = 0; idx < nvals; idx++) {
                index i = rows[idx];
                index j = cols[idx];

                CHECK_RAISE_ERROR(i < getNrows() && j < getNcols(), InvalidArgument, "Out of matrix bounds value");

                rowsVec[i] += 1;
                colsVec.push_back(j);
            }
        }
        else {
            // Create tmp buffer and pack indices into pairs
            using CuBoolPairAlloc = details::HostAllocator<Pair>;

            std::vector<Pair, CuBoolPairAlloc> values;
            values.reserve(nvals);

            for (size_t idx = 0; idx < nvals; idx++) {
                index i = rows[idx];
                index j = cols[idx];

                CHECK_RAISE_ERROR(i < getNrows() && j < getNcols(), InvalidArgument, "Out of matrix bounds value");

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
        mMatrixImpl = std::move(MatrixImplType(std::move(colsDeviceVec), std::move(rowsDeviceVec), getNrows(), getNcols(), nvals));
    }

    void MatrixCsr::extract(index *rows, index *cols, size_t &nvals) {
        assert(nvals >= getNvals());

        // Set nvals to the exact number of nnz values
        nvals = getNvals();

        if (nvals > 0) {
            auto& rowsDeviceVec = mMatrixImpl.m_row_index;
            auto& colsDeviceVec = mMatrixImpl.m_col_index;

            // Copy data to the host
            thrust::host_vector<index, HostAlloc<index>> rowsVec = rowsDeviceVec;
            thrust::host_vector<index, HostAlloc<index>> colsVec = colsDeviceVec;

            // Iterate over csr formatted data
            size_t idx = 0;
            for (index i = 0; i < getNrows(); i++) {
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

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed matrix does not belong to csr matrix class");

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

    void MatrixCsr::resizeStorageToDim() const {
        if (mMatrixImpl.is_zero_dim()) {
            // If actual storage was not allocated, allocate one for an empty matrix
            mMatrixImpl = std::move(MatrixImplType(mNrows, mNcols));
        }
    }

    index MatrixCsr::getNrows() const {
        return mNrows;
    }

    index MatrixCsr::getNcols() const {
        return mNcols;
    }

    index MatrixCsr::getNvals() const {
        return mMatrixImpl.m_vals;
    }

    bool MatrixCsr::isStorageEmpty() const {
        return mMatrixImpl.is_zero_dim();
    }

    bool MatrixCsr::isMatrixEmpty() const {
        return mMatrixImpl.m_vals == 0;
    }

}