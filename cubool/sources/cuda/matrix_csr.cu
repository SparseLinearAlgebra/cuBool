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
#include <utils/exclusive_scan.hpp>
#include <algorithm>

namespace cubool {

    MatrixCsr::MatrixCsr(size_t nrows, size_t ncols, Instance &instance) : mInstance(instance) {
        mNrows = nrows;
        mNcols = ncols;
    }

    void MatrixCsr::setElement(index i, index j) {
        RAISE_ERROR(NotImplemented, "This function is not supported for this matrix class");
    }

    void MatrixCsr::build(const index *rows, const index *cols, size_t nvals, bool isSorted, bool noDuplicates) {
        if (nvals == 0) {
            mMatrixImpl.zero_dim();  // no content, empty matrix
            return;
        }

        thrust::host_vector<index, HostAlloc<index>> rowOffsets;
        rowOffsets.resize(getNrows() + 1, 0);

        thrust::host_vector<index, HostAlloc<index>> colIndices;
        colIndices.resize(nvals);

        // Compute nnz per row
        for (size_t idx = 0; idx < nvals; idx++) {
            index i = rows[idx];
            index j = cols[idx];

            CHECK_RAISE_ERROR(i < getNrows() && j < getNcols(), InvalidArgument, "Out of matrix bounds value");

            rowOffsets[i] += 1;
        }

        // Exclusive scan to eval rows offsets
        ::cubool::exclusive_scan(rowOffsets.begin(), rowOffsets.end(), 0);

        // Write offsets for cols
        std::vector<size_t> writeOffsets(getNrows(), 0);

        for (size_t idx = 0; idx < nvals; idx++) {
            index i = rows[idx];
            index j = cols[idx];

            colIndices[rowOffsets[i] + writeOffsets[i]] = j;
            writeOffsets[i] += 1;
        }

        if (!isSorted) {
            for (size_t i = 0; i < getNrows(); i++) {
                auto begin = rowOffsets[i];
                auto end = rowOffsets[i + 1];

                // Sort col values within row
                thrust::sort(colIndices.begin() + begin, colIndices.begin() + end, [](const index& a, const index& b) {
                    return a < b;
                });
            }
        }

        // Reduce duplicated values
        if (!noDuplicates) {
            size_t unique = 0;
            for (size_t i = 0; i < getNrows(); i++) {
                index prev = std::numeric_limits<index>::max();

                for (size_t k = rowOffsets[i]; k < rowOffsets[i + 1]; k++) {
                    if (prev != colIndices[k]) {
                        unique += 1;
                    }

                    prev = colIndices[k];
                }
            }

            thrust::host_vector<index, HostAlloc<index>> rowOffsetsReduced;
            rowOffsetsReduced.resize(getNrows() + 1, 0);

            thrust::host_vector<index, HostAlloc<index>> colIndicesReduced;
            colIndicesReduced.reserve(unique);

            for (size_t i = 0; i < getNrows(); i++) {
                index prev = std::numeric_limits<index>::max();

                for (size_t k = rowOffsets[i]; k < rowOffsets[i + 1]; k++) {
                    if (prev != colIndices[k]) {
                        rowOffsetsReduced[i] += 1;
                        colIndicesReduced.push_back(colIndices[k]);
                    }

                    prev = colIndices[k];
                }
            }

            // Exclusive scan to eval rows offsets
            ::cubool::exclusive_scan(rowOffsetsReduced.begin(), rowOffsetsReduced.end(), 0);

            // Now result in respective place
            std::swap(rowOffsets, rowOffsetsReduced);
            std::swap(colIndices, colIndicesReduced);
        }

        // Create device buffers and copy data from the cpu side
        thrust::device_vector<index, DeviceAlloc<index>> rowsDeviceVec = rowOffsets;
        thrust::device_vector<index, DeviceAlloc<index>> colsDeviceVec = colIndices;

        // Move actual data to the matrix implementation
        mMatrixImpl = std::move(MatrixImplType(std::move(colsDeviceVec), std::move(rowsDeviceVec), getNrows(), getNcols(), colIndices.size()));
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