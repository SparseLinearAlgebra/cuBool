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

#include <cubool/matrix_csr.cuh>
#include <cubool/details/error.hpp>
#include <algorithm>

namespace cubool {

    MatrixCsr::MatrixCsr(Instance &instance): Super(instance), mMatrixImpl(DeviceAlloc(instance)) {

    }

    void MatrixCsr::resize(CuBoolSize_t nrows, CuBoolSize_t ncols) {
        if ((mNumRows == nrows) && (mNumCols == ncols))
            return;

        mNumRows = nrows;
        mNumCols = ncols;
        mMatrixImpl.empty();  // no content, empty matrix
    }

    void MatrixCsr::build(const CuBoolIndex_t *rows, const CuBoolIndex_t *cols, CuBoolSize_t nvals) {
        // Create tmp buffer and pack indices into pairs
        using CuBoolPairAlloc = details::HostAllocator<CuBoolPair>;
        std::vector<CuBoolPair, CuBoolPairAlloc> values((CuBoolPairAlloc(getInstance())));
        values.reserve(nvals);

        for (CuBoolSize_t idx = 0; idx < nvals; idx++) {
            CuBoolIndex_t i = rows[idx];
            CuBoolIndex_t j = cols[idx];

            if ((i >= getNumRows()) || (j >= getNumCols()))
                throw details::InvalidArgument(std::string{"Out of matrix bounds value"});

            values.push_back({i, j});
        }

        // Sort pairs to ensure that the data in the proper format
        std::sort(values.begin(), values.end(), [](const CuBoolPair& a, const CuBoolPair& b) {
            return a.i < b.i || (a.i == b.i && a.j < b.j);
        });

        std::vector<IndexType, HostAlloc> rowsVec((HostAlloc(getInstance())));
        rowsVec.resize(getNumRows() + 1);

        std::vector<IndexType, HostAlloc> colsVec((HostAlloc(getInstance())));
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
        thrust::device_vector<IndexType, DeviceAlloc> rowsDeviceVec((DeviceAlloc(getInstance())));
        rowsDeviceVec = rowsVec;

        thrust::device_vector<IndexType, DeviceAlloc> colsDeviceVec((DeviceAlloc(getInstance())));
        colsDeviceVec = colsVec;

        // Move actual data to the matrix implementation
        mMatrixImpl = std::move(MatrixImplType(std::move(colsDeviceVec), std::move(rowsDeviceVec), getNumRows(), getNumCols(), nvals));
    }

    void MatrixCsr::extract(CuBoolIndex_t **rows, CuBoolIndex_t **cols, CuBoolSize_t *nvals) const {
        // Allocate host memory vectors and copy data from the device side
        auto& rowsDeviceVec = mMatrixImpl.m_row_index;
        auto& colsDeviceVec = mMatrixImpl.m_col_index;

        // Ensure that we allocated area for the copy operations
        std::vector<IndexType, HostAlloc> rowsVec((HostAlloc(getInstance())));
        rowsVec.resize(rowsDeviceVec.size());

        std::vector<IndexType, HostAlloc> colsVec((HostAlloc(getInstance())));
        colsVec.resize(colsDeviceVec.size());

        thrust::copy(rowsDeviceVec.begin(), rowsDeviceVec.end(), rowsVec.begin());
        thrust::copy(colsDeviceVec.begin(), colsDeviceVec.end(), colsVec.begin());

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

    void MatrixCsr::clone(const MatrixBase &other) {
        throw details::NotImplemented("This function is not implemented");
    }

    void MatrixCsr::multiplySum(const MatrixBase &a, const MatrixBase &b, const MatrixBase &c) {
        throw details::NotImplemented("This function is not implemented");
    }

    void MatrixCsr::multiplyAdd(const MatrixBase &a, const MatrixBase &b) {
        throw details::NotImplemented("This function is not implemented");
    }

}