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

#include <core/error.hpp>
#include <cuda/cuda_matrix.hpp>
#include <cuda/cuda_vector.hpp>
#include <cuda/kernels/bin_search.cuh>
#include <limits>
#include <utils/data_utils.hpp>

#include <thrust/sort.h>

namespace cubool {

    CudaVector::CudaVector(size_t nrows, CudaInstance& instance)
        : mVectorImpl(nrows), mInstance(instance) {
    }

    void CudaVector::setElement(index i) {
        RAISE_ERROR(NotImplemented, "This function is not supported for this vector class");
    }

    void CudaVector::build(const index* rows, size_t nvals, bool isSorted, bool noDuplicates) {
        if (nvals == 0) {
            // Empty vector, no values (but preserve dim)
            mVectorImpl = VectorImplType(getNrows());
            return;
        }

        // Validate data, sort, remove duplicates and etc.
        std::vector<index> data;
        DataUtils::buildVectorFromData(getNrows(), rows, nvals, data, isSorted, noDuplicates);

        // Transfer data to GPU
        thrust::device_vector<index, DeviceAlloc<index>> deviceData(data.size());
        thrust::copy(data.begin(), data.end(), deviceData.begin());

        // New vec instance
        mVectorImpl = VectorImplType(std::move(deviceData), getNrows(), data.size());
    }

    void CudaVector::extract(index* rows, size_t& nvals) {
        assert(nvals >= getNvals());

        nvals = getNvals();

        if (nvals > 0) {
            assert(rows);

            // Transfer data from GPU
            thrust::copy(mVectorImpl.m_rows_index.begin(), mVectorImpl.m_rows_index.end(), rows);
        }
    }

    void CudaVector::extractSubVector(const VectorBase& otherBase, index i, index nrows, bool checkTime) {
        const auto* v = dynamic_cast<const CudaVector*>(&otherBase);

        CHECK_RAISE_ERROR(v != nullptr, InvalidArgument, "Passed vector does not belong to cuda vector class");

        assert(i + nrows <= v->getNrows());
        assert(getNrows() == nrows);
        assert(nrows > 0);

        // If source is empty
        if (v->getNvals() == 0) {
            mVectorImpl = std::move(VectorImplType(nrows));
            return;
        }

        // If source has too small values
        index last = v->mVectorImpl.m_rows_index.back();
        if (i > last) {
            mVectorImpl = std::move(VectorImplType(nrows));
            return;
        }

        auto& vec = v->mVectorImpl;

        thrust::device_vector<index, DeviceAlloc<index>> region(2);
        thrust::fill_n(region.begin(), 1, std::numeric_limits<index>::max());
        thrust::fill_n(region.begin() + 1, 1, 0);

        thrust::for_each(thrust::counting_iterator<index>(0), thrust::counting_iterator<index>(vec.m_vals),
                         [first = region.data(), size = region.data() + 1,
                          i = i, last = i + nrows, rowIndex = vec.m_rows_index.data()] __device__(index id) {
                             auto rowId = rowIndex[id];

                             if (i <= rowId && rowId < last) {
                                 atomicAdd(size.get(), 1);
                                 atomicMin(first.get(), id);
                             }
                         });

        index resultSize = region.back();

        // If no values for result
        if (resultSize == 0) {
            mVectorImpl = std::move(VectorImplType(nrows));
            return;
        }

        // Copy region to the result
        index firstToCopy = region.front();

        VectorImplType::container_type result(resultSize);
        thrust::transform(vec.m_rows_index.begin() + firstToCopy, vec.m_rows_index.begin() + firstToCopy + resultSize, result.begin(),
                          [i] __device__(index id) { return id - i; });

        // Update this impl data
        mVectorImpl = std::move(VectorImplType(std::move(result), nrows, resultSize));
    }

    void CudaVector::extractRow(const class MatrixBase& matrixBase, index i) {
        auto matrix = dynamic_cast<const CudaMatrix*>(&matrixBase);

        CHECK_RAISE_ERROR(matrix != nullptr, InvalidArgument, "Provided matrix does not belongs to cuda matrix class");

        assert(getNrows() == matrix->getNcols());
        assert(i <= matrix->getNrows());

        auto& m = matrix->mMatrixImpl;

        index beginOffset = m.m_row_index[i];
        index endOffset   = m.m_row_index[i + 1];

        auto size  = endOffset - beginOffset;
        auto being = m.m_col_index.begin() + beginOffset;
        auto end   = m.m_col_index.begin() + endOffset;

        VectorImplType::container_type result(size);
        thrust::copy(being, end, result.begin());

        mVectorImpl = std::move(VectorImplType(std::move(result), m.m_cols, size));
    }

    void CudaVector::extractCol(const class MatrixBase& matrixBase, index j) {
        auto matrix = dynamic_cast<const CudaMatrix*>(&matrixBase);

        CHECK_RAISE_ERROR(matrix != nullptr, InvalidArgument, "Provided matrix does not belongs to cuda matrix class");

        assert(getNrows() == matrix->getNrows());
        assert(j <= matrix->getNcols());

        auto& m = matrix->mMatrixImpl;

        VectorImplType::container_type nnz(1);
        thrust::fill(nnz.begin(), nnz.end(), (index) 0);

        thrust::for_each(thrust::counting_iterator<index>(0), thrust::counting_iterator<index>(m.m_rows),
                         [rowOffset = m.m_row_index.data(), colIndex = m.m_col_index.data(),
                          j, nnz                                     = nnz.data()] __device__(index i) {
                             auto size  = rowOffset[i + 1] - rowOffset[i];
                             auto begin = colIndex + rowOffset[i];
                             auto end   = colIndex + rowOffset[i + 1];

                             auto r = kernels::find<index>(begin, end, j);

                             if (r != end && *r == j)
                                 atomicAdd(nnz.get(), 1);
                         });

        index                          size = nnz.back();
        VectorImplType::container_type result(size);

        thrust::fill(nnz.begin(), nnz.end(), (index) 0);

        thrust::for_each(thrust::counting_iterator<index>(0), thrust::counting_iterator<index>(m.m_rows),
                         [rowOffset = m.m_row_index.data(), colIndex = m.m_col_index.data(),
                          j, nnz = nnz.data(), result = result.data()] __device__(index i) {
                             auto size  = rowOffset[i + 1] - rowOffset[i];
                             auto begin = colIndex + rowOffset[i];
                             auto end   = colIndex + rowOffset[i + 1];

                             auto r = kernels::find<index>(begin, end, j);

                             if (r != end && *r == j) {
                                 auto order    = atomicAdd(nnz.get(), 1);
                                 result[order] = i;
                             }
                         });

        thrust::sort(result.begin(), result.end());

        mVectorImpl = std::move(VectorImplType(std::move(result), m.m_rows, size));
    }

    void CudaVector::clone(const VectorBase& otherBase) {
        auto other = dynamic_cast<const CudaVector*>(&otherBase);

        CHECK_RAISE_ERROR(other != nullptr, InvalidArgument, "Passed vector does not belong to vector class");
        CHECK_RAISE_ERROR(other != this, InvalidArgument, "Vectors must differ");

        assert(this->getNrows() == other->getNrows());
        this->mVectorImpl = other->mVectorImpl;
    }

    void CudaVector::reduce(index& result, bool checkTime) {
        result = getNvals();
    }

    index CudaVector::getNrows() const {
        return mVectorImpl.m_rows;
    }

    index CudaVector::getNvals() const {
        return mVectorImpl.m_vals;
    }

}// namespace cubool