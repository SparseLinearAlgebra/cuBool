/**********************************************************************************/
/*                                                                                */
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
/*                                                                                */
/**********************************************************************************/

#ifndef CUBOOL_SPTRANSPOSE2_CUH
#define CUBOOL_SPTRANSPOSE2_CUH

#include <cuda/kernels/bin_search.cuh>
#include <thrust/device_vector.h>
#include <nsparse/matrix.h>

namespace cubool {
    namespace kernels {

        /**
         * Transpose based on
         *
         * @tparam IndexType
         * @tparam AllocType
         */
        template <typename IndexType, typename AllocType>
        class SpTranspose2Functor {
        public:
            template<typename T>
            using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
            using MatrixType = nsparse::matrix<bool, IndexType, AllocType>;

            /**
             * evaluates r = a^T, were r equals transposed matrix a.
             *
             * @param a Csr matrix
             *
             * @return a^T
             */
            MatrixType operator()(const MatrixType& a) {
                auto nrows = a.m_rows;
                auto ncols = a.m_cols;
                auto nvals = a.m_vals;
                auto aRowIndex = a.m_row_index;
                auto aColIndex = a.m_col_index;

                // Create row index and column index to store res matrix
                // Note: transposed matrix has exact the same nnz
                ContainerType<IndexType> rowIndices(nvals);
                ContainerType<IndexType> colIndices(nvals);
                ContainerType<IndexType> colOrder(nvals);

                // Copy col indices of a as row indices or aT
                thrust::copy(aColIndex.begin(), aColIndex.end(), rowIndices.begin());

                // Compute column indices of the aT for each value of a matrix
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(nvals),
                    [aRowIndex = aRowIndex.data(), nrows,
                     colIndices = colIndices.data()]
                    __device__ (IndexType valueId){
                        IndexType rowId = findNearestRowIdx<IndexType>(valueId, (IndexType) nrows, aRowIndex);
                        colIndices[valueId] = rowId;
                });

                // Sort row-col indices
                thrust::sort_by_key(rowIndices.begin(), rowIndices.end(), colIndices.begin());

                // Compute row offsets, based on row indices
                ContainerType<IndexType> rowOffsetsTmp(ncols + 1);
                ContainerType<IndexType> rowOffsets(ncols + 1);

                thrust::fill(rowOffsets.begin(), rowOffsets.end(), (IndexType) 0);
                thrust::for_each(rowIndices.begin(), rowIndices.end(), [rowOffsetsTmp = rowOffsetsTmp.data()]__device__(IndexType rowId) {
                    atomicAdd((rowOffsetsTmp + rowId).get(), 1);
                });

                // Compute actual offsets
                thrust::exclusive_scan(rowOffsetsTmp.begin(), rowOffsetsTmp.end(), rowOffsets.begin(), 0, thrust::plus<IndexType>());

                assert(colIndices.size() == nvals);
                assert(rowOffsets.size() == ncols + 1);
                assert(rowOffsets.back() == colIndices.size());

                return MatrixType(std::move(colIndices), std::move(rowOffsets), ncols, nrows, nvals);
            }
        };

    }
}

#endif //CUBOOL_SPTRANSPOSE2_CUH
