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

#ifndef CUBOOL_SPSUBMATRIX_CUH
#define CUBOOL_SPSUBMATRIX_CUH

#include <cuda/kernels/bin_search.cuh>
#include <thrust/device_vector.h>
#include <nsparse/matrix.h>

namespace cubool {
    namespace kernels {

        template <typename IndexType, typename AllocType>
        class SpSubMatrix {
        public:
            template<typename T>
            using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
            using MatrixType = nsparse::matrix<bool, IndexType, AllocType>;

            /**
             * Extract sub-matrix of the provided matrix.
             *
             * @param a Input matrix
             * @param i First row idx
             * @param j First col idx
             * @param nrows Num of rows to extract
             * @param ncols Num of cols to extract
             *
             * @return Sub-matrix
             */
            MatrixType operator()(const MatrixType& a, IndexType i, IndexType j, IndexType nrows, IndexType ncols) {
                auto& aRows = a.m_row_index;
                auto& aCols = a.m_col_index;
                auto aNrows = a.m_rows;

                auto max = std::numeric_limits<IndexType>::max();

                // Find region of suspected values
                IndexType firstValueId = aRows[i];
                IndexType lastValueId = aRows[i + nrows];

                rRowsTmp.resize(nrows + 1);
                rFirstColIndexOffset.resize(nrows);

                // To count nnz
                thrust::fill(rRowsTmp.begin(), rRowsTmp.end(), (IndexType) 0);

                // To count first nzz per row
                thrust::fill(rFirstColIndexOffset.begin(), rFirstColIndexOffset.end(), max);

                // Evaluate num of nnz per result row and eval relative offsets of first col idx within each row of sum-matrix
                thrust::for_each(thrust::counting_iterator<IndexType>(firstValueId), thrust::counting_iterator<IndexType>(lastValueId),
                    [aRows = aRows.data(), aCols = aCols.data(), aNrows,
                     rRowsTmp = rRowsTmp.data(),
                     rFirstColIndexOffset = rFirstColIndexOffset.data(),
                     i, j, nrows, ncols, firstValueId]
                    __device__ (IndexType valueId) {
                        IndexType colIdx = aCols[valueId];

                        if (colIdx >= j && colIdx < j + ncols) {
                            // Value a row id
                            IndexType aRowId = findNearestRowIdx(valueId, aNrows, aRows);
                            // Row id in result
                            IndexType rRowId = aRowId - i;
                            // Inc values count for row
                            atomicAdd((rRowsTmp + rRowId).get(), 1);

                            // First col index in sub-matrix
                            IndexType offset = valueId - aRows[aRowId];
                            atomicMin((rFirstColIndexOffset + rRowId).get(), offset);
                        }
                    }
                );

                ContainerType<IndexType> rRows(nrows + 1);

                // Evaluate rows offsets
                thrust::exclusive_scan(rRowsTmp.begin(), rRowsTmp.end(), rRows.begin(), 0, thrust::plus<IndexType>());

                // Get num of nnz and allocate actual storage
                size_t nvals = rRows.back();
                ContainerType<IndexType> rCols(nvals);

                // Write actual columns values in proper result matrix column indices and keep the order
                thrust::for_each(thrust::counting_iterator<IndexType>(firstValueId), thrust::counting_iterator<IndexType>(lastValueId),
                    [aRows = aRows.data(), aCols = aCols.data(), aNrows,
                     rRows = rRows.data(), rCols = rCols.data(),
                     rFirstColIndexOffset = rFirstColIndexOffset.data(),
                     i, j, nrows, ncols, firstValueId]
                    __device__ (IndexType valueId) {
                        IndexType colIdx = aCols[valueId];

                        if (colIdx >= j && colIdx < j + ncols) {
                            // Value a row id
                            IndexType aRowId = findNearestRowIdx(valueId, aNrows, aRows);
                            // Row id in result
                            IndexType rRowId = aRowId - i;

                            // Value col index offset
                            IndexType offset = valueId - aRows[aRowId];
                            IndexType baseOffset = rFirstColIndexOffset[rRowId];
                            IndexType relativeOffset = offset - baseOffset;

                            // Write value in the concrete place, do not break the order
                            IndexType rowOffset = rRows[rRowId];
                            rCols[rowOffset + relativeOffset] = aCols[valueId] - j;
                        }
                    }
                );

                assert(rCols.size() == nvals);
                assert(rRows.size() == nrows + 1);
                assert(rRows.back() == nvals);

                return MatrixType(std::move(rCols), std::move(rRows), nrows, ncols, nvals);
            }

        protected:
            ContainerType<IndexType> rRowsTmp;
            ContainerType<IndexType> rFirstColIndexOffset;
        };

    }
}

#endif //CUBOOL_SPSUBMATRIX_CUH
