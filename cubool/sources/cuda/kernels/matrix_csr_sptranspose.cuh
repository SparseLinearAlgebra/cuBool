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

#ifndef CUBOOL_MATRIX_CSR_SPTRANSPOSE_HPP
#define CUBOOL_MATRIX_CSR_SPTRANSPOSE_HPP

#include <nsparse/matrix.h>
#include <thrust/device_vector.h>

namespace cubool {
    namespace kernels {

        template <typename IndexType, typename AllocType>
        class SpTransposeFunctor {
        public:
            template<typename T>
            using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
            using MatrixType = nsparse::matrix<bool, IndexType, AllocType>;

            static __device__ IndexType findNearestRowIdx(IndexType value, IndexType range, thrust::device_ptr<const IndexType> rowIndex) {
                IndexType left = 0;
                IndexType right = range - 1;

                while (left <= right) {
                    IndexType i = (left + right) / 2;

                    if (value < rowIndex[i]) {
                        // value is not in =>row i range
                        right = i - 1;
                    } else if (value < rowIndex[i + 1]) {
                        // value is in =>row i range and value actually in row i
                        return i;
                    } else {
                        // value is in =>row i+1 range
                        left = i + 1;
                    }
                }

                // never goes here since kron row index always has value index
                return range;
            }

            static __device__ void sort(thrust::device_ptr<IndexType> buffer, IndexType size) {
                if (size > 1) {
                    for (int i = 1; i < size; i++) {
                        for (int j = 0; j < size - i; i++) {
                            if (buffer[j] > buffer[j + 1]) {
                                IndexType tmp = buffer[j];
                                buffer[j] = buffer[j + 1];
                                buffer[j + 1] = tmp;
                            }
                        }
                    }
                }
            }

            /**
             * evaluates r = a^T, were r equals transposed matrix a
             *
             * @param a Csr matrix
             *
             * @return a^T
             */
            MatrixType operator()(const MatrixType& a) {
                auto nrows = a.m_rows;
                auto ncols = a.m_cols;
                auto nvals = a.m_vals;

                // Create row index and column index to store res matrix
                // Note: transposed matrix has exact the same nnz
                ContainerType<IndexType> tmpBuffer(ncols + 1);
                ContainerType<IndexType> rowIndex(ncols + 1);
                ContainerType<IndexType> colIndex(nvals);
                ContainerType<IndexType> colOrder(nvals);

                // Will increment this values in the following kernel
                thrust::fill(tmpBuffer.begin(), tmpBuffer.end(), 0);

                // Evaluate number of nnz for each row of the result matrix M^T
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(nvals),
                                 [rRowIndex = tmpBuffer.data(), aColIndex = a.m_col_index.data(),
                                  order = colOrder.data()]

                __device__  (IndexType valueId) {
                    IndexType colIndex = aColIndex[valueId];
                    order[valueId] = atomicAdd((rRowIndex + colIndex).get(), (IndexType) 1);
                }
                );

                // Use prefix sum to evaluate row index offsets
                thrust::exclusive_scan(tmpBuffer.begin(), tmpBuffer.end(), rowIndex.begin(), 0, thrust::plus<IndexType>());

                // For each (i,j) value write i indices into j rows of the result
                // Note: use order matrix to avoid address overlap
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(nvals),
                                 [nrows, rRowIndex = rowIndex.data(), rColIndex = colIndex.data(),
                                  aRowIndex = a.m_row_index.data(), aColIndex = a.m_col_index.data(),
                                  order = colOrder.data()]

                __device__  (IndexType valueId) {
                    // Find the row id of the current value in the matrix a
                    IndexType aRowId = findNearestRowIdx(valueId, nrows, aRowIndex);
                    IndexType aColId = aColIndex[valueId];

                    // Order to write the row index of the transposed column
                    IndexType relativeOffset = order[valueId];

                    // Get start of the row values of M^T
                    IndexType rRowOffset = rRowIndex[aColId];

                    // Write in row j index i
                    rColIndex[rRowOffset + relativeOffset] = aRowId;
                }
                );

                // Sort indices inside each row of the result M^T matrix
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(ncols),
                                 [rRowIndex = rowIndex.data(), rColIndex = colIndex.data()]
                __device__  (IndexType rowId) {
                    IndexType rowBegin = rRowIndex[rowId];
                    IndexType rowLength = rRowIndex[rowId + 1] - rowBegin;

                    // Sort column indices of the row rowId
                    sort(&rColIndex[rowBegin], rowLength);
                });

                assert(colIndex.size() == nvals);
                assert(rowIndex.size() == ncols + 1);
                assert(rowIndex.back() == colIndex.size());

                return MatrixType(std::move(colIndex), std::move(rowIndex), ncols, nrows, nvals);
            }
        };

    }
}

#endif //CUBOOL_MATRIX_CSR_SPTRANSPOSE_HPP
