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

#ifndef CUBOOL_MATRIX_CSR_SPKRON_CUH
#define CUBOOL_MATRIX_CSR_SPKRON_CUH

#include <cuda/kernels/bin_search.cuh>
#include <thrust/device_vector.h>
#include <nsparse/matrix.h>

namespace cubool {
    namespace kernels {

        template <typename IndexType, typename AllocType>
        class SpKronFunctor {
        public:
            template<typename T>
            using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
            using MatrixType = nsparse::matrix<bool, IndexType, AllocType>;

            /**
             * evaluates r = a `kron` b, where `kron` is the Kronecker product of csr matrices.
             *
             * @param a Csr matrix
             * @param b Csr matrix
             *
             * @return a `kron` b
             */
            MatrixType operator()(const MatrixType& a, const MatrixType& b) {
                auto aNrows = a.m_rows;
                auto aNcols = a.m_cols;
                auto bNrows = b.m_rows;
                auto bNcols = b.m_cols;

                auto rNvals = a.m_vals * b.m_vals;
                auto rNrows = aNrows * bNrows;
                auto rNcols = aNcols * bNcols;

                ContainerType<IndexType> tmpBuffer(rNrows + 1, 0);

                // Evaluate number of values for each row of the result matrix
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(rNrows),
                    [bNrows, rTmpRowIndex = tmpBuffer.data(), aRowIndex = a.m_row_index.data(), bRowIndex = b.m_row_index.data()]
                    __device__  (IndexType rRowId) {
                        // Get block row index and row index within block (note: see kronecker product)
                        IndexType blockId = rRowId / bNrows;
                        IndexType bRowId = rRowId % bNrows;

                        // Gen values count for a and b row (note: format is csr)
                        IndexType valsCountInRowA = aRowIndex[blockId + 1] - aRowIndex[blockId];
                        IndexType valsCountInRowB = bRowIndex[bRowId + 1] - bRowIndex[bRowId];

                        // Store number of values in the column of the row idx of the result matrix
                        rTmpRowIndex[rRowId] = valsCountInRowA * valsCountInRowB;
                    }
                );

                // Create row index and column index to store res matrix
                ContainerType<IndexType> rowIndex(rNrows + 1);
                ContainerType<IndexType> colIndex(rNvals);

                // Use prefix sum to evaluate row index offsets
                thrust::exclusive_scan(tmpBuffer.begin(), tmpBuffer.end(), rowIndex.begin(), 0, thrust::plus<IndexType>());

                // Explicitly release resources
                tmpBuffer.clear();

                // Copy values of the b matrix to the proper places with evaluated offsets into col index of the result matrix
                // Evaluate number of values for each row of the result matrix
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(rNvals),
                    [bNrows, bNcols, rNrows, rRowIndex = rowIndex.data(), rColIndex = colIndex.data(),
                     aRowIndex = a.m_row_index.data(), aColIndex = a.m_col_index.data(),
                     bRowIndex = b.m_row_index.data(), bColIndex = b.m_col_index.data()]
                    __device__  (IndexType valueId) {
                        // Find the index of the r row where to write this value
                        IndexType rRowId = findNearestRowIdx<IndexType>(valueId, rNrows, rRowIndex);

                        // Get row index within block (note: see kronecker product)
                        IndexType blockId = rRowId / bNrows;
                        IndexType bRowId = rRowId % bNrows;

                        IndexType valsCountInRowB = bRowIndex[bRowId + 1] - bRowIndex[bRowId];

                        // Get value relative index within the r matrix row
                        IndexType valueIdInRowR = valueId - rRowIndex[rRowId];

                        // Get value index within the a matrix row
                        IndexType valueIdInRowA = valueIdInRowR / valsCountInRowB;

                        // Get value index within the b matrix row
                        IndexType valueIdInRowB = valueIdInRowR % valsCountInRowB;

                        // Store value in the proper place
                        rColIndex[valueId] = bNcols * aColIndex[aRowIndex[blockId] + valueIdInRowA] + bColIndex[bRowIndex[bRowId] + valueIdInRowB];
                    }
                );

                assert(colIndex.size() == rNvals);
                assert(rowIndex.size() == rNrows + 1);
                assert(rowIndex.back() == colIndex.size());

                return MatrixType(std::move(colIndex), std::move(rowIndex), rNrows, rNcols, rNvals);
            }
        };

    }
}

#endif //CUBOOL_MATRIX_CSR_SPKRON_CUH