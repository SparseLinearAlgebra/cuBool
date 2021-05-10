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

#ifndef CUBOOL_SPGEMTV_HPP
#define CUBOOL_SPGEMTV_HPP

#include <cuda/details/sp_vector.hpp>
#include <cuda/details/meta.hpp>
#include <cuda/kernels/bin_search.cuh>
#include <nsparse/matrix.h>
#include <nsparse/detail/meta.h>
#include <limits>

namespace cubool {
    namespace kernels {

        template<typename IndexType, typename AllocType>
        struct SpGEMVT {
            template<typename T>
            using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
            using MatrixType = nsparse::matrix<bool, IndexType, AllocType>;
            using VectorType = details::SpVector<IndexType, AllocType>;

            /**
             * Compute r = M^t x v
             *
             * @param v Sparse vector
             * @param m Sparse matrix
             *
             * @return Sparse vector
             */
            VectorType operator()(const VectorType &v, const MatrixType &m) {
                auto N = m.m_cols;
                auto vnvals = v.m_vals;

                assert(m.m_rows == v.m_rows);

                // Empty result case
                if (v.m_vals == 0 || m.m_vals == 0)
                    return VectorType(N);

                // Resize cached buffers to store r as dense vectors
                if (mOutput.size() < N)
                    mOutput.resize(N);

                // Empty out buffer
                thrust::fill_n(mOutput.begin(), N, (IndexType) 0);

                // Count number of rows of matrix m to process (based on v)
                ContainerType<IndexType> configTmp(vnvals + 1);
                ContainerType<IndexType> config(vnvals + 1);

                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(vnvals),
                        [vIndices = v.m_rows_index.data(), mRowOffset = m.m_row_index.data(), config = configTmp.data()]
                        __device__(IndexType id) {
                    auto rowId = vIndices[id];
                    auto valuesCount = mRowOffset[rowId + 1] - mRowOffset[rowId];
                    config[id] = valuesCount;
                });

                // Offset of each row min-group
                thrust::exclusive_scan(configTmp.begin(), configTmp.end(), config.begin(), 0, thrust::plus<IndexType>());

                IndexType totalToProcess = config.back();

                // For each value in selected rows run and compute non-zero output result values
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(totalToProcess),
                        [result = mOutput.data(), config = config.data(), rows = v.m_vals,
                         rowOffsets = m.m_row_index.data(), colIndex = m.m_col_index.data(), vIndices = v.m_rows_index.data()]
                        __device__(IndexType id) {
                        auto configRow = kernels::findNearestRowIdx<IndexType>(id, rows, config); // Find config slot to process
                        auto valueInRow = id - config[configRow];                                 // Find value relative id in row to process
                        auto rowId = vIndices[configRow];                                         // Find row to process
                        auto valueIdx = colIndex[rowOffsets[rowId] + valueInRow];                 // Get actual col index

                        result[valueIdx] = 0x1u;                                              // This value in result vector is non-zero
                });

                // Nnz of the result
                auto resultSize = thrust::reduce(mOutput.begin(), mOutput.begin() + N, (IndexType) 0);

                ContainerType<index> result(resultSize);

                // Order where to write indices
                ContainerType<index> order(1);
                thrust::fill(order.begin(), order.end(), (IndexType) 0);

                // For each value of the output write in the result buffer if it not zero
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(N),
                                 [output = mOutput.data(), result = result.data(), order = order.data()]
                                         __device__(IndexType i) {
                                     if (output[i] > 0) {
                                         auto offset = atomicAdd(order.get(), (IndexType) 1);
                                         auto colId = i;

                                         result[offset] = colId;
                                     }
                                 }
                );

                // Sort indices
                thrust::sort(result.begin(), result.end());

                return VectorType(std::move(result), N, resultSize);
            }

        private:
            ContainerType<IndexType> mOutput;
        };

    }
}

#endif //CUBOOL_SPGEMTV_HPP
