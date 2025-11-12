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

#ifndef CUBOOL_SPEWISEMULT_CUH
#define CUBOOL_SPEWISEMULT_CUH

#include <cuda/details/sp_vector.hpp>
#include <cuda/kernels/bin_search.cuh>
#include <nsparse/matrix.h>
#include <cmath>

#include <thrust/set_operations.h>

namespace cubool {
    namespace kernels {

        template <typename IndexType, typename AllocType>
        struct SpVectorEWiseMult {
            template<typename T>
            using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
            using VectorType = details::SpVector<IndexType, AllocType>;

            VectorType operator()(const VectorType& a, const VectorType& b) {
                auto aNvals = a.m_vals;
                auto bNvals = b.m_vals;
                auto worst = std::min(aNvals, bNvals);

                // Allocate memory for the worst case scenario
                ContainerType<index> mCacheBuffer(worst);

                // Intersect sorted arrays
                auto out = thrust::set_intersection(a.m_rows_index.begin(), a.m_rows_index.end(),
                                                    b.m_rows_index.begin(), b.m_rows_index.end(),
                                                    mCacheBuffer.begin());

                // Count result nvals count
                auto nvals = thrust::distance(mCacheBuffer.begin(), out);

                // Fill the result buffer
                ContainerType<index> rowIndex(nvals);
                thrust::copy(mCacheBuffer.begin(), out, rowIndex.begin());

                return VectorType(std::move(rowIndex), a.m_rows, nvals);
            }
        };

        template <typename IndexType, typename AllocType>
        struct SpMatrixEWiseMult {
            template<typename T>
            using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
            using MatrixType = nsparse::matrix<bool, IndexType, AllocType>;
            using LargeIndexType = unsigned long;

            static_assert(sizeof(LargeIndexType) > sizeof(IndexType), "Values intersection index must be larger");

            static void fillIndices(const MatrixType& m, ContainerType<LargeIndexType>& out) {
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(m.m_vals),
                        [rowOffset=m.m_row_index.data(), colIndex=m.m_col_index.data(),
                         outIndices=out.data(), nrows=m.m_rows, ncols=m.m_cols] __device__ (IndexType valueId) {
                    LargeIndexType row = findNearestRowIdx<index>(valueId, nrows, rowOffset);
                    LargeIndexType col = colIndex[valueId];
                    LargeIndexType index = row * ncols + col;
                    outIndices[valueId] = index;
                });
            }

            MatrixType operator()(const MatrixType& a, const MatrixType& b) {
                auto aNvals = a.m_vals;
                auto bNvals = b.m_vals;
                auto worst = std::min(aNvals, bNvals);

                // Allocate memory for the worst case scenario
                ContainerType<LargeIndexType> inputA(aNvals);
                ContainerType<LargeIndexType> inputB(bNvals);

                fillIndices(a, inputA);
                fillIndices(b, inputB);

                ContainerType<LargeIndexType> intersected(worst);

                auto out = thrust::set_intersection(inputA.begin(), inputA.end(),
                                                    inputB.begin(), inputB.end(),
                                                    intersected.begin());

                // Count result nvals count
                auto nvals = thrust::distance(intersected.begin(), out);

                ContainerType<index> rowOffsetTmp(a.m_rows + 1);
                ContainerType<index> colIndex(nvals);

                thrust::fill(rowOffsetTmp.begin(), rowOffsetTmp.end(), 0);

                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(nvals),
                        [rowOffset=rowOffsetTmp.data(), colIndex=colIndex.data(), intersected=intersected.data(),
                         nrows=a.m_rows, ncols=a.m_cols] __device__ (IndexType valueId) {
                    LargeIndexType i = intersected[valueId];
                    LargeIndexType row = i / ncols;
                    LargeIndexType col = i % ncols;
                    atomicAdd((rowOffset + row).get(), 1);
                    colIndex[valueId] = (IndexType) col;
                });

                ContainerType<index> rowOffset(a.m_rows + 1);
                thrust::exclusive_scan(rowOffsetTmp.begin(), rowOffsetTmp.end(), rowOffset.begin(), 0, thrust::plus<index>());

                assert(nvals == rowOffset.back());

                return MatrixType(std::move(colIndex), std::move(rowOffset), a.m_rows, a.m_cols, nvals);
            }
        };

    }
}

#endif //CUBOOL_SPEWISEMULT_CUH
