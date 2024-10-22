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

#ifndef CUBOOL_SPGEMV_CUH
#define CUBOOL_SPGEMV_CUH

#include <cuda/details/sp_vector.hpp>
#include <cuda/details/meta.hpp>
#include <cuda/kernels/bin_search.cuh>
#include <nsparse/matrix.h>
#include <nsparse/detail/meta.h>
#include <limits>

#include <thrust/sort.h>

namespace cubool {
    namespace kernels {

        template<typename IndexType>
        __device__ IndexType __get_vec_value(thrust::device_ptr<const IndexType> v, IndexType vSize, IndexType value) {
            auto begin = v;
            auto end = v + vSize;
            auto result = kernels::find(begin, end, value);

            return result == end? 0x0u: 0x1;
        }

        template<typename IndexType, size_t threads, size_t blockSize>
        __global__ void __spgemv(thrust::device_ptr<const IndexType> rowOffsets,  // Input csr matrix rows
                                 thrust::device_ptr<const IndexType> colIndices,  // Input csr matrix col indices
                                 thrust::device_ptr<const IndexType> v,           // Input sparse v vector
                                 IndexType vSize,                                 // Input sparse v vector size
                                 thrust::device_ptr<IndexType> x,                 // Output dense x vector (x = M*v)
                                 thrust::device_ptr<const IndexType> rowConfig,   // Rows to process for each bin
                                 IndexType rowsCount) {                           // Num of rows to process
            // Split block into number of groups of size `threads`.
            // Each group process its own row.

            IndexType id = threadIdx.x % threads;                                           // id of the thread withing row processing group
            IndexType interBlockId = threadIdx.x / threads;                                 // id of the group (number of `threads` belong to the same group)
            IndexType assignedOrder = blockIdx.x * (blockSize / threads) + interBlockId;    // row, which is process by number of `threads`

            __shared__ IndexType tmp_accum[blockSize];

            if (assignedOrder >= rowsCount)
                assignedOrder = rowsCount - 1;

            IndexType i = rowConfig[assignedOrder];    // Row to process

            size_t rowBegin = rowOffsets[i];
            size_t rowSize = rowOffsets[i + 1] - rowBegin;

            // Initial zero
            tmp_accum[threadIdx.x] = 0;

            // Each thread accum nnz values
            for (size_t k = id; k < rowSize; k += threads) {
                tmp_accum[threadIdx.x] |= __get_vec_value<IndexType>(v, vSize, colIndices[rowBegin + k]);
            }

            // Reduce accum to single value
            #pragma unroll
            for (size_t s = 1; s < threads && warpSize; s *= 2) {
                __syncwarp();

                if (id % (2 * s) == 0) {
                    tmp_accum[threadIdx.x] |= tmp_accum[threadIdx.x + s];
                }
            }

            #pragma unroll
            for (size_t s = warpSize; s < threads; s *= 2) {
                __syncthreads();

                if (id % (2 * s) == 0) {
                    tmp_accum[threadIdx.x] |= tmp_accum[threadIdx.x + s];
                }
            }

            // 0-thread for each min-thread group saves result
            if (id == 0) {
                if (tmp_accum[threadIdx.x] > 0) {
                    x[i] = tmp_accum[threadIdx.x];
                }
            }
        }

        template<typename IndexType, typename AllocType>
        struct SpGEMV {
            template<typename T>
            using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
            using MatrixType = nsparse::matrix<bool, IndexType, AllocType>;
            using VectorType = details::SpVector<IndexType, AllocType>;

            template<typename ... Bins>
            void dispatch(StreamsWrapper<Config<Bins...>> &streamsWrapper,
                          thrust::device_ptr<const IndexType> rowOffsets,    // Input csr matrix rows
                          thrust::device_ptr<const IndexType> colIndices,    // Input csr matrix col indices
                          thrust::device_ptr<const IndexType> v,             // Input sparse v vector
                          IndexType vSize,                                   // Input sparse v vector size
                          thrust::device_ptr<IndexType> x,                   // Output dense x vector (x = M*v)
                          const std::vector<IndexType> &binSizes,            // Size of bin in rowConfig
                          const std::vector<IndexType> &binOffset,           // Offset of bin in rowConfig
                          thrust::device_ptr<const IndexType> rowConfig) {   // Rows to process for each bin)

                EXPAND_SIDE_EFFECTS(
                    (binSizes[Bins::id] > 0 ?
                    __spgemv<IndexType, Bins::threads, Bins::blockSize>
                    <<<binSizes[Bins::id] / Bins::dispatchRatio + (binSizes[Bins::id] % Bins::dispatchRatio ? 1 : 0), Bins::blockSize, 0, streamsWrapper.streams[Bins::id]>>>
                    (rowOffsets, colIndices, v, vSize, x, rowConfig + binOffset[Bins::id], binSizes[Bins::id])
                    : void())
                );
            }

            /**
             * Compute r = M x v
             *
             * Matrix-vector multiplication algorithm:
             * 1. Assign for each row its computation group (group defines number of threads used for processing)
             * 2. Run each group (larger row - more threads must be assigned)
             *
             * @param m Sparse matrix
             * @param v Sparse vector
             *
             * @return Sparse vector
             */
            VectorType operator()(const MatrixType &m, const VectorType &v) {
                static constexpr size_t max = std::numeric_limits<size_t>::max();

                auto M = m.m_rows;

                assert(m.m_cols == v.m_rows);

                // Empty result case
                if (v.m_vals == 0 || m.m_vals == 0)
                    return VectorType(M);

                // Resize cached buffers to store r as dense vector
                if (mOutput.size() < M)
                    mOutput.resize(M);

                // Empty out buffer
                thrust::fill_n(mOutput.begin(), M, (IndexType) 0);

                using ConfigType = Config<Bin<4, 32, 1, 8, 0>,
                        Bin<8, 32, 8, 16, 1>,
                        Bin<16, 32, 16, 32, 2>,
                        Bin<32, 32, 32, 64, 3>,
                        Bin<64, 64, 64, 128, 4>,
                        Bin<128, 128, 128, 256, 5>,
                        Bin<256, 256, 256, max, 6>>;
                ConfigType config;

                mRowsConfig.resize(M);

                mBinsSize.resize(config.binsCount());
                mBinsOffsets.resize(config.binsCount());

                thrust::fill(mBinsSize.begin(), mBinsSize.end(), (IndexType) 0);

                // Eval bins size for each row
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(M),
                                 [config, binSize = mBinsSize.data(), rowsIndices = m.m_row_index.data()]
                                         __device__(IndexType i) {
                                     auto valsInRow = rowsIndices[i + 1] - rowsIndices[i];
                                     auto binId = config.selectBin(valsInRow);

                                     if (binId == config.unusedBinId())
                                         // Ignore empty rows
                                         return;

                                     atomicAdd((binSize + binId).get(), 1);
                                 });

                // Offsets for each bin (each bin will have its own section in permutation buffer)
                thrust::exclusive_scan(mBinsSize.begin(), mBinsSize.end(), mBinsOffsets.begin(), 0,
                                       thrust::plus<IndexType>());

                // Reset bin sizes (use as offsets for permutation id assignments)
                thrust::fill(mBinsSize.begin(), mBinsSize.end(), (IndexType) 0);

                // Assign rows for its bins
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(M),
                                 [config, binSize = mBinsSize.data(), binOffset = mBinsOffsets.data(), rowsConfig = mRowsConfig.data(),
                                         rowsIndices = m.m_row_index.data()]
                                         __device__(IndexType i) {
                                     auto valsInRow = rowsIndices[i + 1] - rowsIndices[i];
                                     auto binId = config.selectBin(valsInRow);

                                     if (binId == config.unusedBinId())
                                         // Ignore empty rows
                                         return;

                                     auto order = atomicAdd((binSize + binId).get(), 1);
                                     rowsConfig[binOffset[binId] + order] = i;
                                 });

                // Bring to the host bin sizes and offsets
                std::vector<IndexType> binOffsets(mBinsOffsets.size());
                thrust::copy(mBinsOffsets.begin(), mBinsOffsets.end(), binOffsets.begin());

                std::vector<IndexType> binSizes(mBinsSize.size());
                thrust::copy(mBinsSize.begin(), mBinsSize.end(), binSizes.begin());

                // Stream for each bin
                StreamsWrapper<ConfigType> streamsWrapper;

                dispatch(streamsWrapper,
                         m.m_row_index.data(),
                         m.m_col_index.data(),
                         v.m_rows_index.data(),
                         v.m_vals,
                         mOutput.data(),
                         binSizes,
                         binOffsets,
                         mRowsConfig.data());

                cudaDeviceSynchronize();

                // Nnz of the result
                auto resultSize = thrust::reduce(mOutput.begin(), mOutput.end(), (IndexType) 0);

                ContainerType<index> result(resultSize);

                // Order where to write indices
                ContainerType<index> order(1);
                thrust::fill(order.begin(), order.end(), (IndexType) 0);

                // For each value of the output write in the result buffer if it not zero
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(M),
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

                return VectorType(std::move(result), M, resultSize);
            }

        private:
            ContainerType<index> mRowsConfig;
            ContainerType<index> mBinsSize;
            ContainerType<index> mBinsOffsets;
            ContainerType<index> mOutput;
        };

    }
}

#endif //CUBOOL_SPGEMV_CUH
