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
#include <nsparse/matrix.h>
#include <nsparse/detail/meta.h>
#include <limits>

namespace cubool {
    namespace kernels {

        template<typename IndexType, size_t blockSize>
        __global__ void spgemv(thrust::device_ptr<const IndexType> rowOffsets,  // Input csr matrix rows
                               thrust::device_ptr<const IndexType> colIndices,  // Input csr matrix col indices
                               thrust::device_ptr<const IndexType> v,           // Input dense v vector
                               thrust::device_ptr<IndexType> x,                 // Output dense x vector (x = M*v)
                               thrust::device_ptr<const IndexType> rowConfig) { // Rows to process for each bin
            IndexType assignedOrder = blockIdx.x;
            IndexType id = threadIdx.x;

            IndexType i = rowConfig[assignedOrder];    // Row to process

            size_t rowSize = rowOffsets[i + 1] - rowOffsets[i];
            size_t rowBegin = rowOffsets[i];

            __shared__ IndexType tmp_accum[blockSize];

            // Initial zero
            tmp_accum[id] = 0;
            __syncthreads();

            // Each thread accum nnz values
            for (size_t k = id; k < rowSize; k += blockSize) {
                tmp_accum[id] |= v[colIndices[rowBegin + k]];
            }
            __syncthreads();

            // Reduce accum to single value
            for (size_t s = 1; s < blockSize; s *= 2) {
                if (id % (2 * s) == 0) {
                    tmp_accum[id] |= tmp_accum[id + s];
                }

                __syncthreads();
            }

            // 0-thread saves result
            if (id == 0) {
                if (tmp_accum[0] > 0) {
                    x[i] = tmp_accum[0];
                }
            }
        }

        template <typename IndexType, typename AllocType>
        struct SpGEMV {
            template<typename T>
            using ContainerType = thrust::device_vector<T, typename AllocType::template rebind<T>::other>;
            using MatrixType = nsparse::matrix<bool, IndexType, AllocType>;
            using VectorType = details::SpVector<IndexType, AllocType>;

            template<typename ... Bins>
            void dispatch(StreamsWrapper<Config<Bins...>> &streamsWrapper,
                          thrust::device_ptr<const IndexType> rowOffsets,    // Input csr matrix rows
                          thrust::device_ptr<const IndexType> colIndices,    // Input csr matrix col indices
                          thrust::device_ptr<const IndexType> v,             // Input dense v vector
                          thrust::device_ptr<IndexType> x,                   // Output dense x vector (x = M*v)
                          const std::vector<IndexType>& binSizes,            // Size of bin in rowConfig
                          const std::vector<IndexType>& binOffset,           // Offset of bin in rowConfig
                          thrust::device_ptr<const IndexType> rowConfig) {   // Rows to process for each bin)

                EXPAND_SIDE_EFFECTS(
            (binSizes[Bins::id] > 0?
                      spgemv<IndexType, Bins::blockSize>
                      <<<binSizes[Bins::id], Bins::blockSize, 0, streamsWrapper.streams[Bins::id]>>>
                     (rowOffsets, colIndices, v, x, rowConfig + binOffset[Bins::id])
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
            VectorType operator()(const MatrixType& m, const VectorType& v) {
                static constexpr size_t max = std::numeric_limits<size_t>::max();

                auto M = m.m_rows;
                auto N = m.m_cols;

                assert(N == v.m_rows);

                // Resize cached buffers to store v and r as dense vectors
                if (mInput.size() < N)
                    mInput.resize(N);

                if (mOutput.size() < M) {
                    mOutput.resize(M);
                    mOutputWriteOffsets.resize(M);
                }

                // Copy v to dense vector
                thrust::fill_n(mInput.begin(), N, (IndexType) 0);
                thrust::for_each(v.m_rows_index.begin(), v.m_rows_index.end(),
                    [input = mInput.data()]
                    __device__(IndexType i) {
                    input[i] = 1;
                });

                // Empty out buffer
                thrust::fill_n(mOutput.begin(), M, (IndexType) 0);

                using ConfigType = Config<Bin<32, 1,  32, 0>,
                                          Bin<64, 32, 64, 1>,
                                          Bin<128,64, 128,2>,
                                          Bin<256,128,max,3>>;
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
                thrust::exclusive_scan(mBinsSize.begin(), mBinsSize.end(), mBinsOffsets.begin(), 0, thrust::plus<IndexType>());

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

                dispatch(streamsWrapper, m.m_row_index.data(), m.m_col_index.data(), mInput.data(), mOutput.data(), binSizes, binOffsets, mRowsConfig.data());
                cudaDeviceSynchronize();

                // Nnz of the result
                auto resultSize = thrust::reduce(mOutput.begin(), mOutput.end(), (IndexType) 0);

                ContainerType<index> result(resultSize);

                // Compute offset for each value
                thrust::exclusive_scan(mOutput.begin(), mOutput.end(), mOutputWriteOffsets.begin(), (IndexType) 0, thrust::plus<IndexType>());

                // For each value of the output write in the result buffer if it not zero
                thrust::for_each(thrust::counting_iterator<IndexType>(0), thrust::counting_iterator<IndexType>(M),
                     [output = mOutput.data(), result = result.data(), offsets = mOutputWriteOffsets.data()]
                     __device__ (IndexType i) {
                         if (output[i] > 0) {
                             auto offset = offsets[i];
                             auto colId = i;

                             result[offset] = colId;
                         }
                     }
                );

                return VectorType(std::move(result), M, resultSize);
            }

        private:
            ContainerType<index> mRowsConfig;
            ContainerType<index> mBinsSize;
            ContainerType<index> mBinsOffsets;
            ContainerType<index> mInput;
            ContainerType<index> mOutput;
            ContainerType<index> mOutputWriteOffsets;
        };

    }
}

#endif //CUBOOL_SPGEMV_CUH
