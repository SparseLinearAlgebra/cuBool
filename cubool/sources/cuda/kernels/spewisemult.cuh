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
#include <cmath>

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
                auto worst = std::max(aNvals, bNvals);

                // Allocate memory for the worst case scenario
                if (mCacheBuffer.size() < worst)
                    mCacheBuffer.resize(worst);

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

        private:
            ContainerType<index> mCacheBuffer;
        };

    }
}

#endif //CUBOOL_SPEWISEMULT_CUH
