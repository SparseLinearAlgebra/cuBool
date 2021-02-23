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

#ifndef CUBOOL_MATRIX_CSR_SPMERGE_CUH
#define CUBOOL_MATRIX_CSR_SPMERGE_CUH

#include <core/config.hpp>
#include <nsparse/matrix.h>
#include <nsparse/detail/merge.h>

namespace cubool {
    namespace kernels {

        template <typename IndexType, typename AllocType>
        class SpMergeFunctor {
        public:
            using MatrixType = nsparse::matrix<bool, IndexType, typename AllocType::template rebind<IndexType>::other>;

            /**
             * evaluates r = a + b, where '+' is boolean semiring plus operation
             *
             * @param a Csr matrix
             * @param b Csr matrix
             *
             * @return a + b
             */
            MatrixType operator()(const MatrixType& a, const MatrixType& b) {
                using namespace nsparse::meta;
                constexpr size_t max = std::numeric_limits<size_t>::max();

                assert(a.m_rows == b.m_rows);
                assert(a.m_cols == b.m_cols);

                IndexType rows = a.m_rows;
                IndexType cols = a.m_rows;

                constexpr auto config_merge =
                    make_bin_seq<
                        bin_info_t<merge_conf_t<128>, 64, max>,
                        bin_info_t<merge_conf_t<64>, 32, 64>,
                        bin_info_t<merge_conf_t<32>, 0, 32>
                    >;

                auto merge_res = uniqueMergeFunctor(a.m_row_index, a.m_col_index, b.m_row_index, b.m_col_index, config_merge);

                auto& rpt_result = merge_res.first;
                auto& col_result = merge_res.second;
                IndexType vals = col_result.size();

                assert(rpt_result.size() == rows + 1);
                assert(col_result.size() == rpt_result.back());

                return MatrixType(std::move(col_result), std::move(rpt_result), rows, cols, vals);
            }

        private:
            nsparse::unique_merge_functor_t<IndexType, AllocType> uniqueMergeFunctor;
        };

    }
}




#endif //CUBOOL_MATRIX_CSR_SPMERGE_CUH
