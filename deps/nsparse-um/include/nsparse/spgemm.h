#pragma once

#include <cassert>
#include <nsparse/matrix.h>

#include <nsparse/detail/merge.h>

#include <nsparse/detail/merge_path.cuh>

#include <thrust/iterator/counting_iterator.h>

#include <nsparse/detail/count_nz.h>
#include <nsparse/detail/fill_nz.h>
#include <nsparse/unified_allocator.h>

namespace nsparse {

    template<typename ValueType, typename IndexType, typename AllocType>
    struct spgemm_functor_t;

    template<typename index_type, typename alloc_type>
    struct spgemm_functor_t<bool, index_type, alloc_type> {
        /*
         * returns c + a * b
         */
        matrix<bool, index_type, alloc_type> operator()(const matrix<bool, index_type, alloc_type> &c,
                                                        const matrix<bool, index_type, alloc_type> &a,
                                                        const matrix<bool, index_type, alloc_type> &b) {
            assert(a.m_cols == b.m_rows);
            assert(c.m_rows == a.m_rows);
            assert(c.m_cols == b.m_cols);

            index_type rows = a.m_rows;
            index_type cols = b.m_cols;

            constexpr size_t max = std::numeric_limits<size_t>::max();

            using namespace meta;
            constexpr auto config_find_nz = make_bin_seq<bin_info_t<nz_conf_t<global_row, 1024>, 4096, max>,
                    bin_info_t<nz_conf_t<block_row, 512>, 2048, 4096>,
                    bin_info_t<nz_conf_t<block_row, 256>, 1024, 2048>,
                    bin_info_t<nz_conf_t<block_row, 128>, 512, 1024>,
                    bin_info_t<nz_conf_t<block_row, 128>, 256, 512>,
                    bin_info_t<nz_conf_t<block_row, 128>, 128, 256>,
                    bin_info_t<nz_conf_t<block_row, 64>, 64, 128>,
                    bin_info_t<nz_conf_t<block_row, 32>, 32, 64>,
                    bin_info_t<nz_conf_t<pwarp_row, 256>, 0, 32>>;

            typename count_nz_functor_t<index_type, alloc_type>::row_index_res_t res =
                    count_nz_functor(rows, cols, c.m_col_index, c.m_row_index, a.m_col_index, a.m_row_index,
                                     b.m_col_index, b.m_row_index, config_find_nz);

            constexpr auto config_fill_nz = make_bin_seq<bin_info_t<nz_conf_t<block_row, 512>, 2048, 4096>,
                    bin_info_t<nz_conf_t<block_row, 256>, 1024, 2048>,
                    bin_info_t<nz_conf_t<block_row, 128>, 512, 1024>,
                    bin_info_t<nz_conf_t<block_row, 128>, 256, 512>,
                    bin_info_t<nz_conf_t<block_row, 128>, 128, 256>,
                    bin_info_t<nz_conf_t<block_row, 64>, 64, 128>,
                    bin_info_t<nz_conf_t<block_row, 32>, 32, 64>,
                    bin_info_t<nz_conf_t<pwarp_row, 256>, 0, 32>>;

            thrust::device_vector<index_type, alloc_type> col_index =
                    fill_nz_functor(rows, c.m_col_index, c.m_row_index, a.m_col_index, a.m_row_index,
                                    b.m_col_index, b.m_row_index, res.row_index, config_fill_nz);

            reuse_global_hash_table(res.row_index, col_index, res.global_hash_table_state);

            //    validate_order<index_type><<<rows, 128>>>(res.row_index.data(), col_index.data());
            //    validate_order<index_type><<<rows, 128>>>(c.m_row_index.data(), c.m_col_index.data());

            if (c.m_vals == 0) {
                auto vals = col_index.size();
                return {std::move(col_index), std::move(res.row_index), rows, cols, (index_type) vals};
            }

            constexpr auto config_merge =
                    make_bin_seq<
                        bin_info_t<merge_conf_t<128>, 64, max>,
                        bin_info_t<merge_conf_t<64>, 32, 64>,
                        bin_info_t<merge_conf_t<32>, 0, 32>>;

            auto merge_res = unique_merge_functor(res.row_index, col_index, c.m_row_index, c.m_col_index, config_merge);

            auto &rpt_result = merge_res.first;
            auto &col_result = merge_res.second;

            assert(rpt_result.size() == rows + 1);
            assert(col_result.size() == rpt_result.back());
            index_type vals = col_result.size();

            return {std::move(col_result), std::move(rpt_result), rows, cols, vals};
        }

    private:
        count_nz_functor_t<index_type, alloc_type> count_nz_functor{};
        fill_nz_functor_t<index_type, alloc_type> fill_nz_functor{};
        unique_merge_functor_t<index_type, alloc_type> unique_merge_functor{};
    };

}  // namespace nsparse