#pragma once
#include <nsparse/detail/fill_nz.cuh>
#include <nsparse/detail/util.h>
#include <nsparse/detail/meta.h>
#include <nsparse/detail/count_nz.h>

#include <iostream>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <utility>

namespace nsparse {

template <typename index_type, typename alloc_type>
struct fill_nz_functor_t {
  template <typename T>
  using container_t = thrust::device_vector<T, typename alloc_type::template rebind<T>::other>;

  template <typename... Borders>
  void exec_pwarp_row(
      const container_t<index_type>& c_col_idx, const container_t<index_type>& c_row_idx,
      const container_t<index_type>& a_col_idx, const container_t<index_type>& a_row_idx,
      const container_t<index_type>& b_col_idx, const container_t<index_type>& b_row_idx,
      const container_t<index_type>& permutation_buffer, const container_t<index_type>& bin_offset,
      const container_t<index_type>& bin_size, container_t<index_type>& col_idx,
      const container_t<index_type>& row_idx, std::tuple<Borders...>) {
    constexpr index_type pwarp = 4;
    EXPAND_SIDE_EFFECTS(
        (bin_size[Borders::bin_index] > 0
             ? fill_nz_pwarp_row<index_type, pwarp, Borders::config_t::block_size, Borders::max_border>
             <<<util::div(bin_size[Borders::bin_index] * pwarp, (uint)Borders::config_t::block_size), Borders::config_t::block_size>>>(
                 c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(),
                 b_row_idx.data(), b_col_idx.data(),
                 permutation_buffer.data() + bin_offset[Borders::bin_index], col_idx.data(),
                 row_idx.data(), bin_size[Borders::bin_index])
             : void()));
  }

  template <typename... Borders>
  void exec_block_row(
      const container_t<index_type>& c_col_idx, const container_t<index_type>& c_row_idx,
      const container_t<index_type>& a_col_idx, const container_t<index_type>& a_row_idx,
      const container_t<index_type>& b_col_idx, const container_t<index_type>& b_row_idx,
      const container_t<index_type>& permutation_buffer, const container_t<index_type>& bin_offset,
      const container_t<index_type>& bin_size, container_t<index_type>& col_idx,
      const container_t<index_type>& row_idx, std::tuple<Borders...>) {
    static_assert(meta::all_of<(Borders::config_t::block_size % 32 == 0)...>);

    EXPAND_SIDE_EFFECTS(
        (bin_size[Borders::bin_index] > 0 ? fill_nz_block_row<index_type, Borders::max_border>
             <<<(index_type)bin_size[Borders::bin_index], Borders::config_t::block_size>>>(
                 c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(),
                 b_row_idx.data(), b_col_idx.data(),
                 permutation_buffer.data() + bin_offset[Borders::bin_index], col_idx.data(),
                 row_idx.data())
                                          : void()));
  }

  template <typename... Borders>
  void exec_global_row(
      const container_t<index_type>& c_col_idx, const container_t<index_type>& c_row_idx,
      const container_t<index_type>& a_col_idx, const container_t<index_type>& a_row_idx,
      const container_t<index_type>& b_col_idx, const container_t<index_type>& b_row_idx,
      const container_t<index_type>& permutation_buffer, const container_t<index_type>& bin_offset,
      const container_t<index_type>& bin_size, container_t<index_type>& col_idx,
      const container_t<index_type>& row_idx, std::tuple<Borders...>) {
    static_assert(sizeof...(Borders) <= 1);

    constexpr index_type block_sz = 1024;

    static_assert(block_sz % 32 == 0);

    EXPAND_SIDE_EFFECTS((bin_size[Borders::bin_index] > 0 ? fill_nz_block_row_global<index_type>
                             <<<(index_type)bin_size[Borders::bin_index], block_sz>>>(
                                 c_row_idx.data(), c_col_idx.data(), a_row_idx.data(),
                                 a_col_idx.data(), b_row_idx.data(), b_col_idx.data(),
                                 permutation_buffer.data() + bin_offset[Borders::bin_index],
                                 col_idx.data(), row_idx.data())
                                                          : void()));
  }

  template <typename... Borders>
  container_t<index_type> operator()(index_type n_rows, const container_t<index_type>& c_col_idx,
                                     const container_t<index_type>& c_row_idx,
                                     const container_t<index_type>& a_col_idx,
                                     const container_t<index_type>& a_row_idx,
                                     const container_t<index_type>& b_col_idx,
                                     const container_t<index_type>& b_row_idx,
                                     const container_t<index_type>& row_idx,
                                     std::tuple<Borders...>) {
    constexpr size_t bin_count = sizeof...(Borders);
    constexpr size_t unused_bin = meta::max_bin<Borders...> + 1;

    util::resize_and_fill_zeros(bin_size, bin_count);
    bin_offset.resize(bin_count);
    permutation_buffer.resize(n_rows);

    thrust::for_each(
        thrust::counting_iterator<index_type>(0), thrust::counting_iterator<index_type>(n_rows),
        [row_per_bin = bin_size.data(), rpt = row_idx.data()] __device__(index_type tid) {
          size_t prod = rpt[tid + 1] - rpt[tid];

          size_t bin = meta::select_bin<Borders...>(prod, unused_bin);

          if (bin != unused_bin)
            atomicAdd(row_per_bin.get() + bin, 1);
        });

    thrust::exclusive_scan(bin_size.begin(), bin_size.end(), bin_offset.begin());

    thrust::fill(bin_size.begin(), bin_size.end(), 0);

    thrust::for_each(
        thrust::counting_iterator<index_type>(0), thrust::counting_iterator<index_type>(n_rows),
        [rpt = row_idx.data(), bin_offset = bin_offset.data(), bin_size = bin_size.data(),
         rows_in_bins = permutation_buffer.data()] __device__(index_type tid) {
          auto prod = rpt[tid + 1] - rpt[tid];

          int bin = meta::select_bin<Borders...>(prod, unused_bin);

          if (bin == unused_bin)
            return;

          auto curr_bin_size = atomicAdd(bin_size.get() + bin, 1);
          rows_in_bins[bin_offset[bin] + curr_bin_size] = tid;
        });

    index_type values_count = row_idx.back();

    container_t<index_type> col_idx(values_count, std::numeric_limits<index_type>::max());

    exec_pwarp_row(c_col_idx, c_row_idx, a_col_idx, a_row_idx, b_col_idx, b_row_idx,
                   permutation_buffer, bin_offset, bin_size, col_idx, row_idx,
                   meta::filter<meta::pwarp_row, Borders...>);

    exec_block_row(c_col_idx, c_row_idx, a_col_idx, a_row_idx, b_col_idx, b_row_idx,
                   permutation_buffer, bin_offset, bin_size, col_idx, row_idx,
                   meta::filter<meta::block_row, Borders...>);

    exec_global_row(c_col_idx, c_row_idx, a_col_idx, a_row_idx, b_col_idx, b_row_idx,
                    permutation_buffer, bin_offset, bin_size, col_idx, row_idx,
                    meta::filter<meta::global_row, Borders...>);

    return std::move(col_idx);
  }

 private:
  container_t<index_type> bin_size;
  container_t<index_type> bin_offset;
  container_t<index_type> permutation_buffer;
};

template <typename index_type, typename alloc_type>
void reuse_global_hash_table(
    const thrust::device_vector<index_type, alloc_type>& row_idx,
    thrust::device_vector<index_type, alloc_type>& col_idx,
    const typename count_nz_functor_t<index_type, alloc_type>::global_hash_table_state_t& state) {
  constexpr index_type block_sz = 1024;
  auto hashed_row_count = state.hashed_row_indices.size();

  if (hashed_row_count > 0) {
    filter_hash_table<index_type><<<hashed_row_count, block_sz>>>(
        row_idx.data(), state.hash_table.data(), state.hashed_row_offsets.data(),
        state.hashed_row_indices.data(), col_idx.data());
  }
}

}  // namespace nsparse
