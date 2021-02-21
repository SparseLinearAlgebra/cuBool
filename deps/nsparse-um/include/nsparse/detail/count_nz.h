#pragma once
#include <nsparse/detail/meta.h>
#include <nsparse/detail/util.h>
#include <nsparse/detail/count_nz.cuh>

#include <cub/cub.cuh>

#include <iostream>

#include <algorithm>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <utility>
#include <vector>

namespace nsparse {

template <typename index_type, typename alloc_type>
struct count_nz_functor_t {
  template <typename T>
  using container_t = thrust::device_vector<T, typename alloc_type::template rebind<T>::other>;

  cudaStream_t streams[9];

  count_nz_functor_t() {
    for (auto& s: streams) {
      cudaStreamCreate( &s);
    }
  }

  ~count_nz_functor_t() {
    for (auto& s: streams) {
      cudaStreamDestroy(s);
    }
  }

  struct global_hash_table_state_t {
    container_t<index_type> hash_table;
    container_t<index_type> hashed_row_offsets;
    container_t<index_type> hashed_row_indices;
  };

  struct row_index_res_t {
    container_t<index_type> row_index;
    global_hash_table_state_t global_hash_table_state;
  };

  template <typename... Borders>
  void exec_pwarp_row(
      const container_t<index_type>& c_col_idx, const container_t<index_type>& c_row_idx,
      const container_t<index_type>& a_col_idx, const container_t<index_type>& a_row_idx,
      const container_t<index_type>& b_col_idx, const container_t<index_type>& b_row_idx,
      const container_t<index_type>& permutation_buffer, const container_t<index_type>& bin_offset,
      const container_t<index_type>& bin_size, container_t<index_type>& row_idx,
      std::tuple<Borders...>) {
    constexpr size_t pwarp = 4;

    EXPAND_SIDE_EFFECTS(
        (bin_size[Borders::bin_index] > 0
             ? count_nz_pwarp_row<index_type, pwarp, Borders::config_t::block_size,
                                  Borders::max_border>
             <<<util::div(bin_size[Borders::bin_index] * pwarp, Borders::config_t::block_size),
                Borders::config_t::block_size>>>(
                 c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(),
                 b_row_idx.data(), b_col_idx.data(),
                 permutation_buffer.data() + bin_offset[Borders::bin_index], row_idx.data(),
                 bin_size[Borders::bin_index])
             : void()));
  }

  template <typename... Borders>
  void exec_block_row(
      const container_t<index_type>& c_col_idx, const container_t<index_type>& c_row_idx,
      const container_t<index_type>& a_col_idx, const container_t<index_type>& a_row_idx,
      const container_t<index_type>& b_col_idx, const container_t<index_type>& b_row_idx,
      const container_t<index_type>& permutation_buffer, thrust::host_vector<index_type> bin_offset,
      thrust::host_vector<index_type> bin_size, container_t<index_type>& row_idx,
      std::tuple<Borders...>) {
    static_assert(meta::all_of<(Borders::config_t::block_size % 32 == 0)...>);

    EXPAND_SIDE_EFFECTS(
        (bin_size[Borders::bin_index] > 0 ? count_nz_block_row<index_type, Borders::max_border>
             <<<(index_type)bin_size[Borders::bin_index], Borders::config_t::block_size, 0, streams[Borders::bin_index]>>>(
                 c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(),
                 b_row_idx.data(), b_col_idx.data(),
                 permutation_buffer.data() + bin_offset[Borders::bin_index], row_idx.data())
                                          : void()));
  }

  template <typename Border>
  global_hash_table_state_t exec_global_row(
      index_type n_cols, const container_t<index_type>& c_col_idx,
      const container_t<index_type>& c_row_idx, const container_t<index_type>& a_col_idx,
      const container_t<index_type>& a_row_idx, const container_t<index_type>& b_col_idx,
      const container_t<index_type>& b_row_idx, const container_t<index_type>& permutation_buffer,
      const container_t<index_type>& bin_offset, const container_t<index_type>& bin_size,
      container_t<index_type>& row_idx, std::tuple<Border>) {
    index_type size = bin_size[Border::bin_index];

    if (size == 0)
      return {};

    container_t<index_type> aka_fail_stat(
        permutation_buffer.begin() + bin_offset[Border::bin_index],
        permutation_buffer.begin() + bin_offset[Border::bin_index] + size);

    container_t<index_type> hash_table_offsets(size + 1);

    thrust::transform(aka_fail_stat.begin(), aka_fail_stat.end(), hash_table_offsets.begin(),
                      [prod = row_idx.data()] __device__(auto row_id) { return prod[row_id]; });

    thrust::exclusive_scan(hash_table_offsets.begin(), hash_table_offsets.end(),
                           hash_table_offsets.begin());

    using namespace util;

    util::resize_and_fill_max(hash_table, hash_table_offsets.back());

    count_nz_block_row_large<index_type><<<size, Border::config_t::block_size>>>(
        n_cols, c_row_idx.data(), c_col_idx.data(), a_row_idx.data(), a_col_idx.data(),
        b_row_idx.data(), b_col_idx.data(), aka_fail_stat.data(), hash_table_offsets.data(),
        hash_table.data(), row_idx.data());

    container_t<index_type> sorted_hash_table(hash_table.size());

    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortKeys(
        nullptr, temp_storage_bytes, thrust::raw_pointer_cast(hash_table.data()),
        thrust::raw_pointer_cast(sorted_hash_table.data()), hash_table.size(), size,
        thrust::raw_pointer_cast(hash_table_offsets.data()),
        thrust::raw_pointer_cast(hash_table_offsets.data()) + 1);

    storage.resize(temp_storage_bytes);

    cub::DeviceSegmentedRadixSort::SortKeys(
        thrust::raw_pointer_cast(storage.data()), temp_storage_bytes,
        thrust::raw_pointer_cast(hash_table.data()),
        thrust::raw_pointer_cast(sorted_hash_table.data()), hash_table.size(), size,
        thrust::raw_pointer_cast(hash_table_offsets.data()),
        thrust::raw_pointer_cast(hash_table_offsets.data()) + 1);

    return {std::move(sorted_hash_table), std::move(hash_table_offsets), std::move(aka_fail_stat)};
  }

  template <typename... Borders>
  row_index_res_t operator()(index_type n_rows, index_type n_cols,
                             const container_t<index_type>& c_col_idx,
                             const container_t<index_type>& c_row_idx,
                             const container_t<index_type>& a_col_idx,
                             const container_t<index_type>& a_row_idx,
                             const container_t<index_type>& b_col_idx,
                             const container_t<index_type>& b_row_idx, std::tuple<Borders...>) {
    constexpr size_t bin_count = sizeof...(Borders);
    constexpr size_t unused_bin = meta::max_bin<Borders...> + 1;

    container_t<index_type> products_per_row(n_rows + 1, 0);
    util::resize_and_fill_zeros(bin_size, bin_count);
    bin_offset.resize(bin_count);
    permutation_buffer.resize(n_rows);

    util::kernel_call(
        n_rows, 32,
        [rpt_a = a_row_idx.data(), col_a = a_col_idx.data(), rpt_b = b_row_idx.data(),
         col_b = b_col_idx.data(), rpt_c = c_row_idx.data(), row_per_bin = bin_size.data(),
         max_c_cols = n_cols, prod_per_row = products_per_row.data()] __device__() {
          auto rid = blockIdx.x;
          auto tid = threadIdx.x;

          index_type prod = 0;

          index_type a_begin = rpt_a[rid];
          index_type a_end = rpt_a[rid + 1];

          for (size_t j = a_begin + tid; j < a_end; j += blockDim.x) {
            index_type val_a = col_a[j];
            prod += rpt_b[val_a + 1] - rpt_b[val_a];
          }

          prod = util::warpReduceSum(prod);
          prod = min(max_c_cols, prod);

          if (tid == 0) {
            prod_per_row[rid] = prod;
            size_t bin = meta::select_bin<Borders...>(prod, unused_bin);
            if (bin != unused_bin)
              atomicAdd(row_per_bin.get() + bin, 1);
          }
        });

    thrust::exclusive_scan(bin_size.begin(), bin_size.end(), bin_offset.begin());

    util::fill_zeros(bin_size, bin_count);

    thrust::for_each(thrust::counting_iterator<index_type>(0),
                     thrust::counting_iterator<index_type>(n_rows),
                     [prod_per_row = products_per_row.data(), bin_offset = bin_offset.data(),
                      bin_size = bin_size.data(),
                      rows_in_bins = permutation_buffer.data()] __device__(index_type tid) {
                       auto prod = prod_per_row[tid];

                       int bin = meta::select_bin<Borders...>(prod, unused_bin);

                       if (bin == unused_bin)
                         return;

                       auto curr_bin_size = atomicAdd(bin_size.get() + bin, 1);
                       rows_in_bins[bin_offset[bin] + curr_bin_size] = tid;
                     });

    exec_pwarp_row(c_col_idx, c_row_idx, a_col_idx, a_row_idx, b_col_idx, b_row_idx,
                   permutation_buffer, bin_offset, bin_size, products_per_row,
                   meta::filter<meta::pwarp_row, Borders...>);

    exec_block_row(c_col_idx, c_row_idx, a_col_idx, a_row_idx, b_col_idx, b_row_idx,
                   permutation_buffer, bin_offset, bin_size, products_per_row,
                   meta::filter<meta::block_row, Borders...>);

    auto global_hash_table_state =
        exec_global_row(n_cols, c_col_idx, c_row_idx, a_col_idx, a_row_idx, b_col_idx, b_row_idx,
                        permutation_buffer, bin_offset, bin_size, products_per_row,
                        meta::filter<meta::global_row, Borders...>);
    cudaDeviceSynchronize();
    thrust::exclusive_scan(products_per_row.begin(), products_per_row.end(),
                           products_per_row.begin());

    return {std::move(products_per_row), std::move(global_hash_table_state)};
  }

 private:
  container_t<index_type> bin_size;
  container_t<index_type> bin_offset;
  container_t<index_type> permutation_buffer;
  container_t<index_type> bucket_count;
  container_t<util::bucket_info_t<index_type>> bucket_info;
  container_t<index_type> hash_table;
  container_t<char> storage;
};

}  // namespace nsparse