#pragma once

#include <nsparse/detail/masked_mult.cuh>
#include <thrust/device_vector.h>
#include <nsparse/detail/util.h>
#include <nsparse/detail/meta.h>
#include <nsparse/unified_allocator.h>

namespace nsparse {

template <typename value_type, typename index_type, value_type zero, typename Mul, typename Add>
struct masked_mult_functor_t {
  template <typename T>
  using container_t = thrust::device_vector<T, nsparse::managed<T>>;

  masked_mult_functor_t(Mul mul, Add add) : m_mul(mul), m_add(add) {
        for (auto& s: streams) {
      cudaStreamCreate( &s);
    }
  }

  cudaStream_t streams[15];

  ~masked_mult_functor_t() {
    for (auto& s: streams) {
      cudaStreamDestroy(s);
    }
  }

  template <typename... Borders>
  void exec_masked_mult(const container_t<index_type>& c_col_idx,
                        const container_t<index_type>& c_row_idx,
                        container_t<value_type>& c_values,
                        const container_t<index_type>& a_col_idx,
                        const container_t<index_type>& a_row_idx,
                        const container_t<value_type>& a_values,
                        const container_t<index_type>& b_col_idx,
                        const container_t<index_type>& b_row_idx,
                        const container_t<value_type>& b_values, std::tuple<Borders...>) {
    thrust::host_vector<index_type> bin_offset_ = bin_offset;
    thrust::host_vector<index_type> bin_size_ = bin_size;

    EXPAND_SIDE_EFFECTS(
        (bin_size_[Borders::bin_index] > 0
             ? masked_mult<value_type, index_type, zero, Borders::config_t::block_size,
                           Borders::config_t::cache_size, Borders::config_t::cache_step>
             <<<(index_type)bin_size_[Borders::bin_index], Borders::config_t::block_size, 0, streams[Borders::bin_index]>>>(
                 c_col_idx.data(), c_row_idx.data(), c_values.data(), a_col_idx.data(),
                 a_row_idx.data(), a_values.data(), b_col_idx.data(), b_row_idx.data(),
                 b_values.data(), permutation_buffer.data() + bin_offset_[Borders::bin_index], m_mul,
                 m_add)
             : void()));
  }

  template <typename... Borders>
  void operator()(index_type n_rows, const container_t<index_type>& c_col_idx,
                  const container_t<index_type>& c_row_idx,
                  container_t<value_type>& c_values,
                  const container_t<index_type>& a_col_idx,
                  const container_t<index_type>& a_row_idx,
                  const container_t<value_type>& a_values,
                  const container_t<index_type>& b_col_idx,
                  const container_t<index_type>& b_row_idx,
                  const container_t<value_type>& b_values, std::tuple<Borders...>) {
    constexpr size_t bin_count = sizeof...(Borders);
    constexpr size_t unused_bin = meta::max_bin<Borders...> + 1;

    util::resize_and_fill_zeros(bin_size, bin_count);
    bin_offset.resize(bin_count);
    permutation_buffer.resize(n_rows);

    thrust::for_each(
        thrust::counting_iterator<index_type>(0), thrust::counting_iterator<index_type>(n_rows),
        [rpt_c = c_row_idx.data(), row_per_bin = bin_size.data()] __device__(index_type rid) {
          index_type width = rpt_c[rid + 1] - rpt_c[rid];

          if (width == 0)
            return;

          size_t bin = meta::select_bin<Borders...>(width, unused_bin);
          assert(bin != unused_bin);
          atomicAdd(row_per_bin.get() + bin, 1);
        });

    thrust::exclusive_scan(bin_size.begin(), bin_size.end(), bin_offset.begin());
    util::fill_zeros(bin_size, bin_count);

    thrust::for_each(
        thrust::counting_iterator<index_type>(0), thrust::counting_iterator<index_type>(n_rows),
        [rpt_c = c_row_idx.data(), row_per_bin = bin_size.data(), bin_offset = bin_offset.data(),
         rows_in_bins = permutation_buffer.data()] __device__(index_type rid) {
          index_type width = rpt_c[rid + 1] - rpt_c[rid];

          if (width == 0)
            return;

          size_t bin = meta::select_bin<Borders...>(width, unused_bin);
          assert(bin != unused_bin);

          auto curr_bin_size = atomicAdd(row_per_bin.get() + bin, 1);
          rows_in_bins[bin_offset[bin] + curr_bin_size] = rid;
        });

    using namespace util;

    exec_masked_mult(c_col_idx, c_row_idx, c_values, a_col_idx, a_row_idx, a_values, b_col_idx,
                     b_row_idx, b_values, std::tuple<Borders...>{});

    cudaDeviceSynchronize();
  }

 private:
  Mul m_mul;
  Add m_add;
  container_t<index_type> bin_size;
  container_t<index_type> bin_offset;
  container_t<index_type> permutation_buffer;
};

}  // namespace nsparse