#pragma once

#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nsparse {

template <typename value_type, typename index_type, value_type Zero, unsigned int block_size,
          unsigned int cache_size, unsigned int cache_step, typename Mul, typename Add>
__global__ void masked_mult(
    thrust::device_ptr<const index_type> c_col_idx, thrust::device_ptr<const index_type> c_row_idx,
    thrust::device_ptr<value_type> c_values, thrust::device_ptr<const index_type> a_col_idx,
    thrust::device_ptr<const index_type> a_row_idx, thrust::device_ptr<const value_type> a_values,
    thrust::device_ptr<const index_type> b_col_idx, thrust::device_ptr<const index_type> b_row_idx,
    thrust::device_ptr<const value_type> b_values, thrust::device_ptr<const index_type> permutation,
    Mul&& mul, Add&& add) {
  constexpr index_type warp_size = 32;

  const index_type wid = threadIdx.x / warp_size;
  const index_type rid = permutation[blockIdx.x];
  const index_type c_row_begin = c_row_idx[rid];
  const index_type c_row_end = c_row_idx[rid + 1];
  const index_type c_row_size = c_row_end - c_row_begin;

  if (c_row_size == 0)
    return;

  constexpr index_type warp_count = block_size / warp_size;
  const index_type tid = threadIdx.x % warp_size;

  __shared__ index_type cache[cache_size];

  index_type j;

  for (j = threadIdx.x; j < cache_size; j += blockDim.x) {
    if (j * cache_step < c_row_size)
      cache[j] = c_col_idx[c_row_begin + j * cache_step];
    else
      cache[j] = std::numeric_limits<index_type>::max();
  }
  __syncthreads();

  for (j = a_row_idx[rid] + wid; j < a_row_idx[rid + 1]; j += warp_count) {
    index_type a_col = a_col_idx[j];
    value_type a_value = a_values[j];

    if (a_value == Zero)
      continue;

    index_type b_row_begin = b_row_idx[a_col];
    index_type b_row_end = b_row_idx[a_col + 1];

    for (index_type k = b_row_begin + tid; k < b_row_end; k += warp_size) {
      index_type b_col = b_col_idx[k];
      value_type b_value = b_values[k];

      if (b_value == Zero)
        continue;

      value_type mult_res = mul(a_value, b_value, a_col);

      int l = 0;
      int r = cache_size;

      while (r - l > 1) {
        int delta = (r - l) / 2;

        bool satisfy = cache[l + delta] <= b_col;

        l += delta * satisfy;
        r -= delta * !satisfy;
      }

      l = l * cache_step + c_row_begin;
      r = min(l + cache_step, c_row_end);

      while (r - l > 1) {
        int delta = (r - l) / 2;

        bool satisfy = c_col_idx[l + delta] <= b_col;

        l += delta * satisfy;
        r -= delta * !satisfy;
      }

      add(c_values.get() + l, mult_res);
    }
  }
}

}  // namespace nsparse