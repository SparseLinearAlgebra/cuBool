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
  constexpr index_type wapr_size = 32;

  __shared__ index_type data[4];
  if (threadIdx.x == 0) {
    data[0] = permutation[blockIdx.x];
    data[1] = c_row_idx[data[0]];
    data[2] = c_row_idx[data[0] + 1];
    data[3] = data[2] - data[1];
  }

//  index_type rid = permutation[blockIdx.x];
//  index_type c_row_begin = c_row_idx[rid];
//  index_type c_row_end = c_row_idx[rid + 1];
//  index_type c_row_size = c_row_end - c_row_begin;

  index_type& rid = data[0];
  index_type& c_row_begin = data[1];
  index_type& c_row_end = data[2];
  index_type& c_row_size = data[3];

  __syncthreads();

  if (c_row_size == 0)
    return;

  const index_type wid = threadIdx.x / wapr_size;
  constexpr index_type warp_count = block_size / wapr_size;
  const index_type tid = threadIdx.x % wapr_size;

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

    for (index_type iter = 0; iter < (b_row_end - b_row_begin + warpSize - 1) / wapr_size; iter++) {
      index_type k = iter * warpSize + tid + b_row_begin;

      index_type b_col;
      value_type b_value;

      bool skip = false;

      if (k < b_row_end) {
        b_col = b_col_idx[k];
        b_value = b_values[k];
      } else {
        b_value = Zero;
      }

      if (b_value == Zero)
        skip = true;

      value_type mult_res = mul(a_value, b_value, a_col);

      int l = 0;
      int r = cache_size;

      if (!skip) {
        while (r - l > 1) {
          int delta = (r - l) / 2;
          assert(l + delta < cache_size);
          bool satisfy = cache[l + delta] <= b_col;

          l += delta * satisfy;
          r -= delta * !satisfy;
        }
      }

      constexpr index_type pwarp_size = std::min(cache_step, wapr_size);
      const index_type pwarp_id = tid / pwarp_size;
      unsigned pwarp_mask = 0xffffffff;         // 11111111
      pwarp_mask >>= (wapr_size - pwarp_size);  // 00000011
      pwarp_mask <<= pwarp_id * pwarp_size;     // 00001100

      for (index_type i = 0; i < pwarp_size; i++) {
        if (__shfl_sync(pwarp_mask, skip, i + pwarp_id * pwarp_size))
          continue;

        index_type b_col_broadcast = __shfl_sync(pwarp_mask, b_col, i + pwarp_id * pwarp_size);
        value_type mult_res_broadcast =
            __shfl_sync(pwarp_mask, mult_res, i + pwarp_id * pwarp_size);
        index_type l_broadcast = __shfl_sync(pwarp_mask, l, i + pwarp_id * pwarp_size);

        index_type search_from = l_broadcast * cache_step + c_row_begin;
        index_type search_to = search_from + cache_step;

        bool was_found = false;
        for (index_type search = search_from + tid % pwarp_size; search < search_to;
             search += pwarp_size) {
          bool vote = (search < c_row_end) ? c_col_idx[search] == b_col_broadcast : false;

          if (unsigned vote_res = __ballot_sync(pwarp_mask, vote)) {
            assert(__ffs(vote_res) > 0);
            if (__ffs(vote_res) - 1 == tid) {
              add(c_values.get() + search, mult_res_broadcast);
            }
            was_found = true;
            break;
          }
        }
        assert(was_found);
      }
    }
  }
}

}  // namespace nsparse