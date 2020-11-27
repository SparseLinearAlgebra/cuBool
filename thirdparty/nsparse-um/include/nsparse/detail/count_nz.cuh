#pragma once

#include <nsparse/detail/util.h>
#include <nsparse/detail/bitonic.cuh>

#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nsparse {

template <typename T>
__global__ void count_nz_block_row_large(
    T n_cols, thrust::device_ptr<const T> rpt_c, thrust::device_ptr<const T> col_c,
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const T> rows_in_bins, thrust::device_ptr<const T> global_table_offsets,
    thrust::device_ptr<T> global_table, thrust::device_ptr<T> row_idx) {
  __shared__ T nz;

  if (threadIdx.x == 0) {
    nz = 0;
  }

  __syncthreads();

  auto rid = blockIdx.x;
  auto wid = threadIdx.x / warpSize;
  auto i = threadIdx.x % warpSize;
  auto warpCount = blockDim.x / warpSize;
  T offset = global_table_offsets[rid];
  T table_sz = global_table_offsets[rid + 1] - offset;

  assert(table_sz <= n_cols);

  rid = rows_in_bins[rid];  // permutation

  for (T j = rpt_a[rid] + wid; j < rpt_a[rid + 1]; j += warpCount) {
    T a_col = col_a[j];

    T b_col_begin = rpt_b[a_col];
    T b_col_end = rpt_b[a_col + 1];

    for (T k = b_col_begin + i; k < b_col_end; k += warpSize) {
      T b_col = col_b[k];

      if (table_sz == n_cols) {
        constexpr T hash_invalidate = std::numeric_limits<T>::max();
        if (atomicCAS(global_table.get() + offset + b_col, hash_invalidate, b_col) ==
            hash_invalidate) {
          atomicAdd(&nz, 1);
        }
      } else {
        global_table[atomicAdd(&nz, 1) + offset] = b_col;
      }
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    row_idx[rid] = nz;
  }
}

template <typename T, unsigned int table_sz>
__global__ void count_nz_block_row(
    thrust::device_ptr<const T> rpt_c, thrust::device_ptr<const T> col_c,
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const T> rows_in_bins, thrust::device_ptr<T> nz_per_row) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  __shared__ T hash_table[table_sz];

  auto rid = blockIdx.x;
  auto wid = threadIdx.x / warpSize;
  auto i = threadIdx.x % warpSize;
  auto warpCount = blockDim.x / warpSize;

  for (auto m = threadIdx.x; m < table_sz; m += blockDim.x) {
    hash_table[m] = hash_invalidated;
  }


  rid = rows_in_bins[rid];  // permutation
  T nz = 0;

  nz_per_row[rid] = 0;

  __syncthreads();

  for (T j = rpt_a[rid] + wid; j < rpt_a[rid + 1]; j += warpCount) {
    T a_col = col_a[j];
    for (T k = rpt_b[a_col] + i; k < rpt_b[a_col + 1]; k += warpSize) {
      T b_col = col_b[k];

      T hash = (b_col * 107) % table_sz;
      T offset = hash;

      while (true) {
        T table_value = hash_table[offset];
        if (table_value == b_col) {
          break;
        } else if (table_value == hash_invalidated) {
          T old_value = atomicCAS(hash_table + offset, hash_invalidated, b_col);
          if (old_value == hash_invalidated) {
            nz++;
            break;
          }
        } else {
          hash = (hash + 1) % table_sz;
          offset = hash;
        }
      }
    }
  }

  atomicAdd(nz_per_row.get() + rid, nz);
}

template <typename T, T pwarp, T block_sz, T max_per_row>
__global__ void count_nz_pwarp_row(
    thrust::device_ptr<const T> rpt_c, thrust::device_ptr<const T> col_c,
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const T> rows_in_bins, thrust::device_ptr<T> nz_per_row, T n_rows) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  static_assert(block_sz % pwarp == 0);
  static_assert(block_sz >= pwarp);

  auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ T hash_table[block_sz / pwarp * max_per_row];

  auto rid = tid / pwarp;
  auto i = tid % pwarp;
  auto local_rid = rid % (blockDim.x / pwarp);

  for (auto j = i; j < max_per_row; j += pwarp) {
    hash_table[local_rid * max_per_row + j] = hash_invalidated;
  }

  __syncwarp();

  if (rid >= n_rows)
    return;

  rid = rows_in_bins[rid];  // permutation
  T nz = 0;

  for (T j = rpt_a[rid] + i; j < rpt_a[rid + 1]; j += pwarp) {
    T a_col = col_a[j];
    for (T k = rpt_b[a_col]; k < rpt_b[a_col + 1]; k++) {
      T b_col = col_b[k];

      T hash = (b_col * 107) % max_per_row;
      T offset = hash + local_rid * max_per_row;

      while (true) {
        T table_value = hash_table[offset];
        if (table_value == b_col) {
          break;
        } else if (table_value == hash_invalidated) {
          T old_value = atomicCAS(hash_table + offset, hash_invalidated, b_col);
          if (old_value == hash_invalidated) {
            nz++;
            break;
          }
        } else {
          hash = (hash + 1) % max_per_row;
          offset = hash + local_rid * max_per_row;
        }
      }
    }
  }

  auto mask = __activemask();
  for (auto j = pwarp / 2; j >= 1; j /= 2) {
    nz += __shfl_xor_sync(mask, nz, j);
  }

  if (i == 0) {
    nz_per_row[rid] = nz;
  }
}
}  // namespace nsparse