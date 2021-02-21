#pragma once

#include <detail/util.h>
#include <detail/bitonic.cuh>

#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nsparse {

template <typename T, T block_per_row>
__global__ void count_nz_block_row_large_global(
    thrust::device_ptr<const T> rpt_c, thrust::device_ptr<const T> col_c,
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const T> rows_in_bins, thrust::device_ptr<T> nz_per_row,
    thrust::device_ptr<T> global_hash_table, thrust::device_ptr<const T> hash_table_offset) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  auto bid = blockIdx.x % block_per_row;
  auto rid = blockIdx.x / block_per_row;
  auto wid = (threadIdx.x + bid * blockDim.x) / warpSize;
  auto i = threadIdx.x % warpSize;
  auto warpCount = blockDim.x * block_per_row / warpSize;

  T* hash_table = global_hash_table.get() + hash_table_offset[rid];
  T table_sz = hash_table_offset[rid + 1] - hash_table_offset[rid];
  __shared__ T snz;

  if (threadIdx.x == 0) {
    snz = 0;
  }

  __syncthreads();

  rid = rows_in_bins[rid];  // permutation
  T nz = 0;

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

  for (auto j = warpSize / 2; j >= 1; j /= 2) {
    nz += __shfl_xor_sync(0xffffffff, nz, j);
  }

  if (i == 0) {
    atomicAdd(&snz, nz);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicAdd(nz_per_row.get() + rid, snz);
  }
}

template <typename T, unsigned int table_sz>
__global__ void count_nz_block_row_large(
    thrust::device_ptr<const T> rpt_c, thrust::device_ptr<const T> col_c,
    thrust::device_ptr<const T> rpt_a, thrust::device_ptr<const T> col_a,
    thrust::device_ptr<const T> rpt_b, thrust::device_ptr<const T> col_b,
    thrust::device_ptr<const util::bucket_info_t<T>> buckets, thrust::device_ptr<T> global_table,
    thrust::device_ptr<T> nz_per_row) {
  constexpr T hash_invalidated = std::numeric_limits<T>::max();

  __shared__ T hash_table[table_sz];
  __shared__ T nz;

  auto bid = blockIdx.x;
  auto wid = threadIdx.x / warpSize;
  auto i = threadIdx.x % warpSize;
  auto warpCount = blockDim.x / warpSize;

  for (auto m = threadIdx.x; m < table_sz; m += blockDim.x) {
    hash_table[m] = hash_invalidated;
  }

  if (threadIdx.x == 0) {
    nz = 0;
  }

  __syncthreads();

  const util::bucket_info_t<T> bucket = buckets[bid];

  for (T j = bucket.a_row_begin + wid; j < bucket.a_row_end; j += warpCount) {
    T a_col = col_a[j];

    bool is_first = j == bucket.a_row_begin;
    bool is_last = j == (bucket.a_row_end - 1);

    T b_begin = is_first * bucket.b_row_begin + !is_first * rpt_b[a_col];
    T b_end = is_last * bucket.b_row_end + !is_last * rpt_b[a_col + 1];

    for (T k = b_begin + i; k < b_end; k += warpSize) {
      T b_col = col_b[k];

      hash_table[atomicAdd(&nz, 1)] = b_col;
    }
  }

  __syncthreads();

  T global_offset = bid * table_sz;

  for (auto i = threadIdx.x; i < table_sz; i += blockDim.x) {
    global_table[global_offset + i] = hash_table[i];
  }

  if (threadIdx.x == 0) {
    atomicAdd(nz_per_row.get() + bucket.row_id, nz);
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

  __syncthreads();

  rid = rows_in_bins[rid];  // permutation
  T nz = 0;

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

  for (auto j = warpSize / 2; j >= 1; j /= 2) {
    nz += __shfl_xor_sync(0xffffffff, nz, j);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    hash_table[0] = 0;
  }

  __syncthreads();

  if (i == 0) {
    atomicAdd(hash_table, nz);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    nz_per_row[rid] = hash_table[0];
  }
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