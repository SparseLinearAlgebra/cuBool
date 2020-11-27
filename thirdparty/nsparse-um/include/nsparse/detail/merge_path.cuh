#pragma once

#include <thrust/device_ptr.h>

#include <nsparse/detail/bitonic.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace nsparse {

template <typename T>
__global__ void validate_order(thrust::device_ptr<const T> rpt_a,
                               thrust::device_ptr<const T> col_a) {
  auto row = blockIdx.x;

  T global_offset_a = rpt_a[row];
  T sz_a = rpt_a[row + 1] - global_offset_a;

  for (auto i = global_offset_a + threadIdx.x; i < sz_a + global_offset_a; i += blockDim.x) {
    if (i > global_offset_a)
      assert(col_a[i - 1] <= col_a[i]);

    assert(col_a[i] != std::numeric_limits<T>::max());
  }
}

template <typename T, T block_size>
__global__ void merge_path_count(thrust::device_ptr<const T> rpt_a,
                                 thrust::device_ptr<const T> col_a,
                                 thrust::device_ptr<const T> rpt_b,
                                 thrust::device_ptr<const T> col_b, thrust::device_ptr<T> rpt_c,
                                 thrust::device_ptr<const T> rows_in_bins) {
  const auto row = rows_in_bins[blockIdx.x];

  constexpr T max_val = std::numeric_limits<T>::max();

  const T global_offset_a = rpt_a[row];
  const T sz_a = rpt_a[row + 1] - global_offset_a;

  const T global_offset_b = rpt_b[row];
  const T sz_b = rpt_b[row + 1] - global_offset_b;

  const T block_count = (sz_a + sz_b + block_size - 1) / block_size;

  T begin_a = 0;
  T begin_b = 0;

  __shared__ T raw_a[block_size + 2];
  __shared__ T raw_b[block_size + 2];
  __shared__ T res[block_size];

  bool dir = true;
  T item_from_prev_chank = max_val;

  for (auto i = 0; i < block_count; i++) {
    __shared__ T max_x_index;
    __shared__ T max_y_index;

    T max_x_index_per_thread = 0;
    T max_y_index_per_thread = 0;

    assert(sz_a >= begin_a);
    assert(sz_b >= begin_b);

    T buf_a_size = min(sz_a - begin_a, block_size);
    T buf_b_size = min(sz_b - begin_b, block_size);

    if (threadIdx.x == 0) {
      max_x_index = 0;
      max_y_index = 0;
    }

    for (auto j = threadIdx.x; j < block_size + 2; j += blockDim.x) {
      if (j > 0 && j - 1 < buf_a_size) {
        raw_a[j] = col_a[global_offset_a + j - 1 + begin_a];
      } else {
        raw_a[j] = max_val;
      }
      if (j > 0 && j - 1 < buf_b_size) {
        raw_b[j] = col_b[global_offset_b + j - 1 + begin_b];
      } else {
        raw_b[j] = max_val;
      }
    }

    __syncthreads();

    const T to_process = min(buf_b_size + buf_a_size, block_size);

    for (auto j = threadIdx.x; j < to_process; j += blockDim.x) {
      const T y = j + 2;
      const T x = 0;

      T l = 0;
      T r = j + 2;

      while (r - l > 1) {
        bool ans = raw_b[y - l - (r - l) / 2] > raw_a[x + l + (r - l) / 2];

        l += (r - l) / 2 * ans;
        r -= (r - l) / 2 * !ans;
      }

      T ans_x = x + l;
      T ans_y = y - l;

      if (ans_y == 1 || ans_x == 0) {
        if (ans_y == 1) {
          res[j] = raw_a[ans_x];
          max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
        } else {
          res[j] = raw_b[ans_y - 1];
          max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
        }
      } else {
        if (raw_b[ans_y - 1] > raw_a[ans_x]) {
          res[j] = raw_b[ans_y - 1];
          max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
        } else {
          res[j] = raw_a[ans_x];
          max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
        }
      }
    }

    atomicMax(&max_x_index, max_x_index_per_thread);
    atomicMax(&max_y_index, max_y_index_per_thread);

    __syncthreads();

    T counter = 0;

    if (dir) {
      for (auto m = threadIdx.x; m < to_process; m += blockDim.x) {
        if (m > 0)
          counter += (res[m] - res[m - 1]) != 0;
        else
          counter += (res[0] - item_from_prev_chank) != 0;
        item_from_prev_chank = res[m];
      }
    } else {
      for (auto m = blockDim.x - 1 - threadIdx.x; m < to_process; m += blockDim.x) {
        if (m > 0)
          counter += (res[m] - res[m - 1]) != 0;
        else
          counter += (res[0] - item_from_prev_chank) != 0;
        item_from_prev_chank = res[m];
      }
    }

    dir = !dir;

    atomicAdd(rpt_c.get() + row, counter);

    begin_a += max_x_index;
    begin_b += max_y_index;

    __syncthreads();
  }
}

template <typename T, T block_size>
__global__ void merge_path_fill(thrust::device_ptr<const T> rpt_a,
                                thrust::device_ptr<const T> col_a,
                                thrust::device_ptr<const T> rpt_b,
                                thrust::device_ptr<const T> col_b,
                                thrust::device_ptr<const T> rpt_c, thrust::device_ptr<T> col_c,
                                thrust::device_ptr<const T> rows_in_bins) {
  const auto row = rows_in_bins[blockIdx.x];

  constexpr T max_val = std::numeric_limits<T>::max();

  const T global_offset_a = rpt_a[row];
  const T sz_a = rpt_a[row + 1] - global_offset_a;

  const T global_offset_b = rpt_b[row];
  const T sz_b = rpt_b[row + 1] - global_offset_b;

  T global_offset_c = rpt_c[row];

  const T block_count = (sz_a + sz_b + block_size - 1) / block_size;

  T begin_a = 0;
  T begin_b = 0;

  __shared__ T raw_a[block_size + 2];
  __shared__ T raw_b[block_size + 2];
  __shared__ T res[block_size];

  bool dir = true;
  T item_from_prev_chank = max_val;

  for (auto i = 0; i < block_count; i++) {
    __shared__ T max_x_index;
    __shared__ T max_y_index;

    T max_x_index_per_thread = 0;
    T max_y_index_per_thread = 0;

    assert(sz_a >= begin_a);
    assert(sz_b >= begin_b);

    T buf_a_size = min(sz_a - begin_a, block_size);
    T buf_b_size = min(sz_b - begin_b, block_size);

    if (threadIdx.x == 0) {
      max_x_index = 0;
      max_y_index = 0;
    }

    for (auto j = threadIdx.x; j < block_size + 2; j += blockDim.x) {
      if (j > 0 && j - 1 < buf_a_size) {
        raw_a[j] = col_a[global_offset_a + j - 1 + begin_a];
      } else {
        raw_a[j] = max_val;
      }
      if (j > 0 && j - 1 < buf_b_size) {
        raw_b[j] = col_b[global_offset_b + j - 1 + begin_b];
      } else {
        raw_b[j] = max_val;
      }
    }

    __syncthreads();

    const T to_process = min(buf_b_size + buf_a_size, block_size);

    T answer = max_val;

    const auto j = dir ? threadIdx.x : blockDim.x - 1 - threadIdx.x;

    if (j < to_process) {
      const T y = j + 2;
      const T x = 0;

      T l = 0;
      T r = j + 2;

      while (r - l > 1) {
        bool ans = raw_b[y - l - (r - l) / 2] > raw_a[x + l + (r - l) / 2];

        l += (r - l) / 2 * ans;
        r -= (r - l) / 2 * !ans;
      }

      T ans_x = x + l;
      T ans_y = y - l;

      if (ans_y == 1 || ans_x == 0) {
        if (ans_y == 1) {
          answer = raw_a[ans_x];
          max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
        } else {
          answer = raw_b[ans_y - 1];
          max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
        }
      } else {
        if (raw_b[ans_y - 1] > raw_a[ans_x]) {
          answer = raw_b[ans_y - 1];
          max_y_index_per_thread = max(max_y_index_per_thread, ans_y - 1);
        } else {
          answer = raw_a[ans_x];
          max_x_index_per_thread = max(max_x_index_per_thread, ans_x);
        }
      }
    }

    atomicMax(&max_x_index, max_x_index_per_thread);
    atomicMax(&max_y_index, max_y_index_per_thread);

    res[j] = answer;

    __syncthreads();

    bool take = j < to_process;
    if (j > 0)
      take = take && (answer - res[j - 1]) != 0;
    else
      take = take && (answer - item_from_prev_chank) != 0;

    item_from_prev_chank = answer;

    __syncthreads();

    res[j] = take;

    __syncthreads();

    auto warpId = threadIdx.x / warpSize;
    auto laneId = threadIdx.x % warpSize;

    {
      T item = res[threadIdx.x];
      for (auto i = 1; i <= warpSize; i *= 2) {
        T other = __shfl_up_sync(0xffffffff, item, i);

        if (laneId >= i) {
          item += other;
        }
      }
      res[threadIdx.x] = item;
    }

    __syncthreads();

    if (warpId == 0) {
      for (auto i = 1; i < block_size / warpSize; i++) {
        T accum = res[i * warpSize - 1];
        res[i * warpSize + laneId] += accum;
      }
    }

    __syncthreads();

    if (take) {
      col_c[global_offset_c + (j == 0 ? 0 : res[j - 1])] = answer;
    }

    global_offset_c += res[block_size - 1];

    dir = !dir;

    begin_a += max_x_index;
    begin_b += max_y_index;

    __syncthreads();
  }
}

}  // namespace nsparse