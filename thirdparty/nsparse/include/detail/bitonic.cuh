#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace nsparse {

template <typename T>
__device__ void Comparator(T& keyA, T& keyB, uint dir) {
  T t;

  if ((keyA > keyB) == dir) {
    t = keyA;
    keyA = keyB;
    keyB = t;
  }
}

template <typename T, uint array_size, typename group_t>
__device__ void bitonic_sort_shared(group_t group, T* s_key, uint dir = 1) {
  for (uint size = 2; size < array_size; size <<= 1) {
    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      group.sync();
      for (uint id = group.thread_rank(); id < array_size / 2; id += group.size()) {
        uint ddd = dir ^ ((id & (size / 2)) != 0);

        uint pos = 2 * id - (id & (stride - 1));
        Comparator(s_key[pos + 0], s_key[pos + stride], ddd);
      }
    }
  }

  for (uint stride = array_size / 2; stride > 0; stride >>= 1) {
    group.sync();
    for (uint id = group.thread_rank(); id < array_size / 2; id += group.size()) {
      uint pos = 2 * id - (id & (stride - 1));
      Comparator(s_key[pos + 0], s_key[pos + stride], dir);
    }
  }
  group.sync();
}

template <typename T>
__device__ void bitonicSortGlobal(T* key, T array_size, uint dir = 1) {
  for (uint size = 2; size < array_size; size <<= 1) {
    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      __syncthreads();
      for (uint id = threadIdx.x; id < array_size / 2; id += blockDim.x) {
        uint ddd = dir ^ ((id & (size / 2)) != 0);

        uint pos = 2 * id - (id & (stride - 1));
        Comparator(key[pos + 0], key[pos + stride], ddd);
      }
    }
  }

  for (uint stride = array_size / 2; stride > 0; stride >>= 1) {
    __syncthreads();
    for (uint id = threadIdx.x; id < array_size / 2; id += blockDim.x) {
      uint pos = 2 * id - (id & (stride - 1));
      Comparator(key[pos + 0], key[pos + stride], dir);
    }
  }
}

}  // namespace nsparse
