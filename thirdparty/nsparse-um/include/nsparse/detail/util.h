#pragma once
#include <cmath>
#include <ostream>
#include <thrust/device_vector.h>


namespace nsparse {
namespace util {

template <typename T>
struct bucket_info_t {
  T row_id;
  T a_row_begin;
  T b_row_begin;
  T a_row_end;
  T b_row_end;
};

template <typename T>
T div(T m, T n) {
  return (m + n - 1) / n;
}

template <typename T>
__device__ T nearest_pow_2(T m) {
  T res = 1;
  while (res < m) {
    res *= 2;
  }
  return res;
}

template <typename T>
__device__ T warpReduceSum(T val) {
  for (T offset = warpSize / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

namespace detail {
template <typename F>
__global__ void kernel_call_impl(F functor) {
  functor();
}
}  // namespace detail

template <typename G, typename B, typename F>
void kernel_call(G grid, B block, F&& functor) {
  detail::kernel_call_impl<<<grid, block>>>(std::forward<decltype(functor)>(functor));
}

template <typename T, typename Alloc>
void resize_and_fill_max(thrust::device_vector<T, Alloc>& vec, size_t size) {
  vec.resize(size);
  cudaMemsetAsync(thrust::raw_pointer_cast(vec.data()), -1, sizeof(T) * size);
}

template <typename T, typename Alloc>
void resize_and_fill_zeros(thrust::device_vector<T, Alloc>& vec, size_t size) {
  vec.resize(size);
  cudaMemsetAsync(thrust::raw_pointer_cast(vec.data()), 0, sizeof(T) * size);
}

template <typename T, typename Alloc>
void fill_zeros(thrust::device_vector<T, Alloc>& vec, size_t size) {
  cudaMemsetAsync(thrust::raw_pointer_cast(vec.data()), 0, sizeof(T) * size);
}

template <typename T, typename Alloc>
std::ostream& operator<<(std::ostream& os, const thrust::device_vector<T, Alloc>& vec) {
  thrust::host_vector<T> h_vec = vec;
  for (auto item : h_vec) {
    os << item << " ";
  }
  return os;
}

}  // namespace util

}  // namespace nsparse