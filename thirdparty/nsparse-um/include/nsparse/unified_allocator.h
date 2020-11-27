#pragma once

#include <thrust/detail/config.h>
#include <thrust/device_ptr.h>
#include <thrust/mr/allocator.h>
#include <thrust/memory/detail/device_system_resource.h>

#include <limits>
#include <stdexcept>
#include <iostream>

namespace nsparse {
inline cudaError_t cudaMallocManagedPrefetch(void** ptr, std::size_t bytes) {
  auto status = thrust::system::cuda::detail::cudaMallocManaged(ptr, bytes);
  if (status != cudaSuccess) {
    return status;
  }

  int device = -1;
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaMemPrefetchAsync(*ptr, bytes, device, NULL);
  return status;
}

using universal_prefetched_memory_resource =
    thrust::system::cuda::detail::cuda_memory_resource<cudaMallocManagedPrefetch, cudaFree,
                                                       thrust::cuda::pointer<void>>;
}  // namespace nsparse

namespace thrust {

template <typename T>
class device_unified_allocator : public thrust::mr::stateless_resource_allocator<
                                     T, device_ptr_memory_resource<nsparse::universal_prefetched_memory_resource>> {
  typedef thrust::mr::stateless_resource_allocator<
      T, device_ptr_memory_resource<nsparse::universal_prefetched_memory_resource>>
      base;

 public:
  /*! The \p rebind metafunction provides the type of a \p device_allocator
   *  instantiated with another type.
   *
   *  \tparam U the other type to use for instantiation.
   */
  template <typename U>
  struct rebind {
    /*! The typedef \p other gives the type of the rebound \p device_allocator.
     */
    typedef device_unified_allocator<U> other;
  };

  /*! Default constructor has no effect. */
  __host__ device_unified_allocator() {
  }

  /*! Copy constructor has no effect. */
  __host__ device_unified_allocator(const device_unified_allocator& other) : base(other) {
  }

  /*! Constructor from other \p device_allocator has no effect. */
  template <typename U>
  __host__ device_unified_allocator(const device_unified_allocator<U>& other) : base(other) {
  }

  /*! Destructor has no effect. */
  __host__ ~device_unified_allocator() {
  }
};

}  // namespace thrust

namespace nsparse {

template <typename T>
using managed = thrust::device_unified_allocator<T>;

//template <typename T>
//using managed_vector = thrust::device_vector<T, managed<T>>;

}  // namespace nsparse