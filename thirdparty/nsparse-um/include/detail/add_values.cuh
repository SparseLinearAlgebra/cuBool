#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace nsparse {

template <typename value_type, typename index_type>
__global__ void add_values(thrust::device_ptr<const index_type> skeleton_col_idx,
                           thrust::device_ptr<const index_type> skeleton_row_idx,
                           thrust::device_ptr<value_type> values,
                           thrust::device_ptr<const index_type> edges_col_idx,
                           thrust::device_ptr<const index_type> edges_row_idx, value_type value) {
  auto rid = blockIdx.x;

  index_type skeleton_row_begin = skeleton_row_idx[rid];
  index_type skeleton_row_end = skeleton_row_idx[rid + 1];

  index_type edges_row_begin = edges_row_idx[rid];
  index_type edges_row_end = edges_row_idx[rid + 1];

  for (auto i = edges_row_begin + threadIdx.x; i < edges_row_end; i += blockDim.x) {
    index_type edges_column = edges_col_idx[i];

    bool was_found = false;
    for (auto j = skeleton_row_begin; j < skeleton_row_end; j++) {
      index_type skeleton_column = skeleton_col_idx[j];

      if (skeleton_column == edges_column) {
        values[j] = value;
        was_found = true;
        break;
      }
    }
    assert(was_found);
  }
}

}  // namespace nsparse
