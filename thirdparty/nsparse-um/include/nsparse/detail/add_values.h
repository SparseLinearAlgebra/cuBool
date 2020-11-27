#pragma once

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <nsparse/detail/util.h>

#include <nsparse/detail/add_values.cuh>

namespace nsparse {

template <typename value_type, typename index_type>
void add_values(index_type rows, const thrust::device_vector<index_type>& skeleton_col_idx,
                const thrust::device_vector<index_type>& skeleton_row_idx,
                thrust::device_vector<value_type>& values,
                const thrust::device_vector<index_type>& edges_col_idx,
                const thrust::device_vector<index_type>& edges_row_idx, value_type value) {
  if (rows > 0)
    add_values<<<rows, 64>>>(skeleton_col_idx.data(), skeleton_row_idx.data(), values.data(),
                             edges_col_idx.data(), edges_row_idx.data(), value);
}

}  // namespace nsparse