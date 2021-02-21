#pragma once
#include <matrix.h>
#include <detail/add_values.h>

namespace nsparse {

template <typename ValueType, typename IndexType>
class masked_matrix {
 public:
  typedef IndexType index_type;
  typedef ValueType value_type;

  masked_matrix(matrix<bool, index_type> skeleton, int default_value)
      : m_skeleton(std::move(skeleton)) {
    m_values.resize(m_skeleton.m_vals);
    cudaMemsetAsync(thrust::raw_pointer_cast(m_values.data()), default_value,
                    sizeof(value_type) * m_skeleton.m_vals);
  }

  masked_matrix(matrix<bool, index_type> skeleton, thrust::device_vector<value_type> values)
      : m_skeleton(std::move(skeleton)), m_values(std::move(values)) {
    assert(m_values.size() == m_skeleton.m_vals);
  }

  static masked_matrix identity(index_type n, value_type default_value) {
    return masked_matrix(matrix<bool, index_type>::identity(n), default_value);
  }

  void set_values(const matrix<bool, index_type>& pos, value_type value) {
    add_values(m_skeleton.m_rows, m_skeleton.m_col_index, m_skeleton.m_row_index, m_values,
               pos.m_col_index, pos.m_row_index, value);
  }

  matrix<bool, index_type> m_skeleton;
  thrust::device_vector<value_type> m_values;
};

}  // namespace nsparse