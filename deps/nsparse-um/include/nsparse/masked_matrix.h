#pragma once
#include <nsparse/matrix.h>
#include <nsparse/detail/add_values.h>
#include <nsparse/unified_allocator.h>

namespace nsparse {

template <typename ValueType, typename IndexType>
class masked_matrix {
 public:
  typedef IndexType index_type;
  typedef ValueType value_type;

  masked_matrix(matrix<bool, index_type, nsparse::managed<index_type>> skeleton, int default_value)
      : m_skeleton(std::move(skeleton)) {
    m_values.resize(m_skeleton.m_vals);
    cudaMemsetAsync(thrust::raw_pointer_cast(m_values.data()), default_value,
                    sizeof(value_type) * m_skeleton.m_vals);
  }

  masked_matrix(matrix<bool, index_type, nsparse::managed<index_type>> skeleton,
                thrust::device_vector<value_type, nsparse::managed<value_type>> values)
      : m_skeleton(std::move(skeleton)), m_values(std::move(values)) {
    assert(m_values.size() == m_skeleton.m_vals);
  }

  static masked_matrix identity(index_type n, value_type default_value) {
    return masked_matrix(matrix<bool, index_type, nsparse::managed<index_type>>::identity(n), default_value);
  }

  auto vals() const {
    return m_values.size();
  }

  matrix<bool, index_type, nsparse::managed<index_type>> m_skeleton;
  thrust::device_vector<value_type, nsparse::managed<value_type>> m_values;
};

}  // namespace nsparse