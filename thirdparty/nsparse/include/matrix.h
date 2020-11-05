#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cassert>

namespace nsparse {

template <typename ValueType, typename IndexType>
class matrix;

template <typename IndexType>
class matrix<bool, IndexType> {
 public:
  typedef IndexType index_type;
  typedef bool value_type;

  matrix() : m_col_index{}, m_row_index{}, m_rows{0}, m_cols{0}, m_vals{0} {
  }

  matrix(index_type rows, index_type cols)
      : m_col_index{}, m_row_index{rows + 1, 0}, m_rows{rows}, m_cols{cols}, m_vals{0} {
  }

  static matrix identity(index_type n) {
    thrust::device_vector<index_type> col_index(thrust::counting_iterator<index_type>(0),
                                                thrust::counting_iterator<index_type>(n));
    thrust::device_vector<index_type> row_index(thrust::counting_iterator<index_type>(0),
                                                thrust::counting_iterator<index_type>(n + 1));
    return matrix(std::move(col_index), std::move(row_index), n, n, n);
  }

  matrix(thrust::device_vector<index_type> col_index, thrust::device_vector<index_type> row_index,
         index_type rows, index_type cols, index_type vals)
      : m_col_index{std::move(col_index)},
        m_row_index{std::move(row_index)},
        m_rows{rows},
        m_cols{cols},
        m_vals{vals} {
    assert(m_col_index.size() == vals);
    assert(m_row_index.size() == rows + 1);
  }

  thrust::device_vector<index_type> m_col_index;
  thrust::device_vector<index_type> m_row_index;
  index_type m_rows;
  index_type m_cols;
  index_type m_vals;
};

}  // namespace nsparse