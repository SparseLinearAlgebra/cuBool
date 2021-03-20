#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <nsparse/unified_allocator.h>
#include <cassert>

namespace nsparse {

// Edited by Egor Orachyov
//
// Add template argument <AllocType> for the allocator, since we want to
// allow runtime specific changes (not only compile time) in the type of
// allocated memory: device only or managed

    template<typename ValueType, typename IndexType, typename AllocType>
    class matrix;

    template<typename IndexType, typename AllocType>
    class matrix<bool, IndexType, AllocType> {
    public:
        typedef IndexType index_type;
        typedef bool value_type;
        typedef AllocType alloc_type;

        matrix(alloc_type alloc) : m_col_index(alloc), m_row_index(alloc), m_rows{0}, m_cols{0}, m_vals{0} {

        }

        matrix() : m_col_index{}, m_row_index{}, m_rows{0}, m_cols{0}, m_vals{0} {
        }

        matrix(index_type rows, index_type cols)
                : m_col_index{}, m_row_index{rows + 1, 0}, m_rows{rows}, m_cols{cols}, m_vals{0} {
        }

        static matrix identity(index_type n) {
            thrust::device_vector<index_type, alloc_type>
                    col_index(thrust::counting_iterator<index_type>(0), thrust::counting_iterator<index_type>(n));

            thrust::device_vector<index_type, alloc_type>
                    row_index(thrust::counting_iterator<index_type>(0), thrust::counting_iterator<index_type>(n + 1));

            return matrix(std::move(col_index), std::move(row_index), n, n, n);
        }

        matrix(thrust::device_vector<index_type, alloc_type> col_index,
               thrust::device_vector<index_type, alloc_type> row_index, index_type rows,
               index_type cols, index_type vals)
                : m_col_index{std::move(col_index)},
                  m_row_index{std::move(row_index)},
                  m_rows{rows},
                  m_cols{cols},
                  m_vals{vals} {
            assert(m_col_index.size() == vals);
            assert(m_row_index.size() == rows + 1);
        }

        void zero_dim() {
            m_col_index.clear();
            m_row_index.clear();
            m_rows = 0;
            m_cols = 0;
            m_vals = 0;
        }

        bool is_zero_dim() const {
            return m_rows == 0;
        }



    public:
        thrust::device_vector<index_type, alloc_type> m_col_index;
        thrust::device_vector<index_type, alloc_type> m_row_index;
        index_type m_rows;
        index_type m_cols;
        index_type m_vals;
    };

}  // namespace nsparse