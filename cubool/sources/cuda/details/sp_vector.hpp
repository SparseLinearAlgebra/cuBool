/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020, 2021 JetBrains-Research                                    */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files (the "Software"), to deal  */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions:                       */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#ifndef CUBOOL_SP_VECTOR_HPP
#define CUBOOL_SP_VECTOR_HPP

#include <thrust/device_vector.h>

namespace cubool {
    namespace details {

        template<typename IndexType, typename AllocType>
        struct SpVector {
        public:
            typedef bool value_type;
            typedef IndexType index_type;
            typedef AllocType alloc_type;

            SpVector()
                : m_rows_index{}, m_rows{0}, m_vals{0} {}

            SpVector(index_type rows)
                : m_rows_index{}, m_rows (rows), m_vals{0} {}

            SpVector(thrust::device_vector<index_type, alloc_type> rows_index, index_type rows, index_type vals)
                : m_rows_index{ std::move(rows_index) }, m_rows{ rows }, m_vals{ vals }  {
                assert(m_rows > 0);
                assert(m_rows_index.size() == m_vals);
            }

        public:
            thrust::device_vector<index_type, alloc_type> m_rows_index;
            index_type m_rows;
            index_type m_vals;
        };
    }
}

#endif //CUBOOL_SP_VECTOR_HPP
