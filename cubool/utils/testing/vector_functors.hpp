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

#ifndef CUBOOL_VECTOR_FUNCTORS_HPP
#define CUBOOL_VECTOR_FUNCTORS_HPP

#include <testing/matrix.hpp>
#include <testing/vector.hpp>

namespace testing {

    struct VectorEWiseAddFunctor {
    public:
        Vector operator()(const Vector& a, const Vector& b) {
            std::unordered_set<cuBool_Index> values;

            for (auto v : a.index)
                values.emplace(v);

            for (auto v : b.index)
                values.emplace(v);

            std::vector<cuBool_Index> sorted(values.size());
            std::copy(values.begin(), values.end(), sorted.begin());

            std::sort(sorted.begin(), sorted.end(), [](cuBool_Index i, cuBool_Index j) { return i < j; });

            Vector out;
            out.nrows = a.nrows;
            out.nvals = sorted.size();
            out.index = std::move(sorted);

            return out;
        }
    };

    struct VectorEWiseMultFunctor {
    public:
        Vector operator()(const Vector& a, const Vector& b) {
            std::unordered_set<cuBool_Index> values;
            std::vector<cuBool_Index>        result;

            for (auto v : a.index)
                values.emplace(v);

            for (auto v : b.index) {
                if (values.find(v) != values.end())
                    result.push_back(v);
            }

            Vector out;
            out.nrows = a.nrows;
            out.nvals = result.size();
            out.index = std::move(result);

            return out;
        }
    };

    struct MatrixVectorMultiplyFunctor {
    public:
        Vector operator()(const Matrix& m, const Vector& v) {
            m.computeRowOffsets();

            std::unordered_set<cuBool_Index> buffer;

            for (auto i : v.index)
                buffer.emplace(i);

            std::vector<cuBool_Index> result;

            for (cuBool_Index i = 0; i < m.nrows; i++) {
                for (cuBool_Index k = m.rowOffsets[i]; k < m.rowOffsets[i + 1]; k++) {
                    cuBool_Index j = m.colsIndex[k];

                    if (buffer.find(j) != buffer.end()) {
                        result.push_back(i);
                        break;
                    }
                }
            }

            Vector r;
            r.nvals = result.size();
            r.index = std::move(result);

            return r;
        }
    };

    struct VectorMatrixMultiplyFunctor {
    public:
        Vector operator()(const Vector& v, const Matrix& m) {
            m.computeRowOffsets();

            std::unordered_set<cuBool_Index> result;

            for (cuBool_Index i : v.index) {
                for (cuBool_Index k = m.rowOffsets[i]; k < m.rowOffsets[i + 1]; k++) {
                    result.emplace(m.colsIndex[k]);
                }
            }

            Vector r;
            r.nvals = result.size();
            r.index.resize(result.size());

            std::copy(result.begin(), result.end(), r.index.begin());
            std::sort(r.index.begin(), r.index.end(), [](cuBool_Index a, cuBool_Index b) { return a < b; });

            return r;
        }
    };

}// namespace testing

#endif//CUBOOL_VECTOR_FUNCTORS_HPP
