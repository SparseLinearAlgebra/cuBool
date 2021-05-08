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

#ifndef CUBOOL_VECTOR_HPP
#define CUBOOL_VECTOR_HPP

#include <cubool/cubool.h>
#include <testing/pair.hpp>
#include <testing/matrix_generator.hpp>
#include <algorithm>
#include <chrono>
#include <cassert>

namespace testing {

    struct Vector {
        std::vector<cuBool_Index> index;
        size_t nvals = 0;
        size_t nrows = 0;

        Vector subVector(cuBool_Index i, cuBool_Index m) const {
            assert(i + m <= nrows);

            Vector out;

            for (auto v: index) {
                if (i <= v && v < (i + m))
                    out.index.push_back(v - i);
            }

            out.nrows = m;
            out.nvals = out.index.size();

            return out;
        }

        bool areEqual(cuBool_Vector vector) const {
            cuBool_Index extNvals;

            if (cuBool_Vector_Nvals(vector, &extNvals) != CUBOOL_STATUS_SUCCESS)
                return false;

            if (extNvals != nvals)
                return false;

            std::vector<cuBool_Index> ext(extNvals);

            if (cuBool_Vector_ExtractValues(vector, ext.data(), &extNvals) != CUBOOL_STATUS_SUCCESS)
                return false;

            return ext == index;
        }

        static Vector generateSparse(size_t nrows, double density) {
            Vector vec;
            vec.nrows = nrows;

            std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());

            size_t totalVals = nrows;
            size_t valsToGen = totalVals * density;

            auto dist = std::uniform_real_distribution<float>(0.0, 1.0);

            std::unordered_set<cuBool_Index> indices;

            for (size_t id = 0; id < valsToGen; id++) {
                auto pr = dist(engine);

                auto r = (cuBool_Index) (pr * (float) nrows);

                r = std::min(r, (cuBool_Index)nrows - 1);

                indices.emplace(r);
            }

            std::vector<cuBool_Index> toSort(indices.size());
            std::copy(indices.begin(), indices.end(), toSort.begin());
            std::sort(toSort.begin(), toSort.end(), [](cuBool_Index a, cuBool_Index b){ return a < b; });

            vec.nvals = toSort.size();
            vec.index = std::move(toSort);

            return vec;
        }

        static Vector empty(size_t nrows) {
            Vector out;
            out.nrows = nrows;
            return out;
        }
    };

}

#endif //CUBOOL_VECTOR_HPP
