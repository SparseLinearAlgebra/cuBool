/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2021 JetBrains-Research                                          */
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

#ifndef CUBOOL_TESTING_MATRIX_HPP
#define CUBOOL_TESTING_MATRIX_HPP

#include <cubool/cubool.h>
#include <testing/pair.hpp>
#include <testing/matrix_generator.hpp>
#include <algorithm>
#include <chrono>
#include <cassert>

namespace testing {

    // Default sizes
    const cuBoolIndex SMALL_M = 50, SMALL_N = 60;
    const cuBoolIndex MEDIUM_M = 500, MEDIUM_N = 600;
    const cuBoolIndex LARGE_M = 1500, LARGE_N = 2600;

    struct Matrix {
        std::vector<cuBoolIndex> mRowsIndex;
        std::vector<cuBoolIndex> mColsIndex;
        size_t mNvals = 0;
        size_t mNrows = 0;
        size_t mNcols = 0;

        Matrix transpose() const {
            Matrix result;
            result.mNrows = mNcols;
            result.mNcols = mNrows;
            result.mNvals = mNvals;
            result.mRowsIndex.reserve(result.mNvals);
            result.mRowsIndex.reserve(result.mNvals);

            std::vector<Pair> vals;
            vals.reserve(result.mNvals);

            for (auto i = 0; i < mNvals; i++) {
                vals.push_back(Pair{mColsIndex[i], mRowsIndex[i]});
            }

            std::sort(vals.begin(), vals.end(), PairCmp{});

            for (auto& p: vals) {
                result.mRowsIndex.push_back(p.i);
                result.mColsIndex.push_back(p.j);
            }

            return std::move(result);
        }

        bool areEqual(cuBoolMatrix matrix) const {
            cuBoolIndex extNvals;

            if (cuBool_Matrix_Nvals(matrix, &extNvals) != CUBOOL_STATUS_SUCCESS)
                return false;

            std::vector<cuBoolIndex> extRows(extNvals);
            std::vector<cuBoolIndex> extCols(extNvals);

            if (cuBool_Matrix_ExtractPairs(matrix, extRows.data(), extCols.data(), &extNvals) != CUBOOL_STATUS_SUCCESS)
                return false;

            if (extNvals != mNvals)
                return false;

            std::vector<Pair> extracted(mNvals);
            std::vector<Pair> reference(mNvals);

            for (cuBoolIndex idx = 0; idx < mNvals; idx++) {
                extracted[idx] = Pair{extRows[idx], extCols[idx]};
                reference[idx] = Pair{mRowsIndex[idx], mColsIndex[idx]};
            }

            std::sort(extracted.begin(), extracted.end(), PairCmp());
            std::sort(reference.begin(), reference.end(), PairCmp());

            return extracted == reference;
        }

        template<typename Condition>
        static Matrix generate(size_t nrows, size_t ncols, Condition&& condition) {
            Matrix matrix;
            matrix.mNrows = nrows;
            matrix.mNcols = ncols;
            generateTestData(nrows, ncols, matrix.mRowsIndex, matrix.mColsIndex, matrix.mNvals, std::forward<Condition>(condition));
            return matrix;
        }

        static Matrix generateSparse(size_t nrows, size_t ncols, double density) {
            Matrix matrix;
            matrix.mNrows = nrows;
            matrix.mNcols = ncols;

            std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());

            size_t totalVals = nrows * ncols;
            size_t valsToGen = totalVals * density;

            auto dist = std::uniform_real_distribution<float>(0.0, 1.0);

            std::unordered_set<Pair,PairHash,PairEq> indices;

            for (size_t id = 0; id < valsToGen; id++) {
                auto pr = dist(engine);
                auto pc = dist(engine);

                auto r = (cuBoolIndex) (pr * (float) nrows);
                auto c = (cuBoolIndex) (pc * (float) ncols);

                r = std::min(r, (cuBoolIndex)nrows - 1);
                c = std::min(c, (cuBoolIndex)ncols - 1);

                indices.emplace(Pair{r, c});
            }

            matrix.mNvals = indices.size();
            matrix.mRowsIndex.reserve(matrix.mNvals);
            matrix.mColsIndex.reserve(matrix.mNvals);

            for (auto& p: indices) {
                matrix.mRowsIndex.push_back(p.i);
                matrix.mColsIndex.push_back(p.j);
            }

            return std::move(matrix);
        }

    };
    
}

#endif //CUBOOL_TESTING_MATRIX_HPP