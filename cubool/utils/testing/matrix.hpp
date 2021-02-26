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

#ifndef CUBOOL_TESTING_MATRIX_HPP
#define CUBOOL_TESTING_MATRIX_HPP

#include <cubool/cubool.h>
#include <testing/pair.hpp>
#include <testing/matrix_generator.hpp>
#include <algorithm>
#include <chrono>
#include <cassert>

namespace testing {

    struct Matrix {
        mutable std::vector<cuBool_Index> rowOffsets;
        mutable bool hasRowOffsets = false;

        std::vector<cuBool_Index> rowsIndex;
        std::vector<cuBool_Index> colsIndex;
        size_t nvals = 0;
        size_t nrows = 0;
        size_t ncols = 0;

        void computeRowOffsets() const {
            if (!hasRowOffsets) {
                rowOffsets.resize(nrows + 1, 0);

                for (size_t k = 0; k < nvals; k++) {
                    rowOffsets[rowsIndex[k]]++;
                }

                cuBool_Index sum = 0;
                for (auto& rowOffset: rowOffsets) {
                    cuBool_Index next = sum + rowOffset;
                    rowOffset = sum;
                    sum = next;
                }

                assert(nvals == rowOffsets.back());

                hasRowOffsets = true;
            }
        }

        Matrix transpose() const {
            Matrix result;
            result.nrows = ncols;
            result.ncols = nrows;
            result.nvals = nvals;
            result.rowsIndex.reserve(result.nvals);
            result.rowsIndex.reserve(result.nvals);

            std::vector<Pair> vals;
            vals.reserve(result.nvals);

            for (auto i = 0; i < nvals; i++) {
                vals.push_back(Pair{colsIndex[i], rowsIndex[i]});
            }

            std::sort(vals.begin(), vals.end(), PairCmp{});

            for (auto& p: vals) {
                result.rowsIndex.push_back(p.i);
                result.colsIndex.push_back(p.j);
            }

            return std::move(result);
        }

        Matrix reduce() const {
            Matrix result;
            result.nrows = nrows;
            result.ncols = 1;

            std::vector<uint8_t> rows(nrows, 0);

            for (size_t i = 0; i < nvals; i++) {
                rows[rowsIndex[i]] |= 0x1u;
            }

            for (auto i: rows) {
                result.nvals += i;
            }

            result.rowsIndex.reserve(result.nvals);
            result.colsIndex.reserve(result.nvals);

            for (size_t i = 0; i < nrows; i++) {
                if (rows[i] != 0) {
                    result.rowsIndex.push_back(i);
                    result.colsIndex.push_back(0);
                }
            }

            return std::move(result);
        }

        Matrix subMatrix(cuBool_Index i, cuBool_Index j, cuBool_Index m, cuBool_Index n) const {
            Matrix result;
            result.nrows = m;
            result.ncols = n;

            for (size_t id = 0; id < nvals; id++) {
                auto r = rowsIndex[id];
                auto c = colsIndex[id];

                if (i <= r && r < i + m && j <= c && c < j + n) {
                    result.rowsIndex.push_back(r - i);
                    result.colsIndex.push_back(c - j);
                    result.nvals += 1;
                }
            }

            return std::move(result);
        }

        bool areEqual(cuBool_Matrix matrix) const {
            cuBool_Index extNvals;

            if (cuBool_Matrix_Nvals(matrix, &extNvals) != CUBOOL_STATUS_SUCCESS)
                return false;

            if (extNvals != nvals)
                return false;

            std::vector<cuBool_Index> extRows(extNvals);
            std::vector<cuBool_Index> extCols(extNvals);

            if (cuBool_Matrix_ExtractPairs(matrix, extRows.data(), extCols.data(), &extNvals) != CUBOOL_STATUS_SUCCESS)
                return false;

            std::vector<Pair> extracted(nvals);
            std::vector<Pair> reference(nvals);

            for (cuBool_Index idx = 0; idx < nvals; idx++) {
                extracted[idx] = Pair{extRows[idx], extCols[idx]};
                reference[idx] = Pair{rowsIndex[idx], colsIndex[idx]};
            }

            std::sort(extracted.begin(), extracted.end(), PairCmp());
            std::sort(reference.begin(), reference.end(), PairCmp());

            return extracted == reference;
        }

        template<typename Condition>
        static Matrix generatet(size_t nrows, size_t ncols, Condition&& condition) {
            Matrix matrix;
            matrix.nrows = nrows;
            matrix.ncols = ncols;
            generateTestData(nrows, ncols, matrix.rowsIndex, matrix.colsIndex, matrix.nvals, std::forward<Condition>(condition));
            return matrix;
        }

        static Matrix generateSparse(size_t nrows, size_t ncols, double density) {
            Matrix matrix;
            matrix.nrows = nrows;
            matrix.ncols = ncols;

            std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());

            size_t totalVals = nrows * ncols;
            size_t valsToGen = totalVals * density;

            auto dist = std::uniform_real_distribution<float>(0.0, 1.0);

            std::unordered_set<Pair,PairHash,PairEq> indices;

            for (size_t id = 0; id < valsToGen; id++) {
                auto pr = dist(engine);
                auto pc = dist(engine);

                auto r = (cuBool_Index) (pr * (float) nrows);
                auto c = (cuBool_Index) (pc * (float) ncols);

                r = std::min(r, (cuBool_Index)nrows - 1);
                c = std::min(c, (cuBool_Index)ncols - 1);

                indices.emplace(Pair{r, c});
            }

            std::vector<Pair> toSort(indices.size());
            std::copy(indices.begin(), indices.end(), toSort.begin());
            std::sort(toSort.begin(), toSort.end(), PairCmp());

            matrix.nvals = toSort.size();
            matrix.rowsIndex.reserve(matrix.nvals);
            matrix.colsIndex.reserve(matrix.nvals);

            for (auto& p: toSort) {
                matrix.rowsIndex.push_back(p.i);
                matrix.colsIndex.push_back(p.j);
            }

            return std::move(matrix);
        }

    };
    
}

#endif //CUBOOL_TESTING_MATRIX_HPP