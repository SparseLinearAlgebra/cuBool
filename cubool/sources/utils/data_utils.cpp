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

#include <utils/data_utils.hpp>
#include <utils/algo_utils.hpp>
#include <core/error.hpp>
#include <algorithm>
#include <cassert>

namespace cubool {

    void DataUtils::buildFromData(size_t nrows, size_t ncols,
                                  const index *rows, const index *cols, size_t nvals,
                                  std::vector<index> &rowOffsets, std::vector<index> &colIndices,
                                  bool isSorted, bool noDuplicates) {

        rowOffsets.resize(nrows + 1, 0);
        colIndices.resize(nvals);

        std::fill(rowOffsets.begin(), rowOffsets.end(), 0);

        if (nvals == 0)
            return;

        assert(rows);
        assert(cols);

        for (size_t k = 0; k < nvals; k++) {
            auto i = rows[k];
            auto j = cols[k];

            CHECK_RAISE_ERROR(i < nrows, InvalidArgument, "Index out of matrix bounds");
            CHECK_RAISE_ERROR(j < ncols, InvalidArgument, "Index out of matrix bounds");

            rowOffsets[i]++;
        }

        exclusive_scan(rowOffsets.begin(), rowOffsets.end(), 0);

        std::vector<size_t> writeOffset(nrows, 0);
        for (size_t k = 0; k < nvals; k++) {
            auto i = rows[k];
            auto j = cols[k];

            colIndices[rowOffsets[i] + writeOffset[i]] = j;
            writeOffset[i] += 1;
        }

        if (!isSorted) {
            for (size_t i = 0; i < nrows; i++) {
                auto begin = rowOffsets[i];
                auto end = rowOffsets[i + 1];

                // Sort col values within row
                std::sort(colIndices.begin() + begin, colIndices.begin() + end, [](const index& a, const index& b) {
                    return a < b;
                });
            }
        }

        if (!noDuplicates) {
            size_t unique = 0;
            for (size_t i = 0; i < nrows; i++) {
                index prev = std::numeric_limits<index>::max();

                for (size_t k = rowOffsets[i]; k < rowOffsets[i + 1]; k++) {
                    if (prev != colIndices[k]) {
                        unique += 1;
                    }

                    prev = colIndices[k];
                }
            }

            std::vector<index> rowOffsetsReduced;
            rowOffsetsReduced.resize(nrows + 1, 0);

            std::vector<index> colIndicesReduced;
            colIndicesReduced.reserve(unique);

            for (size_t i = 0; i < nrows; i++) {
                index prev = std::numeric_limits<index>::max();

                for (size_t k = rowOffsets[i]; k < rowOffsets[i + 1]; k++) {
                    if (prev != colIndices[k]) {
                        rowOffsetsReduced[i] += 1;
                        colIndicesReduced.push_back(colIndices[k]);
                    }

                    prev = colIndices[k];
                }
            }

            // Exclusive scan to eval rows offsets
            exclusive_scan(rowOffsetsReduced.begin(), rowOffsetsReduced.end(), 0);

            // Now result in respective place
            std::swap(rowOffsets, rowOffsetsReduced);
            std::swap(colIndices, colIndicesReduced);
        }
    }

    void DataUtils::extractData(size_t nrows, size_t ncols,
                                index *rows, index *cols, size_t nvals,
                                const std::vector<index> &rowOffsets, const std::vector<index> &colIndices) {
        assert(rows);
        assert(cols);

        size_t id = 0;
        for (index i = 0; i < nrows; i++) {
            for (index k = rowOffsets[i]; k < rowOffsets[i + 1]; k++) {
                rows[id] = i;
                cols[id] = colIndices[k];
                id += 1;
            }
        }
    }

    void DataUtils::buildVectorFromData(size_t nrows, const index *rows, size_t nvals, std::vector<index> &values,
                                        bool isSorted, bool noDuplicates) {
        values.resize(nvals);
        std::copy(rows, rows + nvals, values.begin());

        if (nvals == 0)
            return;

        assert(rows);

        std::copy(rows, rows + nvals, values.begin());

        auto checkBounds = [&]() {
            for (auto value: values) {
                CHECK_RAISE_ERROR(value < nrows, InvalidArgument, "Index out of vector bounds");
            }

            return true;
        };

        assert(checkBounds());

        if (!isSorted) {
            std::sort(values.begin(), values.end(), [](const index& a, const index& b) {
                return a < b;
            });
        }

        if (!noDuplicates) {
            size_t unique = 0;
            index prev = std::numeric_limits<index>::max();

            for (auto value : values) {
                if (prev != value) {
                    unique += 1;
                }

                prev = value;
            }

            std::vector<index> reduced;
            reduced.reserve(unique);

            prev = std::numeric_limits<index>::max();
            for (auto value : values) {
                if (prev != value) {
                    reduced.push_back(value);
                }

                prev = value;
            }

            std::swap(values, reduced);
        }
    }

}