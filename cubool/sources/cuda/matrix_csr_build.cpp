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

#include <cuda/matrix_csr.hpp>
#include <utils/exclusive_scan.hpp>
#include <vector>
#include <algorithm>

namespace cubool {

    void MatrixCsr::build(const index *rows, const index *cols, size_t nvals, bool isSorted, bool noDuplicates) {
        if (nvals == 0) {
            mMatrixImpl.zero_dim();  // no content, empty matrix
            return;
        }

        std::vector<index> rowOffsets;
        rowOffsets.resize(getNrows() + 1, 0);

        std::vector<index> colIndices;
        colIndices.resize(nvals);

        // Compute nnz per row
        for (size_t idx = 0; idx < nvals; idx++) {
            index i = rows[idx];
            index j = cols[idx];

            CHECK_RAISE_ERROR(i < getNrows() && j < getNcols(), InvalidArgument, "Out of matrix bounds value");

            rowOffsets[i] += 1;
        }

        // Exclusive scan to eval rows offsets
        ::cubool::exclusive_scan(rowOffsets.begin(), rowOffsets.end(), 0);

        // Write offsets for cols
        std::vector<size_t> writeOffsets(getNrows(), 0);

        for (size_t idx = 0; idx < nvals; idx++) {
            index i = rows[idx];
            index j = cols[idx];

            colIndices[rowOffsets[i] + writeOffsets[i]] = j;
            writeOffsets[i] += 1;
        }

        if (!isSorted) {
            for (size_t i = 0; i < getNrows(); i++) {
                auto begin = rowOffsets[i];
                auto end = rowOffsets[i + 1];

                // Sort col values within row
                std::sort(colIndices.begin() + begin, colIndices.begin() + end, [](const index& a, const index& b) {
                    return a < b;
                });
            }
        }

        // Reduce duplicated values
        if (!noDuplicates) {
            size_t unique = 0;
            for (size_t i = 0; i < getNrows(); i++) {
                index prev = std::numeric_limits<index>::max();

                for (size_t k = rowOffsets[i]; k < rowOffsets[i + 1]; k++) {
                    if (prev != colIndices[k]) {
                        unique += 1;
                    }

                    prev = colIndices[k];
                }
            }

            std::vector<index> rowOffsetsReduced;
            rowOffsetsReduced.resize(getNrows() + 1, 0);

            std::vector<index> colIndicesReduced;
            colIndicesReduced.reserve(unique);

            for (size_t i = 0; i < getNrows(); i++) {
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
            ::cubool::exclusive_scan(rowOffsetsReduced.begin(), rowOffsetsReduced.end(), 0);

            // Now result in respective place
            std::swap(rowOffsets, rowOffsetsReduced);
            std::swap(colIndices, colIndicesReduced);
        }

        // Move actual data to the matrix implementation
        this->transferToDevice(rowOffsets, colIndices);
    }

}