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

#include <limits>
#include <sequential/sq_reduce.hpp>
#include <utils/algo_utils.hpp>

namespace cubool {

    void sq_reduce(const CsrData& a, CsrData& out) {
        out.rowOffsets.resize(a.nrows + 1);

        for (index i = 0; i < a.nrows; i++) {
            index nnz = a.rowOffsets[i + 1] - a.rowOffsets[i];

            if (nnz > 0) {
                out.rowOffsets[i] = 1;
            }
        }

        exclusive_scan(out.rowOffsets.begin(), out.rowOffsets.end(), 0);

        out.nvals = out.rowOffsets.back();
        out.colIndices.clear();
        out.colIndices.resize(out.nvals, 0);
    }

    void sq_reduce(const CsrData& a, VecData& out) {
        size_t size = 0;

        for (index i = 0; i < a.nrows; i++) {
            index nnz = a.rowOffsets[i + 1] - a.rowOffsets[i];

            if (nnz > 0) {
                size += 1;
            }
        }

        out.indices.reserve(size);
        out.nvals = size;

        for (index i = 0; i < a.nrows; i++) {
            index nnz = a.rowOffsets[i + 1] - a.rowOffsets[i];

            if (nnz > 0) {
                out.indices.push_back(i);
            }
        }
    }

    void sq_reduce_transposed(const CsrData& a, VecData& out) {
        const auto         max = std::numeric_limits<index>::max();
        std::vector<index> mask(a.ncols, max);

        for (auto j : a.colIndices) {
            mask[j] = j;
        }

        size_t size = 0;
        for (auto v : mask) {
            if (v != max) {
                size += 1;
            }
        }

        out.indices.reserve(size);
        out.nvals = size;

        for (auto v : mask) {
            if (v != max) {
                out.indices.push_back(v);
            }
        }
    }

}// namespace cubool