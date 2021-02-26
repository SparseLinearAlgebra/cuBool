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

#include <sequential/sq_spgemm.hpp>
#include <sequential/sq_exclusive_scan.hpp>
#include <algorithm>
#include <limits>

namespace cubool {

    void sq_spgemm(const CsrData& a, const CsrData& b, CsrData& out) {
        index max = std::numeric_limits<index>::max();

        // Evaluate total nnz and nnz per row
        size_t nvals = 0;
        out.rowOffsets.resize(a.nrows + 1);
        std::vector<index> mask(b.ncols, max);

        for (index i = 0; i < a.nrows; i++) {
            size_t nvalsInRow = 0;

            for (index ak = a.rowOffsets[i]; ak < a.rowOffsets[i + 1]; ak++) {
                index k = a.colIndices[ak];

                for (index bk = b.rowOffsets[k]; bk < b.rowOffsets[k + 1]; bk++) {
                    index j = b.colIndices[bk];

                    // Do not compute col nnz twice
                    if (mask[j] != i) {
                        mask[j] = i;
                        nvalsInRow += 1;
                    }
                }
            }

            nvals += nvalsInRow;
            out.rowOffsets[i] = nvalsInRow;
        }

        // Row offsets
        sq_exclusive_scan(out.rowOffsets.begin(), out.rowOffsets.end(), 0);

        out.colIndices.resize(nvals);

        // Fill column indices per row and sort
        mask.clear();
        mask.resize(b.ncols, max);

        for (index i = 0; i < a.nrows; i++) {
            size_t id = 0;
            size_t first = out.rowOffsets[i];
            size_t last = out.rowOffsets[i + 1];

            for (index ak = a.rowOffsets[i]; ak < a.rowOffsets[i + 1]; ak++) {
                index k = a.colIndices[ak];

                for (index bk = b.rowOffsets[k]; bk < b.rowOffsets[k + 1]; bk++) {
                    index j = b.colIndices[bk];

                    // Do not compute col nnz twice
                    if (mask[j] != i) {
                        mask[j] = i;
                        out.colIndices[first + id] = j;
                        id += 1;
                    }
                }
            }

            // Sort indices within row
            std::sort(out.colIndices.begin() + first, out.colIndices.begin() + last);
        }
    }

}