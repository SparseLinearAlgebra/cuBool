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

#include <sequential/sq_submatrix.hpp>
#include <sequential/sq_exclusive_scan.hpp>

namespace cubool {

    void sq_submatrix(const CsrData& a, CsrData& sub, index i, index j, index nrows, index ncols) {
        index first = i;
        index last = i + nrows;
        size_t nvals = 0;

        for (index ai = first; ai < last; ai++) {
            for (index k = a.rowOffsets[ai]; k < a.rowOffsets[ai + 1]; k++) {
                index aj = a.colIndices[k];

                if (i <= ai && ai < i + nrows && j <= aj && aj < j + ncols) {
                    nvals += 1;
                }
            }
        }

        sub.nvals = nvals;
        sub.rowOffsets.clear();
        sub.rowOffsets.resize(nrows + 1, 0);
        sub.colIndices.resize(nvals);

        size_t idx = 0;
        for (index ai = first; ai < last; ai++) {
            for (index k = a.rowOffsets[ai]; k < a.rowOffsets[ai + 1]; k++) {
                index aj = a.colIndices[k];

                if (i <= ai && ai < i + nrows && j <= aj && aj < j + ncols) {
                    index ri = ai - i;
                    index rj = aj - j;

                    sub.rowOffsets[ri] += 1;
                    sub.colIndices[idx] = rj;

                    idx += 1;
                }
            }
        }

        sq_exclusive_scan(sub.rowOffsets.begin(), sub.rowOffsets.end(), 0);
    }

}