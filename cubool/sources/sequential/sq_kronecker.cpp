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

#include <sequential/sq_kronecker.hpp>
#include <utils/exclusive_scan.hpp>

namespace cubool {

    void sq_kronecker(const CsrData& a, const CsrData& b, CsrData& out) {
        size_t nvals = a.nvals * b.nvals;

        out.nvals = nvals;
        out.rowOffsets.clear();
        out.rowOffsets.resize(a.nrows * b.nrows + 1, 0);
        out.colIndices.resize(nvals);

        size_t id = 0;

        for (index ai = 0; ai < a.nrows; ai++) {
            for (index bi = 0; bi < b.nrows; bi++) {
                index rowId = ai * b.nrows + bi;

                for (index k = a.rowOffsets[ai]; k < a.rowOffsets[ai + 1]; k++) {
                    index colIdBase = a.colIndices[k] * b.ncols;

                    for (index l = b.rowOffsets[bi]; l < b.rowOffsets[bi + 1]; l++) {
                        index colId = colIdBase + b.colIndices[l];

                        out.rowOffsets[rowId]++;
                        out.colIndices[id] = colId;
                        id += 1;
                    }
                }
            }
        }

        exclusive_scan(out.rowOffsets.begin(), out.rowOffsets.end(), 0);
    }

}