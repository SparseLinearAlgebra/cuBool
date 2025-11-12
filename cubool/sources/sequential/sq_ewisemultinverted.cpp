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

#include <sequential/sq_ewisemult.hpp>
#include <utils/algo_utils.hpp>

namespace cubool {

    void sq_ewisemultinverted(const CsrData& a, const CsrData& b, CsrData& out) {
        out.rowOffsets.resize(a.nrows + 1, 0);

        size_t nvals = 0;

        // Count nnz of the result matrix to allocate memory
        for (index i = 0; i < a.nrows; i++) {
            index ak = a.rowOffsets[i];
            index bk = b.rowOffsets[i];
            index asize = a.rowOffsets[i + 1] - ak;
            index bsize = b.rowOffsets[i + 1] - bk;

            const index* ar = &a.colIndices[ak];
            const index* br = &b.colIndices[bk];
            const index* arend = ar + asize;
            const index* brend = br + bsize;

            index nvalsInRow = 0;

            while (ar != arend && br != brend) {
                if (*ar == *br) {
                    ar++;
                    br++;
                }
                else if (*ar < *br) {
                    nvalsInRow++;
                    ar++;
                }
                else {
                    br++;
                }
            }
            nvalsInRow += arend - ar;

            nvals += nvalsInRow;
            out.rowOffsets[i] = nvalsInRow;
        }

        // Eval row offsets
        exclusive_scan(out.rowOffsets.begin(), out.rowOffsets.end(), 0);

        // Allocate memory for values
        out.nvals = nvals;
        out.colIndices.resize(nvals);

        // Fill sorted column indices
        size_t k = 0;
        for (index i = 0; i < a.nrows; i++) {
            const index* ar = &a.colIndices[a.rowOffsets[i]];
            const index* br = &b.colIndices[b.rowOffsets[i]];
            const index* arend = &a.colIndices[a.rowOffsets[i + 1]];
            const index* brend = &b.colIndices[b.rowOffsets[i + 1]];

            while (ar != arend && br != brend) {
                if (*ar == *br) {
                    ar++;
                    br++;
                }
                else if (*ar < *br) {
                    out.colIndices[k] = *ar;
                    k++;
                    ar++;
                }
                else {
                    br++;
                }
            }

            while (ar != arend) {
                out.colIndices[k] = *ar;
                k++;
                ar++;
            }
        }
    }
}
