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

namespace cubool {

    void scan(std::vector<index> &v) {
        index sum = 0;
        for (auto& val: v) {
            index next = sum + val;
            val = sum;
            sum = next;
        }
    }

    void sq_kronecker(const CooData& a, const CooData& b, CooData& out) {
        size_t nvals = a.mNvals * b.mNvals;

        out.mNvals = nvals;
        out.mRowIndices.resize(nvals);
        out.mColIndices.resize(nvals);

        std::vector<index> aoffsets(a.mNrows + 1, 0);
        std::vector<index> boffsets(b.mNrows + 1, 0);

        for (index k = 0; k < a.mNvals; k++) {
            aoffsets[a.mRowIndices[k]]++;
        }

        for (index k = 0; k < b.mNvals; k++) {
            boffsets[b.mRowIndices[k]]++;
        }

        scan(aoffsets);
        scan(boffsets);

        size_t id = 0;

        for (index ai = 0; ai < a.mNrows; ai++) {
            for (index bi = 0; bi < b.mNrows; bi++) {
                index rowId = ai * b.mNrows + bi;

                for (index k = aoffsets[ai]; k < aoffsets[ai + 1]; k++) {
                    index colIdBase = a.mColIndices[k] * b.mNcols;

                    for (index l = boffsets[bi]; l < boffsets[bi + 1]; l++) {
                        index colId = colIdBase + b.mColIndices[l];

                        out.mRowIndices[id] = rowId;
                        out.mColIndices[id] = colId;
                        id += 1;
                    }
                }
            }
        }
    }

}