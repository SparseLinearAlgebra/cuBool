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

namespace cubool {

    void sq_submatrix(const CooData& a, CooData& sub, index i, index j, index nrows, index ncols) {
        index first = a.mNvals;
        index last = 0;
        size_t nvals = 0;

        for (index k = 0; k < a.mNvals; k++) {
            auto ai = a.mRowIndices[k];
            auto aj = a.mColIndices[k];

            if (i <= ai && ai < i + nrows && j <= aj && aj < j + ncols) {
                first = std::min(first, k);
                last = std::max(last, k);
                nvals += 1;
            }
        }

        sub.mNvals = nvals;
        sub.mRowIndices.resize(nvals);
        sub.mColIndices.resize(nvals);

        size_t idx = 0;
        for (index k = first; k <= last; k++) {
            auto ai = a.mRowIndices[k];
            auto aj = a.mColIndices[k];

            if (i <= ai && ai < i + nrows && j <= aj && aj < j + ncols) {
                sub.mRowIndices[idx] = ai - i;
                sub.mColIndices[idx] = aj - j;
                idx += 1;
            }
        }
    }

}