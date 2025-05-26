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

namespace cubool {

    void sq_spgemv(const CsrData& a, const VecData& b, VecData& out) {
        std::vector<index> result;

        for (index i = 0; i < a.nrows; i++) {
            const index* ar = a.colIndices.data() + a.rowOffsets[i];
            const index* vr = b.indices.data();

            const index* arend = ar + (a.rowOffsets[i + 1] - a.rowOffsets[i]);
            const index* vrend = vr + b.nvals;

            bool nonZero = false;

            while (ar != arend && vr != vrend) {
                if (*ar == *vr) {
                    nonZero = true;
                    break;
                } else if (*ar < *vr) {
                    ar++;
                } else {
                    vr++;
                }
            }

            if (nonZero) {
                result.push_back(i);
            }
        }

        out.nvals   = result.size();
        out.indices = std::move(result);
    }

    void sq_spgemv_transposed(const CsrData& a, const VecData& b, VecData& out) {
        std::vector<bool> mask(a.ncols, false);

        for (index i : b.indices) {
            for (index k = a.rowOffsets[i]; k < a.rowOffsets[i + 1]; k++) {
                mask[a.colIndices[k]] = true;
            }
        }

        std::vector<index> result;

        for (index i = 0; i < mask.size(); i++) {
            if (mask[i])
                result.push_back(i);
        }

        out.nvals   = result.size();
        out.indices = std::move(result);
    }

}// namespace cubool