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

namespace cubool {

    void sq_ewisemult(const CsrData& a, const CsrData& b, CsrData& out) {

    }

    void sq_ewisemult(const VecData& a, const VecData& b, VecData& out) {
        out.nrows = a.nrows;

        const index* aP = a.indices.data();
        const index* bP = b.indices.data();

        const index* aEnd = a.indices.data() + a.indices.size();
        const index* bEnd = b.indices.data() + b.indices.size();

        while (aP != aEnd && bP != bEnd) {
            if (*aP == *bP) {
                out.indices.push_back(*aP);
                aP++;
                bP++;
            }
            else if (*aP < *bP) {
                aP++;
            }
            else {
                bP++;
            }
        }

        out.nvals = out.indices.size();
    }

}
