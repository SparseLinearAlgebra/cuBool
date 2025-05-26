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

#ifndef CUBOOL_TESTING_MATRIXEWISEADD_HPP
#define CUBOOL_TESTING_MATRIXEWISEADD_HPP

#include <testing/matrix.hpp>

namespace testing {

    struct MatrixEWiseAddFunctor {
        Matrix operator()(const Matrix& a, const Matrix& b) {
            assert(a.nrows == b.nrows);
            assert(a.ncols == b.ncols);

            a.computeRowOffsets();
            b.computeRowOffsets();

            Matrix out;
            out.nrows = a.nrows;
            out.ncols = a.ncols;

            out.rowOffsets.resize(a.nrows + 1, 0);

            size_t nvals = 0;

            // Count nnz of the result matrix to allocate memory
            for (cuBool_Index i = 0; i < a.nrows; i++) {
                cuBool_Index ak    = a.rowOffsets[i];
                cuBool_Index bk    = b.rowOffsets[i];
                cuBool_Index asize = a.rowOffsets[i + 1] - ak;
                cuBool_Index bsize = b.rowOffsets[i + 1] - bk;

                const cuBool_Index* ar    = &a.colsIndex[ak];
                const cuBool_Index* br    = &b.colsIndex[bk];
                const cuBool_Index* arend = ar + asize;
                const cuBool_Index* brend = br + bsize;

                cuBool_Index nvalsInRow = 0;

                while (ar != arend && br != brend) {
                    if (*ar == *br) {
                        nvalsInRow++;
                        ar++;
                        br++;
                    } else if (*ar < *br) {
                        nvalsInRow++;
                        ar++;
                    } else {
                        nvalsInRow++;
                        br++;
                    }
                }

                nvalsInRow += (size_t) (arend - ar);
                nvalsInRow += (size_t) (brend - br);

                nvals += nvalsInRow;
                out.rowOffsets[i] = nvalsInRow;
            }

            // Eval row offsets
            cuBool_Index sum = 0;
            for (auto& rowOffset : out.rowOffsets) {
                cuBool_Index next = sum + rowOffset;
                rowOffset         = sum;
                sum               = next;
            }
            out.hasRowOffsets = true;

            // Allocate memory for values
            out.nvals = nvals;
            out.rowsIndex.resize(nvals);
            out.colsIndex.resize(nvals);

            // Fill sorted column indices
            size_t k = 0;
            for (cuBool_Index i = 0; i < a.nrows; i++) {
                const cuBool_Index* ar    = &a.colsIndex[a.rowOffsets[i]];
                const cuBool_Index* br    = &b.colsIndex[b.rowOffsets[i]];
                const cuBool_Index* arend = &a.colsIndex[a.rowOffsets[i + 1]];
                const cuBool_Index* brend = &b.colsIndex[b.rowOffsets[i + 1]];

                while (ar != arend && br != brend) {
                    if (*ar == *br) {
                        out.rowsIndex[k] = i;
                        out.colsIndex[k] = *ar;
                        k++;
                        ar++;
                        br++;
                    } else if (*ar < *br) {
                        out.rowsIndex[k] = i;
                        out.colsIndex[k] = *ar;
                        k++;
                        ar++;
                    } else {
                        out.rowsIndex[k] = i;
                        out.colsIndex[k] = *br;
                        k++;
                        br++;
                    }
                }

                while (ar != arend) {
                    out.rowsIndex[k] = i;
                    out.colsIndex[k] = *ar;
                    k++;
                    ar++;
                }

                while (br != brend) {
                    out.rowsIndex[k] = i;
                    out.colsIndex[k] = *br;
                    k++;
                    br++;
                }
            }

            return std::move(out);
        }
    };

}// namespace testing

#endif//CUBOOL_TESTING_MATRIXEWISEADD_HPP
