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

#ifndef CUBOOL_MATRIX_EWISEMULT_HPP
#define CUBOOL_MATRIX_EWISEMULT_HPP

#include <testing/matrix.hpp>

namespace testing {

    struct MatrixEWiseMultFunctor {
        Matrix operator()(const Matrix& a, const Matrix& b) {
            assert(a.nrows == b.nrows);
            assert(a.ncols == b.ncols);

            std::unordered_set<uint64_t> values;

            for (size_t i = 0; i < a.nvals; i++) {
                uint64_t row = a.rowsIndex[i];
                uint64_t col = a.colsIndex[i];
                uint64_t index = row * a.ncols + col;

                values.insert(index);
            }

            Matrix out;
            out.nrows = a.nrows;
            out.ncols = a.ncols;

            for (size_t i = 0; i < b.nvals; i++) {
                uint64_t row = b.rowsIndex[i];
                uint64_t col = b.colsIndex[i];
                uint64_t index = row * b.ncols + col;

                if (values.find(index) != values.end()) {
                    out.rowsIndex.push_back(row);
                    out.colsIndex.push_back(col);
                }
            }

            out.nvals = out.rowsIndex.size();

            return out;
        }
    };

}

#endif //CUBOOL_MATRIX_EWISEMULT_HPP
