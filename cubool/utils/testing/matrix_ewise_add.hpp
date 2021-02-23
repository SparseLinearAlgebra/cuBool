/**********************************************************************************/
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2021 JetBrains-Research                                          */
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
        Matrix operator()(const Matrix& ma, const Matrix& mb) {
            auto m = ma.mNrows;
            auto n = ma.mNcols;

            assert(ma.mNrows == mb.mNrows);
            assert(ma.mNcols == mb.mNcols);

            std::vector<std::vector<uint8_t>> r(m, std::vector<uint8_t>(n, 0));

            for (size_t i = 0; i < ma.mNvals; i++) {
                r[ma.mRowsIndex[i]][ma.mColsIndex[i]] = 1;
            }

            for (size_t i = 0; i < mb.mNvals; i++) {
                r[mb.mRowsIndex[i]][mb.mColsIndex[i]] = 1;
            }

            Matrix result;
            result.mNrows = m;
            result.mNcols = n;

            for (size_t i = 0; i < m; i++) {
                for (size_t j = 0; j < n; j++) {
                    if (r[i][j] != 0) {
                        result.mRowsIndex.push_back(i);
                        result.mColsIndex.push_back(j);
                        result.mNvals += 1;
                    }
                }
            }

            return std::move(result);
        }
    };

}

#endif //CUBOOL_TESTING_MATRIXEWISEADD_HPP
