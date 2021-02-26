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

#ifndef CUBOOL_TESTING_MATRIXKRONECKER_HPP
#define CUBOOL_TESTING_MATRIXKRONECKER_HPP

#include <testing/matrix.hpp>

namespace testing {

    struct MatrixKroneckerFunctor {
        Matrix operator()(const Matrix& ma, const Matrix& mb) {
            auto m = ma.nrows;
            auto n = ma.ncols;
            auto k = mb.nrows;
            auto t = mb.ncols;

            Matrix result;
            result.nrows = m * k;
            result.ncols = n * t;
            result.nvals = ma.nvals * mb.nvals;
            result.rowsIndex.reserve(result.nvals);
            result.colsIndex.reserve(result.nvals);

            std::vector<Pair> vals;
            vals.reserve(result.nvals);

            for (cuBool_Index i = 0; i < ma.nvals; i++) {
                auto blockI = ma.rowsIndex[i];
                auto blockJ = ma.colsIndex[i];

                for (cuBool_Index j = 0; j < mb.nvals; j++) {
                    auto valueI = mb.rowsIndex[j];
                    auto valueJ = mb.colsIndex[j];

                    cuBool_Index idI = k * blockI + valueI;
                    cuBool_Index idJ = t * blockJ + valueJ;

                    vals.push_back(Pair{idI, idJ});
                }
            }

            std::sort(vals.begin(), vals.end(), PairCmp{});

            for (auto& p: vals) {
                result.rowsIndex.push_back(p.i);
                result.colsIndex.push_back(p.j);
            }

            return std::move(result);
        }
    };

}

#endif //CUBOOL_TESTING_MATRIXKRONECKER_HPP
