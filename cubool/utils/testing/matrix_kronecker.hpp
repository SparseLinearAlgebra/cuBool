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
            auto m = ma.mNrows;
            auto n = ma.mNcols;
            auto k = mb.mNrows;
            auto t = mb.mNcols;

            Matrix result;
            result.mNrows = m * k;
            result.mNcols = n * t;
            result.mNvals = ma.mNvals * mb.mNvals;
            result.mRowsIndex.reserve(result.mNvals);
            result.mColsIndex.reserve(result.mNvals);

            std::vector<Pair> vals;
            vals.reserve(result.mNvals);

            for (cuBool_Index i = 0; i < ma.mNvals; i++) {
                auto blockI = ma.mRowsIndex[i];
                auto blockJ = ma.mColsIndex[i];

                for (cuBool_Index j = 0; j < mb.mNvals; j++) {
                    auto valueI = mb.mRowsIndex[j];
                    auto valueJ = mb.mColsIndex[j];

                    cuBool_Index idI = k * blockI + valueI;
                    cuBool_Index idJ = t * blockJ + valueJ;

                    vals.push_back(Pair{idI, idJ});
                }
            }

            std::sort(vals.begin(), vals.end(), PairCmp{});

            for (auto& p: vals) {
                result.mRowsIndex.push_back(p.i);
                result.mColsIndex.push_back(p.j);
            }

            return std::move(result);
        }
    };

}

#endif //CUBOOL_TESTING_MATRIXKRONECKER_HPP
