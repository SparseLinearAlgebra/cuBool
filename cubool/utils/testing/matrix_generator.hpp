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

#ifndef CUBOOL_TESTING_MATRIXGENERATOR_HPP
#define CUBOOL_TESTING_MATRIXGENERATOR_HPP

#include <testing/pair.hpp>
#include <vector>
#include <random>
#include <ctime>

namespace testing {

    struct Condition1 {
        bool operator()(cuBool_Index i, cuBool_Index j) {
            return (((i - 1) & i) == 0 && ((j - 1) & j) == 0);
        }
    };

    struct Condition2 {
        bool operator()(cuBool_Index i, cuBool_Index j) {
            return !(i % 5) && !(j % 7);
        }
    };

    struct Condition3 {
    public:
        explicit Condition3(float density): mDensity(density) {
            mRandomEngine.seed(std::time(0));
        }
        bool operator()(cuBool_Index i, cuBool_Index j) {
            return std::uniform_real_distribution<float>(0.0f, 1.0f)(mRandomEngine) <= mDensity;
        }
    private:
        std::default_random_engine mRandomEngine;
        float mDensity = 1.0f;
    };

    template<typename Condition>
    static void generateTestData(size_t rows, size_t columns, std::vector<Pair> &values, Condition&& condition) {
        for (cuBool_Index i = 0; i < rows; i++) {
            for (cuBool_Index j = 0; j < columns; j++) {
                // is i and j power of two or 0
                if (condition(i, j)) {
                    values.push_back(Pair{i, j});
                }
            }
        }
    }

    template<typename Condition>
    static void generateTestData(size_t nrows, size_t ncols, std::vector<cuBool_Index> &rows, std::vector<cuBool_Index> &cols, size_t& nvals, Condition&& condition) {
        nvals = 0;
        for (cuBool_Index i = 0; i < nrows; i++) {
            for (cuBool_Index j = 0; j < ncols; j++) {
                // is i and j power of two or 0
                if (condition(i, j)) {
                    rows.push_back(i);
                    cols.push_back(j);
                    nvals += 1;
                }
            }
        }
    }
    
}

#endif //CUBOOL_TESTING_MATRIXGENERATOR_HPP
