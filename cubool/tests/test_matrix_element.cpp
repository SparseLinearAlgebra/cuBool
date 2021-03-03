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

#include <testing/testing.hpp>

// Fills sparse matrix with random data and tests whether the transfer works correctly
void testMatrixSetElement(cuBool_Index m, cuBool_Index n, float density) {
    cuBool_Matrix matrix = nullptr;

    testing::Matrix tmatrix = std::move(testing::Matrix::generateSparse(m, n, density));

    auto I = tmatrix.rowsIndex;
    auto J = tmatrix.colsIndex;
    size_t nvals = tmatrix.nvals;
    size_t dups = nvals * 0.10;

    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    auto dist = std::uniform_real_distribution<float>(0.0, 1.0);

    if (nvals > 0) {
        for (size_t k = 0; k < dups; k++) {
            size_t id = std::min<size_t>(dist(engine) * nvals, nvals - 1);
            auto i = I[id], j = J[id];
            I.push_back(i);
            J.push_back(j);
        }
    }

    ASSERT_EQ(cuBool_Matrix_New(&matrix, m, n), CUBOOL_STATUS_SUCCESS);

    nvals = I.size();
    for (size_t i = 0; i < nvals; i++) {
        ASSERT_EQ(cuBool_Matrix_SetElement(matrix, I[i], J[i]), CUBOOL_STATUS_SUCCESS);
    }

    // Compare test matrix and library one
    ASSERT_TRUE(tmatrix.areEqual(matrix));

    // Remember to release resources
    ASSERT_EQ(cuBool_Matrix_Free(matrix), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Index n, cuBool_Hints setup) {
    ASSERT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixSetElement(m, n, 0.001f + (0.05f) * ((float) i));
    }

    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Matrix, SetElementSmall) {
    cuBool_Index m = 60, n = 100;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, SetElementMedium) {
    cuBool_Index m = 500, n = 1000;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, SetElementLarge) {
    cuBool_Index m = 1000, n = 2000;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, SetElementSmallFallback) {
    cuBool_Index m = 60, n = 100;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, SetElementMediumFallback) {
    cuBool_Index m = 500, n = 1000;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);

}

TEST(cuBool_Matrix, SetElementLargeFallback) {
    cuBool_Index m = 1000, n = 2000;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

CUBOOL_GTEST_MAIN