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

void testMatrixExtractSubMatrix(cuBool_Index m, cuBool_Index n, cuBool_Index N, float density, cuBool_Hints flags) {
    cuBool_Matrix r, a;

    auto ta = testing::Matrix::generateSparse(m, n, density);

    ASSERT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
            auto k = m / N;
            auto l = n / N;

            ASSERT_EQ(cuBool_Matrix_New(&r, k, l), CUBOOL_STATUS_SUCCESS);
            ASSERT_EQ(cuBool_Matrix_ExtractSubMatrix(r, a, i * k, j * l, k, l, flags), CUBOOL_STATUS_SUCCESS);

            auto tr = ta.subMatrix(i * k, j * l, k, l);

            ASSERT_TRUE(tr.areEqual(r));
            ASSERT_EQ(cuBool_Matrix_Free(r), CUBOOL_STATUS_SUCCESS);
        }
    }

    ASSERT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Index n, float step, cuBool_Hints setup) {
    // Setup library
    EXPECT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    auto N = 10;

    for (size_t i = 0; i < N; i++) {
        testMatrixExtractSubMatrix(m, n, N, 0.01f + step * ((float) i), CUBOOL_HINT_NO);
    }

    // Finalize library
    EXPECT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Matrix, SubMatrixExtractSmall) {
    cuBool_Index m = 100, n = 200;
    float        step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, SubMatrixExtractMedium) {
    cuBool_Index m = 400, n = 700;
    float        step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, SubMatrixExtractLarge) {
    cuBool_Index m = 2000, n = 4000;
    float        step = 0.01f;
    testRun(m, n, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, SubMatrixExtractSmallFallback) {
    cuBool_Index m = 100, n = 200;
    float        step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, SubMatrixExtractMediumFallback) {
    cuBool_Index m = 400, n = 700;
    float        step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, SubMatrixExtractLargeFallback) {
    cuBool_Index m = 2000, n = 4000;
    float        step = 0.01f;
    testRun(m, n, step, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, SubMatrixExtractSmallManaged) {
    cuBool_Index m = 100, n = 200;
    float        step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Matrix, SubMatrixExtractMediumManaged) {
    cuBool_Index m = 400, n = 700;
    float        step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Matrix, SubMatrixExtractLargeManaged) {
    cuBool_Index m = 2000, n = 4000;
    float        step = 0.01f;
    testRun(m, n, step, CUBOOL_HINT_GPU_MEM_MANAGED);
}

CUBOOL_GTEST_MAIN