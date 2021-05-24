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

void testMatrixExtractRow(cuBool_Index m, cuBool_Index n, float density, cuBool_Hints flags) {
    cuBool_Matrix a;

    auto ta = testing::Matrix::generateSparse(m, n, density);

    ASSERT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    for (cuBool_Index i = 0; i < m; i++) {
        cuBool_Vector r;

        ASSERT_EQ(cuBool_Vector_New(&r, n), CUBOOL_STATUS_SUCCESS);
        ASSERT_EQ(cuBool_Matrix_ExtractRow(r, a, i, CUBOOL_HINT_NO), CUBOOL_STATUS_SUCCESS);

        auto tr = ta.extractRow(i);

        EXPECT_TRUE(tr.areEqual(r));
        ASSERT_EQ(cuBool_Vector_Free(r), CUBOOL_STATUS_SUCCESS);
    }

    ASSERT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
}

void testMatrixExtractCol(cuBool_Index m, cuBool_Index n, float density, cuBool_Hints flags) {
    cuBool_Matrix a;

    auto ta = testing::Matrix::generateSparse(m, n, density);
    auto taT = ta.transpose();

    ASSERT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    for (cuBool_Index j = 0; j < m; j++) {
        cuBool_Vector r;

        ASSERT_EQ(cuBool_Vector_New(&r, m), CUBOOL_STATUS_SUCCESS);
        ASSERT_EQ(cuBool_Matrix_ExtractCol(r, a, j, CUBOOL_HINT_NO), CUBOOL_STATUS_SUCCESS);

        auto tr = taT.extractRow(j);

        EXPECT_TRUE(tr.areEqual(r));
        ASSERT_EQ(cuBool_Vector_Free(r), CUBOOL_STATUS_SUCCESS);
    }

    ASSERT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Index n, cuBool_Hints setup) {
    // Setup library
    EXPECT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    auto N = 4;

    for (size_t i = 0; i < N; i++) {
        testMatrixExtractRow(m, n, 0.01f + (0.5f) * ((float) i + 1.0f) / ((float) n), CUBOOL_HINT_NO);
    }

    for (size_t i = 0; i < N; i++) {
        testMatrixExtractCol(m, n, 0.01f + (0.5f) * ((float) i + 1.0f) / ((float) n), CUBOOL_HINT_NO);
    }

    // Finalize library
    EXPECT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Matrix, MatrixExtractVectorSmall) {
    cuBool_Index m = 1000, n = 2000;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, MatrixExtractVectorMedium) {
    cuBool_Index m = 4000, n = 7000;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, MatrixExtractVectorLarge) {
    cuBool_Index m = 8000, n = 10000;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, MatrixExtractVectorSmallFallback) {
    cuBool_Index m = 1000, n = 2000;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, MatrixExtractVectorMediumFallback) {
    cuBool_Index m = 4000, n = 7000;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, MatrixExtractVectorLargeFallback) {
    cuBool_Index m = 8000, n = 10000;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, MatrixExtractVectorSmallManaged) {
    cuBool_Index m = 1000, n = 2000;
    testRun(m, n, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Matrix, MatrixExtractVectorMediumManaged) {
    cuBool_Index m = 4000, n = 7000;
    testRun(m, n, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Matrix, MatrixExtractVectorLargeManaged) {
    cuBool_Index m = 8000, n = 10000;
    testRun(m, n, CUBOOL_HINT_GPU_MEM_MANAGED);
}

CUBOOL_GTEST_MAIN