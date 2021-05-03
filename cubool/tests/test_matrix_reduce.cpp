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

void testMatrixToVectorReduce(cuBool_Index m, cuBool_Index n, float density, cuBool_Hints flags) {
    cuBool_Matrix a;
    cuBool_Vector r;

    auto ta = testing::Matrix::generateSparse(m, n, density);

    ASSERT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&r, m), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_Reduce(r, a, flags), CUBOOL_STATUS_SUCCESS);

    auto tr = ta.reduceToVector();

    ASSERT_TRUE(tr.areEqual(r));

    ASSERT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(r), CUBOOL_STATUS_SUCCESS);
}

void testMatrixToVectorReduceTransposed(cuBool_Index m, cuBool_Index n, float density, cuBool_Hints flags) {
    cuBool_Matrix a;
    cuBool_Vector r;

    auto ta = testing::Matrix::generateSparse(m, n, density);

    ASSERT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&r, n), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_Reduce(r, a, CUBOOL_HINT_TRANSPOSE | flags), CUBOOL_STATUS_SUCCESS);

    auto tr = ta.transpose().reduceToVector();

    ASSERT_TRUE(tr.areEqual(r));

    ASSERT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(r), CUBOOL_STATUS_SUCCESS);
}

void testMatrixReduce2(cuBool_Index m, cuBool_Index n, float density, cuBool_Hints flags) {
    cuBool_Matrix r, a;

    auto ta = testing::Matrix::generateSparse(m, n, density);

    ASSERT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_New(&r, m, 1), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_Reduce2(r, a, flags), CUBOOL_STATUS_SUCCESS);

    auto tr = ta.reduce();

    ASSERT_TRUE(tr.areEqual(r));

    ASSERT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Free(r), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Index n, float step, cuBool_Hints setup) {
    // Setup library
    EXPECT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixToVectorReduce(m, n, 0.01f + step * ((float) i), CUBOOL_HINT_NO);
    }

    for (size_t i = 0; i < 10; i++) {
        testMatrixToVectorReduceTransposed(m, n, 0.01f + step * ((float) i), CUBOOL_HINT_NO);
    }

    for (size_t i = 0; i < 10; i++) {
        testMatrixReduce2(m, n, 0.01f + step * ((float) i), CUBOOL_HINT_NO);
    }

    // Finalize library
    EXPECT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Matrix, ReduceSmall) {
    cuBool_Index m = 100, n = 200;
    float step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, ReduceMedium) {
    cuBool_Index m = 400, n = 700;
    float step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, ReduceLarge) {
    cuBool_Index m = 2000, n = 4000;
    float step = 0.01f;
    testRun(m, n, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, ReduceSmallFallback) {
    cuBool_Index m = 100, n = 200;
    float step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, ReduceMediumFallback) {
    cuBool_Index m = 400, n = 700;
    float step = 0.05f;
    testRun(m, n, step, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, ReduceLargeFallback) {
    cuBool_Index m = 2000, n = 4000;
    float step = 0.01f;
    testRun(m, n, step, CUBOOL_HINT_CPU_BACKEND);
}

CUBOOL_GTEST_MAIN