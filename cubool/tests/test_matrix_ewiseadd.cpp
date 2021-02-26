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

void testMatrixAdd(cuBool_Index m, cuBool_Index n, float density) {
    cuBool_Matrix r, a, b;

    testing::Matrix ta = std::move(testing::Matrix::generateSparse(m, n, density));
    testing::Matrix tb = std::move(testing::Matrix::generateSparse(m, n, density));

    // Allocate input matrices and resize to fill with input data
    ASSERT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_New(&b, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_New(&r, m, n), CUBOOL_STATUS_SUCCESS);

    // Transfer input data into input matrices
    ASSERT_EQ(cuBool_Matrix_Build(a, ta.mRowsIndex.data(), ta.mColsIndex.data(), ta.mNvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Build(b, tb.mRowsIndex.data(), tb.mColsIndex.data(), tb.mNvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    // Evaluate r = a + b
    ASSERT_EQ(cuBool_Matrix_EWiseAdd(r, a, b), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive r += a on the cpu to compare results
    testing::MatrixEWiseAddFunctor functor;
    auto tr = std::move(functor(ta, tb));

    // Compare results
    ASSERT_EQ(tr.areEqual(r), true);

    // Deallocate matrices
    ASSERT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Free(b), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Free(r), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Index n, cuBool_Hints setup) {
    // Setup library
    ASSERT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 5; i++) {
        testMatrixAdd(m, n, 0.1f + (0.05f) * ((float) i));
    }

    // Finalize library
    EXPECT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Matrix, AddSmall) {
    cuBool_Index m = 60, n = 80;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, AddMedium) {
    cuBool_Index m = 500, n = 800;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, AddLarge) {
    cuBool_Index m = 2500, n = 1500;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, AddSmallFallback) {
    cuBool_Index m = 60, n = 80;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, AddMediumFallback) {
    cuBool_Index m = 500, n = 800;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, AddLargeFallback) {
    cuBool_Index m = 2500, n = 1500;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

CUBOOL_GTEST_MAIN