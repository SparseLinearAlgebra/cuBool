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

void testMatrixKronecker(cuBool_Index m, cuBool_Index n, cuBool_Index k, cuBool_Index t, float density) {
    cuBool_Matrix r, a, b;

    testing::Matrix ta = std::move(testing::Matrix::generate(m, n, testing::Condition3(density)));
    testing::Matrix tb = std::move(testing::Matrix::generate(k, t, testing::Condition3(density)));

    // Allocate input matrices and resize to fill with input data
    EXPECT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(cuBool_Matrix_New(&b, k, t), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(cuBool_Matrix_New(&r, m * k, n * t), CUBOOL_STATUS_SUCCESS);

    // Transfer input data into input matrices
    EXPECT_EQ(cuBool_Matrix_Build(a, ta.mRowsIndex.data(), ta.mColsIndex.data(), ta.mNvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(cuBool_Matrix_Build(b, tb.mRowsIndex.data(), tb.mColsIndex.data(), tb.mNvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    // Evaluate r = a `kron` b
    EXPECT_EQ(cuBool_Kronecker(r, a, b), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive r = a `kron` b on the cpu to compare results
    testing::MatrixKroneckerFunctor functor;
    testing::Matrix tr = std::move(functor(ta, tb));

    // Compare results
    EXPECT_EQ(tr.areEqual(r), true);

    // Deallocate matrices
    EXPECT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(cuBool_Matrix_Free(b), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(cuBool_Matrix_Free(r), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Index n, cuBool_Index k, cuBool_Index t, float step, cuBool_Hints setup) {
    // Setup library
    EXPECT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixKronecker(m, n, k, t, 0.01f + step * ((float) i));
    }

    // Finalize library
    EXPECT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Matrix, KroneckerSmall) {
    cuBool_Index m = 10, n = 20;
    cuBool_Index k = 5, t = 15;
    float step = 0.05f;
    testRun(m, n, k, t, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, KroneckerMedium) {
    cuBool_Index m = 100, n = 40;
    cuBool_Index k = 30, t = 80;
    float step = 0.02f;
    testRun(m, n, k, t, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, KroneckerLarge) {
    cuBool_Index m = 1000, n = 400;
    cuBool_Index k = 300, t = 800;
    float step = 0.001f;
    testRun(m, n, k, t, step, CUBOOL_HINT_NO);
}

CUBOOL_GTEST_MAIN