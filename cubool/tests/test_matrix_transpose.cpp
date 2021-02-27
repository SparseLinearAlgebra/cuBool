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

void testMatrixTranspose(cuBool_Index m, cuBool_Index n, float density) {
    cuBool_Matrix r, a;

    testing::Matrix ta = std::move(testing::Matrix::generateSparse(m, n, density));

    // Allocate input matrices and resize to fill with input data
    EXPECT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(cuBool_Matrix_New(&r, n, m), CUBOOL_STATUS_SUCCESS);

    // Transfer input data into input matrices
    EXPECT_EQ(cuBool_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    // Evaluate r = a ^ T
    EXPECT_EQ(cuBool_Matrix_Transpose(r, a), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive r = a ^ T on the cpu to compare results
    testing::Matrix tr = ta.transpose();

    // Compare results
    EXPECT_EQ(tr.areEqual(r), true);

    // Deallocate matrices
    EXPECT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(cuBool_Matrix_Free(r), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Index n, cuBool_Hints setup) {
    // Setup library
    EXPECT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 5; i++) {
        testMatrixTranspose(m, n, 0.1f + (0.05f) * ((float) i));
    }

    // Finalize library
    EXPECT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Matrix, TransposeSmall) {
    cuBool_Index m = 60, n = 80;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, TransposeMedium) {
    cuBool_Index m = 500, n = 800;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, TransposeLarge) {
    cuBool_Index m = 2500, n = 1500;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, TransposeSmallFallback) {
    cuBool_Index m = 60, n = 80;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, TransposeMediumFallback) {
    cuBool_Index m = 500, n = 800;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, TransposeLargeFallback) {
    cuBool_Index m = 2500, n = 1500;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

CUBOOL_GTEST_MAIN
