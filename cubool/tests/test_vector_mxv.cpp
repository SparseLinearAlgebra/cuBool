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

void testMatrixVectorMultiplyAdd(cuBool_Index m, cuBool_Index n, float density) {
    cuBool_Matrix a;
    cuBool_Vector v, r;

    // Generate test data with specified density
    testing::Matrix ta = std::move(testing::Matrix::generateSparse(m, n, density));
    testing::Vector tv = std::move(testing::Vector::generateSparse(n, density));

    // Allocate input matrices and resize to fill with input data
    ASSERT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&v, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&r, m), CUBOOL_STATUS_SUCCESS);

    // Transfer input data into input matrices
    ASSERT_EQ(cuBool_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED & CUBOOL_HINT_NO_DUPLICATES), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Build(v, tv.index.data(), tv.nvals, CUBOOL_HINT_VALUES_SORTED & CUBOOL_HINT_NO_DUPLICATES), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive r += a x b on the cpu to compare results
    testing::MatrixVectorMultiplyFunctor functor;
    testing::Vector tr = functor(ta, tv);

    // Evaluate r += a x b
    ASSERT_EQ(cuBool_MxV(r, a, v, CUBOOL_HINT_NO), CUBOOL_STATUS_SUCCESS);

    // Compare results
    ASSERT_EQ(tr.areEqual(r), true);

    // Deallocate matrices
    ASSERT_EQ(cuBool_Matrix_Free(a), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(v), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(r), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Index n, cuBool_Hints setup) {
    // Setup library
    ASSERT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 5; i++) {
        testMatrixVectorMultiplyAdd(m, n, 0.01f + (0.5f) * ((float) i + 1.0f) / ((float) n));
    }

    // Finalize library
    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Matrix, MultiplyMatrixVectorSmall) {
    cuBool_Index m = 600, n = 800;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, MultiplyMatrixVectorMedium) {
    cuBool_Index m = 2500, n = 4000;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, MultiplyMatrixVectorLarge) {
    cuBool_Index m = 10000, n = 5000;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Matrix, MultiplyMatrixVectorSmallFallback) {
    cuBool_Index m = 600, n = 800;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, MultiplyMatrixVectorMediumFallback) {
    cuBool_Index m = 2500, n = 4000;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Matrix, MultiplyMatrixVectorLargeFallback) {
    cuBool_Index m = 10000, n = 5000;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

CUBOOL_GTEST_MAIN