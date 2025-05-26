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

void testVectorMatrixMultiplyAdd(cuBool_Index m, cuBool_Index n, float density) {
    cuBool_Matrix a;
    cuBool_Vector v, r;

    // Generate test data with specified density
    testing::Matrix ta = std::move(testing::Matrix::generateSparse(m, n, density));
    testing::Vector tv = std::move(testing::Vector::generateSparse(m, density));

    // Allocate input matrices and resize to fill with input data
    ASSERT_EQ(cuBool_Matrix_New(&a, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&v, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&r, n), CUBOOL_STATUS_SUCCESS);

    // Transfer input data into input matrices
    ASSERT_EQ(cuBool_Matrix_Build(a, ta.rowsIndex.data(), ta.colsIndex.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED & CUBOOL_HINT_NO_DUPLICATES), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Build(v, tv.index.data(), tv.nvals, CUBOOL_HINT_VALUES_SORTED & CUBOOL_HINT_NO_DUPLICATES), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive on the cpu to compare results
    testing::VectorMatrixMultiplyFunctor functor;
    testing::Vector                      tr = functor(tv, ta);

    // Evaluate r = v x M
    ASSERT_EQ(cuBool_VxM(r, v, a, CUBOOL_HINT_NO), CUBOOL_STATUS_SUCCESS);

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
        testVectorMatrixMultiplyAdd(m, n, 0.01f + (0.5f) * ((float) i + 1.0f) / ((float) n));
    }

    // Finalize library
    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Vector, MultiplyVectorMatrixSmall) {
    cuBool_Index m = 600, n = 800;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, MultiplyVectorMatrixMedium) {
    cuBool_Index m = 2500, n = 4000;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, MultiplyVectorMatrixLarge) {
    cuBool_Index m = 5000, n = 10000;
    testRun(m, n, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, MultiplyVectorMatrixSmallFallback) {
    cuBool_Index m = 600, n = 800;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, MultiplyVectorMatrixMediumFallback) {
    cuBool_Index m = 2500, n = 4000;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, MultiplyVectorMatrixLargeFallback) {
    cuBool_Index m = 5000, n = 10000;
    testRun(m, n, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, MultiplyVectorMatrixSmallManaged) {
    cuBool_Index m = 600, n = 800;
    testRun(m, n, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Vector, MultiplyVectorMatrixMediumManaged) {
    cuBool_Index m = 2500, n = 4000;
    testRun(m, n, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Vector, MultiplyVectorMatrixLargeManaged) {
    cuBool_Index m = 5000, n = 10000;
    testRun(m, n, CUBOOL_HINT_GPU_MEM_MANAGED);
}

CUBOOL_GTEST_MAIN