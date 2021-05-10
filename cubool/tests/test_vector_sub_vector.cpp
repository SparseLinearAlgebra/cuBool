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

void testVectorExtractSubVector(cuBool_Index m, cuBool_Index N, float density, cuBool_Hints flags) {
    cuBool_Vector r, a;

    auto ta = testing::Vector::generateSparse(m, density);

    ASSERT_EQ(cuBool_Vector_New(&a, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Build(a, ta.index.data(), ta.nvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < N; i++) {
        auto k = m / N;

        ASSERT_EQ(cuBool_Vector_New(&r, k), CUBOOL_STATUS_SUCCESS);
        ASSERT_EQ(cuBool_Vector_ExtractSubVector(r, a, i * k, k, flags), CUBOOL_STATUS_SUCCESS);

        auto tr = ta.subVector(i * k, k);

        ASSERT_TRUE(tr.areEqual(r));
        ASSERT_EQ(cuBool_Vector_Free(r), CUBOOL_STATUS_SUCCESS);
    }

    ASSERT_EQ(cuBool_Vector_Free(a), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, float step, cuBool_Hints setup) {
    // Setup library
    EXPECT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    auto N = 10;

    for (size_t i = 0; i < N; i++) {
        testVectorExtractSubVector(m, N, 0.1f + step * ((float) i), CUBOOL_HINT_NO);
    }

    // Finalize library
    EXPECT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Vector, SubVectorExtractSmall) {
    cuBool_Index m = 10000;
    float step = 0.05f;
    testRun(m, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, SubVectorExtractMedium) {
    cuBool_Index m = 50000;
    float step = 0.05f;
    testRun(m, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, SubVectorExtractLarge) {
    cuBool_Index m = 100000;
    float step = 0.05f;
    testRun(m, step, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, SubVectorExtractSmallFallback) {
    cuBool_Index m = 10000;
    float step = 0.05f;
    testRun(m, step, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, SubVectorExtractMediumFallback) {
    cuBool_Index m = 50000;
    float step = 0.05f;
    testRun(m, step, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, SubVectorExtractLargeFallback) {
    cuBool_Index m = 100000;
    float step = 0.05f;
    testRun(m, step, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, SubVectorExtractSmallManaged) {
    cuBool_Index m = 10000;
    float step = 0.05f;
    testRun(m, step, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Vector, SubVectorExtractMediumManaged) {
    cuBool_Index m = 50000;
    float step = 0.05f;
    testRun(m, step, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Vector, SubVectorExtractLargeManaged) {
    cuBool_Index m = 100000;
    float step = 0.05f;
    testRun(m, step, CUBOOL_HINT_GPU_MEM_MANAGED);
}

CUBOOL_GTEST_MAIN