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

void testVectorAdd(cuBool_Index m, float density, cuBool_Hints flags) {
    cuBool_Vector r, a, b;

    testing::Vector ta = std::move(testing::Vector::generateSparse(m, density));
    testing::Vector tb = std::move(testing::Vector::generateSparse(m, density));

    // Allocate input matrices and resize to fill with input data
    ASSERT_EQ(cuBool_Vector_New(&a, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&b, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_New(&r, m), CUBOOL_STATUS_SUCCESS);

    // Transfer input data into input matrices
    ASSERT_EQ(cuBool_Vector_Build(a, ta.index.data(), ta.nvals, 0), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Build(b, tb.index.data(), tb.nvals, 0), CUBOOL_STATUS_SUCCESS);

    // Evaluate r = a + b
    ASSERT_EQ(cuBool_Vector_EWiseAdd(r, a, b, flags), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive r += a on the cpu to compare results
    testing::VectorEWiseAddFunctor functor;
    auto tr = std::move(functor(ta, tb));

    // Compare results
    ASSERT_EQ(tr.areEqual(r), true);

    // Deallocate matrices
    ASSERT_EQ(cuBool_Vector_Free(a), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(b), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(r), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Hints setup) {
    // Setup library
    ASSERT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testVectorAdd(m, 0.2f + (0.05f) * ((float) i), CUBOOL_HINT_NO);
    }

    // Finalize library
    EXPECT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Vector, EWiseAddSmall) {
    cuBool_Index m = 60;
    testRun(m, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, EWiseAddMedium) {
    cuBool_Index m = 500;
    testRun(m, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, EWiseAddLarge) {
    cuBool_Index m = 2500;
    testRun(m, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, EWiseAddSmallFallback) {
    cuBool_Index m = 60;
    testRun(m, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, EWiseAddMediumFallback) {
    cuBool_Index m = 500;
    testRun(m, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, EWiseAddLargeFallback) {
    cuBool_Index m = 2500;
    testRun(m, CUBOOL_HINT_CPU_BACKEND);
}

CUBOOL_GTEST_MAIN