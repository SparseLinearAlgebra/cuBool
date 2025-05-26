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

// Fills sparse vector with random data and tests whether the transfer works correctly
void testVectorSetElement(cuBool_Index m, float density) {
    cuBool_Vector vector = nullptr;

    testing::Vector tvector = testing::Vector::generateSparse(m, density);

    auto   I     = tvector.index;
    size_t nvals = tvector.nvals;
    size_t dups  = nvals * 0.10;

    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    auto                       dist = std::uniform_real_distribution<float>(0.0, 1.0);

    if (nvals > 0) {
        for (size_t k = 0; k < dups; k++) {
            size_t id = std::min<size_t>(dist(engine) * nvals, nvals - 1);
            auto   i  = I[id];
            I.push_back(i);
        }
    }

    ASSERT_EQ(cuBool_Vector_New(&vector, m), CUBOOL_STATUS_SUCCESS);

    nvals = I.size();
    for (size_t i = 0; i < nvals; i++) {
        ASSERT_EQ(cuBool_Vector_SetElement(vector, I[i]), CUBOOL_STATUS_SUCCESS);
    }

    // Compare test vector and library one
    ASSERT_TRUE(tvector.areEqual(vector));

    // Remember to release resources
    ASSERT_EQ(cuBool_Vector_Free(vector), CUBOOL_STATUS_SUCCESS);
}

void testVectorAppendElement(cuBool_Index m, float density) {
    cuBool_Vector vector = nullptr;

    testing::Vector tvector = testing::Vector::generateSparse(m, density);

    ASSERT_EQ(cuBool_Vector_New(&vector, m), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Build(vector, tvector.index.data(), (cuBool_Index) tvector.nvals, CUBOOL_HINT_VALUES_SORTED | CUBOOL_HINT_NO_DUPLICATES), CUBOOL_STATUS_SUCCESS);

    std::vector<cuBool_Index> I;
    size_t                    nvals = tvector.nvals;
    size_t                    dups  = nvals * 0.20;

    std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    auto                       dist = std::uniform_real_distribution<float>(0.0, 1.0);

    if (nvals > 0) {
        for (size_t k = 0; k < dups; k++) {
            size_t id = std::min<size_t>(dist(engine) * nvals, nvals - 1);
            auto   i  = tvector.index[id];
            I.push_back(i);
        }
    }

    nvals = I.size();
    for (size_t i = 0; i < nvals; i++) {
        ASSERT_EQ(cuBool_Vector_SetElement(vector, I[i]), CUBOOL_STATUS_SUCCESS);
    }

    // Compare test vector and library one
    ASSERT_TRUE(tvector.areEqual(vector));

    // Remember to release resources
    ASSERT_EQ(cuBool_Vector_Free(vector), CUBOOL_STATUS_SUCCESS);
}

void testVectorPostAppendElement(cuBool_Index m, float density) {
    cuBool_Vector vector     = nullptr;
    cuBool_Vector duplicated = nullptr;

    testing::Vector tvector = std::move(testing::Vector::generateSparse(m, density));

    ASSERT_EQ(cuBool_Vector_New(&vector, m), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < tvector.nvals; i++) {
        ASSERT_EQ(cuBool_Vector_SetElement(vector, tvector.index[i]), CUBOOL_STATUS_SUCCESS);
    }

    ASSERT_EQ(cuBool_Vector_Duplicate(vector, &duplicated), CUBOOL_STATUS_SUCCESS);

    // Compare test vector and library one
    ASSERT_TRUE(tvector.areEqual(duplicated));

    // Remember to release resources
    ASSERT_EQ(cuBool_Vector_Free(vector), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Vector_Free(duplicated), CUBOOL_STATUS_SUCCESS);
}

void testRun(cuBool_Index m, cuBool_Hints setup) {
    ASSERT_EQ(cuBool_Initialize(setup), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testVectorSetElement(m, 0.1f + (0.05f) * ((float) i));
    }

    for (size_t i = 0; i < 10; i++) {
        testVectorAppendElement(m, 0.1f + (0.05f) * ((float) i));
    }

    for (size_t i = 0; i < 10; i++) {
        testVectorPostAppendElement(m, 0.1f + (0.05f) * ((float) i));
    }

    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(cuBool_Vector, SetElementSmall) {
    cuBool_Index m = 1000;
    testRun(m, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, SetElementMedium) {
    cuBool_Index m = 20000;
    testRun(m, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, SetElementLarge) {
    cuBool_Index m = 90000;
    testRun(m, CUBOOL_HINT_NO);
}

TEST(cuBool_Vector, SetElementSmallFallback) {
    cuBool_Index m = 1000;
    testRun(m, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, SetElementMediumFallback) {
    cuBool_Index m = 20000;
    testRun(m, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, SetElementLargeFallback) {
    cuBool_Index m = 90000;
    testRun(m, CUBOOL_HINT_CPU_BACKEND);
}

TEST(cuBool_Vector, SetElementSmallManaged) {
    cuBool_Index m = 1000;
    testRun(m, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Vector, SetElementMediumManaged) {
    cuBool_Index m = 20000;
    testRun(m, CUBOOL_HINT_GPU_MEM_MANAGED);
}

TEST(cuBool_Vector, SetElementLargeManaged) {
    cuBool_Index m = 90000;
    testRun(m, CUBOOL_HINT_GPU_MEM_MANAGED);
}

CUBOOL_GTEST_MAIN