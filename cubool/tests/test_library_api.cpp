/**********************************************************************************/
/*                                                                                */
/* MIT License                                                                    */
/*                                                                                */
/* Copyright (c) 2020 JetBrains-Research                                          */
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
/*                                                                                */
/**********************************************************************************/

#include <gtest/gtest.h>
#include <testing/testing.hpp>

// Query library version info
TEST(CuBoolVersion, Query) {
    int major;
    int minor;
    int version;

    cuBool_Version_Get(&major, &minor, &version);

    std::cout << "Major: " << major << std::endl;
    std::cout << "Minor: " << minor << std::endl;
    std::cout << "Version: " << version << std::endl;

    EXPECT_NE(version, 0);
}

// Test cubool library instance creation and destruction
TEST(CuBoolInstance, Setup) {
    cuBool_Status error;

    error = cuBool_Initialize(CUBOOL_HINT_NO);
    ASSERT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = cuBool_Finalize();
    ASSERT_EQ(error, CUBOOL_STATUS_SUCCESS);
}

/**
 * Performs transitive closure for directed graph
 *
 * @param A Adjacency matrix of the graph
 * @param T Reference to the handle where to allocate and store result
 *
 * @return Status on this operation
 */
cuBool_Status TransitiveClosure(cuBool_Matrix A, cuBool_Matrix* T) {
    cuBool_Matrix_Duplicate(A, T);                              /* Duplicate A to result T */

    cuBool_Index total = 0;
    cuBool_Index current;

    cuBool_Matrix_Nvals(*T, &current);                          /* Query current nvals value */

    while (current != total) {                                  /* Iterate, while new values are added */
        total = current;
        cuBool_MxM(*T, *T, *T, CUBOOL_HINT_ACCUMULATE);  /* T += T x T */
        cuBool_Matrix_Nvals(*T, &current);
    }

    return CUBOOL_STATUS_SUCCESS;
}

TEST(CuBoolInstance, Example) {
    cuBool_Matrix A = nullptr;
    cuBool_Matrix T = nullptr;

    cuBool_Index n = 100;

    cuBool_Initialize(CUBOOL_HINT_NO);
    cuBool_Matrix_New(&A, n, n);

    testing::Matrix ta = testing::Matrix::generateSparse(n , n, 0.2);

    cuBool_Matrix_Build(A, ta.mRowsIndex.data(), ta.mColsIndex.data(), ta.mNvals, CUBOOL_HINT_VALUES_SORTED);

    TransitiveClosure(A, &T);

    testing::Matrix tr = ta;
    size_t total = 0;

    do {
        total = tr.mNvals;

        testing::MatrixMultiplyFunctor functor;
        tr = std::move(functor(tr, tr, tr));
    }
    while (tr.mNvals != total);

    ASSERT_TRUE(tr.areEqual(T));

    cuBool_Matrix_Free(A);
    cuBool_Matrix_Free(T);
    cuBool_Finalize();
}

CUBOOL_GTEST_MAIN