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
#include <test_common.hpp>


TEST(MatrixCsr, CreateDestroy) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolMatrix matrix = nullptr;

    testing::details::setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Matrix_New(instance, &matrix, 0, 0), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Free(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, Resize) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolMatrix matrix = nullptr;
    CuBoolSize_t rows = 1024, columns = 1024;

    testing::details::setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Matrix_New(instance, &matrix, rows, columns), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Free(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, Duplicate) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolMatrix matrix = nullptr, duplicated = nullptr;
    CuBoolSize_t m = 900, n = 600;
    float density = 0.31;

    testing::Matrix tmatrix = std::move(testing::Matrix::generate(m, n, testing::details::Condition3(density)));

    testing::details::setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Matrix_New(instance, &matrix, m, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Build(instance, matrix, tmatrix.mRowsIndex.data(), tmatrix.mColsIndex.data(), tmatrix.mNvals), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Matrix_Duplicate(instance, matrix, &duplicated), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(tmatrix.areEqual(duplicated, instance), true);

    EXPECT_EQ(CuBool_Matrix_Free(instance, matrix), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Free(instance, duplicated), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, PropertyQuery) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolMatrix matrix = nullptr;
    CuBoolSize_t m = 900, n = 600;
    float density = 0.21;

    testing::Matrix tmatrix = std::move(testing::Matrix::generate(m, n, testing::details::Condition3(density)));

    testing::details::setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Matrix_New(instance, &matrix, m, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Build(instance, matrix, tmatrix.mRowsIndex.data(), tmatrix.mColsIndex.data(), tmatrix.mNvals), CUBOOL_STATUS_SUCCESS);

    CuBoolIndex_t nrows;
    CuBoolIndex_t ncols;
    CuBoolSize_t nvals;

    EXPECT_EQ(CuBool_Matrix_Nvals(instance, matrix, &nvals), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Nrows(instance, matrix, &nrows), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Ncols(instance, matrix, &ncols), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(nvals, tmatrix.mNvals);
    EXPECT_EQ(nrows, tmatrix.mNrows);
    EXPECT_EQ(ncols, tmatrix.mNcols);

    EXPECT_EQ(CuBool_Matrix_Free(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

// Fills sparse matrix with random data and tests whether the transfer works correctly
void testMatrixFilling(CuBoolSize_t m, CuBoolSize_t n, float density, CuBoolInstance instance) {
    CuBoolMatrix matrix = nullptr;

    testing::Matrix tmatrix = std::move(testing::Matrix::generate(m, n, testing::details::Condition3(density)));

    EXPECT_EQ(CuBool_Matrix_New(instance, &matrix, m, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Build(instance, matrix, tmatrix.mRowsIndex.data(), tmatrix.mColsIndex.data(), tmatrix.mNvals), CUBOOL_STATUS_SUCCESS);

    // Compare test matrix and library one
    EXPECT_EQ(tmatrix.areEqual(matrix, instance), true);

    // Remember to release resources
    EXPECT_EQ(CuBool_Matrix_Free(instance, matrix), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, FillingSmall) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};

    CuBoolSize_t m = 60, n = 100;

    testing::details::setupInstanceDesc(instanceDesc);
    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixFilling(m, n, 0.001f + (0.05f) * ((float) i), instance);
    }

    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, FillingMedium) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};

    CuBoolSize_t m = 500, n = 1000;

    testing::details::setupInstanceDesc(instanceDesc);
    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixFilling(m, n, 0.001f + (0.05f) * ((float) i), instance);
    }

    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, FillingLarge) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};

    CuBoolSize_t m = 1000, n = 2000;

    testing::details::setupInstanceDesc(instanceDesc);
    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixFilling(m, n, 0.001f + (0.05f) * ((float) i), instance);
    }

    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

void testMatrixMultiplyAdd(CuBoolSize_t m, CuBoolSize_t t, CuBoolSize_t n, float density, CuBoolInstance instance) {
    CuBoolMatrix a, b, r;

    // Generate test data with specified density
    testing::Matrix ta = std::move(testing::Matrix::generate(m, t, testing::details::Condition3(density)));
    testing::Matrix tb = std::move(testing::Matrix::generate(t, n, testing::details::Condition3(density)));
    testing::Matrix tr = std::move(testing::Matrix::generate(m, n, testing::details::Condition3(density)));

    // Allocate input matrices and resize to fill with input data
    EXPECT_EQ(CuBool_Matrix_New(instance, &a, m, t), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_New(instance, &b, t, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_New(instance, &r, m, n), CUBOOL_STATUS_SUCCESS);

    // Transfer input data into input matrices
    EXPECT_EQ(CuBool_Matrix_Build(instance, a, ta.mRowsIndex.data(), ta.mColsIndex.data(), ta.mNvals), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Build(instance, b, tb.mRowsIndex.data(), tb.mColsIndex.data(), tb.mNvals), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Build(instance, r, tr.mRowsIndex.data(), tr.mColsIndex.data(), tr.mNvals), CUBOOL_STATUS_SUCCESS);

    // Evaluate r += a x b
    EXPECT_EQ(CuBool_MxM(instance, r, a, b), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive r += a x b on the cpu to compare results
    testing::MatrixMultiplyAdd functor;
    tr = std::move(functor(ta, tb, tr));

    // Compare results
    EXPECT_EQ(tr.areEqual(r, instance), true);

    // Deallocate matrices
    EXPECT_EQ(CuBool_Matrix_Free(instance, a), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Free(instance, b), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Free(instance, r), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, MultiplyAddSmall) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolSize_t m = 60, t = 100, n = 80;

    // Setup instance
    testing::details::setupInstanceDesc(instanceDesc);
    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 5; i++) {
        testMatrixMultiplyAdd(m, t, n, 0.1f + (0.05f) * ((float) i), instance);
    }

    // Destroy instance
    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, MultiplyAddMedium) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolSize_t m = 500, t = 1000, n = 800;

    // Setup instance
    testing::details::setupInstanceDesc(instanceDesc);
    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 5; i++) {
        testMatrixMultiplyAdd(m, t, n, 0.1f + (0.05f) * ((float) i), instance);
    }

    // Destroy instance
    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, MultiplyAddLarge) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolSize_t m = 1000, t = 2000, n = 500;

    // Setup instance
    testing::details::setupInstanceDesc(instanceDesc);
    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 5; i++) {
        testMatrixMultiplyAdd(m, t, n, 0.1f + (0.05f) * ((float) i), instance);
    }

    // Destroy instance
    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
