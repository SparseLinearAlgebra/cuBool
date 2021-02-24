/**********************************************************************************/
/*                                                                                */
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
/*                                                                                */
/**********************************************************************************/

#include <testing/testing.hpp>

TEST(MatrixCsr, Duplicate) {
    cuBool_Matrix matrix = nullptr, duplicated = nullptr;
    cuBool_Index m = 900, n = 600;
    float density = 0.31;

    testing::Matrix tmatrix = std::move(testing::Matrix::generate(m, n, testing::Condition3(density)));

    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_NO), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_New(&matrix, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Build(matrix, tmatrix.mRowsIndex.data(), tmatrix.mColsIndex.data(), tmatrix.mNvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_Duplicate(matrix, &duplicated), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(tmatrix.areEqual(duplicated), true);

    ASSERT_EQ(cuBool_Matrix_Free(matrix), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Free(duplicated), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, PropertyQuery) {
    cuBool_Matrix matrix = nullptr;
    cuBool_Index m = 900, n = 600;
    float density = 0.21;

    testing::Matrix tmatrix = std::move(testing::Matrix::generate(m, n, testing::Condition3(density)));

    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_NO), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_New(&matrix, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Build(matrix, tmatrix.mRowsIndex.data(), tmatrix.mColsIndex.data(), tmatrix.mNvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    cuBool_Index nrows;
    cuBool_Index ncols;
    cuBool_Index nvals;

    ASSERT_EQ(cuBool_Matrix_Nvals(matrix, &nvals), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Nrows(matrix, &nrows), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Ncols(matrix, &ncols), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(nvals, tmatrix.mNvals);
    ASSERT_EQ(nrows, tmatrix.mNrows);
    ASSERT_EQ(ncols, tmatrix.mNcols);

    ASSERT_EQ(cuBool_Matrix_Free(matrix), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, ExtractPairs) {
    cuBool_Matrix matrix = nullptr;
    cuBool_Index m = 900, n = 600;
    float density = 0.21;

    testing::Matrix tmatrix = std::move(testing::Matrix::generate(m, n, testing::Condition3(density)));

    ASSERT_EQ(cuBool_Initialize(CUBOOL_HINT_NO), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(cuBool_Matrix_New(&matrix, m, n), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Matrix_Build(matrix, tmatrix.mRowsIndex.data(), tmatrix.mColsIndex.data(), tmatrix.mNvals, CUBOOL_HINT_VALUES_SORTED), CUBOOL_STATUS_SUCCESS);

    cuBool_Index nvals = tmatrix.mNvals;
    std::vector<cuBool_Index> rows(tmatrix.mNvals);
    std::vector<cuBool_Index> cols(tmatrix.mNvals);

    ASSERT_EQ(cuBool_Matrix_ExtractPairs(matrix, rows.data(), cols.data(), &nvals), CUBOOL_STATUS_SUCCESS);

    ASSERT_EQ(nvals, tmatrix.mNvals);
    ASSERT_EQ(cuBool_Matrix_Free(matrix), CUBOOL_STATUS_SUCCESS);
    ASSERT_EQ(cuBool_Finalize(), CUBOOL_STATUS_SUCCESS);
}

CUBOOL_GTEST_MAIN
