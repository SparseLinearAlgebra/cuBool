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

    setupInstanceDesc(instanceDesc);

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

    setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Matrix_New(instance, &matrix, rows, columns), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Free(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

// Fills sparse matrix with random data and tests whether the transfer works correctly
void testMatrixFilling(CuBoolSize_t m, CuBoolSize_t n, float density, CuBoolInstance instance) {
    CuBoolMatrix matrix = nullptr;

    std::vector<CuBoolIndex_t> rows;
    std::vector<CuBoolIndex_t> cols;
    CuBoolSize_t nvals;

    generateTestData(m, n, rows, cols, nvals, Condition3{density});

    EXPECT_EQ(CuBool_Matrix_New(instance, &matrix, m, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Build(instance, matrix, rows.data(), cols.data(), nvals), CUBOOL_STATUS_SUCCESS);

    CuBoolIndex_t* extRows;
    CuBoolIndex_t* extCols;
    CuBoolSize_t extNvals;

    EXPECT_EQ(CuBool_Matrix_ExtractPairs(instance, matrix, &extRows, &extCols, &extNvals), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(nvals, extNvals);

    if (nvals == extNvals) {
        std::unordered_set<CuBoolPair,CuBoolPairHash,CuBoolPairEq> cmpSet;
        packCmpSet(rows, cols, cmpSet);

        for (CuBoolSize_t k = 0; k < extNvals; k++) {
            EXPECT_EQ(cmpSet.find(CuBoolPair{extRows[k], extCols[k]}) != cmpSet.end(), true);
        }
    }

    // Remember to release exposed array buffer
    EXPECT_EQ(CuBool_Vals_Free(instance, extRows), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Vals_Free(instance, extCols), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Free(instance, matrix), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixCsr, FillingSmall) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};

    CuBoolSize_t m = 60, n = 100;

    setupInstanceDesc(instanceDesc);
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

    setupInstanceDesc(instanceDesc);
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

    setupInstanceDesc(instanceDesc);
    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    for (size_t i = 0; i < 10; i++) {
        testMatrixFilling(m, n, 0.001f + (0.05f) * ((float) i), instance);
    }

    EXPECT_EQ(CuBool_Instance_Free(instance), CUBOOL_STATUS_SUCCESS);
}

void testMatrixMultiplyAdd(CuBoolSize_t m, CuBoolSize_t t, CuBoolSize_t n, float density, CuBoolInstance instance) {
    CuBoolMatrix a, b, r;

    std::vector<CuBoolIndex_t> arows;
    std::vector<CuBoolIndex_t> acols;
    std::vector<CuBoolIndex_t> brows;
    std::vector<CuBoolIndex_t> bcols;
    std::vector<CuBoolIndex_t> rrows;
    std::vector<CuBoolIndex_t> rcols;

    CuBoolSize_t anvals;
    CuBoolSize_t bnvals;
    CuBoolSize_t rnvals;

    // Generate test data with specified density
    generateTestData(m, t, arows, acols, anvals, Condition3{density});
    generateTestData(t, n, brows, bcols, bnvals, Condition3{density});
    generateTestData(m, n, rrows, rcols, rnvals, Condition3{density});

    // Allocate input matrices and resize to fill with input data
    EXPECT_EQ(CuBool_Matrix_New(instance, &a, m, t), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_New(instance, &b, t, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_New(instance, &r, m, n), CUBOOL_STATUS_SUCCESS);

    // Transfer input data into input matrices
    EXPECT_EQ(CuBool_Matrix_Build(instance, a, arows.data(), acols.data(), anvals), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Build(instance, b, brows.data(), bcols.data(), bnvals), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Matrix_Build(instance, r, rrows.data(), rcols.data(), rnvals), CUBOOL_STATUS_SUCCESS);

    // Evaluate r += a x b
    EXPECT_EQ(CuBool_MxM(instance, r, a, b), CUBOOL_STATUS_SUCCESS);

    CuBoolIndex_t* extRows;
    CuBoolIndex_t* extCols;
    CuBoolSize_t extNvals;
    std::unordered_set<CuBoolPair,CuBoolPairHash,CuBoolPairEq> resultSet;

    // Extract result
    EXPECT_EQ(CuBool_Matrix_ExtractPairs(instance, r, &extRows, &extCols, &extNvals), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive r = a x b + c on the cpu to compare results
    evaluateMultiplyAdd(m, t, n, arows, acols, brows, bcols, rrows, rcols, resultSet);

    EXPECT_EQ(resultSet.size(), extNvals);

    if (resultSet.size() == extNvals) {
        for (CuBoolSize_t k = 0; k < extNvals; k++) {
            EXPECT_EQ(resultSet.find(CuBoolPair{extRows[k], extCols[k]}) != resultSet.end(), true);
        }
    }

    // Remember to release exposed array buffer
    EXPECT_EQ(CuBool_Vals_Free(instance, extRows), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_Vals_Free(instance, extCols), CUBOOL_STATUS_SUCCESS);

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
    setupInstanceDesc(instanceDesc);
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
    setupInstanceDesc(instanceDesc);
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
    setupInstanceDesc(instanceDesc);
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
