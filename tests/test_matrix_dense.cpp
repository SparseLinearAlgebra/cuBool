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

// Test dense matrix create/destroy functions
TEST(MatrixDense, CreateDestroy) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolMatrixDense matrix = nullptr;

    setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_MatrixDense_New_(instance, &matrix), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Delete(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Instance_Delete(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixDense, Resize) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolMatrixDense matrix = nullptr;
    CuBoolSize_t rows = 1024, columns = 1024;

    setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_MatrixDense_New_(instance, &matrix), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Resize(instance, matrix, rows, columns), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Delete(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Instance_Delete(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixDense, Filling) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolMatrixDense matrix = nullptr;
    CuBoolSize_t m = 1024, n = 2048;

    CuBoolSize_t resultCount;
    CuBoolPair* resultValues;
    std::vector<CuBoolPair> inputValues;
    std::unordered_set<CuBoolPair, CuBoolPairHash, CuBoolPairEq> inputValuesSet;

    setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_MatrixDense_New_(instance, &matrix), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Resize(instance, matrix, m, n), CUBOOL_STATUS_SUCCESS);

    generateTestData(m, n, inputValues, Condition1{});

    EXPECT_EQ(CuBool_MatrixDense_Build_(instance, matrix, inputValues.size(), inputValues.data()), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_ExtractPairs(instance, matrix, &resultCount, &resultValues), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(resultCount, inputValues.size());

    if (resultCount == inputValues.size()) {
        packCmpSet(inputValues, inputValuesSet);

        for (CuBoolSize_t k = 0; k < resultCount; k++) {
            EXPECT_EQ(inputValuesSet.find(resultValues[k]) != inputValuesSet.end(), true);
        }
    }

    // Remember to release exposed array buffer
    EXPECT_EQ(CuBool_Vals_Delete(instance, resultValues), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Delete(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBool_Instance_Delete(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixDense, MultiplyAdd) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolSize_t m = 145, t = 2456, n = 1753;

    CuBoolMatrixDense a, b, c, r;

    CuBoolSize_t resultCount = 0;
    CuBoolPair* resultValues = nullptr;
    std::vector<CuBoolPair> aval;
    std::vector<CuBoolPair> bval;
    std::vector<CuBoolPair> cval;
    std::unordered_set<CuBoolPair, CuBoolPairHash, CuBoolPairEq> resultValuesSet;

    setupInstanceDesc(instanceDesc);

    // Setup instance
    EXPECT_EQ(CuBool_Instance_New(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    // Allocate input matrices and resize to fill with input data
    EXPECT_EQ(CuBool_MatrixDense_New_(instance, &a), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Resize(instance, a, m, t), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_New_(instance, &b), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Resize(instance, b, t, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_New_(instance, &c), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Resize(instance, c, m, n), CUBOOL_STATUS_SUCCESS);

    // Allocate result matrix. No resize needed, since the data will be placed automatically
    EXPECT_EQ(CuBool_MatrixDense_New_(instance, &r), CUBOOL_STATUS_SUCCESS);

    generateTestData(m, t, aval, Condition2{});
    generateTestData(t, n, bval, Condition2{});
    generateTestData(m, n, cval, Condition2{});

    // Transfer input data into input matrices
    EXPECT_EQ(CuBool_MatrixDense_Build_(instance, a, aval.size(), aval.data()), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Build_(instance, b, bval.size(), bval.data()), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Build_(instance, c, cval.size(), cval.data()), CUBOOL_STATUS_SUCCESS);

    // Evaluate r = a x b + c
    EXPECT_EQ(CuBool_MatrixDense_MultAdd(instance, r, a, b, c), CUBOOL_STATUS_SUCCESS);

    // Transfer result values into host arrays
    EXPECT_EQ(CuBool_MatrixDense_ExtractPairs(instance, r, &resultCount, &resultValues), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive r = a x b + c on the cpu to compare results
    evaluateMultiplyAdd(m, t, n, aval, bval, cval, resultValuesSet);

    // Validate size and compare results
    EXPECT_EQ(resultCount, resultValuesSet.size());

    std::cout << "Got: " << resultCount << " Expected: " << resultValuesSet.size() << std::endl;

    for (CuBoolSize_t k = 0; k < resultCount; k++) {
        EXPECT_EQ(resultValuesSet.find(resultValues[k]) != resultValuesSet.end(), true);
    }

    // Manually deallocate array, returned by read* function
    EXPECT_EQ(CuBool_Vals_Delete(instance, resultValues), CUBOOL_STATUS_SUCCESS);

    // Release matrices
    EXPECT_EQ(CuBool_MatrixDense_Delete(instance, a), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Delete(instance, b), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Delete(instance, c), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBool_MatrixDense_Delete(instance, r), CUBOOL_STATUS_SUCCESS);

    // Destroy instance
    EXPECT_EQ(CuBool_Instance_Delete(instance), CUBOOL_STATUS_SUCCESS);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
