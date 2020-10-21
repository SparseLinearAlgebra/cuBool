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
#include <cubool/cubool.h>

#include <vector>
#include <iostream>
#include <unordered_set>

struct CuBoolPairHash {
public:
    std::size_t operator()(const CuBoolPair &x) const {
        return std::hash<CuBoolSize_t>()(x.i) ^ std::hash<CuBoolSize_t>()(x.j);
    }
};

struct CuBoolPairEq {
public:
    bool operator()(const CuBoolPair &a, const CuBoolPair& b)  const {
        return a.i == b.i && a.j == b.j;
    }
};

static void testMsgFun(CuBoolStatus error, const char* message, void* _) {
    std::cout << "CuBool: " << message << std::endl;
}

static CuBoolCpuPtr_t testAllocateFun(CuBoolSize_t size, void* _) {
    CuBoolCpuPtr_t ptr = malloc(size);
    std::cout << "Cubool: Allocate: " << size << " " << ptr << std::endl;
    return ptr;
}

static void testDeallocateFun(CuBoolCpuPtr_t ptr, void* _) {
    std::cout << "Cubool: Deallocate: " << ptr << std::endl;
    free(ptr);
}

struct Condition1 {
    bool operator()(CuBoolSize_t i, CuBoolSize_t j) {
        return (((i - 1) & i) == 0 && ((j - 1) & j) == 0);
    }
};

struct Condition2 {
    bool operator()(CuBoolSize_t i, CuBoolSize_t j) {
        return !(i % 5) && !(j % 7);
    }
};

template<typename Condition>
static void generateTestData(CuBoolSize_t rows, CuBoolSize_t columns, std::vector<CuBoolPair> &values, Condition&& condition) {
    for (CuBoolSize_t i = 0; i < rows; i++) {
        for (CuBoolSize_t j = 0; j < columns; j++) {
            // is i and j power of two or 0
            if (condition(i, j)) {
                values.push_back(CuBoolPair{ i, j });
            }
        }
    }
}

static void packCmpSet(const std::vector<CuBoolPair>& values, std::unordered_set<CuBoolPair,CuBoolPairHash,CuBoolPairEq>& set) {
    for (auto p: values) {
        set.emplace(p);
    }
}

static void setupInstanceDesc(CuBoolInstanceDesc& instanceDesc) {
    instanceDesc.memoryType = CuBoolGpuMemoryType::CUBOOL_GPU_MEMORY_TYPE_GENERIC;
    instanceDesc.errorCallback.userData = nullptr;
    instanceDesc.errorCallback.msgFun = testMsgFun;
    instanceDesc.allocationCallback.userData = nullptr;
    instanceDesc.allocationCallback.allocateFun = testAllocateFun;
    instanceDesc.allocationCallback.deallocateFun = testDeallocateFun;
}

static void evaluateMultiplyAdd(
        CuBoolSize_t m, CuBoolSize_t t, CuBoolSize_t n,
        const std::vector<CuBoolPair> &aval,
        const std::vector<CuBoolPair> &bval,
        const std::vector<CuBoolPair> &cval,
        std::unordered_set<CuBoolPair,CuBoolPairHash,CuBoolPairEq>& result) {

    std::vector<std::vector<uint8_t>> a(m, std::vector<uint8_t>(t, 0));
    std::vector<std::vector<uint8_t>> b(n, std::vector<uint8_t>(t, 0));
    std::vector<std::vector<uint8_t>> r(m, std::vector<uint8_t>(n, 0));

    for (auto& p: aval) {
        a[p.i][p.j] = 1;
    }

    for (auto& p: bval) {
        b[p.j][p.i] = 1;
    }

    for (CuBoolSize_t i = 0; i < m; i++) {
        for (CuBoolSize_t j = 0; j < n; j++) {
            uint8_t v = 0;

            for (CuBoolSize_t k = 0; k < t; k++) {
                v |= a[i][k] & b[j][k] ? 1u: 0u;
            }

            r[i][j] = v;
        }
    }

    for (auto& p: cval) {
        r[p.i][p.j] |= 1u;
    }

    for (CuBoolSize_t i = 0; i < m; i++) {
        for (CuBoolSize_t j = 0; j < n; j++) {
            if (r[i][j] != 0) {
                result.emplace(CuBoolPair{i,j});
            }
        }
    }
}

static void evaluateMultiplyAdd(CuBoolSize_t n, const std::vector<CuBoolPair> &values,  std::unordered_set<CuBoolPair,CuBoolPairHash,CuBoolPairEq>& result) {
    evaluateMultiplyAdd(n, n, n, values, values, values, result);
}

// Test dense matrix create/destroy functions
TEST(MatrixDense, CreateDestroy) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolMatrixDense matrix = nullptr;

    setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBoolInstanceCreate(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBoolMatrixDenseCreate(instance, &matrix), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseDestroy(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBoolInstanceDestroy(instance), CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixDense, Resize) {
    CuBoolInstance instance = nullptr;
    CuBoolInstanceDesc instanceDesc{};
    CuBoolMatrixDense matrix = nullptr;
    CuBoolSize_t rows = 1024, columns = 1024;

    setupInstanceDesc(instanceDesc);

    EXPECT_EQ(CuBoolInstanceCreate(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBoolMatrixDenseCreate(instance, &matrix), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseResize(instance, matrix, rows, columns), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseDestroy(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBoolInstanceDestroy(instance), CUBOOL_STATUS_SUCCESS);
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

    EXPECT_EQ(CuBoolInstanceCreate(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBoolMatrixDenseCreate(instance, &matrix), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseResize(instance, matrix, m, n), CUBOOL_STATUS_SUCCESS);

    generateTestData(m, n, inputValues, Condition1{});

    EXPECT_EQ(CuBoolMatrixDenseWriteData(instance, matrix, inputValues.size(), inputValues.data()), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseReadData(instance, matrix, &resultCount, &resultValues), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(resultCount, inputValues.size());

    if (resultCount == inputValues.size()) {
        packCmpSet(inputValues, inputValuesSet);

        for (CuBoolSize_t k = 0; k < resultCount; k++) {
            EXPECT_EQ(inputValuesSet.find(resultValues[k]) != inputValuesSet.end(), true);
        }
    }

    // Remember to release exposed array buffer
    EXPECT_EQ(CuBoolValuesArrayFree(instance, resultValues), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseDestroy(instance, matrix), CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(CuBoolInstanceDestroy(instance), CUBOOL_STATUS_SUCCESS);
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
    EXPECT_EQ(CuBoolInstanceCreate(&instanceDesc, &instance), CUBOOL_STATUS_SUCCESS);

    // Allocate input matrices and resize to fill with input data
    EXPECT_EQ(CuBoolMatrixDenseCreate(instance, &a), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseResize(instance, a, m, t), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseCreate(instance, &b), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseResize(instance, b, t, n), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseCreate(instance, &c), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseResize(instance, c, m, n), CUBOOL_STATUS_SUCCESS);

    // Allocate result matrix. No resize needed, since the data will be placed automatically
    EXPECT_EQ(CuBoolMatrixDenseCreate(instance, &r), CUBOOL_STATUS_SUCCESS);

    generateTestData(m, t, aval, Condition2{});
    generateTestData(t, n, bval, Condition2{});
    generateTestData(m, n, cval, Condition2{});

    // Transfer input data into input matrices
    EXPECT_EQ(CuBoolMatrixDenseWriteData(instance, a, aval.size(), aval.data()), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseWriteData(instance, b, bval.size(), bval.data()), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseWriteData(instance, c, cval.size(), cval.data()), CUBOOL_STATUS_SUCCESS);

    // Evaluate r = a x b + c
    EXPECT_EQ(CuBoolMatrixDenseMultiplyAdd(instance, r, a, b, c), CUBOOL_STATUS_SUCCESS);

    // Transfer result values into host arrays
    EXPECT_EQ(CuBoolMatrixDenseReadData(instance, r, &resultCount, &resultValues), CUBOOL_STATUS_SUCCESS);

    // Evaluate naive r = a x b + c on the cpu to compare results
    evaluateMultiplyAdd(m, t, n, aval, bval, cval, resultValuesSet);

    // Validate size and compare results
    EXPECT_EQ(resultCount, resultValuesSet.size());

    std::cout << "Got: " << resultCount << " Expected: " << resultValuesSet.size() << std::endl;

    for (CuBoolSize_t k = 0; k < resultCount; k++) {
        EXPECT_EQ(resultValuesSet.find(resultValues[k]) != resultValuesSet.end(), true);
    }

    // Manually deallocate array, returned by read* function
    EXPECT_EQ(CuBoolValuesArrayFree(instance, resultValues), CUBOOL_STATUS_SUCCESS);

    // Release matrices
    EXPECT_EQ(CuBoolMatrixDenseDestroy(instance, a), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseDestroy(instance, b), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseDestroy(instance, c), CUBOOL_STATUS_SUCCESS);
    EXPECT_EQ(CuBoolMatrixDenseDestroy(instance, r), CUBOOL_STATUS_SUCCESS);

    // Destroy instance
    EXPECT_EQ(CuBoolInstanceDestroy(instance), CUBOOL_STATUS_SUCCESS);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
