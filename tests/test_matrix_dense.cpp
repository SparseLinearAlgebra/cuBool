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

static void TestMsgFun(CuBoolStatus error, const char* message, void* _) {
    std::cout << "CuBool: " << message << std::endl;
}

static CuBoolCpuPtr_t TestAllocateFun(CuBoolSize_t size, void* _) {
    CuBoolCpuPtr_t ptr = malloc(size);
    std::cout << "Cubool: Allocate: " << size << " " << ptr << std::endl;
    return ptr;
}

static void TestDeallocateFun(CuBoolCpuPtr_t ptr, void* _) {
    std::cout << "Cubool: Deallocate: " << ptr << std::endl;
    free(ptr);
}

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


TEST(MatrixDense, CreateDestroy) {
    CuBoolStatus error;
    CuBoolInstance instance = nullptr;
    CuBoolMatrixDense matrix = nullptr;

    CuBoolInstanceDesc instanceDesc{};
    instanceDesc.memoryType = CuBoolGpuMemoryType::CUBOOL_GPU_MEMORY_TYPE_GENERIC;
    instanceDesc.errorCallback.userData = nullptr;
    instanceDesc.errorCallback.msgFun = TestMsgFun;
    instanceDesc.allocationCallback.userData = nullptr;
    instanceDesc.allocationCallback.allocateFun = TestAllocateFun;
    instanceDesc.allocationCallback.deallocateFun = TestDeallocateFun;

    error = CuBoolCreateInstance(&instanceDesc, &instance);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolCreateMatrixDense(instance, &matrix);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolDestroyMatrixDense(instance, matrix);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolDestroyInstance(instance);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixDense, Resize) {
    CuBoolStatus error;
    CuBoolInstance instance = nullptr;
    CuBoolMatrixDense matrix = nullptr;
    CuBoolSize_t rows = 1024, columns = 1024;

    CuBoolInstanceDesc instanceDesc{};
    instanceDesc.memoryType = CuBoolGpuMemoryType::CUBOOL_GPU_MEMORY_TYPE_GENERIC;
    instanceDesc.errorCallback.userData = nullptr;
    instanceDesc.errorCallback.msgFun = TestMsgFun;
    instanceDesc.allocationCallback.userData = nullptr;
    instanceDesc.allocationCallback.allocateFun = TestAllocateFun;
    instanceDesc.allocationCallback.deallocateFun = TestDeallocateFun;

    error = CuBoolCreateInstance(&instanceDesc, &instance);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolCreateMatrixDense(instance, &matrix);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolMatrixDenseResize(instance, matrix, rows, columns);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolDestroyMatrixDense(instance, matrix);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolDestroyInstance(instance);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixDense, Filling) {
    CuBoolStatus error;
    CuBoolInstance instance = nullptr;
    CuBoolMatrixDense matrix = nullptr;
    CuBoolSize_t rows = 1024, columns = 1024;

    CuBoolInstanceDesc instanceDesc{};
    instanceDesc.memoryType = CuBoolGpuMemoryType::CUBOOL_GPU_MEMORY_TYPE_GENERIC;
    instanceDesc.errorCallback.userData = nullptr;
    instanceDesc.errorCallback.msgFun = TestMsgFun;
    instanceDesc.allocationCallback.userData = nullptr;
    instanceDesc.allocationCallback.allocateFun = TestAllocateFun;
    instanceDesc.allocationCallback.deallocateFun = TestDeallocateFun;

    error = CuBoolCreateInstance(&instanceDesc, &instance);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolCreateMatrixDense(instance, &matrix);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolMatrixDenseResize(instance, matrix, rows, columns);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    std::vector<CuBoolPair> values;

    for (CuBoolSize_t i = 0; i < rows; i++) {
        for (CuBoolSize_t j = 0; j < columns; j++) {
            // is i and j power of two or 0
            if (((i - 1) & i) == 0 && ((j - 1) & j) == 0) {
                values.push_back(CuBoolPair{ i, j });
            }
        }
    }

    error = CuBoolDenseMatrixWriteData(instance, matrix, values.size(), values.data());
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    CuBoolSize_t readCount;
    CuBoolPair* readValues;

    error = CuBoolDenseMatrixReadData(instance, matrix, &readCount, &readValues);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    EXPECT_EQ(readCount, values.size());

    if (readCount == values.size()) {
        std::unordered_set<CuBoolPair, CuBoolPairHash, CuBoolPairEq> a;
        for (auto p: values) {
            a.emplace(p);
        }

        for (CuBoolSize_t k = 0; k < readCount; k++) {
            EXPECT_EQ(a.find(readValues[k]) != a.end(), true);
        }
    }

    error = CuBoolReleaseValuesArray(instance, readValues);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolDestroyMatrixDense(instance, matrix);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    error = CuBoolDestroyInstance(instance);
    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);
}

TEST(MatrixDense, MultiplyAdd) {

}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
