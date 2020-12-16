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

// Query library version info
TEST(CuBoolVersion, Query) {
    int major;
    int minor;
    int version;

    CuBool_Version_Get(&major, &minor, &version);

    std::cout << "Major: " << major << std::endl;
    std::cout << "Minor: " << minor << std::endl;
    std::cout << "Version: " << version << std::endl;

    EXPECT_NE(version, 0);
}

// Test cubool library instance creation and destruction
TEST(CuBoolInstance, Setup) {
    CuBoolStatus error;
    CuBoolInstance instance = nullptr;

    CuBoolInstanceDesc instanceDesc{};
    instanceDesc.memoryType = CuBoolGpuMemoryType::CUBOOL_GPU_MEMORY_TYPE_GENERIC;
    instanceDesc.errorCallback.userData = nullptr;
    instanceDesc.errorCallback.msgFun = testing::details::testMsgFun;
    instanceDesc.allocationCallback.userData = nullptr;
    instanceDesc.allocationCallback.allocateFun = testing::details::testAllocateFun;
    instanceDesc.allocationCallback.deallocateFun = testing::details::testDeallocateFun;

    error = CuBool_Instance_New(&instanceDesc, &instance);

    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);
    EXPECT_NE(instance, nullptr);

    error = CuBool_Instance_Free(instance);

    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    instance = nullptr;
}

// Test cubool library instance creation and destruction
TEST(CuBoolInstance, SetupExt) {
    CuBoolStatus error;
    CuBoolInstance instance = nullptr;

    CuBoolInstanceDescExt instanceDesc{};
    instanceDesc.memoryType = CuBoolGpuMemoryType::CUBOOL_GPU_MEMORY_TYPE_GENERIC;

    error = CuBool_Instance_NewExt(&instanceDesc, &instance);

    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);
    EXPECT_NE(instance, nullptr);

    error = CuBool_Instance_Free(instance);

    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    instance = nullptr;
}

/**
 * Performs transitive closure for directed graph
 *
 * @param Inst Library instance, which provides context for operations
 * @param A Adjacency matrix of the graph
 * @param T Reference to the handle where to allocate and store result
 *
 * @return Status on this operation
 */
CuBoolStatus TransitiveClosure(CuBoolInstance Inst, CuBoolMatrix A, CuBoolMatrix* T) {
    CuBool_Matrix_Duplicate(Inst, A, T);         /** Create result matrix and copy initial values */

    CuBoolSize_t total = 0;
    CuBoolSize_t current;
    CuBool_Matrix_Nvals(Inst, *T, &current);     /** Query current number on non-zero elements */

    while (current != total) {                   /** Loop while values are added */
        total = current;
        CuBool_MxM(Inst, *T, *T, *T);            /** T += T * T */
        CuBool_Matrix_Nvals(Inst, *T, &current);
    }

    return CUBOOL_STATUS_SUCCESS;
}

TEST(CuBoolInstance, Example) {
    CuBoolInstanceDescExt descExt{};
    CuBoolInstance Inst;
    CuBoolMatrix A;
    CuBoolMatrix T;

    CuBoolIndex_t n = 100;

    descExt.memoryType = CUBOOL_GPU_MEMORY_TYPE_GENERIC;

    CuBool_Instance_NewExt(&descExt, &Inst);
    CuBool_Matrix_New(Inst, &A, n, n);

    testing::Matrix ta = testing::Matrix::generateSparse(n , n, 0.2);

    CuBool_Matrix_Build(Inst, A, ta.mRowsIndex.data(), ta.mColsIndex.data(), ta.mNvals);

    TransitiveClosure(Inst, A, &T);

    testing::Matrix tr = ta;
    size_t total = 0;

    do {
        total = tr.mNvals;

        testing::MatrixMultiplyAddFunctor functor;
        tr = std::move(functor(tr, tr, tr));
    }
    while (tr.mNvals != total);

    ASSERT_TRUE(tr.areEqual(T, Inst));

    CuBool_Matrix_Free(Inst, A);
    CuBool_Matrix_Free(Inst, T);
    CuBool_Instance_Free(Inst);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}