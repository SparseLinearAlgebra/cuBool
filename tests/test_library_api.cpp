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
    instanceDesc.errorCallback.msgFun = testMsgFun;
    instanceDesc.allocationCallback.userData = nullptr;
    instanceDesc.allocationCallback.allocateFun = testAllocateFun;
    instanceDesc.allocationCallback.deallocateFun = testDeallocateFun;

    error = CuBool_Instance_New(&instanceDesc, &instance);

    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);
    EXPECT_NE(instance, nullptr);

    error = CuBool_Instance_Free(instance);

    EXPECT_EQ(error, CUBOOL_STATUS_SUCCESS);

    instance = nullptr;
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}