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
#include <GpuMatrix.h> // Naive gpu implementation for dense boolean matrix multiplication

#include <chrono>
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
    return malloc(size);
}

static void testDeallocateFun(CuBoolCpuPtr_t ptr, void* _) {
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

static void setupInstanceDesc(CuBoolInstanceDesc& instanceDesc) {
    instanceDesc.memoryType = CuBoolGpuMemoryType::CUBOOL_GPU_MEMORY_TYPE_GENERIC;
    instanceDesc.errorCallback.userData = nullptr;
    instanceDesc.errorCallback.msgFun = testMsgFun;
    instanceDesc.allocationCallback.userData = nullptr;
    instanceDesc.allocationCallback.allocateFun = testAllocateFun;
    instanceDesc.allocationCallback.deallocateFun = testDeallocateFun;
}

TEST(Benchmanrk, CuboolDenseMatrix) {
    CuBoolInstance instance;
    CuBoolMatrixDense a, b, c, r;
    CuBoolSize_t m, t, n;

    CuBoolSize_t sizes[] = { 128, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2, 8192 * 4, 8192 * 8 };

    CuBoolInstanceDesc instanceDesc;
    setupInstanceDesc(instanceDesc);

    CuBoolInstanceCreate(&instanceDesc, &instance);

    CuBoolMatrixDenseCreate(instance, &a);
    CuBoolMatrixDenseCreate(instance, &b);
    CuBoolMatrixDenseCreate(instance, &c);
    CuBoolMatrixDenseCreate(instance, &r);

    for (auto s: sizes) {
        m = t = n = s;

        std::vector<CuBoolPair> aval;
        std::vector<CuBoolPair> bval;
        std::vector<CuBoolPair> cval;

        generateTestData(m, t, aval, Condition2());
        generateTestData(t, n, bval, Condition2());
        generateTestData(m, n, cval, Condition2());

        CuBoolMatrixDenseResize(instance, a, m, t);
        CuBoolMatrixDenseResize(instance, b, t, n);
        CuBoolMatrixDenseResize(instance, c, m, n);
        CuBoolMatrixDenseResize(instance, r, m, n); // resize, since we do not want to measure the speed of cuda allocator

        CuBoolMatrixDenseWriteData(instance, a, aval.size(), aval.data());
        CuBoolMatrixDenseWriteData(instance, b, bval.size(), bval.data());
        CuBoolMatrixDenseWriteData(instance, c, cval.size(), cval.data());

        {
            using namespace std::chrono;

            CuBoolSyncHostDevice(instance);
            auto start = high_resolution_clock::now();

            CuBoolMatrixDenseMultiplyAdd(instance, r, a, b, c);

            CuBoolSyncHostDevice(instance);
            auto end = high_resolution_clock::now();

            std::cout << "Operation: Mult-Add [N=" << s << "]: " << (double)duration_cast<nanoseconds>(end - start).count() / 1.0e6 << "ms" << std::endl;
        }
    }

    CuBoolMatrixDenseDestroy(instance, a);
    CuBoolMatrixDenseDestroy(instance, b);
    CuBoolMatrixDenseDestroy(instance, c);
    CuBoolMatrixDenseDestroy(instance, r);

    CuBoolInstanceDestroy(instance);
}

TEST(Benchmark, NaiveGpuDenseMatrix) {
    gpuMatrix *a, *b, *c;

    int sizes[] = { 128, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2, 8192 * 4, 8192 * 8 };

    for (auto s: sizes) {
        int n = s;

        std::vector<CuBoolPair> aval;
        std::vector<CuBoolPair> bval;
        std::vector<CuBoolPair> cval;

        generateTestData(n, n, aval, Condition2());
        generateTestData(n, n, bval, Condition2());
        generateTestData(n, n, cval, Condition2());

        gpuMatrix::set_N(n);

        a = new gpuMatrix(n);
        b = new gpuMatrix(n);
        c = new gpuMatrix(n);

        for (auto p: aval)
            a->set_bit(p.i, p.j);
        for (auto p: bval)
            b->set_bit(p.i, p.j);
        for (auto p: cval)
            c->set_bit(p.i, p.j);

        a->allocate_device_matrix();
        a->transfer_to_gpu();

        b->allocate_device_matrix();
        b->transfer_to_gpu();

        c->allocate_device_matrix();
        c->transfer_to_gpu();

        {
            using namespace std::chrono;

            gpu_lib::synchronize();
            auto start = high_resolution_clock::now();

            c->add_mul(a, b);

            gpu_lib::synchronize();
            auto end = high_resolution_clock::now();

            std::cout << "Operation: Mult-Add [N=" << s << "]: " << (double)duration_cast<nanoseconds>(end - start).count() / 1.0e6 << "ms" << std::endl;
        }

        a->deallocate_device_matrix();
        b->deallocate_device_matrix();
        c->deallocate_device_matrix();

        delete a;
        delete b;
        delete c;
    }
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}