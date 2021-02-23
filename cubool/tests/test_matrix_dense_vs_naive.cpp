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
#include <chrono>

#include <testing/testing.hpp>
#include <naive-gpu/GpuMatrix.h> // Naive gpu implementation for dense boolean matrix multiplication
#include <naive-gpu-shared/GpuMatrix.h> // Naive gpu with small shared mem optimization

static size_t iterations = 1;
static size_t sizes[] = { 128, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2, 8192 * 4, 8192 * 8 };

TEST(Benchmark, NaiveGpuDenseMatrix) {
    using namespace naive_gpu;

    gpuMatrix *a, *b, *c;

    for (auto s: sizes) {
        int n = (int) s;

        std::vector<testing::Pair> aval;
        std::vector<testing::Pair> bval;
        std::vector<testing::Pair> cval;

        testing::generateTestData(n, n, aval, testing::Condition2{});
        testing::generateTestData(n, n, bval, testing::Condition2{});
        testing::generateTestData(n, n, cval, testing::Condition2{});

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

        double executionTimeMs = 0;

        for (size_t i = 0; i < iterations; i++) {
            using namespace std::chrono;

            gpu_lib::synchronize();
            auto start = high_resolution_clock::now();

            c->add_mul(a, b);

            gpu_lib::synchronize();
            auto end = high_resolution_clock::now();

            executionTimeMs += (double)duration_cast<nanoseconds>(end - start).count() / 1.0e6;
        }

        std::cout << "Operation: Mult-Add [N=" << n << "]: " << executionTimeMs / (double)iterations << " ms" << std::endl;

        a->deallocate_device_matrix();
        b->deallocate_device_matrix();
        c->deallocate_device_matrix();

        delete a;
        delete b;
        delete c;
    }
}

TEST(Benchmark, NaiveGpuSharedDenseMatrix) {
    using namespace naive_gpu_shared;

    gpuMatrix *a, *b, *c;

    for (auto s: sizes) {
        int n = (int) s;

        std::vector<testing::Pair> aval;
        std::vector<testing::Pair> bval;
        std::vector<testing::Pair> cval;

        testing::generateTestData(n, n, aval, testing::Condition2{});
        testing::generateTestData(n, n, bval, testing::Condition2{});
        testing::generateTestData(n, n, cval, testing::Condition2{});

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

        double executionTimeMs = 0;

        for (size_t i = 0; i < iterations; i++) {
            using namespace std::chrono;

            gpu_lib::synchronize();
            auto start = high_resolution_clock::now();

            c->add_mul(a, b);

            gpu_lib::synchronize();
            auto end = high_resolution_clock::now();

            executionTimeMs += (double)duration_cast<nanoseconds>(end - start).count() / 1.0e6;
        }

        std::cout << "Operation: Mult-Add [N=" << n << "]: " << executionTimeMs / (double)iterations << " ms" << std::endl;

        a->deallocate_device_matrix();
        b->deallocate_device_matrix();
        c->deallocate_device_matrix();

        delete a;
        delete b;
        delete c;
    }
}

CUBOOL_GTEST_MAIN