/**********************************************************************************/
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
/**********************************************************************************/

#include <testing/testing.hpp>
#include <memory>

// Simple kernel to sum float matrices

__global__ void kernelAdd(unsigned int n, const float* a, const float* b, float* c) {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int idx = n * i + j;

    if (i < n * n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Test cuda device support.
TEST(Cuda, BasicExample) {
    const unsigned int N = 128;
    const unsigned int NxN = N * N;
    const unsigned int THREADS_PER_BLOCK = 8;

    float *a, *device_a;
    float *b, *device_b;
    float *c, *device_c;

    a = (float*) malloc(sizeof(float) * NxN);
    b = (float*) malloc(sizeof(float) * NxN);
    c = (float*) malloc(sizeof(float) * NxN);

    for (int i = 0; i < NxN; i++) {
        a[i] = (float) i / 2.0f;
        b[i] = (float) -i / 4.0f;
    }

    cudaMalloc(&device_a, sizeof(float) * NxN);
    cudaMalloc(&device_b, sizeof(float) * NxN);
    cudaMalloc(&device_c, sizeof(float) * NxN);

    cudaMemcpy(device_a, a, sizeof(float) * NxN, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, sizeof(float) * NxN, cudaMemcpyHostToDevice);

    dim3 blocks(N / THREADS_PER_BLOCK, N / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK, THREADS_PER_BLOCK);

    kernelAdd<<<blocks, threads>>>(N, device_a, device_b, device_c);

    cudaDeviceSynchronize();
    cudaMemcpy(c, device_c, sizeof(float) * NxN, cudaMemcpyDeviceToHost);

    for (int i = 0; i < NxN; i++) {
        ASSERT_EQ(c[i], a[i] + b[i]);
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    free(a);
    free(b);
    free(c);
}

CUBOOL_GTEST_MAIN