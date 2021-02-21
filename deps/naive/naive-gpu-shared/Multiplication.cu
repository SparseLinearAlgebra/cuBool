// This file is part of the JetBrains-Research/CFPQ-on-GPGPU project.
// Project link https://github.com/JetBrains-Research/CFPQ-on-GPGPU
// Project is licensed under MIT license.
// License link https://github.com/JetBrains-Research/CFPQ-on-GPGPU/blob/master/LICENSE
//
// This source code files are used to compare the performance of the
// boolean matrix multiplication between two dense matrix multiplications implementations.

#include <iostream>
#include <naive-gpu-shared/Multiplication.h>
#include <naive-gpu-shared/Parameters.h>

#define cuda_handle_error(ans) { gpuAssert((ans), __FILE__, __LINE__); }

namespace naive_gpu_shared {

    using namespace gpu_lib;

    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
        if (code != cudaSuccess) {
            std::cout << "GPU assert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
            if (abort) {
                exit(code);
            }
        }
    }

    __device__ bool matrix_was_changed;

    int gpu_lib::rows(int N) {
        return N / TYPE_SIZE + (N % TYPE_SIZE ? 1 : 0);
    }

    int gpu_lib::cols(int N) {
        return N;
    }

    inline size_t matrix_memsize(int N) {
        return rows(N) * cols(N) * sizeof(TYPE);
    }

    __device__ TYPE row_column_product(TYPE *A, TYPE *B, int cols) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int rows = cols / TYPE_SIZE + (cols % TYPE_SIZE ? 1 : 0);
        int row_start = blockIdx.y * cols;

        __shared__ TYPE A_shared[THREADS_PER_BLOCK];

        TYPE acc = 0;
        TYPE b_el;
        for (TYPE i = 0; i < rows; ++i) {
            if (i == (rows - 1) && x >= cols) {
                return 0;
            }
            if ((i % (THREADS_PER_BLOCK / TYPE_SIZE)) == 0) {
                A_shared[threadIdx.x] = A[row_start + i * TYPE_SIZE + threadIdx.x];
                if (THREADS_PER_BLOCK > 32) {
                    __syncthreads();
                }
            }

            __syncthreads();

            b_el = B[i * cols + x];
#pragma unroll
            for (TYPE b = 0; b < TYPE_SIZE; ++b) {
                if (b_el & 1) {
                    acc |= A_shared[(i % (THREADS_PER_BLOCK / TYPE_SIZE)) * TYPE_SIZE + b];
                }
                b_el >>= 1;
            }
        }

        return acc;
    }

    __device__ void or_value(TYPE *M, TYPE val) {
        TYPE old_value = *M;
        if (old_value != (val | old_value)) {
            matrix_was_changed = true;
            *M = val | old_value;
        }
    }

    __global__ void matrix_product_add(TYPE *A, TYPE *B, TYPE *C, int cols) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int row_start = blockIdx.y * cols;

        TYPE acc = row_column_product(A, B, cols);

        if (acc == 0) return;

        or_value(&C[row_start + x], acc);
    }

    __global__ void matrix_product(TYPE *A, TYPE *B, TYPE *C, int cols) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int row_start = blockIdx.y * cols;

        TYPE acc = row_column_product(A, B, cols);

        if (acc == 0) return;

        C[row_start + x] = acc;
    }

    __global__ void matrix_add_to_left(TYPE *A, TYPE *B, int cols) {
        int index = blockIdx.y * cols + blockIdx.x * blockDim.x + threadIdx.x;

        if ((blockIdx.x * blockDim.x + threadIdx.x) >= cols) return;

        or_value(&A[index], B[index]);
    }

    void gpu_lib::synchronize() {
        cudaDeviceSynchronize();
    }

    void set_value(int N, TYPE *d_M, int val) {
        cuda_handle_error(cudaMemset(d_M, val, matrix_memsize(N)));
    }

    TYPE *gpu_lib::device_matrix_alloc(int N) {
        TYPE *d_M;
        cuda_handle_error(cudaMalloc(reinterpret_cast<void **>(&d_M), matrix_memsize(N)));

        return d_M;
    }

    void gpu_lib::device_matrix_dealloc(TYPE *M) {
        cuda_handle_error(cudaFree(M));
    }

    TYPE *gpu_lib::host_matrix_calloc(int N) {
        TYPE *M;
        cuda_handle_error(cudaMallocHost(reinterpret_cast<void **>(&M), matrix_memsize(N)));
        set_value(N, M, 0);
        return M;
    }

    void gpu_lib::host_matrix_dealloc(TYPE *M) {
        cuda_handle_error(cudaFreeHost(M));
    }

    void gpu_lib::gpu_to_cpu_transfer_async(int N, TYPE *d_M, TYPE *h_M) {
        cuda_handle_error(cudaMemcpyAsync(h_M, d_M, matrix_memsize(N), cudaMemcpyDeviceToHost));
    }

    void gpu_lib::cpu_to_gpu_transfer_async(int N, TYPE *h_M, TYPE *d_M) {
        cuda_handle_error(cudaMemcpyAsync(d_M, h_M, matrix_memsize(N), cudaMemcpyHostToDevice));
    }

    void set_flag() {
        bool flag = false;
        cuda_handle_error(cudaMemcpyToSymbol(matrix_was_changed, &flag, sizeof(bool)));
    }

    bool get_flag() {
        bool flag;
        cuda_handle_error(cudaMemcpyFromSymbol(&flag, matrix_was_changed, sizeof(bool)));

        return flag;
    }

    bool gpu_lib::matrix_product_add_wrapper(TYPE *A, TYPE *B, TYPE *C, int N, TYPE *tmp_matrix) {
        bool safe = (A == C) || (B == C);
        dim3 threads(THREADS_PER_BLOCK);
        dim3 blocks(cols(N) / THREADS_PER_BLOCK + (cols(N) % THREADS_PER_BLOCK ? 1 : 0), rows(N));

        set_flag();
        if (safe) {
            matrix_product <<<blocks, threads>>>(A, B, tmp_matrix, cols(N));
            synchronize();
            cuda_handle_error(cudaGetLastError());
            matrix_add_to_left <<<blocks, threads>>>(C, tmp_matrix, cols(N));
            synchronize();
            cuda_handle_error(cudaGetLastError());
        } else {
            matrix_product_add <<<blocks, threads>>>(A, B, C, cols(N));
            synchronize();
            cuda_handle_error(cudaGetLastError());
        }

        return get_flag();
    }

}
