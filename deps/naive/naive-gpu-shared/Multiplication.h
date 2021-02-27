// This file is part of the JetBrains-Research/CFPQ-on-GPGPU project.
// Project link https://github.com/JetBrains-Research/CFPQ-on-GPGPU
// Project is licensed under MIT license.
// License link https://github.com/JetBrains-Research/CFPQ-on-GPGPU/blob/master/LICENSE
//
// This source code files are used to compare the performance of the
// boolean matrix multiplication between two dense matrix multiplications implementations.

#ifndef NAIVE_GPU_SHARED_MULTIPLICATION_H
#define NAIVE_GPU_SHARED_MULTIPLICATION_H

#include <naive-gpu-shared/Parameters.h>

namespace naive_gpu_shared {

    namespace gpu_lib {
        int rows(int N);
        int cols(int N);

        void synchronize();

        TYPE *device_matrix_alloc(int N);
        void device_matrix_dealloc(TYPE *M);

        TYPE *host_matrix_calloc(int N);
        void host_matrix_dealloc(TYPE *M);

        void gpu_to_cpu_transfer_async(int N, TYPE *d_M, TYPE *h_M);
        void cpu_to_gpu_transfer_async(int N, TYPE *h_M, TYPE *d_M);

        bool matrix_product_add_wrapper(TYPE *A, TYPE *B, TYPE *C, int N, TYPE *tmp);
    }

}

#endif //NAIVE_GPU_SHARED_MULTIPLICATION_H