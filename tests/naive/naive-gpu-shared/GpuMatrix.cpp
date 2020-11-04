// This file is part of the JetBrains-Research/CFPQ-on-GPGPU project.
// Project link https://github.com/JetBrains-Research/CFPQ-on-GPGPU
// Project is licensed under MIT license.
// License link https://github.com/JetBrains-Research/CFPQ-on-GPGPU/blob/master/LICENSE
//
// This source code files are used to compare the performance of the
// boolean matrix multiplication between two dense matrix multiplications implementations.

#include <naive-gpu-shared/Multiplication.h>
#include <naive-gpu-shared/Matrix.h>
#include <naive-gpu-shared/GpuMatrix.h>
#include <naive-gpu-shared/Parameters.h>

namespace naive_gpu_shared {

    int gpuMatrix::N;

    TYPE *gpuMatrix::tmp_matrix;

    void gpuMatrix::set_N(int n) {
        N = n;
    }

    void gpuMatrix::set_bit(unsigned int row, unsigned int col) {
        matrix_host[(row / TYPE_SIZE) * gpu_lib::cols(N) + col]
                |= 1U << (row % TYPE_SIZE);
    }

    unsigned int gpuMatrix::get_bit(unsigned int row, unsigned int col) {
        return (matrix_host[(row / TYPE_SIZE) * gpu_lib::cols(N) + col] &
                (1U << (row % TYPE_SIZE))) ? 1 : 0;
    }

    bool gpuMatrix::add_mul(Matrix *left, Matrix *right) {
        auto *A = dynamic_cast<gpuMatrix *>(left);
        auto *B = dynamic_cast<gpuMatrix *>(right);
        return gpu_lib::matrix_product_add_wrapper(A->matrix_device, B->matrix_device, this->matrix_device, N,
                                                   tmp_matrix);
    }

    void gpuMatrix::allocate_device_matrix() {
        matrix_device = gpu_lib::device_matrix_alloc(N);
    }

    void gpuMatrix::deallocate_device_matrix() {
        gpu_lib::device_matrix_dealloc(matrix_device);
    }

    void gpuMatrix::allocate_tmp_matrix() {
        gpuMatrix::tmp_matrix = gpu_lib::device_matrix_alloc(N);
    }

    void gpuMatrix::deallocate_tmp_matrix() {
        gpu_lib::device_matrix_dealloc(tmp_matrix);
    }

    void gpuMatrix::transfer_to_gpu() {
        gpu_lib::cpu_to_gpu_transfer_async(N, matrix_host, matrix_device);
    }

    void gpuMatrix::transfer_from_gpu() {
        gpu_lib::gpu_to_cpu_transfer_async(N, matrix_device, matrix_host);
    }


    void gpuMatricesEnv::environment_preprocessing(const std::vector<Matrix *> &matrices) {
        for (auto *matrix: matrices) {
            auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
            gpu_matrix->allocate_device_matrix();
            gpu_matrix->transfer_to_gpu();
        }
        gpuMatrix::allocate_tmp_matrix();
        gpu_lib::synchronize();
    };

    void gpuMatricesEnv::environment_postprocessing(const std::vector<Matrix *> &matrices) {
        gpuMatrix::deallocate_tmp_matrix();
        for (auto *matrix: matrices) {
            auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
            gpu_matrix->transfer_from_gpu();
        }
        gpu_lib::synchronize();
        for (auto *matrix: matrices) {
            auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
            gpu_matrix->deallocate_device_matrix();
        }
    };

}
