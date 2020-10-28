// This file is part of the JetBrains-Research/CFPQ-on-GPGPU project.
// Project link https://github.com/JetBrains-Research/CFPQ-on-GPGPU
// Project is licensed under MIT license.
// License link https://github.com/JetBrains-Research/CFPQ-on-GPGPU/blob/master/LICENSE
//
// This source code files are used to compare the performance of the
// boolean matrix multiplication between two dense matrix multiplications implementations.

#ifndef GPU_MATRIX_H
#define GPU_MATRIX_H

#include "Multiplication.h"
#include "Parameters.h"
#include "Matrix.h"

class gpuMatrix : public Matrix {
    TYPE *matrix_host = nullptr;
    TYPE *matrix_device = nullptr;
    
public:
    static int N;
    static TYPE *tmp_matrix;
    bool changed_prev = false;
    bool changed = false;

    explicit gpuMatrix(unsigned n) : Matrix(n) { matrix_host = gpu_lib::host_matrix_calloc(N); };
    ~gpuMatrix() override { gpu_lib::host_matrix_dealloc(matrix_host); };

    void set_bit(unsigned int row, unsigned int col) override;
    unsigned int get_bit(unsigned int row, unsigned int col) override;

    bool add_mul(Matrix *left, Matrix *right) override;

    void allocate_device_matrix();
    void deallocate_device_matrix();

    void transfer_to_gpu();
    void transfer_from_gpu();

    static void set_N(int n);

    static void allocate_tmp_matrix();
    static void deallocate_tmp_matrix();
};

class gpuMatricesEnv : public MatricesEnv {
public:
    gpuMatricesEnv() = default;
    ~gpuMatricesEnv() override = default;

    void environment_preprocessing(const std::vector<Matrix *> &matrices) override;
    void environment_postprocessing(const std::vector<Matrix *> &matrices) override;
};

#endif //GPU_MATRIX_H