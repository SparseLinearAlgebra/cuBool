// This file is part of the JetBrains-Research/CFPQ-on-GPGPU project.
// Project link https://github.com/JetBrains-Research/CFPQ-on-GPGPU
// Project is licensed under MIT license.
// License link https://github.com/JetBrains-Research/CFPQ-on-GPGPU/blob/master/LICENSE
//
// This source code files are used to compare the performance of the
// boolean matrix multiplication between two dense matrix multiplications implementations.

#ifndef NAIVE_GPU_SHARED_MATRIX_H
#define NAIVE_GPU_SHARED_MATRIX_H

#include <vector>

namespace naive_gpu_shared {

    class Matrix {
    public:
        explicit Matrix(unsigned int n) {};

        virtual ~Matrix() = default;

        virtual void set_bit(unsigned int row, unsigned col) = 0;

        virtual unsigned int get_bit(unsigned int row, unsigned col) = 0;

        virtual bool add_mul(Matrix *left, Matrix *right) = 0;

        static unsigned int toBoolVector(unsigned int number) {
            return ((unsigned int) 0x1) << number;
        }

        static unsigned char toBoolVector8(unsigned int number) {
            return ((unsigned char) 0x1) << number;
        }
    };

    class MatricesEnv {
    public:
        MatricesEnv() = default;

        virtual ~MatricesEnv() = default;

        virtual void environment_preprocessing(const std::vector<Matrix *> &matrices) {};

        virtual void environment_postprocessing(const std::vector<Matrix *> &matrices) {};
    };

}

#endif //NAIVE_GPU_SHARED_MATRIX_H