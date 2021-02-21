// This file is part of the JetBrains-Research/CFPQ-on-GPGPU project.
// Project link https://github.com/JetBrains-Research/CFPQ-on-GPGPU
// Project is licensed under MIT license.
// License link https://github.com/JetBrains-Research/CFPQ-on-GPGPU/blob/master/LICENSE
//
// This source code files are used to compare the performance of the
// boolean matrix multiplication between two dense matrix multiplications implementations.

#ifndef NAIVE_GPU_SHARED_PARAMETERS_H
#define NAIVE_GPU_SHARED_PARAMETERS_H

#include <stdint.h>

#define TYPE uint32_t
#define TYPE_SIZE (sizeof(TYPE) * 8)
#define THREADS_PER_BLOCK 32

#endif //NAIVE_GPU_SHARED_PARAMETERS_H
