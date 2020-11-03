# Fast boolean semiring matrix multiplication for CFPQ

This `naive-gpu` dense boolean matrix multiplication implementation is part of 
the JetBrains-Research CFPQ-on-GPGPU project, which can found at 
[link](https://github.com/JetBrains-Research/CFPQ-on-GPGPU).

This project is licensed under MIT license. Original license text can be found at
[link](https://github.com/JetBrains-Research/CFPQ-on-GPGPU/blob/master/LICENSE).

This source code files are used to compare the performance of the 
boolean matrix multiplication between two dense matrix multiplications implementations.

## Benchmark

The performance was measured for multiply-add operation. 
All the device/host memory allocations were performed prior the
benchmark test, therefore its impact is out of the measured time scope.

Device capabilities: Nvidia Quadro T2000 7.7 with 4008128 KiB on-board VRAM.

| Matrix size as N x N  | CuBool (ms)   | Naive-Gpu (ms)    |
|:----------------------|:--------------|:------------------|
| 128 x 128             | 0.024949      | 0.038519          |
| 256 x 256             | 0.020462      | 0.037075          |
| 512 x 512             | 0.031247      | 0.043692          |
| 1024 x 1024           | 0.170258      | 0.253172          |
| 2048 x 2048           | 1.24371       | 1.47968           |
| 4096 x 4096           | 10.0039       | 13.0728           |
| 8192 x 8192           | 77.7092       | 107.093           |
| 16384 x 16384         | 498.655       | 901.763           |
| 32768 x 32768         | 3421.29       | 6858.15           |
| 65536 x 65536         | 27157.7       | 54583.8           |

> The benchmark is performed in the *naive-way* and it is not intended 
> to perform precise comparison test.