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
| 128                   | 0.020615      | 0.038519          |
| 256                   | 0.016495      | 0.037075          |
| 512                   | 0.046521      | 0.043692          |
| 1024                  | 0.292983      | 0.253172          |
| 2048                  | 2.23964       | 1.4917            |
| 4096                  | 17.6718       | 13.0728           |
| 8192                  | 130.779       | 107.093           |
| 16384                 | 824.735       | 901.763           |
| 32768                 | 6193.05       | 6858.15           |
| 65536                 | 49873.4       | 54583.8           |

> The benchmark is performed in the *naive-way* and it is not intended 
> to perform precise comparison test.