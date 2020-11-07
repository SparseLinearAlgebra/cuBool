# Fast boolean semiring matrix multiplication for CFPQ

This `naive-gpu` dense boolean matrix multiplication implementation is part of 
the JetBrains-Research CFPQ-on-GPGPU project, which can found at 
[link](https://github.com/JetBrains-Research/CFPQ-on-GPGPU).

This project is licensed under MIT license. Original license text can be found at
[link](https://github.com/JetBrains-Research/CFPQ-on-GPGPU/blob/master/LICENSE).

This source code files are used to compare the performance of the 
boolean matrix multiplication between two dense matrix multiplications implementations.

## Benchmark

Performance was measured on a multiply-add operation. The results were averaged over 
multiple runs of the calculation. All device / host memory allocations were made 
prior to measurements, so they do not affect the timing in the benchmark.

Device capabilities: Nvidia Quadro T2000 7.7 with 4008128 KiB on-board VRAM.

| Matrix size as N x N  | CuBool (ms) | Naive-Gpu (ms) | Naive-Gpu-Shared (ms) |
|:----------------------|:------------|:---------------|:----------------------|
| 128 x 128             | 0.0142447   | 0.0254926      | 0.018274              |
| 256 x 256             | 0.0179503   | 0.0329124      | 0.0226946             |
| 512 x 512             | 0.0290219   | 0.0472873      | 0.0285077             |
| 1024 x 1024           | 0.167151    | 0.249645       | 0.108185              |
| 2048 x 2048           | 1.25771     | 1.48071        | 0.760774              |
| 4096 x 4096           | 9.70362     | 13.0619        | 6.36253               |
| 8192 x 8192           | 57.5565     | 107.084        | 52.665                |
| 16384 x 16384         | 420.027     | 861.083        | 433.531               |
| 32768 x 32768         | 3357.27     | 6824.81        | 3423.82               |
| 65536 x 65536         | 27137.1     | 54565.9        | 28265.2               |