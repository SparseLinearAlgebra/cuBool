# NSPARSE-UNIFIED-MEMORY

Sparse boolean matrix multiplication and addition implementation
for matrix in compressed-sparse row (csr) format, which employs
cuda unified memory allocator for allocating large gpu resources.

Original source code is hosted [here](https://github.com/YaccConstructor/RedisGraph).
For more info view branches `CFPQ-gpu` and `CFPQ-gpu-um`, where the source code
is stored at path `deps/cfpq/algorithms/cuda/nsparse`.

## Useful links

- High-performance and Memory-saving
  Sparse General Matrix-Matrix Multiplication for NVIDIA Pascal GPU 
  [paper](https://ieeexplore.ieee.org/document/8025284)
- GPU Merge Path - A GPU Merging Algorithm 
  [paper](https://www.researchgate.net/publication/254462662_GPU_merge_path_a_GPU_merging_algorithm)
- Context-Free Path Querying with Single-Path
  Semantics by Matrix Multiplication 
  [paper](https://www.researchgate.net/publication/342164347_Context-Free_Path_Querying_with_Single-Path_Semantics_by_Matrix_Multiplication)
  
## Also

The author of the original implementation is Artyom Khoroshev 
(Github: [profile](https://github.com/akhoroshev)).