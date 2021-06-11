# Changelog

All notable changes to this project are documented in this file.
Changes format: version name and date, minor notes, new features, fixes, what removed.

Release are available at [link](https://github.com/JetBrains-Research/cuBool/releases). 

## v1.1.0 - May 8, 2021

### Summary

Introduce sparse vector of boolean values support for Cuda and CPU computations.
Added new primitive type as well as supported common operations,
which involve matrix-vector operations and vector manipulation.
Added new vector C API, exposed vector primitive into python-package.

#### New features

- Sparse vector support
- Vector implementation for Cuda backend
- Vector implementation for Sequential backend
- Vector creation (empty, from data, with random data)
- Matrix-vector operations (matrix-vector and vector-matrix multiplication)
- Vector-vector operations (element-wise addition)
- Vector data extraction (as list of indices)
- Vector syntax sugar (pretty string printing, slicing, iterating through non-zero indices)
- Matrix operations (extract row or matrix column as sparse vector, reduce matrix (optionally transposed) to vector)

#### Deprecated

- Matrix reduce to matrix (compatibility feature)

## v1.0.0 - Apr 2, 2021

### Summary 

cuBool is a sparse linear Boolean algebra for Nvidia Cuda computations. 
Library provides C compatible API, the core of the library is written on C++ with CUDA C/C++ API and 
Thrust for Cuda related computations. Library also supports CPU backend for non-Cuda devices. 
pycubool Python-package is shipped with the library source code and it provides high-level safe 
and efficient access to the library within Python runtime.

#### New features

- Cuda backend
- Sequential (CPU) backend
- Sparse matrix support
- Matrix creation (empty, from data, with random data)
- Matrix-matrix operations (multiplication, element-wise addition, kronecker product)
- Matrix operations (equality, transpose, reduce to vector, extract sub-matrix)
- Matrix data extraction (as lists, as list of pairs)
- Matrix syntax sugar (pretty string printing, slicing, iterating through non-zero values)
- IO (import/export matrix from/to .mtx file format)
- GraphViz (export single matrix or set of matrices as a graph with custom color and label settings)
- Debug (matrix string debug markers, logging, operations time profiling)
