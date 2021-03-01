# pycubool

[![JB Research](https://jb.gg/badges/research-flat-square.svg)](https://research.jetbrains.org/)
[![Ubuntu](https://github.com/JetBrains-Research/cuBool/workflows/Ubuntu/badge.svg?branch=master)](https://github.com/JetBrains-Research/cuBool/actions)
[![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/JetBrains-Research/cuBool/blob/master/LICENSE)

`pycubool` is a python wrapper for [cuBool](https://github.com/JetBrains-Research/cuBool) library.

**cuBool** is a linear Boolean algebra library primitives and operations for 
work with sparse matrices written on the NVIDIA CUDA platform. The primary 
goal of the library is implementation, testing and profiling algorithms for
solving *formal-language-constrained problems*, such as *context-free* 
and *regular* path queries with various semantics for graph databases.
The library provides C-compatible API, written in the GraphBLAS style,
as well as python high-level wrapper with automated resources management and fancy syntax sugar.

**The primary library primitive** is a sparse boolean matrix. The library provides 
the most popular operations for matrix manipulation, such as construction from
values, transpose, sub-matrix extraction, matrix-to-vector reduce, matrix-matrix
element-wise addition, matrix-matrix multiplication and Kronecker product.  

**As a fallback** library provides sequential backend for mentioned above operations
for computations on CPU side only. This backend is selected automatically
if Cuda compatible device is not presented in the system. This can be quite handy for 
prototyping algorithms on a local computer for later running on a powerful server.  

## Example

The following Python code snippet demonstrates, how the `pycubool`
wrapper can be used to compute the transitive closure problem for the
directed graph within python environment:

```python
import pycubool

def transitive_closure(a: pycubool.Matrix):
    """
    Evaluates transitive closure for the provided
    adjacency matrix of the graph.

    :param a: Adjacency matrix of the graph
    :return: The transitive closure adjacency matrix
    """

    t = a.dup()                           # Duplicate matrix where to store result
    total = 0                             # Current number of values

    while total != t.nvals:
        total = t.nvals
        t.mxm(t, out=t, accumulate=True)  # t += t * t

    return t
```

## Directory structure

```
pycubool
├── docs - documents, text files and various helpful stuff
├── demo - examples and basic programs
├── pycubool - package with `cuBool` C API wrapper
└── tests - tests for python wrapper
```

## License

This project is licensed under MIT License. License text can be found in the 
[license file](https://github.com/JetBrains-Research/cuBool/blob/master/python/LICENSE.md).

