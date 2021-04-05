![Project logo](https://github.com/JetBrains-Research/cuBool/blob/master/docs/logo/cubool_logo_trsp.png?raw=true)

# cuBool

[![JB Research](https://jb.gg/badges/research-flat-square.svg)](https://research.jetbrains.org/)
[![Ubuntu](https://github.com/JetBrains-Research/cuBool/workflows/Ubuntu/badge.svg?branch=master)](https://github.com/JetBrains-Research/cuBool/actions)
[![License](https://img.shields.io/badge/license-MIT-orange)](https://github.com/JetBrains-Research/cuBool/blob/master/LICENSE)
[![Package](https://img.shields.io/badge/pypi%20package-1.0.0-%233776ab)](https://pypi.org/project/pycubool/)

**cuBool** is a linear Boolean algebra library primitives and operations for 
work with sparse matrices written on the NVIDIA CUDA platform. The primary 
goal of the library is implementation, testing and profiling algorithms for
solving *formal-language-constrained problems*, such as *context-free* 
and *regular* path queries with various semantics for graph databases.
The library provides C-compatible API, written in the GraphBLAS style.

**The library** is shipped with python package **pycubool** - wrapper for
cuBool library C API. This package exports library features and primitives 
in high-level format with automated resources management and fancy syntax sugar.

**The primary library primitive** is a sparse boolean matrix. The library provides 
the most popular operations for matrix manipulation, such as construction from
values, transpose, sub-matrix extraction, matrix-to-vector reduce, matrix-matrix
element-wise addition, matrix-matrix multiplication and Kronecker product.  

**As a fallback** library provides sequential backend for mentioned above operations
for computations on CPU side only. This backend is selected automatically
if Cuda compatible device is not presented in the system. This can be quite handy for 
prototyping algorithms on a local computer for later running on a powerful server.  

**PyPI package web page** is following [link](https://pypi.org/project/pycubool/).

### Features

- [X] Library C interface
- [X] Library setup
- [X] Sparse matrix
- [X] Sparse matrix multiplication
- [X] Sparse matrix element-wise addition
- [X] Sparse matrix kronecker
- [X] Sparse matrix transpose
- [X] Sparse matrix extract sub-matrix
- [X] Sparse matrix reduce
- [X] Sparse matrix slicing
- [X] Matrix cached filling
- [X] Sequential fallback backend for CPU
- [X] Device capabilities query
- [X] IO matrix loading from mtx file
- [X] IO matrix saving into mtx file
- [X] IO matrix saving into gviz format
- [X] IO user-defined file logging
- [X] Wrapper for Python API
- [X] Wrapper syntax sugar
- [X] Tests for Python wrapper
- [X] Pip package
- [X] Code examples
- [X] User guide
- [X] Unit Tests collection
- [X] Publish built artifacts and shared libs
- [X] Publish stable source code archives

## Simple example

Create sparse matrices, compute matrix-matrix product and print the result to the output:

```python
import pycubool as cb

a = cb.Matrix.empty(shape=(2, 3))
a[0, 0] = True
a[1, 2] = True

b = cb.Matrix.empty(shape=(3, 4))
b[0, 1] = True
b[0, 2] = True
b[1, 3] = True
b[2, 1] = True

print(a, b, a.mxm(b), sep="\n")
```

## Installation

If you are running **Linux based** OS (tested on Ubuntu 20.04) you can download the official
PyPI **pycubool** python package, which includes compiled library source code
with Cuda and Sequential computations support. Installation process 
requires only `python3` to be installed on your machine. Python can be installed 
as follows [link](https://phoenixnap.com/kb/how-to-install-python-3-ubuntu).

If all requirements are satisfied, run the following command to install PyPI package:

```shell script
$ python3 -m pip install pycubool
``` 

## Getting Started

This section gives instructions to build the library from sources.
These steps are required if you want to build library for your specific platform with custom build settings.

### Requirements

- Linux based (tested on Ubuntu 20.04)
- CMake Version 3.15 or higher
- CUDA Compatible GPU device (to run Cuda computations)
- GCC Compiler 
- NVIDIA CUDA toolkit (to build Cuda backend)
- Python 3 (for `pycubool` library)
- Git (to get source code)

### Cuda & Compiler Setup

> Skip this section if you want to build library with only sequential backend
> without cuda backend support.

Before the CUDA setup process, validate your system NVIDIA driver with `nvidia-smi`
command. if it is need, install required driver via `ubuntu-drivers devices` and 
`apt install <driver>` commands respectively.

The following commands grubs the required GCC compilers for the CC and CXX compiling 
respectively. CUDA toolkit, shipped in the default Ubuntu package manager, has version 
number 10 and supports only GCC of the version 8.4 or less.  

```shell script
$ sudo apt update
$ sudo apt install gcc-8 g++-8
$ sudo apt install nvidia-cuda-toolkit
$ sudo apt install nvidia-cuda-dev 
$ nvcc --version
```

If everything successfully installed, the last version command will output 
something like this:

```shell script
$ nvcc: NVIDIA (R) Cuda compiler driver
$ Copyright (c) 2005-2019 NVIDIA Corporation
$ Built on Sun_Jul_28_19:07:16_PDT_2019
$ Cuda compilation tools, release 10.1, V10.1.243
```

**Bonus Step:** In order to have CUDA support in the CLion IDE, you will have to
overwrite global alias for the `gcc` and `g++` compilers:

```shell script
$ sudo rm /usr/bin/gcc
$ sudo rm /usr/bin/g++
$ sudo ln -s /usr/bin/gcc-8 /usr/bin/gcc
$ sudo ln -s /usr/bin/g++-8 /usr/bin/g++
```

This step can be easily undone by removing old aliases and creating new one 
for the desired gcc version on your machine. Also you can safely omit this step
if you want to build library from the command line only. 

**Useful links:**
- [NVIDIA Drivers installation Ubuntu](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux)
- [CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [CUDA Hello world program](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [CUDA CMake tutorial](https://developer.nvidia.com/blog/building-cuda-applications-cmake/)

### Get the source code and run

Run the following commands in the command shell to download the repository,
make `build` directory, configure `cmake build` and run compilation process.
First of all, get the source code and project dependencies:

```shell script
$ git clone https://github.com/JetBrains-Research/cuBool.git
$ cd cuBool
$ git submodule update --init --recursive
```

Make the build directory and go into it:

```shell script
$ mkdir build
$ cd build
```

Configure build in Release mode with tests and run actual compilation process:

```shell script
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DCUBOOL_BUILD_TESTS=ON
$ cmake --build . --target all -j `nproc`
$ bash ./scripts/run_tests_all.sh
```

By default, the following cmake options will be automatically enabled:

- `CUBOOL_WITH_CUDA` - build library with actual cuda backend
- `CUBOOL_WITH_CPU` - build library witt cpu based backend
- `CUBOOL_WITH_TESTS` - build library unit-tests collection

> Note: in order to provide correct GCC version for CUDA sources compiling,
> you will have to provide custom paths to the CC and CXX compilers before 
> the actual compilation process as follows:
>
> ```shell script
> $ export CC=/usr/bin/gcc-8
> $ export CXX=/usr/bin/g++-8
> $ export CUDAHOSTCXX=/usr/bin/g++-8
> ```

### Python Wrapper

**Export** env variable `PYTHONPATH="/build_dir_path/python/:$PYTHONPATH"` if
you want to use `pycubool` without installation into default python packages dir.
This variable will help python find package if you import it as `import pycubool` in your python scripts.

**To run regression tests** within your build directory, open folder `/build_dir_path/python` and
run the following command:

```shell script
$ export PYTHONPATH="`pwd`:$PYTHONPATH"
$ cd tests
$ python3 -m unittest discover -v
```

**Note:** after the build process, the shared library object `libcubool.so` will be placed
inside the build directory in the folder with python wrapper `python/pycubool/`. 
So, the wrapper will be able to automatically locate required lib file.

## Usage 

The following C++ code snipped demonstrates, how library functions and
primitives can be used for the transitive closure evaluation of the directed
graph, represented as an adjacency matrix with boolean values. The transitive
closure provides info about reachable vertices in the graph:

```c++
/**
 * Performs transitive closure for directed graph
 *
 * @param A Adjacency matrix of the graph
 * @param T Reference to the handle where to allocate and store result
 *
 * @return Status on this operation
 */
cuBool_Status TransitiveClosure(cuBool_Matrix A, cuBool_Matrix* T) {
    cuBool_Matrix_Duplicate(A, T);                       /* Duplicate A to result T */

    cuBool_Index total = 0;
    cuBool_Index current;

    cuBool_Matrix_Nvals(*T, &current);                   /* Query current nvals value */

    while (current != total) {                           /* Iterate, while new values are added */
        total = current;
        cuBool_MxM(*T, *T, *T, CUBOOL_HINT_ACCUMULATE);  /* T += T x T */
        cuBool_Matrix_Nvals(*T, &current);
    }

    return CUBOOL_STATUS_SUCCESS;
}
```

The following Python code snippet demonstrates, how the library python
wrapper can be used to compute the same transitive closure problem for the
directed graph within python environment:

```python
import pycubool as cb

def transitive_closure(a: cb.Matrix):
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
cuBool
├── .github - GitHub Actions CI setup 
├── docs - documents, text files and various helpful stuff
├── scripts - short utility programs 
├── cubool - library core source code
│   ├── include - library public C API 
│   ├── sources - source-code for implementation
│   │   ├── core - library core and state management
│   │   ├── backend - common interfaces
│   │   ├── cuda - cuda backend
│   │   └── sequential - fallback cpu backend
│   ├── utils - testing utilities
│   └── tests - gtest-based unit-tests collection
├── python - pycubool related sources
│   ├── pycubool - cubool library wrapper for python (similar to pygraphblas)
│   ├── tests - regression tests for python wrapper
│   └── data - generate data for pycubool regression tests
├── deps - project dependencies
│   ├── cub - cuda utility, required for nsparse
│   ├── gtest - google test framework for unit testing
│   ├── naive - GEMM implementation for squared dense boolean matrices
│   ├── nsparse - SpGEMM implementation for csr matrices
│   └── nsparse-um - SpGEMM implementation for csr matrices with unified memory (configurable)
└── CMakeLists.txt - library cmake config, add this as sub-directory to your project
```

## Contributors

- Egor Orachyov (Github: [EgorOrachyov](https://github.com/EgorOrachyov))
- Pavel Alimov (Github : [Krekep](https://github.com/Krekep))
- Semyon Grigorev (Github: [gsvgit](https://github.com/gsvgit))

## Citation 

```ignorelang
@MISC{cuBool,
  author = {Orachyov, Egor and Alimov, Pavel and Grigorev, Semyon},
  title = {cuBool: sparse Boolean linear algebra for Nvidia Cuda},
  year = 2020,
  url = {https://github.com/JetBrains-Research/cuBool},
  note = {Version Alpha}
}
```

## License

This project is licensed under MIT License. License text can be found in the 
[license file](https://github.com/JetBrains-Research/cuBool/blob/master/LICENSE.md).

## Acknowledgments

This is a research project of the Programming Languages and Tools Laboratory
at JetBrains-Research. Laboratory website [link](https://research.jetbrains.org/groups/plt_lab/).

## Also

The name of the library is formed by a combination of words *Cuda* and *Boolean*,
what literally means *Cuda with Boolean* and sounds very similar to the name of 
the programming language *COBOL*.
