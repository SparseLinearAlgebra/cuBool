# CUBOOL

[![Ubuntu](https://github.com/JetBrains-Research/cuBool/workflows/Ubuntu/badge.svg?branch=master)](https://github.com/JetBrains-Research/cuBool/actions)

CuBool is a C++ linear boolean algebra library primitives and operations for 
work with dense and sparse matrices on the NVIDIA CUDA platform. The primary 
goal of the library is implementation, testing and profiling algorithms for
solving *formal-language-constrained problems*, such as *context-free* 
and *recursive* path queries with various semantics for graph databases.

The name of the library is formed by a combination of words *Cuda* and *Boolean*,
what literally means *Cuda with Boolean* and sounds very similar to the name of 
the programming language *COBOL*.

## Features

- [ ] Generic matrix representation
- [ ] Generic serialization
- [ ] Dense boolean matrix
- [ ] Dense boolean matrix operations
- [ ] Dense multiplication 
- [ ] Dense addition
- [ ] Dense kronecker product 
- [ ] Sparse CSR matrix
- [ ] Sparse CSR matrix operations
- [ ] Sparse CSR multiplication
- [ ] Sparse CSR addition
- [ ] Sparse CSR kronecker
- [ ] Wrapper for Python API 
- [ ] User guide

## Requirements

- Linux Ubuntu 18 or higher (current version is 20.04)
- CMake Version 3.10 or higher
- CUDA Compatible GPU device
- GCC 8 Compiler 
- G++ 8 Compiler
- NVIDIA CUDA 10 toolkit

## Setup

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
for the desired gcc version on your machine.

Useful links:
- [NVIDIA Drivers installation Ubuntu](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-20-04-focal-fossa-linux)
- [CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- [CUDA Hello world program](https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/)
- [CUDA CMake tutorial](https://developer.nvidia.com/blog/building-cuda-applications-cmake/)

## Get and run

Run the following commands in the command shell to download the repository,
make `build` directory, configure `cmake build` and run compilation process:

```shell script
$ git clone --recurse-submodules https://github.com/JetBrains-Research/cuBool.git
$ cd cuBool
$ mkdir build
$ cd build
$ cmake .. -DCUBOOL_BUILD_TESTS=YES
$ cmake --build .
```

> Note: in order to provide correct GCC version for CUDA sources compiling,
> you will have to provide custom paths to the CC and CXX compilers before 
> the actual compilation process as follows:
>
> ```shell script
> $ export CC=/usr/bin/gcc-8
> $ export CXX=/usr/bin/g++-8
> $ export CUDAHOSTCXX=/usr/bin/g++-8
> ```

## Directory structure

```
cuBool
├── src - library source-code
├── docs - documents, text files and various helpful stuff
├── tests - gtest-based unit-tests collection
└── thirdparty - project dependencies
```

## License

This project is licensed under MIT License. The full license text could be found in
the [license file](https://github.com/JetBrains-Research/cuBool/blob/master/LICENSE).

## Contributors

- Semyon Grigorev (Github: [gsvgit](https://github.com/gsvgit))
- Egor Orachyov (Github: [EgorOrachyov](https://github.com/EgorOrachyov))