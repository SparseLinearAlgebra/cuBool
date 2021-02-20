# PyCuBool

Python wrapper for cubool C API shared library functions.
Provides safe and managed access to library resources as well
as python syntax sugar and various fancy functions.

## Exported primitives

- Boolean sparse matrix 
- Matrix read and write with values arrays
- Matrix MxM
- Matrix Kron
- Matrix Add

## Things to consider

- This folder is supposed to store pycubool python module files, init.py, matrix.py and etc.
- In order to run this module, the actual libcubool.so (shared library) is
required, so will have to provide path to this library, or use `ENV` variable,
such as `CUBOOL_PATH = ${some location in my computer}`
- Remember about resource, `unmanaged` resource must be released by `__del__`
- Use `ctypes` for wrapping native C functions

## Useful links

- [Python ctypes](https://docs.python.org/3/library/ctypes.html)
- [Ctypes example](https://habr.com/ru/post/466499/)
- [About __del__](https://www.geeksforgeeks.org/python-__delete__-vs-__del__/)
- [PyGraphBLAS Repository](https://github.com/michelp/pygraphblas)
