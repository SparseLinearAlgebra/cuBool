import ctypes

from . import wrapper
from . import dll

__all__ = [
    "Matrix"
]


class Matrix:
    """
    Wrapper for CuBool Sparse boolean matrix type
    """

    def __init__(self, nrows: int, ncols: int):
        self.hnd = ctypes.c_void_p(0)
        self.wrapper = wrapper.singleton

        status = wrapper.loaded_dll.CuBool_Matrix_New(wrapper.instance,
                                                      ctypes.byref(self.hnd),
                                                      ctypes.c_uint(nrows),
                                                      ctypes.c_uint(ncols))

        dll.check(status)

    def __del__(self):
        dll.check(self.wrapper.loaded_dll.CuBool_Matrix_Free(wrapper.instance, self.hnd))

    def resize(self, nrows, ncols):
        status = wrapper.loaded_dll.matrix_resizer(wrapper.instance,
                                                   self.hnd,
                                                   ctypes.c_uint(nrows),
                                                   ctypes.c_uint(ncols))

        dll.check(status)

    def build(self, rows, cols, nvals):
        if len(rows) != len(cols) or len(rows) != nvals:
            raise Exception("Size of rows and cols arrays must match the nval values")

        t_rows = (ctypes.c_uint * len(rows))(*rows)
        t_cols = (ctypes.c_uint * len(cols))(*cols)

        status = wrapper.loaded_dll.CuBool_Matrix_Build(wrapper.instance,
                                                        self.hnd,
                                                        t_rows,
                                                        t_cols,
                                                        ctypes.c_size_t(nvals))

        dll.check(status)

    @property
    def nrows(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.CuBool_Matrix_Nrows(wrapper.instance,
                                                        self.hnd,
                                                        ctypes.byref(result))

        dll.check(status)
        return int(result.value)

    @property
    def nclos(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.CuBool_Matrix_Ncols(wrapper.instance,
                                                        self.hnd,
                                                        ctypes.byref(result))

        dll.check(status)
        return int(result.value)

    @property
    def nvals(self) -> int:
        result = ctypes.c_size_t(0)

        status = wrapper.loaded_dll.CuBool_Matrix_Nvals(wrapper.instance,
                                                        self.hnd,
                                                        ctypes.byref(result))

        dll.check(status)
        return int(result.value)

    @property
    def shape(self) -> (int, int):
        return self.nrows, self.nclos

    def to_lists(self):
        values_count = self.nvals
        rows = (ctypes.c_uint * values_count)()
        cols = (ctypes.c_uint * values_count)()
        nvals = ctypes.c_size_t(values_count)

        status = wrapper.loaded_dll.CuBool_Matrix_ExtractPairs(wrapper.instance,
                                                               self.hnd,
                                                               rows,
                                                               cols,
                                                               ctypes.byref(nvals))

        dll.check(status)

        return rows, cols

    def duplicate(self):
        result_matrix = Matrix(self.nrows, self.nclos)
        status = wrapper.loaded_dll.CuBool_Matrix_Duplicate(wrapper.instance,
                                                            self.hnd,
                                                            result_matrix.hnd)

        dll.check(status)
        return result_matrix
