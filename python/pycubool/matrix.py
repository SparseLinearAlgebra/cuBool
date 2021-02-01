import ctypes

from . import wrapper
from . import bridge

__all__ = [
    "Matrix"
]


class Matrix:
    """
    Wrapper for CuBool Sparse boolean matrix type
    """

    __slots__ = ["hnd"]

    def __init__(self, hnd):
        self.hnd = hnd

    @classmethod
    def empty(cls, shape):
        hnd = ctypes.c_void_p(0)

        nrows = shape[0]
        ncols = shape[1]

        status = wrapper.loaded_dll.CuBool_Matrix_New(wrapper.instance,
                                                      ctypes.byref(hnd),
                                                      ctypes.c_uint(nrows),
                                                      ctypes.c_uint(ncols))

        bridge.check(status)

        return Matrix(hnd)

    def __del__(self):
        bridge.check(wrapper.loaded_dll.CuBool_Matrix_Free(wrapper.instance, self.hnd))

    def resize(self, nrows, ncols):
        status = wrapper.loaded_dll.matrix_resizer(wrapper.instance,
                                                   self.hnd,
                                                   ctypes.c_uint(nrows),
                                                   ctypes.c_uint(ncols))

        bridge.check(status)

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

        bridge.check(status)

    def duplicate(self):
        hnd = ctypes.c_void_p(0)

        status = wrapper.loaded_dll.CuBool_Matrix_Duplicate(wrapper.instance,
                                                            self.hnd,
                                                            ctypes.byref(hnd))

        bridge.check(status)
        return Matrix(hnd)

    def transpose(self):
        shape = (self.ncols, self.nrows)
        result = Matrix.empty(shape)

        status = wrapper.loaded_dll.CuBool_Matrix_Transpose(wrapper.instance,
                                                            result.hnd,
                                                            self.hnd)

        bridge.check(status)
        return result

    @property
    def nrows(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.CuBool_Matrix_Nrows(wrapper.instance,
                                                        self.hnd,
                                                        ctypes.byref(result))

        bridge.check(status)
        return int(result.value)

    @property
    def ncols(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.CuBool_Matrix_Ncols(wrapper.instance,
                                                        self.hnd,
                                                        ctypes.byref(result))

        bridge.check(status)
        return int(result.value)

    @property
    def nvals(self) -> int:
        result = ctypes.c_size_t(0)

        status = wrapper.loaded_dll.CuBool_Matrix_Nvals(wrapper.instance,
                                                        self.hnd,
                                                        ctypes.byref(result))

        bridge.check(status)
        return int(result.value)

    @property
    def shape(self) -> (int, int):
        return self.nrows, self.ncols

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

        bridge.check(status)

        return rows, cols
