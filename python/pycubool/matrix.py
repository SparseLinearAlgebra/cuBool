import ctypes

from . import wrapper
from . import bridge

__all__ = [
    "Matrix"
]


class Matrix:
    """
    Wrapper for CuBool Sparse boolean matrix type.
    """

    __slots__ = ["hnd"]

    def __init__(self, hnd):
        self.hnd = hnd

    @classmethod
    def empty(cls, shape):
        hnd = ctypes.c_void_p(0)

        nrows = shape[0]
        ncols = shape[1]

        status = wrapper.loaded_dll.cuBool_Matrix_New(ctypes.byref(hnd),
                                                      ctypes.c_uint(nrows),
                                                      ctypes.c_uint(ncols))

        bridge.check(status)

        return Matrix(hnd)

    def __del__(self):
        bridge.check(wrapper.loaded_dll.cuBool_Matrix_Free(self.hnd))

    def build(self, rows, cols, nvals, is_sorted=False):
        if len(rows) != len(cols) or len(rows) != nvals:
            raise Exception("Size of rows and cols arrays must match the nval values")

        t_rows = (ctypes.c_uint * len(rows))(*rows)
        t_cols = (ctypes.c_uint * len(cols))(*cols)

        status = wrapper.loaded_dll.cuBool_Matrix_Build(self.hnd,
                                                        t_rows,
                                                        t_cols,
                                                        ctypes.c_uint(nvals),
                                                        ctypes.c_uint(bridge.get_build_hints(is_sorted)))

        bridge.check(status)

    def duplicate(self):
        hnd = ctypes.c_void_p(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Duplicate(self.hnd,
                                                            ctypes.byref(hnd))

        bridge.check(status)
        return Matrix(hnd)

    def transpose(self):
        shape = (self.ncols, self.nrows)
        result = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Matrix_Transpose(result.hnd,
                                                            self.hnd)

        bridge.check(status)
        return result

    @property
    def nrows(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Nrows(self.hnd,
                                                        ctypes.byref(result))

        bridge.check(status)
        return int(result.value)

    @property
    def ncols(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Ncols(self.hnd,
                                                        ctypes.byref(result))

        bridge.check(status)
        return int(result.value)

    @property
    def nvals(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Nvals(self.hnd,
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
        nvals = ctypes.c_uint(values_count)

        status = wrapper.loaded_dll.cuBool_Matrix_ExtractPairs(self.hnd,
                                                               rows,
                                                               cols,
                                                               ctypes.byref(nvals))

        bridge.check(status)

        return rows, cols
