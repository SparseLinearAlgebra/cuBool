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

    def __del__(self):
        bridge.check(wrapper.loaded_dll.cuBool_Matrix_Free(self.hnd))

    @classmethod
    def empty(cls, shape):
        """
        Creates empty matrix of specified `shape`.

        :param shape: Pair with two values with rows and cols count of the matrix
        :return: Created empty matrix
        """

        hnd = ctypes.c_void_p(0)

        nrows = shape[0]
        ncols = shape[1]

        status = wrapper.loaded_dll.cuBool_Matrix_New(
            ctypes.byref(hnd), ctypes.c_uint(nrows), ctypes.c_uint(ncols))

        bridge.check(status)

        return Matrix(hnd)

    @classmethod
    def from_lists(cls, shape, rows, cols, is_sorted=False):
        """
        Build matrix from provided `shape` and non-zero values data.

        :param shape: Matrix shape
        :param rows: List with row indices
        :param cols: List with column indices
        :param is_sorted: True if values are sorted in row-col order
        :return: Created matrix filled with data
        """

        out = cls.empty(shape)
        out.build(rows, cols, is_sorted=is_sorted)
        return out

    def build(self, rows, cols, nvals, is_sorted=False):
        if len(rows) != len(cols) or len(rows) != nvals:
            raise Exception("Size of rows and cols arrays must match the nval values")

        t_rows = (ctypes.c_uint * len(rows))(*rows)
        t_cols = (ctypes.c_uint * len(cols))(*cols)

        status = wrapper.loaded_dll.cuBool_Matrix_Build(
            self.hnd, t_rows, t_cols, ctypes.c_uint(nvals),
            ctypes.c_uint(bridge.get_build_hints(is_sorted)))

        bridge.check(status)

    def dup(self):
        hnd = ctypes.c_void_p(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Duplicate(
            self.hnd, ctypes.byref(hnd))

        bridge.check(status)
        return Matrix(hnd)

    def transpose(self):
        shape = (self.ncols, self.nrows)
        out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Matrix_Transpose(
            out.hnd, self.hnd)

        bridge.check(status)
        return out

    @property
    def nrows(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Nrows(
            self.hnd, ctypes.byref(result))

        bridge.check(status)
        return int(result.value)

    @property
    def ncols(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Ncols(
            self.hnd, ctypes.byref(result))

        bridge.check(status)
        return int(result.value)

    @property
    def nvals(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Nvals(
            self.hnd, ctypes.byref(result))

        bridge.check(status)
        return int(result.value)

    @property
    def shape(self) -> (int, int):
        return self.nrows, self.ncols

    def to_lists(self):
        """
        Read matrix data as lists of `rows` and `clos` indices.

        :return: Pair with `rows` and `cols` lists
        """

        values_count = self.nvals

        rows = (ctypes.c_uint * values_count)()
        cols = (ctypes.c_uint * values_count)()
        nvals = ctypes.c_uint(values_count)

        status = wrapper.loaded_dll.cuBool_Matrix_ExtractPairs(
            self.hnd, rows, cols, ctypes.byref(nvals))

        bridge.check(status)

        return rows, cols

    def mxm(self, other, out=None, accumulate=True):
        """
        Matrix-matrix multiplication in boolean semiring with "x = and" and "+ = or" operations.
        Returns `self` multiplied to `other` matrix.

        Pass optional `out` matrix to store result.
        Pass `accumulate`=True to sum the multiplication result with `out` matrix.

        :param other: Input matrix for multiplication
        :param out: Optional out matrix to store result
        :param accumulate: Set in true to accumulate the result with `out` matrix
        :return: Matrix-matrix multiplication result (with possible accumulation to `out` if provided)
        """

        if out is None:
            shape = (self.nrows, other.ncols)
            out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_MxM(
            out.hnd, self.hnd, other.hnd,
            ctypes.c_uint(bridge.get_mxm_hints(accumulate)))

        bridge.check(status)
        return out

    def kronecker(self, other):
        """
        Matrix-matrix kronecker product with boolean "x = and" operation.
        Returns kronecker product of `self` and `other` matrices.

        :param other: Input matrix
        :return: Matrices kronecker product matrix
        """

        shape = (self.nrows * other.nrows, self.ncols * other.ncols)
        out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Kronecker(
            out.hnd, self.hnd, other.hnd)

        bridge.check(status)
        return out

    def ewiseadd(self, other):
        """
        Element-wise matrix-matrix addition with boolean "+ = or" operation.
        Returns element-wise sum of `self` and `other` matrix.

        :param other: Input matrix to sum
        :return: Element-wise matrix-matrix sum
        """

        shape = (self.nrows, self.ncols)
        out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Matrix_EWiseAdd(
            out.hnd, self.hnd, other.hnd)

        bridge.check(status)
        return out

    def reduce(self):
        """
        Reduce matrix to vector with boolean "+ = or" operation.
        Return `self` reduced matrix.

        :return: Reduced matrix (matrix with M x 1 shape)
        """

        shape = (self.nrows, 1)
        out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Matrix_Reduce(
            out.hnd, self.hnd)

        bridge.check(status)
        return out
