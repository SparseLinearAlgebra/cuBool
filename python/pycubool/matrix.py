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
            ctypes.byref(hnd), ctypes.c_uint(nrows), ctypes.c_uint(ncols)
        )

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

    def build(self, rows, cols, is_sorted=False):
        if len(rows) != len(cols):
            raise Exception("Size of rows and cols arrays must match the nval values")

        nvals = len(rows)
        t_rows = (ctypes.c_uint * len(rows))(*rows)
        t_cols = (ctypes.c_uint * len(cols))(*cols)

        status = wrapper.loaded_dll.cuBool_Matrix_Build(
            self.hnd, t_rows, t_cols,
            ctypes.c_uint(nvals),
            ctypes.c_uint(bridge.get_build_hints(is_sorted))
        )

        bridge.check(status)

    def dup(self):
        hnd = ctypes.c_void_p(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Duplicate(
            self.hnd, ctypes.byref(hnd)
        )

        bridge.check(status)
        return Matrix(hnd)

    def transpose(self):
        shape = (self.ncols, self.nrows)
        out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Matrix_Transpose(
            out.hnd, self.hnd
        )

        bridge.check(status)
        return out

    @property
    def nrows(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Nrows(
            self.hnd, ctypes.byref(result)
        )

        bridge.check(status)
        return int(result.value)

    @property
    def ncols(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Ncols(
            self.hnd, ctypes.byref(result)
        )

        bridge.check(status)
        return int(result.value)

    @property
    def nvals(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Nvals(
            self.hnd, ctypes.byref(result)
        )

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

        count = self.nvals

        rows = (ctypes.c_uint * count)()
        cols = (ctypes.c_uint * count)()
        nvals = ctypes.c_uint(count)

        status = wrapper.loaded_dll.cuBool_Matrix_ExtractPairs(
            self.hnd, rows, cols, ctypes.byref(nvals)
        )

        bridge.check(status)

        return rows, cols

    def to_string(self, width=3):
        """
        Return a string representation of the matrix.

        >>> matrix = Matrix.from_lists((4, 4), [0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True)
        >>> print(matrix)
        `
                0   1   2   3
          0 |   1   .   .   . |   0
          1 |   .   1   .   . |   1
          2 |   .   .   1   . |   2
          3 |   1   .   .   . |   3
                0   1   2   3
        `

        :param width: Width of the field in chars where to put numbers of rows and columns
        :return: Matrix string representation
        """

        nrows = self.nrows
        ncols = self.ncols
        nvals = self.nvals
        rows, cols = self.to_lists()

        cell_empty = "."
        cell_filled = "1"
        cell_sep = " "
        format_str = "{:>%s}" % width

        header = format_str.format("") + "  " + cell_sep + "".join(format_str.format(j) + cell_sep for j in range(ncols))
        result = header + "\n"

        v = 0
        for i in range(nrows):
            line = format_str.format(i) + " |" + cell_sep
            for j in range(ncols):
                if v < nvals and rows[v] == i and cols[v] == j:
                    line += format_str.format(cell_filled) + cell_sep
                    v += 1
                else:
                    line += format_str.format(cell_empty) + cell_sep
            line += "| " + format_str.format(i) + "\n"
            result += line

        result += header
        return result

    def extract_matrix(self, i, j, shape, out=None):
        """
        Extract a sub-matrix.

        :param i: First row index to extract
        :param j: First column index to extract
        :param shape: Shape of the sub-matrix
        :param out: Optional matrix where to store result
        :return: Sub-matrix
        """

        if out is None:
            out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Matrix_ExtractSubMatrix(
            out.hnd, self.hnd,
            ctypes.c_uint(i),
            ctypes.c_uint(j),
            ctypes.c_uint(shape[0]),
            ctypes.c_uint(shape[1]),
            bridge.get_extract_hints()
        )

        bridge.check(status)
        return out

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
            ctypes.c_uint(bridge.get_mxm_hints(accumulate))
        )

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
            out.hnd, self.hnd, other.hnd
        )

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
            out.hnd, self.hnd, other.hnd
        )

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
            out.hnd, self.hnd
        )

        bridge.check(status)
        return out

    def __str__(self):
        return self.to_string()

    def __getitem__(self, item):
        """
        Extract sub-matrix from `self`.
        Supported only tuple `item` with two slices. Step in slices is not supported.

        :param item: Tuple of two slices for rows and cols regions
        :return: Extracted sub-matrix
        """

        if isinstance(item, tuple):
            first = item[0]
            second = item[1]

            if isinstance(first, slice) and isinstance(second, slice):
                i = first.start
                iend = first.stop

                j = second.start
                jend = second.stop

                assert first.step is None
                assert second.step is None

                if i is None:
                    i = 0
                if j is None:
                    j = 0

                assert 0 <= i < self.nrows
                assert 0 <= j < self.ncols

                if iend is None:
                    iend = self.nrows
                if jend is None:
                    jend = self.ncols

                shape = (iend - i, jend - j)
                return self.extract_matrix(i, j, shape)

        raise Exception("Invalid matrix slicing")
