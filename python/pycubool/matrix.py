"""
Matrix primitive.
"""

import ctypes
import random

from . import wrapper
from . import bridge
from . import vector


__all__ = [
    "Matrix"
]


class Matrix:
    """
    Wrapper for cuBool Sparse boolean matrix type.

    Matrix class supports all cuBool C API Matrix functions.
    Also Matrix class provides additional fancy functions/operators for better user experience.

    Matrix creation:
    - empty
    - from lists data
    - random generated

    Matrix operations:
    - mxm
    - ewiseadd
    - kronecker
    - reduce
    - transpose
    - matrix extraction

    Matrix functions:
    - to string
    - values iterating
    - equality check

    Debug features:
    - string markers
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
    def from_lists(cls, shape, rows, cols, is_sorted=False, no_duplicates=False):
        """
        Create matrix from provided `shape` and non-zero values data.

        >>> matrix = Matrix.from_lists((4, 4), [0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True, no_duplicates=True)
        >>> print(matrix)
        '
                0   1   2   3
          0 |   1   .   .   . |   0
          1 |   .   1   .   . |   1
          2 |   .   .   1   . |   2
          3 |   1   .   .   . |   3
                0   1   2   3
        '

        :param shape: Matrix shape
        :param rows: List with row indices
        :param cols: List with column indices
        :param is_sorted: True if values are sorted in row-col order
        :param no_duplicates: True if provided values has no duplicates
        :return: Created matrix filled with data
        """

        out = cls.empty(shape)
        out.build(rows, cols, is_sorted=is_sorted, no_duplicates=no_duplicates)
        return out

    @classmethod
    def generate(cls, shape, density: float):
        """
        Generate matrix of the specified shape with desired values density.

        >>> matrix = Matrix.generate(shape=(4, 4), density=0.5)
        >>> print(matrix)
        '
                0   1   2   3
          0 |   .   1   .   . |   0
          1 |   .   .   1   1 |   1
          2 |   1   1   .   . |   2
          3 |   .   1   .   . |   3
                0   1   2   3
        '

        :param shape: Matrix shape to generate
        :param density: Matrix values density, must be within [0, 1] bounds
        :return: Generated matrix
        """

        density = min(1.0, max(density, 0.0))
        nvals_max = shape[0] * shape[1]
        nvals_to_gen = int(nvals_max * density)

        m, n = shape
        rows, cols = list(), list()

        for i in range(nvals_to_gen):
            rows.append(random.randrange(0, m))
            cols.append(random.randrange(0, n))

        return Matrix.from_lists(shape=shape, rows=rows, cols=cols, is_sorted=False, no_duplicates=False)

    def build(self, rows, cols, is_sorted=False, no_duplicates=False):
        """
        Build sparse matrix of boolean values from provided arrays of non-zero rows and columns.

        >>> matrix = Matrix.empty(shape=(4,4))
        >>> matrix.build([0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True, no_duplicates=True)
        >>> print(matrix)
        '
                0   1   2   3
          0 |   1   .   .   . |   0
          1 |   .   1   .   . |   1
          2 |   .   .   1   . |   2
          3 |   1   .   .   . |   3
                0   1   2   3
        '

        :param rows: Array of values rows indices
        :param cols: Array of values column indices
        :param is_sorted: True if values are sorted in row-col order
        :param no_duplicates: True if provided values has no duplicates
        :return:
        """

        if len(rows) != len(cols):
            raise Exception("Rows and cols arrays must have equal size")

        nvals = len(rows)
        t_rows = (ctypes.c_uint * len(rows))(*rows)
        t_cols = (ctypes.c_uint * len(cols))(*cols)

        status = wrapper.loaded_dll.cuBool_Matrix_Build(
            self.hnd, t_rows, t_cols,
            ctypes.c_uint(nvals),
            ctypes.c_uint(bridge.get_build_hints(is_sorted, no_duplicates))
        )

        bridge.check(status)

    def dup(self):
        """
        Creates new matrix instance, the exact copy of the `self`

        >>> a = Matrix.from_lists((4, 4), [0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True, no_duplicates=True)
        >>> b = a.dup()
        >>> b[3, 3] = True
        >>> print(a, b, sep="")
        '
                0   1   2   3
          0 |   1   .   .   . |   0
          1 |   .   1   .   . |   1
          2 |   .   .   1   . |   2
          3 |   1   .   .   . |   3
                0   1   2   3
                0   1   2   3
          0 |   1   .   .   . |   0
          1 |   .   1   .   . |   1
          2 |   .   .   1   . |   2
          3 |   1   .   .   1 |   3
                0   1   2   3
        '

        :return: New matrix instance with `self` copied data
        """

        hnd = ctypes.c_void_p(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Duplicate(
            self.hnd, ctypes.byref(hnd)
        )

        bridge.check(status)
        return Matrix(hnd)

    def transpose(self, time_check=False):
        """
        Creates new transposed `self` matrix.

        >>> a = Matrix.from_lists((4, 4), [0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True, no_duplicates=True)
        >>> b = a.transpose()
        >>> print(a, b, sep="")
        '
                0   1   2   3
          0 |   1   .   .   . |   0
          1 |   .   1   .   . |   1
          2 |   .   .   1   . |   2
          3 |   1   .   .   . |   3
                0   1   2   3
                0   1   2   3
          0 |   1   .   .   1 |   0
          1 |   .   1   .   . |   1
          2 |   .   .   1   . |   2
          3 |   .   .   .   . |   3
                0   1   2   3
        '

        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: New matrix instance with `self` transposed data
        """

        shape = (self.ncols, self.nrows)
        out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Matrix_Transpose(
            out.hnd,
            self.hnd,
            ctypes.c_uint(bridge.get_transpose_hints(time_check=time_check))
        )

        bridge.check(status)
        return out

    def set_marker(self, marker: str):
        """
        Sets to the matrix specific debug string marker.
        This marker will appear in the log messages as string identifier of the matrix.

        >>> a = Matrix.empty(shape=(4, 4))
        >>> print(a.marker)
        '0x1a767b0'
        >>> a.set_marker("meow")
        >>> print(a.marker)
        'meow (0x1a767b0)'

        :param marker: String marker to set
        :return:
        """

        assert marker is not None

        status = wrapper.loaded_dll.cuBool_Matrix_SetMarker(
            self.hnd, marker.encode("utf-8")
        )

        bridge.check(status)

    @property
    def marker(self):
        """
        Allows to get matrix debug string marker.

        >>> a = Matrix.empty(shape=(4, 4))
        >>> print(a.marker)
        '0x1a767b0'
        >>> a.set_marker("meow")
        >>> print(a.marker)
        'meow (0x1a767b0)'

        :return: String matrix marker.
        """

        size = ctypes.c_uint(0)
        status = wrapper.loaded_dll.cuBool_Matrix_Marker(
            self.hnd, ctypes.POINTER(ctypes.c_char)(), ctypes.byref(size)
        )

        bridge.check(status)

        c_buffer = (ctypes.c_char * int(size.value))()
        status = wrapper.loaded_dll.cuBool_Matrix_Marker(
            self.hnd, c_buffer, ctypes.byref(size)
        )

        bridge.check(status)
        return c_buffer.value.decode("utf-8")

    @property
    def nrows(self) -> int:
        """
        Query number of rows of the `self` matrix.
        :return: Number of rows
        """

        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Nrows(
            self.hnd, ctypes.byref(result)
        )

        bridge.check(status)
        return int(result.value)

    @property
    def ncols(self) -> int:
        """
        Query number of columns of the `self` matrix.
        :return: Number of columns
        """

        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Ncols(
            self.hnd, ctypes.byref(result)
        )

        bridge.check(status)
        return int(result.value)

    @property
    def nvals(self) -> int:
        """
        Query number of non-zero values of the `self` matrix.
        :return: Number of non-zero values
        """

        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Matrix_Nvals(
            self.hnd, ctypes.byref(result)
        )

        bridge.check(status)
        return int(result.value)

    @property
    def shape(self) -> (int, int):
        """
        Query shape of `self` matrix as (nrows, ncols) tuple.
        :return: Return tuple of (nrows, ncols)
        """

        return self.nrows, self.ncols

    def to_lists(self):
        """
        Read matrix data as lists of `rows` and `clos` indices.

        >>> a = Matrix.empty(shape=(4, 4))
        >>> a[0, 0] = True
        >>> a[1, 3] = True
        >>> a[1, 0] = True
        >>> a[2, 2] = True
        >>> rows, cols = a.to_lists()
        >>> print(list(rows), list(cols))
        '[0, 1, 1, 2] [0, 0, 3, 2]'

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

    def to_list(self):
        """
        Read matrix values as list of (i,j) pairs.

        >>> a = Matrix.empty(shape=(4, 4))
        >>> a[0, 0] = True
        >>> a[1, 3] = True
        >>> a[1, 0] = True
        >>> a[2, 2] = True
        >>> vals = a.to_list()
        >>> print(vals)
        '[(0, 0), (1, 0), (1, 3), (2, 2)]'

        :return: List of (i, j) pairs
        """

        I, J = self.to_lists()
        return list(zip(I, J))

    def to_string(self, width=3):
        """
        Return a string representation of the matrix.

        >>> matrix = Matrix.from_lists((4, 4), [0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True, no_duplicates=True)
        >>> print(matrix)
        '
                0   1   2   3
          0 |   1   .   .   . |   0
          1 |   .   1   .   . |   1
          2 |   .   .   1   . |   2
          3 |   1   .   .   . |   3
                0   1   2   3
        '

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

        result += header + "\n"
        return result

    def extract_matrix(self, i, j, shape, out=None, time_check=False):
        """
        Extract a sub-matrix.

        >>> matrix = Matrix.from_lists((4, 4), [0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True)
        >>> print(matrix.extract_matrix(0, 1, shape=(3, 3)))
        '
                0   1   2
          0 |   .   .   . |   0
          1 |   1   .   . |   1
          2 |   .   1   . |   2
                0   1   2
        '

        :param i: First row index to extract
        :param j: First column index to extract
        :param shape: Shape of the sub-matrix
        :param out: Optional matrix where to store result
        :param time_check: Pass True to measure and log elapsed time of the operation
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
            ctypes.c_uint(bridge.get_sub_matrix_hints(time_check=time_check))
        )

        bridge.check(status)
        return out

    def extract_row(self, i, out=None):
        """
        Extract specified `self` matrix row as sparse vector.

        >>> matrix = Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
        >>> print(matrix.extract_row(1))
        '
          0 |   . |   0
          1 |   1 |   1
          2 |   . |   2
          3 |   . |   3
        '

        :param i: Row index to extract
        :param out: Optional out vector to store result
        :return: Return extracted row
        """

        if out is None:
            out = vector.Vector.empty(self.ncols)

        status = wrapper.loaded_dll.cuBool_Matrix_ExtractRow(
            out.hnd,
            self.hnd,
            ctypes.c_uint(i),
            ctypes.c_uint(0)
        )

        bridge.check(status)
        return out

    def extract_col(self, j, out=None):
        """
        Extract specified `self` matrix column as sparse vector.

        >>> matrix = Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
        >>> print(matrix.extract_col(1))
        '
          0 |   . |   0
          1 |   1 |   1
          2 |   1 |   2
          3 |   . |   3
          4 |   . |   4
        '

        :param j: Column index to extract
        :param out: Optional out vector to store result
        :return: Return extracted column
        """

        if out is None:
            out = vector.Vector.empty(self.nrows)

        status = wrapper.loaded_dll.cuBool_Matrix_ExtractCol(
            out.hnd,
            self.hnd,
            ctypes.c_uint(j),
            ctypes.c_uint(0)
        )

        bridge.check(status)
        return out

    def mxm(self, other, out=None, accumulate=False, time_check=False):
        """
        Matrix-matrix multiplication in boolean semiring with "x = and" and "+ = or" operations.
        Returns `self` multiplied to `other` matrix.

        Pass optional `out` matrix to store result.
        Pass `accumulate`=True to sum the multiplication result with `out` matrix.

        >>> a = Matrix.from_lists((4, 4), [0, 1, 2], [2, 3, 0])
        >>> b = Matrix.from_lists((4, 4), [0, 1, 3], [2, 3, 0])
        >>> print(a.mxm(b, out=a, accumulate=True))
        '
                0   1   2   3
          0 |   .   .   1   . |   0
          1 |   1   .   .   1 |   1
          2 |   1   .   1   . |   2
          3 |   .   .   .   . |   3
                0   1   2   3
        '

        :param other: Input matrix for multiplication
        :param out: Optional out matrix to store result
        :param accumulate: Set in true to accumulate the result with `out` matrix
        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Matrix-matrix multiplication result (with possible accumulation to `out` if provided)
        """

        if out is None:
            shape = (self.nrows, other.ncols)
            out = Matrix.empty(shape)
            accumulate = False

        status = wrapper.loaded_dll.cuBool_MxM(
            out.hnd,
            self.hnd,
            other.hnd,
            ctypes.c_uint(bridge.get_mxm_hints(is_accumulated=accumulate, time_check=time_check))
        )

        bridge.check(status)
        return out

    def mxv(self, other, out=None, time_check=False):
        """
        Matrix-vector multiply.

        Multiply `this` matrix by column `other` vector `on the right`.
        For row vector-matrix multiplication "on the left" see `Vector.vxm`.

        >>> matrix = Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
        >>> vector = Vector.from_list(4, [0, 1, 2])
        >>> print(matrix.mxv(vector))
        '
          0 |   1 |   0
          1 |   1 |   1
          2 |   1 |   2
          3 |   . |   3
          4 |   . |   4
        '

        :param other: Input matrix for multiplication
        :param out: Optional out vector to store result
        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Vector-matrix multiplication result
        """

        if out is None:
            out = vector.Vector.empty(self.nrows)

        status = wrapper.loaded_dll.cuBool_MxV(
            out.hnd,
            self.hnd,
            other.hnd,
            ctypes.c_uint(bridge.get_mxv_hints(time_check=time_check))
        )

        bridge.check(status)
        return out

    def kronecker(self, other, out=None, time_check=False):
        """
        Matrix-matrix kronecker product with boolean "x = and" operation.
        Returns kronecker product of `self` and `other` matrices.

        >>> a = Matrix.from_lists((3, 3), [0, 0, 1, 2, 2], [0, 2, 1, 0, 2])
        >>> b = Matrix.from_lists((3, 3), [0, 1, 1, 2], [1, 0, 2, 1])
        >>> print(a.kronecker(b))
        '
                0   1   2   3   4   5   6   7   8
          0 |   .   1   .   .   .   .   .   1   . |   0
          1 |   1   .   1   .   .   .   1   .   1 |   1
          2 |   .   1   .   .   .   .   .   1   . |   2
          3 |   .   .   .   .   1   .   .   .   . |   3
          4 |   .   .   .   1   .   1   .   .   . |   4
          5 |   .   .   .   .   1   .   .   .   . |   5
          6 |   .   1   .   .   .   .   .   1   . |   6
          7 |   1   .   1   .   .   .   1   .   1 |   7
          8 |   .   1   .   .   .   .   .   1   . |   8
                0   1   2   3   4   5   6   7   8
        '

        :param other: Input matrix
        :param out: Optional out matrix to store result
        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Matrices kronecker product matrix
        """

        if out is None:
            shape = (self.nrows * other.nrows, self.ncols * other.ncols)
            out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Kronecker(
            out.hnd,
            self.hnd,
            other.hnd,
            ctypes.c_uint(bridge.get_kronecker_hints(time_check=time_check))
        )

        bridge.check(status)
        return out

    def ewiseadd(self, other, out=None, time_check=False):
        """
        Element-wise matrix-matrix addition with boolean "+ = or" operation.
        Returns element-wise sum of `self` and `other` matrix.

        >>> a = Matrix.from_lists((4, 4), [0, 1, 2], [2, 3, 0])
        >>> b = Matrix.from_lists((4, 4), [0, 1, 3], [2, 3, 0])
        >>> print(a.ewiseadd(b))
        '
                0   1   2   3
          0 |   .   .   1   . |   0
          1 |   .   .   .   1 |   1
          2 |   1   .   .   . |   2
          3 |   1   .   .   . |   3
                0   1   2   3
        '

        :param other: Input matrix to sum
        :param out: Optional out matrix to store result
        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Element-wise matrix-matrix sum
        """

        if out is None:
            shape = (self.nrows, self.ncols)
            out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Matrix_EWiseAdd(
            out.hnd,
            self.hnd,
            other.hnd,
            ctypes.c_uint(bridge.get_ewiseadd_hints(time_check=time_check))
        )

        bridge.check(status)
        return out

    def reduce(self, out=None, time_check=False):
        """
        Reduce matrix to column matrix with boolean "+ = or" operation.
        Return `self` reduced matrix.

        >>> matrix = Matrix.from_lists((4, 4), [0, 1, 2, 2], [0, 1, 0, 2])
        >>> print(matrix.reduce())
        '
                  0
            0 |   1 |   0
            1 |   1 |   1
            2 |   1 |   2
            3 |   . |   3
                  0
        '

        :param out: Optional out matrix to store result
        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Reduced matrix (matrix with M x 1 shape)
        """

        if out is None:
            shape = (self.nrows, 1)
            out = Matrix.empty(shape)

        status = wrapper.loaded_dll.cuBool_Matrix_Reduce2(
            out.hnd,
            self.hnd,
            ctypes.c_uint(bridge.get_reduce_hints(time_check=time_check))
        )

        bridge.check(status)
        return out

    def reduce_vector(self, out=None, transpose=False, time_check=False):
        """
        Reduce matrix to column vector with boolean "+ = or" operation.
        Return `self` reduced matrix.

        >>> matrix = Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
        >>> print(matrix.reduce_vector(), matrix.reduce_vector(transpose=True), sep="")
        '
          0 |   1 |   0
          1 |   1 |   1
          2 |   1 |   2
          3 |   . |   3
          4 |   1 |   4

          0 |   1 |   0
          1 |   1 |   1
          2 |   . |   2
          3 |   1 |   3
        '

        :param out: Optional out matrix to store result
        :param transpose: Pass True to reduce matrix to row vector
        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Reduced matrix (matrix with M x 1 shape)
        """

        if out is None:
            nrows = self.ncols if transpose else self.nrows
            out = vector.Vector.empty(nrows)

        status = wrapper.loaded_dll.cuBool_Matrix_Reduce(
            out.hnd,
            self.hnd,
            ctypes.c_uint(bridge.get_reduce_vector_hints(transpose=transpose, time_check=time_check))
        )

        bridge.check(status)
        return out

    def equals(self, other) -> bool:
        """
        Compare two matrices. Returns true if they are equal.

        :param other: Other matrix to compare
        :return: True if matrices are equal
        """

        if not self.shape == other.shape:
            return False
        if not self.nvals == other.nvals:
            return False

        self_rows, self_cols = self.to_lists()
        other_rows, other_cols = other.to_lists()

        for i in range(len(self_rows)):
            if self_rows[i] != other_rows[i]:
                return False
        for i in range(len(self_cols)):
            if self_cols[i] != other_cols[i]:
                return False

        return True

    def __str__(self):
        return self.to_string()

    def __iter__(self):
        """
        Iterate over (i, j) tuples of the matrix values.

        >>> matrix = Matrix.from_lists((4, 4), [0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True)
        >>> print(list(iter(matrix)))
        '[(0, 0), (1, 1), (2, 2), (3, 0)]'

        :return: Matrix tuples iterator
        """

        rows, cols = self.to_lists()
        return zip(rows, cols)

    def __getitem__(self, item):
        """
        Extract sub-matrix from `self`.
        Supported only tuple `item` with two slices. Step in slices is not supported.

        >>> matrix = Matrix.from_lists((4, 4), [0, 1, 2, 3], [0, 1, 2, 0], is_sorted=True)
        >>> print(matrix[0:3, 1:])
        '
                0   1   2
          0 |   .   .   . |   0
          1 |   1   .   . |   1
          2 |   .   1   . |   2
                0   1   2
        '

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

    def __setitem__(self, key, value):
        """
        Sets specified `key` = (i, j) value of the matrix to True.

        >>> matrix = Matrix.empty(shape=(4, 4))
        >>> matrix[0, 0] = True
        >>> matrix[1, 1] = True
        >>> matrix[2, 3] = True
        >>> matrix[3, 1] = True
        >>> print(matrix)
        '
                0   1   2   3
          0 |   1   .   .   . |   0
          1 |   .   1   .   . |   1
          2 |   .   .   .   1 |   2
          3 |   .   1   .   . |   3
                0   1   2   3
        '

        :param key: (i, j) pair to set matrix element in True
        :param value: Must be True always
        :return:
        """

        assert value is True

        if isinstance(key, tuple):
            i = key[0]
            j = key[1]

            status = wrapper.loaded_dll.cuBool_Matrix_SetElement(
                self.hnd,
                ctypes.c_uint(i),
                ctypes.c_uint(j)
            )

            bridge.check(status)
            return

        raise Exception("Invalid item assignment")

