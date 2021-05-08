"""
Vector primitive.
"""

import ctypes
import random

from . import wrapper
from . import bridge


__all__ = [
    "Vector"
]


class Vector:
    """
    Wrapper for cuBool sparse boolean vector type.

    Vector class supports all cuBool C API Vector functions.
    Also Vector class provides additional fancy functions/operations for better user experience.

    Vector creation:
    - empty
    - from list data
    - random generated

    Vector operations:
    - vxm
    - ewiseadd
    - reduce
    - vector extraction

    Vector functions:
    - to string
    - values iterating
    - equality check

    Debug features:
    - string markers
    """

    __slots__ = ["hnd"]

    def __init__(self, hnd):
        self.hnd = hnd

    @classmethod
    def empty(cls, nrows):
        """
        Creates empty vector of specified `nrows` size.

        :param nrows: Vector dimension
        :return: Created empty vector
        """

        hnd = ctypes.c_void_p(0)

        status = wrapper.loaded_dll.cuBool_Vector_New(
            ctypes.byref(hnd), ctypes.c_uint(nrows)
        )

        bridge.check(status)

        return Vector(hnd)

    @classmethod
    def from_list(cls, nrows, rows, is_sorted=False, no_duplicates=False):
        """
        Create vector from provided `nrows` size and specified non-zero indices.

        >>> vector = Vector.from_list(4, [0, 1, 3], is_sorted=True, no_duplicates=True)
        >>> print(vector)
        '
          0 |   1 |   0
          1 |   1 |   1
          2 |   . |   2
          3 |   1 |   3
        '

        :param nrows: Vector dimension
        :param rows: List with indices of non-zero vector values
        :param is_sorted: True if values are sorted
        :param no_duplicates: True if provided values has no duplicates
        :return: Created vector filled with data
        """

        out = cls.empty(nrows)
        out.build(rows, is_sorted=is_sorted, no_duplicates=no_duplicates)
        return out

    @classmethod
    def generate(cls, nrows, density: float):
        """
        Generate vector of specified size with desired values density.

        >>> vector = Vector.generate(nrows=4, density=0.5)
        >>> print(vector)
        '
          0 |   1 |   0
          1 |   . |   1
          2 |   . |   2
          3 |   1 |   3
        '

        :param nrows: Vector dimension
        :param density: Vector values density, must be within [0, 1]
        :return: Generated vector
        """

        density = min(1.0, max(density, 0.0))
        nvals_max = nrows
        nvals_to_gen = int(nvals_max * density)

        m = nrows
        rows = list()

        for i in range(nvals_to_gen):
            rows.append(random.randrange(0, m))

        return Vector.from_list(nrows=nrows, rows=rows, is_sorted=False, no_duplicates=False)

    def build(self, rows, is_sorted=False, no_duplicates=False):
        """
        Build sparse vector of boolean values from provided array of non-zero values.

        >>> vector = Vector.empty(4)
        >>> vector.build([0, 1, 3], is_sorted=True, no_duplicates=True)
        >>> print(vector)
        '
          0 |   1 |   0
          1 |   1 |   1
          2 |   . |   2
          3 |   1 |   3
        '

        :param rows: Array of vector values
        :param is_sorted: True if values are sorted
        :param no_duplicates: True if provided values has no duplicates
        :return:
        """

        nvals = len(rows)
        t_rows = (ctypes.c_uint * len(rows))(*rows)

        status = wrapper.loaded_dll.cuBool_Vector_Build(
            self.hnd, t_rows,
            ctypes.c_uint(nvals),
            ctypes.c_uint(bridge.get_build_hints(is_sorted, no_duplicates))
        )

        bridge.check(status)

    def dup(self):
        """
        Creates new vector instance, the exact copy of the `self`.

        >>> a = Vector.from_list(4, [0, 1, 3], is_sorted=True, no_duplicates=True)
        >>> b = a.dup()
        >>> b[2] = True
        >>> print(a, b, sep="")
        '
          0 |   1 |   0
          1 |   1 |   1
          2 |   . |   2
          3 |   1 |   3

          0 |   1 |   0
          1 |   1 |   1
          2 |   1 |   2
          3 |   1 |   3
        '

        :return: New vector instance with `self` copied data
        """

        hnd = ctypes.c_void_p(0)

        status = wrapper.loaded_dll.cuBool_Vector_Duplicate(
            self.hnd, ctypes.byref(hnd)
        )

        bridge.check(status)
        return Vector(hnd)

    def set_marker(self, marker: str):
        """
        Sets to the vector specific debug string marker.
        This marker will appear in the log messages as string identifier of the vector.

        >>> a = Vector.empty(nrows=4)
        >>> print(a.marker)
        '0x2590a78'
        >>> a.set_marker("meow")
        >>> print(a.marker)
        'meow (0x2590a78)'

        :param marker: String marker to set
        :return:
        """

        assert marker is not None

        status = wrapper.loaded_dll.cuBool_Vector_SetMarker(
            self.hnd, marker.encode("utf-8")
        )

        bridge.check(status)

    @property
    def marker(self):
        """
        Allows to get vector debug string marker.

        >>> a = Vector.empty(nrows=4)
        >>> print(a.marker)
        '0x1a767b0'
        >>> a.set_marker("meow")
        >>> print(a.marker)
        'meow (0x1a767b0)'

        :return: String vector marker.
        """

        size = ctypes.c_uint(0)
        status = wrapper.loaded_dll.cuBool_Vector_Marker(
            self.hnd, ctypes.POINTER(ctypes.c_char)(), ctypes.byref(size)
        )

        bridge.check(status)

        c_buffer = (ctypes.c_char * int(size.value))()
        status = wrapper.loaded_dll.cuBool_Vector_Marker(
            self.hnd, c_buffer, ctypes.byref(size)
        )

        bridge.check(status)
        return c_buffer.value.decode("utf-8")

    @property
    def nrows(self) -> int:
        """
        Query number of rows of the `self` vector.
        :return: Number of rows
        """

        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Vector_Nrows(
            self.hnd, ctypes.byref(result)
        )

        bridge.check(status)
        return int(result.value)

    @property
    def nvals(self) -> int:
        """
        Query number of non-zero values of the `self` vector.
        :return: Number of non-zero values
        """

        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Vector_Nvals(
            self.hnd, ctypes.byref(result)
        )

        bridge.check(status)
        return int(result.value)

    def to_list(self):
        """
        Read vector data as list of non-zero vector indices.

        >>> a = Vector.empty(nrows=4)
        >>> a[0] = True
        >>> a[2] = True
        >>> a[3] = True
        >>> rows = a.to_list()
        >>> print(list(rows))
        '[0, 2, 3]'

        :return: List of indices
        """

        count = self.nvals

        rows = (ctypes.c_uint * count)()
        nvals = ctypes.c_uint(count)

        status = wrapper.loaded_dll.cuBool_Vector_ExtractValues(
            self.hnd, rows, ctypes.byref(nvals)
        )

        bridge.check(status)

        return rows

    def to_string(self, width=3):
        """
        Return a string representation of the vector.

        >>> vector = Vector.from_list(4, [0, 1, 3], is_sorted=True, no_duplicates=True)
        >>> print(vector)
        '
          0 |   1 |   0
          1 |   1 |   1
          2 |   . |   2
          3 |   1 |   3
        '

        :param width: Width of the field in chars where to put numbers of rows
        :return: Vector string representation
        """

        nrows = self.nrows
        nvals = self.nvals
        rows = self.to_list()

        cell_empty = "."
        cell_filled = "1"
        cell_sep = " "
        format_str = "{:>%s}" % width

        result = ""

        v = 0
        for i in range(nrows):
            line = format_str.format(i) + " |" + cell_sep
            if v < nvals and rows[v] == i:
                line += format_str.format(cell_filled) + cell_sep
                v += 1
            else:
                line += format_str.format(cell_empty) + cell_sep
            line += "| " + format_str.format(i) + "\n"
            result += line

        result += "\n"
        return result

    def extract_vector(self, i, nrows, out=None, time_check=False):
        """
        Extract a sub-vector.

        >>> vector = Vector.from_list(5, [0, 1, 3])
        >>> print(vector.extract_vector(1, nrows=3))
        '
          0 |   1 |   0
          1 |   . |   1
          2 |   1 |   2
        '

        :param i: First row index to extract
        :param nrows: Number of row of the sub-vector
        :param out: Optional out vector to store result
        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Sub-vector
        """

        if out is None:
            out = Vector.empty(nrows)

        status = wrapper.loaded_dll.cuBool_Vector_ExtractSubVector(
            out.hnd, self.hnd,
            ctypes.c_uint(i),
            ctypes.c_uint(nrows),
            ctypes.c_uint(bridge.get_sub_vector_hints(time_check=time_check))
        )

        bridge.check(status)
        return out

    def vxm(self, other, out=None, time_check=False):
        """
        Vector-matrix multiply.

        Multiply this row vector by `other` matrix `on the left`.
        For column matrix-vector multiplication "on the right" see `Matrix.mxv`.

        >>> matrix = Matrix.from_lists((5, 4), [0, 1, 2, 4], [0, 1, 1, 3])
        >>> vector = Vector.from_list(5, [2, 3, 4])
        >>> print(vector.vxm(matrix))
        '
          0 |   . |   0
          1 |   1 |   1
          2 |   . |   2
          3 |   1 |   3
        '

        :param other: Input matrix for multiplication
        :param out: Optional out vector to store result
        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Vector-matrix multiplication result
        """

        if out is None:
            out = Vector.empty(other.ncols)

        status = wrapper.loaded_dll.cuBool_VxM(
            out.hnd,
            self.hnd,
            other.hnd,
            ctypes.c_uint(bridge.get_vxm_hints(time_check=time_check))
        )

        bridge.check(status)
        return out

    def ewiseadd(self, other, out=None, time_check=False):
        """
        Element-wise vector-vector addition with boolean "+ = or" operation.
        Returns element-wise sum of `self` and `other` vector.

        >>> a = Vector.from_list(4, [0, 1, 3])
        >>> b = Vector.from_list(4, [1, 2, 3])
        >>> print(a.ewiseadd(b))
        '
          0 |   1 |   0
          1 |   1 |   1
          2 |   1 |   2
          3 |   1 |   3
        '

        :param other: Input vector to sum
        :param out: Optional out vector to store result
        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Element-wise vector-vector sum
        """

        if out is None:
            out = Vector.empty(self.nrows)

        status = wrapper.loaded_dll.cuBool_Vector_EWiseAdd(
            out.hnd,
            self.hnd,
            other.hnd,
            ctypes.c_uint(bridge.get_ewiseadd_hints(time_check=time_check))
        )

        bridge.check(status)
        return out

    def reduce(self, time_check=False):
        """
        Reduce vector to int value.

        >>> vector = Vector.from_list(5, [2, 3, 4])
        >>> print(vector.reduce())
        '3'

        :param time_check: Pass True to measure and log elapsed time of the operation
        :return: Int sum of vector values
        """

        value = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Vector_Reduce(
            ctypes.byref(value),
            self.hnd,
            ctypes.c_uint(bridge.get_reduce_hints(time_check=time_check))
        )

        bridge.check(status)
        return int(value.value)

    def equals(self, other) -> bool:
        """
        Compare two vectors. Returns true if they are equal.

        :param other: Other vector to compare
        :return: True if vectors are equal
        """

        if not self.nrows == other.nrows:
            return False
        if not self.nvals == other.nvals:
            return False

        self_rows = self.to_list()
        other_rows = self.to_list()

        for i in range(len(self_rows)):
            if self_rows[i] != other_rows[i]:
                return False

        return True

    def __str__(self):
        return self.to_string()

    def __iter__(self):
        """
        Iterate over (i) values if the vector.

        >>> vector = Vector.from_list(5, [1, 3, 4])
        >>> print(list(iter(vector)))
        '[1, 3, 4]'

        :return: Vector non-zero indices iterator
        """

        return iter(self.to_list())

    def __getitem__(self, item):
        """
        Extract sub-vector from `self`.
        Supported `item` as slice. Step in slices is not supported.

        >>> vector = Vector.from_list(5, [1, 3, 4])
        >>> print(vector[0:3], vector[2:], sep="")
        '
          0 |   . |   0
          1 |   1 |   1
          2 |   . |   2

          0 |   . |   0
          1 |   1 |   1
          2 |   1 |   2
        '

        :param item: Slice for rows region
        :return: Extracted sub-vector
        """

        if isinstance(item, slice):
            i = item.start
            nrows = item.stop

            assert item.step is None

            if i is None:
                i = 0

            assert 0 <= i < self.nrows

            if nrows is None:
                nrows = self.nrows

            return self.extract_vector(i, nrows - i)

        raise Exception("Invalid vector slicing")

    def __setitem__(self, key, value):
        """
        Sets specifies vector index as True.

        >>> vector = Vector.empty(4)
        >>> vector[0] = True
        >>> vector[2] = True
        >>> vector[3] = True
        >>> print(vector)
        '
          0 |   1 |   0
          1 |   . |   1
          2 |   1 |   2
          3 |   1 |   3
        '

        :param key: Index of the vector row to set True.
        :param value: Must be True always.
        :return:
        """

        assert value is True

        if isinstance(key, int):
            i = key

            status = wrapper.loaded_dll.cuBool_Vector_SetElement(
                self.hnd,
                ctypes.c_uint(i)
            )

            bridge.check(status)
            return

        raise Exception("Invalid item assignment")
