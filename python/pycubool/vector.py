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

        hnd = ctypes.c_void_p(0)

        status = wrapper.loaded_dll.cuBool_Vector_New(
            ctypes.byref(hnd), ctypes.c_uint(nrows)
        )

        bridge.check(status)

        return Vector(hnd)

    @classmethod
    def from_list(cls, nrows, rows, is_sorted=False, no_duplicates=False):
        out = cls.empty(nrows)
        out.build(rows, is_sorted=is_sorted, no_duplicates=no_duplicates)
        return out

    @classmethod
    def generate(cls, nrows, density: float):
        density = min(1.0, max(density, 0.0))
        nvals_max = nrows
        nvals_to_gen = int(nvals_max * density)

        m = nrows
        rows = list()

        for i in range(nvals_to_gen):
            rows.append(random.randrange(0, m))

        return Vector.from_list(nrows=nrows, rows=rows, is_sorted=False, no_duplicates=False)

    def build(self, rows, is_sorted=False, no_duplicates=False):

        nvals = len(rows)
        t_rows = (ctypes.c_uint * len(rows))(*rows)

        status = wrapper.loaded_dll.cuBool_Vector_Build(
            self.hnd, t_rows,
            ctypes.c_uint(nvals),
            ctypes.c_uint(bridge.get_build_hints(is_sorted, no_duplicates))
        )

        bridge.check(status)

    def dup(self):
        hnd = ctypes.c_void_p(0)

        status = wrapper.loaded_dll.cuBool_Vector_Duplicate(
            self.hnd, ctypes.byref(hnd)
        )

        bridge.check(status)
        return Vector(hnd)

    def set_marker(self, marker: str):
        assert marker is not None

        status = wrapper.loaded_dll.cuBool_Vector_SetMarker(
            self.hnd, marker.encode("utf-8")
        )

        bridge.check(status)

    @property
    def marker(self):
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
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Vector_Nrows(
            self.hnd, ctypes.byref(result)
        )

        bridge.check(status)
        return int(result.value)

    @property
    def nvals(self) -> int:
        result = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Vector_Nvals(
            self.hnd, ctypes.byref(result)
        )

        bridge.check(status)
        return int(result.value)

    def to_list(self):
        count = self.nvals

        rows = (ctypes.c_uint * count)()
        nvals = ctypes.c_uint(count)

        status = wrapper.loaded_dll.cuBool_Vector_ExtractValues(
            self.hnd, rows, ctypes.byref(nvals)
        )

        bridge.check(status)

        return rows

    def to_string(self, width=3):
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
        if out is None:
            out = Vector.empty(self.nrows)

        status = wrapper.loaded_dll.cuBool_Vector_EWiseAdd(
            out.hnd,
            self.hnd,
            other.hnd,
            ctypes.c_uint(bridge.get_ewiseadd_hints(time_check=time_check))
        )

    def reduce(self, time_check=False):
        value = ctypes.c_uint(0)

        status = wrapper.loaded_dll.cuBool_Vector_Reduce(
            ctypes.byref(value),
            self.hnd,
            ctypes.c_uint(bridge.get_reduce_hints(time_check=time_check))
        )

        bridge.check(status)
        return int(value.value)

    def equals(self, other) -> bool:
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
        return self.to_list()

    def __getitem__(self, item):
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
