from . import wrapper
from . import dll
from . import matrix


__all__ = [
    "add"
]


def add(result_matrix: matrix.Matrix, a_matrix: matrix.Matrix):
    status = wrapper.loaded_dll.CuBool_Matrix_Add(wrapper.instance,
                                                  result_matrix.hnd,
                                                  a_matrix.hnd)

    dll.check(status)
