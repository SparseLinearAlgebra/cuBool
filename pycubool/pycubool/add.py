from . import wrapper
from . import bridge
from . import matrix

__all__ = [
    "add"
]


def add(result_matrix: matrix.Matrix, a_matrix: matrix.Matrix):
    status = wrapper.loaded_dll.cuBool_EWise_Add(wrapper.instance,
                                                  result_matrix.hnd,
                                                  a_matrix.hnd)

    bridge.check(status)
