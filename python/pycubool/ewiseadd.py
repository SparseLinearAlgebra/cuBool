from . import wrapper
from . import bridge
from . import matrix

__all__ = [
    "ewiseadd"
]


def ewiseadd(result_matrix: matrix.Matrix, a_matrix: matrix.Matrix, b_matrix: matrix.Matrix):
    status = wrapper.loaded_dll.cuBool_Matrix_EWiseAdd(result_matrix.hnd,
                                                       a_matrix.hnd,
                                                       b_matrix.hnd)

    bridge.check(status)
