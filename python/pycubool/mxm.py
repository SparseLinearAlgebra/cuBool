from . import wrapper
from . import bridge
from . import matrix
import ctypes

__all__ = [
    "mxm"
]


def mxm(result_matrix: matrix.Matrix, a_matrix: matrix.Matrix, b_matrix: matrix.Matrix, accumulate=True):
    status = wrapper.loaded_dll.cuBool_MxM(result_matrix.hnd,
                                           a_matrix.hnd,
                                           b_matrix.hnd,
                                           ctypes.c_uint(bridge.get_mxm_hints(accumulate)))

    bridge.check(status)
