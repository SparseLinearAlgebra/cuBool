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
