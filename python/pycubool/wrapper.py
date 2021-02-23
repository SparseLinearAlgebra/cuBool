import os
import ctypes
from . import bridge

__all__ = [
    "singleton",
    "loaded_dll",
    "init_wrapper"
]

# Wrapper
singleton = None

# Exposed imported dll (owned by _lib_wrapper)
loaded_dll = None


def init_wrapper():
    global singleton
    global loaded_dll

    singleton = Wrapper()
    loaded_dll = singleton.loaded_dll


class Wrapper:
    """
    Instance of this class represents the main access point
    to the C cubool library. It stores:

     - Ctypes imported dll instance
     - Ctypes functions declarations
     - Import path

    At exit global instance of the class is properly destroyed.
    Instance of the cubool library in released and the dll is closed.
    """

    def __init__(self):
        self.loaded_dll = None

        self.load_path = os.environ["CUBOOL_PATH"]
        self.loaded_dll = bridge.load_and_configure(self.load_path)
        self._setup_library()

    def __del__(self):
        self._release_library()
        pass

    def _setup_library(self):
        status = self.loaded_dll.cuBool_Initialize(ctypes.c_uint(bridge.get_init_hints(False)))
        bridge.check(status)

    def _release_library(self):
        status = self.loaded_dll.cuBool_Finalize()
        bridge.check(status)
