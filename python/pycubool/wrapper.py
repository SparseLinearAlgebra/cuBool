"""
cuBool library state wrapper.
Provides global access points and initialization logic for library API.
"""

import os
import ctypes
import pathlib
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

    At exit the library is properly finalized.
    Instance of the cubool library in released and the shared library is closed.
    """

    def __init__(self):
        self.loaded_dll = None
        self.lib_object_name = "libcubool.so"

        try:
            # Try from config if present
            self.load_path = os.environ["CUBOOL_PATH"]
        except KeyError:
            # Fallback to package directory
            source_path = pathlib.Path(__file__).resolve()
            self.load_path = str(source_path.parent / self.lib_object_name)

        self.loaded_dll = bridge.load_and_configure(self.load_path)
        self._setup_library()

    def __del__(self):
        self._release_library()
        pass

    def _setup_library(self):
        status = self.loaded_dll.cuBool_Initialize(ctypes.c_uint(bridge.get_init_hints(False, False)))
        bridge.check(status)

    def _release_library(self):
        status = self.loaded_dll.cuBool_Finalize()
        bridge.check(status)
