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
        self.is_managed = Wrapper.__get_use_managed_mem()
        self.force_cpu = Wrapper.__get_force_cpu()

        try:
            # Try from config if present
            self.load_path = os.environ["CUBOOL_PATH"]
        except KeyError:
            # Fallback to package directory
            source_path = pathlib.Path(__file__).resolve()
            self.load_path = Wrapper.__get_lib_path(source_path.parent)

        assert self.load_path

        self.loaded_dll = bridge.load_and_configure(self.load_path)
        self.__setup_library()

    def __del__(self):
        self.__release_library()

    def __setup_library(self):
        status = self.loaded_dll.cuBool_Initialize(ctypes.c_uint(bridge.get_init_hints(
            force_cpu_backend=self.force_cpu, is_gpu_mem_managed=self.is_managed)))

        bridge.check(status)

    def __release_library(self):
        status = self.loaded_dll.cuBool_Finalize()
        bridge.check(status)

    @classmethod
    def __get_lib_path(cls, prefix):
        for entry in os.listdir(prefix):
            if "cubool" in str(entry):
                return prefix / entry

        return None

    @classmethod
    def __get_force_cpu(cls):
        try:
            memory = os.environ["CUBOOL_BACKEND"]
            return True if memory.lower() == "cpu" else False
        except KeyError:
            return False

    @classmethod
    def __get_use_managed_mem(cls):
        try:
            memory = os.environ["CUBOOL_MEM"]
            return True if memory.lower() == "managed" else False
        except KeyError:
            return False
