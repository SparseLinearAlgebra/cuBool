import os
import ctypes
from . import bridge

__all__ = [
    "singleton",
    "loaded_dll",
    "instance",
    "init_wrapper"
]

# Wrapper
singleton = None

# Exposed imported dll (owned by _lib_wrapper)
loaded_dll = None

# Exposed instance reference (owned by _lib_wrapper)
instance = None


def init_wrapper():
    global singleton
    global loaded_dll
    global instance

    singleton = Wrapper()
    loaded_dll = singleton.loaded_dll
    instance = singleton.instance


class Wrapper:
    """
    Instance of this class represents the main access point
    to the C cubool library. It stores:

     - Ctypes imported dll instance
     - Ctypes functions declarations
     - Instance of the library
     - Import path

    At exit global instance of the class in properly destroyed.
    Instance of the cubool in released and the dll is closed.
    """

    def __init__(self):
        self.instance = None
        self.loaded_dll = None

        self.load_path = os.environ["CUBOOL_PATH"]
        self.loaded_dll = bridge.load_and_configure(self.load_path)

        self._setup_instance()

    def __del__(self):
        self._release_instance()
        pass

    def _setup_instance(self):
        self.instance = ctypes.c_void_p(None)
        self._descInstance = bridge.configure_instance_desc()
        self._descInstance.memoryType = 0

        status = self.loaded_dll.cuBool_Instance_NewExt(ctypes.byref(self._descInstance),
                                                        ctypes.byref(self.instance))

        bridge.check(status)

    def _release_instance(self):
        status = self.loaded_dll.cuBool_Instance_Free(self.instance)
        bridge.check(status)
