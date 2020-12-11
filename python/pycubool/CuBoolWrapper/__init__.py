import atexit
import os
from ctypes import *
from .ctypesWrapper import *
from .codes import check

_lib_wrapper = None


class InternalWrapper:

    def __init__(self):
        self._load_path = os.environ["CUBOOL_PATH"]
        self._lib = cdll.LoadLibrary(self._load_path)
        self._instance_new()

        self._declarations = FuncDeclaration(self._lib)

    def unload(self):
        self._instance_free()
        del self._lib
        handle = self._lib._handle
        self._lib.dlclose(handle)

    def _instance_new(self):
        self._lib.CuBool_Instance_New.restype = c_int
        self._lib.CuBool_Instance_New.argtypes = [POINTER(CuBoolInstanceDesc),
                                                  POINTER(CuBoolInstance)]
        self._instance = CuBoolInstance()
        self._descInstance = CuBoolInstanceDesc()
        status = self._lib.CuBool_Instance_New(pointer(self._instance),
                                               pointer(self._descInstance))
        check(status.value)
        # if status != c_int(0):
        #     raise Exception("Error at create instance. Code - " + str(status))
        # print("Status of creating instance is", status)

    def _instance_free(self):
        self._lib.CuBool_Instance_Free.restype = c_int
        self._lib.CuBool_Instance_Free.argtypes = [POINTER(CuBoolInstanceDesc),
                                                   POINTER(CuBoolInstance)]
        status = self._lib.CuBool_Instance_Free(self._instance)
        check(status.value)

    def matrix_new(self, nrows, ncols):
        matrix = CuBoolMatrix()
        status = self._declarations.matrix_creator(self._instance,
                                                   pointer(matrix),
                                                   c_int(nrows),
                                                   c_int(ncols))
        return status.value, matrix

    def matrix_free(self, matrix):
        status = self._declarations.matrix_destructor(self._instance,
                                                      matrix)
        return status.value

    def matrix_resize(self, matrix, nrows, ncols):
        status = self._declarations.matrix_resizer(self._instance,
                                                   matrix,
                                                   c_int(nrows),
                                                   c_int(ncols))
        return status.value

    def matrix_build(self, matrix, rows, cols, nvals):
        t_rows = (c_int * len(rows))(*rows)
        t_cols = (c_int * len(cols))(*cols)

        status = self._declarations.matrix_builder(self._instance,
                                                   matrix,
                                                   pointer(t_rows),
                                                   pointer(t_cols),
                                                   c_int(nvals))
        return status.value

    def matrix_add(self, acc_matrix, add_matrix):
        status = self._declarations.matrix_builder(self._instance,
                                                   acc_matrix,
                                                   add_matrix)
        return status.value, acc_matrix


_lib_wrapper = InternalWrapper()


def action_at_exit():
    global _lib_wrapper
    _lib_wrapper.unload()
