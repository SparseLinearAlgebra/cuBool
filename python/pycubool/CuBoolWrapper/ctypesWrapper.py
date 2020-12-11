from ctypes import *


class CuBoolInstance(Structure):
    _fields_ = []


class CuBoolInstanceDesc(Structure):
    _fields_ = []


class CuBoolMatrix(Structure):
    _fields_ = []


class FuncDeclaration:
    def __init__(self, lib):
        self.matrix_creator = FuncDeclaration._matrix_creator(lib)
        self.matrix_destructor = FuncDeclaration._matrix_destructor(lib)
        self.matrix_resizer = FuncDeclaration._matrix_resizer(lib)
        self.matrix_builder = FuncDeclaration._matrix_builder(lib)
        self.matrix_adder = FuncDeclaration._matrix_adder(lib)

    @staticmethod
    def _matrix_creator(lib):
        lib.CuBool_Matrix_New.restype = c_int
        lib.CuBool_Matrix_New.argtypes = [CuBoolInstance,
                                          POINTER(CuBoolMatrix),
                                          c_int,
                                          c_int]
        return lib.CuBool_Matrix_New

    @staticmethod
    def _matrix_destructor(lib):
        lib.CuBool_Matrix_Free.restype = c_int
        lib.CuBool_Matrix_Free.argtypes = [CuBoolInstance,
                                           CuBoolMatrix]
        return lib.CuBool_Matrix_Free

    @staticmethod
    def _matrix_resizer(lib):
        lib.CuBool_Matrix_Resize.restype = c_int
        lib.CuBool_Matrix_Resize.argtypes = [CuBoolInstance,
                                             CuBoolMatrix,
                                             c_int,
                                             c_int]
        return lib.CuBool_Matrix_Resize

    @staticmethod
    def _matrix_builder(lib):
        lib.CuBool_Matrix_Build.restype = c_int
        lib.CuBool_Matrix_Build.argtypes = [CuBoolInstance,
                                            CuBoolMatrix,
                                            POINTER(c_int),
                                            POINTER(c_int),
                                            c_int]
        return lib.CuBool_Matrix_Build

    @staticmethod
    def _matrix_adder(lib):
        lib.CuBool_Matrix_Add.restype = c_int
        lib.CuBool_Matrix_Add.argtypes = [CuBoolInstance,
                                          CuBoolMatrix,
                                          CuBoolMatrix]
        return lib.CuBool_Matrix_Add
