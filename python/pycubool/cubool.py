from ctypes import *

__cuBoolLib = 0

class CuBoolInstanceDesc(Structure):
    _fields_ = []

class CuBoolInstance(Structure):
    _fields_ = []

class CuBoolMatrix(Structure):
    _fields_ = []

__instance = CuBoolInstance()
__instanceDesc = CuBoolInstanceDesc()

def CuBoolLibLoad(path: str):
    global cuBoolLib
    cuBoolLib = cdll.LoadLibrary(path)

def CuBool_Instance_New():
    global __cuBoolLib
    global __instance
    global __instanceDesc
    __cuBoolLib.CuBool_Instance_New.restype = c_int
    __cuBoolLib.CuBool_Instance_New.argtypes = [POINTER(CuBoolInstanceDesc),
                                                POINTER(CuBoolInstance)]
    status = __cuBoolLib.CuBool_Instance_New(pointer(__instanceDesc), pointer(__instance))
    print("Status of creating instance is", status)

def CuBool_Instance_Free():
    global instance
    __cuBoolLib.CuBool_Instance_Free.restype = c_int
    __cuBoolLib.CuBool_Instance_Free.argtypes = [CuBoolInstance]
    status = __cuBoolLib.CuBool_Instance_Free(__instance)
    print("Status of deleting instance is", status)

def CuBool_Matrix_New(nrows: c_int, ncols: c_int):
    global __cuBoolLib
    global __instance

    matrix = CuBoolMatrix()
    __cuBoolLib.CuBool_Matrix_New.restype = c_int
    __cuBoolLib.CuBool_Matrix_New.argtypes = [CuBoolInstance,
                                              POINTER(CuBoolMatrix),
                                              c_int,
                                              c_int]
    status = __cuBoolLib.CuBool_Matrix_New(__instance, pointer(matrix), nrows, ncols)
    print("Status of creating matrix is", status)
    return matrix
