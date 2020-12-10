import cubool
import ctypes

print("Enter path to lib")
path_to_lib = input()

cubool.CuBoolLibLoad(path_to_lib)
cubool.CuBool_Instance_New()

test_matrix = cubool.CuBool_Matrix_New(ctypes.c_int(100), ctypes.c_int(100))

cubool.CuBool_Instance_Free()