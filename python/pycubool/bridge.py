import ctypes

__all__ = [
    "load_and_configure",
    "get_build_hints",
    "get_mxm_hints",
    "check"
]

_hint_no = 0x0
_hint_cpu_backend = 0x1
_hint_gpu_mem_managed = 0x2
_hint_values_sorted = 0x4
_hint_accumulate = 0x8
_hint_relaxed_release = 0x16


def get_init_hints(is_gpu_mem_managed):
    return _hint_relaxed_release | (_hint_gpu_mem_managed if is_gpu_mem_managed else _hint_no)


def get_mxm_hints(is_accumulated):
    return _hint_accumulate if is_accumulated else _hint_no


def get_build_hints(is_sorted):
    return _hint_values_sorted if is_sorted else _hint_no


def load_and_configure(cubool_lib_path: str):
    lib = ctypes.cdll.LoadLibrary(cubool_lib_path)

    hints_t = ctypes.c_uint
    matrix_p = ctypes.c_void_p
    p_to_matrix_p = ctypes.POINTER(matrix_p)

    lib.cuBool_Initialize.restype = ctypes.c_uint
    lib.cuBool_Initialize.argtypes = [hints_t]

    lib.cuBool_Finalize.restype = ctypes.c_uint
    lib.cuBool_Finalize.argtypes = []

    lib.cuBool_Matrix_New.restype = ctypes.c_uint
    lib.cuBool_Matrix_New.argtypes = [p_to_matrix_p,
                                      ctypes.c_uint,
                                      ctypes.c_uint]

    lib.cuBool_Matrix_Free.restype = ctypes.c_uint
    lib.cuBool_Matrix_Free.argtypes = [matrix_p]

    lib.cuBool_Matrix_Build.restype = ctypes.c_uint
    lib.cuBool_Matrix_Build.argtypes = [matrix_p,
                                        ctypes.POINTER(ctypes.c_uint),
                                        ctypes.POINTER(ctypes.c_uint),
                                        ctypes.c_uint,
                                        hints_t]

    lib.cuBool_Matrix_ExtractPairs.restype = ctypes.c_uint
    lib.cuBool_Matrix_ExtractPairs.argtypes = [matrix_p,
                                               ctypes.POINTER(ctypes.c_uint),
                                               ctypes.POINTER(ctypes.c_uint),
                                               ctypes.POINTER(ctypes.c_uint)]

    lib.cuBool_Matrix_Duplicate.restype = ctypes.c_uint
    lib.cuBool_Matrix_Duplicate.argtypes = [matrix_p,
                                            p_to_matrix_p]

    lib.cuBool_Matrix_Transpose.restype = ctypes.c_uint
    lib.cuBool_Matrix_Transpose.argtypes = [matrix_p,
                                            matrix_p]

    lib.cuBool_Matrix_Nrows.restype = ctypes.c_uint
    lib.cuBool_Matrix_Nrows.argtype = [matrix_p,
                                       ctypes.POINTER(ctypes.c_uint)]

    lib.cuBool_Matrix_Ncols.restype = ctypes.c_uint
    lib.cuBool_Matrix_Ncols.argtype = [matrix_p,
                                       ctypes.POINTER(ctypes.c_uint)]

    lib.cuBool_Matrix_Nvals.restype = ctypes.c_uint
    lib.cuBool_Matrix_Nvals.argtype = [matrix_p,
                                       ctypes.POINTER(ctypes.c_size_t)]

    lib.cuBool_Matrix_EWiseAdd.restype = ctypes.c_uint
    lib.cuBool_Matrix_EWiseAdd.argtypes = [matrix_p,
                                           matrix_p,
                                           matrix_p]

    lib.cuBool_MxM.restype = ctypes.c_uint
    lib.cuBool_MxM.argtypes = [matrix_p,
                               matrix_p,
                               matrix_p,
                               hints_t]

    lib.cuBool_Kronecker.restype = ctypes.c_uint
    lib.cuBool_Kronecker.argtypes = [matrix_p,
                                     matrix_p,
                                     matrix_p]

    return lib


"""
/** Possible status codes that can be returned from cubool api */

/** Successful execution of the function */
CUBOOL_STATUS_SUCCESS,

/** Generic error code */
CUBOOL_STATUS_ERROR,

/** No cuda compatible device in the system */
CUBOOL_STATUS_DEVICE_NOT_PRESENT,

/** Device side error */
CUBOOL_STATUS_DEVICE_ERROR,

/** Failed to allocate memory on cpy or gpu side */
CUBOOL_STATUS_MEM_OP_FAILED,

/** Passed invalid argument to some function */
CUBOOL_STATUS_INVALID_ARGUMENT,

/** Call of the function is not possible for some context */
CUBOOL_STATUS_INVALID_STATE

/** Failed to select supported backend for computations */
CUBOOL_STATUS_BACKEND_ERROR

/** Some library feature is not implemented */
CUBOOL_STATUS_NOT_IMPLEMENTED
"""
_status_codes_mappings = {
    0: "CUBOOL_STATUS_SUCCESS",
    1: "CUBOOL_STATUS_ERROR",
    2: "CUBOOL_STATUS_DEVICE_NOT_PRESENT",
    3: "CUBOOL_STATUS_DEVICE_ERROR",
    4: "CUBOOL_STATUS_MEM_OP_FAILED",
    5: "CUBOOL_STATUS_INVALID_ARGUMENT",
    6: "CUBOOL_STATUS_INVALID_STATE",
    7: "CUBOOL_STATUS_BACKEND_ERROR",
    8: "CUBOOL_STATUS_NOT_IMPLEMENTED"
}

_success = 0


def check(status_code):
    if status_code != _success:
        raise Exception(_status_codes_mappings[status_code])
