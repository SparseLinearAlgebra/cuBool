"""
cuBool C API Python bridge.

Wraps native C API details for accessing
this functionality via Python CTypes library.

Functionality:
- Flags wrapping
- Functions definitions
- Error checking
"""

import ctypes

__all__ = [
    "load_and_configure",
    "get_init_hints",
    "get_build_hints",
    "get_sub_matrix_hints",
    "get_transpose_hints",
    "get_reduce_hints",
    "get_kronecker_hints",
    "get_mxm_hints",
    "get_ewiseadd_hints",
    "check"
]

_hint_no = 0
_hint_cpu_backend = 1
_hint_gpu_mem_managed = 2
_hint_values_sorted = 4
_hint_accumulate = 8
_hint_relaxed_release = 16
_hint_log_error = 32
_hint_log_warning = 64
_hint_log_all = 128
_hint_no_duplicates = 256
_hint_time_check = 512
_hint_transpose = 1024


def get_log_hints(default=True, error=False, warning=False):
    hints = _hint_no

    if default:
        hints |= _hint_log_all
    if error:
        hints |= _hint_log_error
    if warning:
        hints |= _hint_log_warning

    return hints


def get_init_hints(force_cpu_backend, is_gpu_mem_managed):
    hints = _hint_relaxed_release

    if force_cpu_backend:
        hints |= _hint_cpu_backend
    if is_gpu_mem_managed:
        hints |= _hint_gpu_mem_managed

    return hints


def get_sub_matrix_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_sub_vector_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_transpose_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_reduce_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_reduce_vector_hints(transpose, time_check):
    hints = _hint_no

    if transpose:
        hints |= _hint_transpose
    if time_check:
        hints |= _hint_time_check

    return hints


def get_kronecker_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_mxm_hints(is_accumulated, time_check):
    hints = _hint_no

    if is_accumulated:
        hints |= _hint_accumulate
    if time_check:
        hints |= _hint_time_check

    return hints


def get_vxm_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_mxv_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_ewiseadd_hints(time_check):
    hints = _hint_no

    if time_check:
        hints |= _hint_time_check

    return hints


def get_build_hints(is_sorted, no_duplicates):
    hints = _hint_no

    if is_sorted:
        hints |= _hint_values_sorted
    if no_duplicates:
        hints |= _hint_no_duplicates

    return hints


def load_and_configure(cubool_lib_path: str):
    lib = ctypes.cdll.LoadLibrary(cubool_lib_path)

    status_t = ctypes.c_uint
    index_t = ctypes.c_uint
    hints_t = ctypes.c_uint
    matrix_p = ctypes.c_void_p
    vector_p = ctypes.c_void_p

    p_to_matrix_p = ctypes.POINTER(matrix_p)
    p_to_vector_p = ctypes.POINTER(vector_p)

    lib.cuBool_SetupLogging.restype = status_t
    lib.cuBool_SetupLogging.argtypes = [
        ctypes.POINTER(ctypes.c_char),
        hints_t
    ]

    lib.cuBool_Initialize.restype = status_t
    lib.cuBool_Initialize.argtypes = [
        hints_t
    ]

    lib.cuBool_Finalize.restype = status_t
    lib.cuBool_Finalize.argtypes = []

    lib.cuBool_Matrix_New.restype = status_t
    lib.cuBool_Matrix_New.argtypes = [
        p_to_matrix_p,
        index_t,
        index_t
    ]

    lib.cuBool_Matrix_Free.restype = status_t
    lib.cuBool_Matrix_Free.argtypes = [
        matrix_p
    ]

    lib.cuBool_Matrix_Build.restype = status_t
    lib.cuBool_Matrix_Build.argtypes = [
        matrix_p,
        ctypes.POINTER(index_t),
        ctypes.POINTER(index_t),
        index_t,
        hints_t
    ]

    lib.cuBool_Matrix_SetElement.restype = status_t
    lib.cuBool_Matrix_SetElement.argtypes = [
        matrix_p,
        index_t,
        index_t
    ]

    lib.cuBool_Matrix_SetMarker.restype = status_t
    lib.cuBool_Matrix_SetMarker.argtypes = [
        matrix_p,
        ctypes.POINTER(ctypes.c_char)
    ]

    lib.cuBool_Matrix_Marker.restype = status_t
    lib.cuBool_Matrix_Marker.argtypes = [
        matrix_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(index_t)
    ]

    lib.cuBool_Matrix_ExtractPairs.restype = status_t
    lib.cuBool_Matrix_ExtractPairs.argtypes = [
        matrix_p,
        ctypes.POINTER(index_t),
        ctypes.POINTER(index_t),
        ctypes.POINTER(index_t)
    ]

    lib.cuBool_Matrix_ExtractSubMatrix.restype = status_t
    lib.cuBool_Matrix_ExtractSubMatrix.argtypes = [
        matrix_p,
        matrix_p,
        index_t,
        index_t,
        index_t,
        index_t,
        hints_t
    ]

    lib.cuBool_Matrix_ExtractRow.restype = status_t
    lib.cuBool_Matrix_ExtractRow.argtypes = [
        vector_p,
        matrix_p,
        index_t,
        hints_t
    ]

    lib.cuBool_Matrix_ExtractCol.restype = status_t
    lib.cuBool_Matrix_ExtractCol.argtypes = [
        vector_p,
        matrix_p,
        index_t,
        hints_t
    ]

    lib.cuBool_Matrix_Duplicate.restype = status_t
    lib.cuBool_Matrix_Duplicate.argtypes = [
        matrix_p,
        p_to_matrix_p
    ]

    lib.cuBool_Matrix_Transpose.restype = status_t
    lib.cuBool_Matrix_Transpose.argtypes = [
        matrix_p,
        matrix_p,
        hints_t
    ]

    lib.cuBool_Matrix_Nrows.restype = status_t
    lib.cuBool_Matrix_Nrows.argtype = [
        matrix_p,
        ctypes.POINTER(index_t)
    ]

    lib.cuBool_Matrix_Ncols.restype = status_t
    lib.cuBool_Matrix_Ncols.argtype = [
        matrix_p,
        ctypes.POINTER(index_t)
    ]

    lib.cuBool_Matrix_Nvals.restype = status_t
    lib.cuBool_Matrix_Nvals.cuBool_Matrix_Reduce2 = [
        matrix_p,
        ctypes.POINTER(ctypes.c_size_t)
    ]

    lib.cuBool_Matrix_Reduce.restype = status_t
    lib.cuBool_Matrix_Reduce.argtype = [
        vector_p,
        matrix_p,
        hints_t
    ]

    lib.cuBool_Matrix_Reduce2.restype = status_t
    lib.cuBool_Matrix_Reduce2.argtype = [
        matrix_p,
        matrix_p,
        hints_t
    ]

    lib.cuBool_Matrix_EWiseAdd.restype = status_t
    lib.cuBool_Matrix_EWiseAdd.argtypes = [
        matrix_p,
        matrix_p,
        matrix_p,
        hints_t
    ]

    lib.cuBool_Vector_New.restype = status_t
    lib.cuBool_Vector_New.argtypes = [
        p_to_vector_p,
        index_t
    ]

    lib.cuBool_Vector_Build.restype = status_t
    lib.cuBool_Vector_Build.argtypes = [
        vector_p,
        ctypes.POINTER(index_t),
        index_t,
        hints_t
    ]

    lib.cuBool_Vector_SetElement.restype = status_t
    lib.cuBool_Vector_SetElement.argtypes = [
        vector_p,
        index_t
    ]

    lib.cuBool_Vector_SetMarker.restype = status_t
    lib.cuBool_Vector_SetMarker.argtypes = [
        vector_p,
        ctypes.POINTER(ctypes.c_char)
    ]

    lib.cuBool_Vector_Marker.restype = status_t
    lib.cuBool_Vector_Marker.argtypes = [
        vector_p,
        ctypes.POINTER(ctypes.c_char),
        ctypes.POINTER(index_t)
    ]

    lib.cuBool_Vector_ExtractValues.restype = status_t
    lib.cuBool_Vector_ExtractValues.argtypes = [
        vector_p,
        ctypes.POINTER(index_t),
        ctypes.POINTER(index_t)
    ]

    lib.cuBool_Vector_ExtractSubVector.restype = status_t
    lib.cuBool_Vector_ExtractSubVector.argtypes = [
        vector_p,
        vector_p,
        index_t,
        index_t,
        hints_t
    ]

    lib.cuBool_Vector_Duplicate.restype = status_t
    lib.cuBool_Vector_Duplicate.argtypes = [
        vector_p,
        p_to_vector_p
    ]

    lib.cuBool_Vector_Nvals.restype = status_t
    lib.cuBool_Vector_Nvals.argtypes = [
        vector_p,
        ctypes.POINTER(index_t)
    ]

    lib.cuBool_Vector_Nrows.restype = status_t
    lib.cuBool_Vector_Nrows.argtypes = [
        vector_p,
        ctypes.POINTER(index_t)
    ]

    lib.cuBool_Vector_Free.restype = status_t
    lib.cuBool_Vector_Free.argtypes = [
        vector_p
    ]

    lib.cuBool_Vector_Reduce.restype = status_t
    lib.cuBool_Vector_Reduce.argtypes = [
        ctypes.POINTER(index_t),
        vector_p,
        hints_t
    ]

    lib.cuBool_Vector_EWiseAdd.restype = status_t
    lib.cuBool_Vector_EWiseAdd.argtypes = [
        vector_p,
        vector_p,
        vector_p,
        hints_t
    ]

    lib.cuBool_MxM.restype = status_t
    lib.cuBool_MxM.argtypes = [
        matrix_p,
        matrix_p,
        matrix_p,
        hints_t
    ]

    lib.cuBool_MxV.restype = status_t
    lib.cuBool_MxV.argtypes = [
        vector_p,
        matrix_p,
        vector_p,
        hints_t
    ]

    lib.cuBool_VxM.restype = status_t
    lib.cuBool_VxM.argtypes = [
        vector_p,
        vector_p,
        matrix_p,
        hints_t
    ]

    lib.cuBool_Kronecker.restype = status_t
    lib.cuBool_Kronecker.argtypes = [
        matrix_p,
        matrix_p,
        matrix_p,
        hints_t
    ]

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
