from CuBoolWrapper import _lib_wrapper, check


def Add(accumulate_matrix, add_matrix):
    temp = _lib_wrapper.matrix_add(accumulate_matrix, add_matrix)
    check(temp[0])
    return temp[1]
