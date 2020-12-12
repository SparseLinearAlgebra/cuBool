from CuBoolWrapper import _lib_wrapper
from codes import check


class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix

    def __del__(self):
        check(_lib_wrapper.matrix_free(self, self.matrix))

    @classmethod
    def sparse(cls, nrows=0, ncols=0):
        temp = _lib_wrapper.matrix_new(nrows, ncols)
        check(temp[0])
        m = cls(temp[1])
        return m

    def resize(self, nrows, ncols):
        check(_lib_wrapper.matrix_resize(self.matrix, nrows, ncols))

    def build(self, rows, cols, nvals):
        check(_lib_wrapper.matrix_build(self.matrix, rows, cols, nvals))
