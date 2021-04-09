"""
An example of writing a matrix to a file in .mtx format
"""

import pycubool as cb

#
#  Matrix initialization
#

a = cb.Matrix.empty(shape=(3, 3))
a[1, 0] = True
a[1, 1] = True
a[1, 2] = True
a[0, 0] = True

#
#  Export matrix to file
#

path = "data/output_matrix.mtx"                 # relative path to target matrix
cb.export_matrix_to_mtx(path, a)                # write matrix to file

#
#  Import this matrix to check the correctness of the writing
#

result = cb.import_matrix_from_mtx(path)        # read matrix from file

print("Result matrix:")
print(result, sep='\n')                         # Matrix output
