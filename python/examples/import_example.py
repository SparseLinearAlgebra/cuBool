"""
An example of reading a matrix from a file in .mtx format
"""

import pycubool as cb


#
#  Import matrix from file
#

path = "data/input_matrix.mtx"              # relative path to target matrix
a = cb.import_matrix_from_mtx(path)         # read matrix from file

print("Result of import matrix from file:")
print(a, sep='\n')                          # Matrix output
