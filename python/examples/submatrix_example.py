"""
Examples of extracting sub-matrices 
"""

import pycubool as cb

#
#  Matrix initialization
#

shape = (3, 3)                          # Matrix shape
a = cb.Matrix.empty(shape=shape)
a[1, 0] = True
a[1, 1] = True
a[1, 2] = True
a[0, 1] = True

#
#  Cut of a 2x2 (third param) matrix
#  below and to the right of the specified index (first and second params)
#  of the original matrix
#

result = a.extract_matrix(1, 1, (2, 2))

print("First result of extract sub-matrix operation:")
print(result, sep='\n')                 # Matrix output


#
#  Create duplicate of original matrix by extract a matrix with 3x3 shape
#

result = a.extract_matrix(0, 0, (3, 3))

print("Second result of extract sub-matrix operation:")
print(result, sep='\n')                 # Matrix output
