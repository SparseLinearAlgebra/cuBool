"""
Example of element-wise addition of two matrices
"""

import pycubool as cb

#
#  Matrix initialization
#

shape = (3, 3)                          # Adjacency matrices shape
a = cb.Matrix.empty(shape=shape)
a[0, 0] = True
a[2, 0] = True

shape = (3, 3)                          # Adjacency matrices shape
b = cb.Matrix.empty(shape=shape)
b[1, 1] = True
b[1, 2] = True

#
#  Matrix element-wise addition
#

result = a.ewiseadd(b)                  # result = a + b

print("Matrix element-wise addition:")
print(result, sep='\n')                 # Matrix output
