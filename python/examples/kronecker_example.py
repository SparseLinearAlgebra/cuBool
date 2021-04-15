"""
Example of kronecker product of two matrices
"""

import pycubool as cb

#
#  Matrix initialization
#

a = cb.Matrix.empty(shape=(2, 2))           # Adjacency matrices shape
a[0, 0] = True
a[1, 0] = True

b = cb.Matrix.empty(shape=(2, 2))           # Adjacency matrices shape
b[1, 1] = True
b[1, 0] = True

#
#  Matrix kronecker product
#

result = a.kronecker(b)                     # result = a x b

print("Matrix kronecker product:")
print(result, sep='\n')                     # Matrix output
