"""
Examples of matrix multiplication of two matrices
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

b = cb.Matrix.empty(shape=(3, 3))
b[0, 1] = True
b[0, 2] = True

#
#  Simple matrix multiplication
#

result = a.mxm(b)  # result = a * b

print("Simple matrix multiplication:")
print(result, sep='\n')                     # Matrix output

#
# Matrix multiplication with accumulate
#

result = a.mxm(b, out=a, accumulate=True)   # result = a + a * b

print("Matrix multiplication with accumulation:")
print(result, sep='\n')                     # Matrix output
