"""
An example of matrix reduction
"""

import pycubool as cb

#
#  Matrix initialization
#

a = cb.Matrix.empty(shape=(3, 3))
a[1, 0] = True
a[1, 1] = True
a[1, 2] = True
a[0, 1] = True

#
#  Matrix reduce operation
#

result = a.reduce()

print("Result of matrix reduce operation:")
print(result, sep='\n')
