"""
Example of iterating over matrix cells 
"""

import pycubool as cb

#
#  Matrix initialization
#

shape = (3, 3)                          # Adjacency matrices shape
a = cb.Matrix.empty(shape=shape)
a[1, 0] = True
a[1, 1] = True
a[1, 2] = True
a[0, 1] = True


#
#  Iterating over matrix elements
#

print("Filled cell indices (row, column):")
for i, j in a:
    print(f"({i}, {j})", end=" ")
