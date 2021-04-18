"""
Examples of extraction matrix values as lists 
"""

import pycubool as cb

#
#  Creation an empty matrix of a known form
#

shape = (3, 3)                          # Matrix shape
a = cb.Matrix.empty(shape=shape)        # Creating matrix
a[1, 0] = True
a[1, 1] = True
a[1, 2] = True
a[0, 0] = True


#
# Extract values as two lists - rows and columns
# By default, a ctypes object is returned
#

rows, columns = a.to_lists()
print(f"Rows - {list(rows)}")
print(f"Columns - {list(columns)}")

#
# Extract values as one list - pair of indices (i, j) - list of edges
#

values = a.to_list()

print(f"Values - {values}")
