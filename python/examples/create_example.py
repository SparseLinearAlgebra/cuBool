"""
Examples of different ways to create matrices 
"""

import pycubool as cb

#
#  Creation an empty matrix of a known form
#

shape = (3, 3)                          # Adjacency matrices shape
a = cb.Matrix.empty(shape=shape)        # Creating matrix

#
#  Filling values by indices
#

a[1, 0] = True                          # Set "True" value to (1, 0) cell
a[1, 1] = True                          # Set "True" value to (1, 1) cell
a[1, 2] = True                          # Set "True" value to (1, 2) cell
a[0, 0] = True                          # Set "True" value to (0, 0) cell

print("First example:")
print(a, sep='\n')                      # Matrix output

#
# Creation an empty matrix of known shape and filling with given arrays of indices
#

shape = (3, 3)                          # Adjacency matrices shape
a = cb.Matrix.empty(shape=shape)        # Creating matrix
rows = [0, 1, 1, 1]                     # Row indices of values
cols = [0, 0, 1, 2]                     # Column indices of values
a.build(rows, cols)                     # Filling matrix

print("Second example:")
print(a, sep='\n')                      # Matrix output

#
# Creating a matrix via shape and arrays of significant element indices
#

shape = (3, 3)                          # Adjacency matrices shape
rows = [0, 1, 1, 1]                     # Row indices of values
cols = [0, 0, 1, 2]                     # Column indices of values
a = cb.Matrix.from_lists(shape=shape, rows=rows, cols=cols)

print("Third example:")
print(a, sep='\n')                      # Matrix output


#
# Generate random matrix by determining the shape and density - the percentage of non-zeros elements
#

shape = (4, 4)                          # Adjacency matrices shape
density = 0.5                           # Matrix filling density
a = cb.Matrix.generate(shape=shape, density=density)

print("Fourth example:")
print(a, sep='\n')                      # Matrix output
