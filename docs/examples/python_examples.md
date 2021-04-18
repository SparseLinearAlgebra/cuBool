# Detailed Example

## Matrix creation example
```Python
"""
Examples of different ways to create matrices 
"""

import pycubool as cb

#
#  Creation an empty matrix of a known form
#

shape = (3, 3)                          # Matrix shape
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

shape = (3, 3)                          # Matrix shape
a = cb.Matrix.empty(shape=shape)        # Creating matrix
rows = [0, 1, 1, 1]                     # Row indices of values
cols = [0, 0, 1, 2]                     # Column indices of values
a.build(rows, cols)                     # Filling matrix

print("Second example:")
print(a, sep='\n')                      # Matrix output

#
# Creating a matrix via shape and arrays of significant element indices
#

shape = (3, 3)                          # Matrix shape
rows = [0, 1, 1, 1]                     # Row indices of values
cols = [0, 0, 1, 2]                     # Column indices of values
a = cb.Matrix.from_lists(shape=shape, rows=rows, cols=cols)

print("Third example:")
print(a, sep='\n')                      # Matrix output


#
# Generate random matrix by determining the shape and density - the percentage of non-zeros elements
#

shape = (4, 4)                          # Matrix shape
density = 0.5                           # Matrix filling density
a = cb.Matrix.generate(shape=shape, density=density)

print("Fourth example:")
print(a, sep='\n')                      # Matrix output
```

Output:
```
First example:
        0   1   2 
  0 |   1   .   . |   0
  1 |   1   1   1 |   1
  2 |   .   .   . |   2
        0   1   2 

Second example:
        0   1   2 
  0 |   1   .   . |   0
  1 |   1   1   1 |   1
  2 |   .   .   . |   2
        0   1   2 

Third example:
        0   1   2 
  0 |   1   .   . |   0
  1 |   1   1   1 |   1
  2 |   .   .   . |   2
        0   1   2 

Fourth example:
        0   1   2   3 
  0 |   1   .   .   . |   0
  1 |   1   1   .   . |   1
  2 |   .   .   .   . |   2
  3 |   .   1   .   . |   3
        0   1   2   3 
```

## Element-wise addition example

```Python
"""
Example of element-wise addition of two matrices
"""

import pycubool as cb

#
#  Matrix initialization
#

shape = (3, 3)                          # Matrix shape
a = cb.Matrix.empty(shape=shape)
a[0, 0] = True
a[2, 0] = True

shape = (3, 3)                          # Matrix shape
b = cb.Matrix.empty(shape=shape)
b[1, 1] = True
b[1, 2] = True

#
#  Matrix element-wise addition
#

result = a.ewiseadd(b)                  # result = a + b

print("Matrix element-wise addition:")
print(result, sep='\n')                 # Matrix output
```

Output:
```
Matrix element-wise addition:
        0   1   2 
  0 |   1   .   . |   0
  1 |   .   1   1 |   1
  2 |   1   .   . |   2
        0   1   2 
```
## Matrix multiplication example
```Python
"""
Examples of matrix multiplication of two matrices
"""

import pycubool as cb

#
#  Matrix initialization
#

a = cb.Matrix.empty(shape=(3, 3))           # Creating an empty matrix of a given shape
a[1, 0] = True
a[1, 1] = True
a[1, 2] = True
a[0, 0] = True

b = cb.Matrix.empty(shape=(3, 3))           # Creating an empty matrix of a given shape
b[0, 1] = True
b[0, 2] = True

#
#  Simple matrix multiplication
#

result = a.mxm(b)                           # result = a * b

print("Simple matrix multiplication:")
print(result, sep='\n')                     # Matrix output

#
# Matrix multiplication with accumulate
#

result = a.mxm(b, out=a, accumulate=True)   # result = a + a * b

print("Matrix multiplication with accumulation:")
print(result, sep='\n')                     # Matrix output
```

Output:
```
Simple matrix multiplication:
        0   1   2 
  0 |   .   1   1 |   0
  1 |   .   1   1 |   1
  2 |   .   .   . |   2
        0   1   2 

Matrix multiplication with accumulation:
        0   1   2 
  0 |   1   1   1 |   0
  1 |   1   1   1 |   1
  2 |   .   .   . |   2
        0   1   2
```

## Matrix duplication example
```Python
"""
Example of matrix duplication 
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
#  Matrix duplicate operation
#

result = a.dup()

print("Result of matrix duplication operation:")
print(result, sep='\n')                 # Matrix output
```

Output:
```
Result of matrix duplication operation:
        0   1   2 
  0 |   .   1   . |   0
  1 |   1   1   1 |   1
  2 |   .   .   . |   2
        0   1   2
```

## Kronecker operation example
```Python
"""
Example of kronecker product of two matrices
"""

import pycubool as cb

#
#  Matrix initialization
#

a = cb.Matrix.empty(shape=(2, 2))           # Creating an empty matrix of a given shape
a[0, 0] = True
a[1, 0] = True

b = cb.Matrix.empty(shape=(2, 2))           # Creating an empty matrix of a given shape
b[1, 1] = True
b[1, 0] = True

#
#  Matrix kronecker product
#

result = a.kronecker(b)                     # result = a x b

print("Matrix kronecker product:")
print(result, sep='\n')                     # Matrix output
```

Output:
```
Matrix kronecker product:
        0   1   2   3 
  0 |   .   .   .   . |   0
  1 |   1   1   .   . |   1
  2 |   .   .   .   . |   2
  3 |   1   1   .   . |   3
        0   1   2   3
```

## Matrix reduce example
```Python
"""
An example of matrix reduction
"""

import pycubool as cb

#
#  Matrix initialization
#

a = cb.Matrix.empty(shape=(3, 3))               # Creating an empty matrix of a given shape
a[1, 0] = True
a[1, 1] = True
a[1, 2] = True
a[0, 1] = True

#
#  Matrix reduce operation
#

result = a.reduce()

print("Result of matrix reduce operation:")
print(result, sep='\n')                         # Matrix output
```

Output:
```
Result of matrix reduce operation:
        0 
  0 |   1 |   0
  1 |   1 |   1
  2 |   . |   2
        0
```

## Matrix extract sub-matrix example
```Python
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
```

Output:
```
First result of extract sub-matrix operation:
        0   1 
  0 |   1   1 |   0
  1 |   .   . |   1
        0   1 

Second result of extract sub-matrix operation:
        0   1   2 
  0 |   .   1   . |   0
  1 |   1   1   1 |   1
  2 |   .   .   . |   2
        0   1   2
```

## Matrix transpose operation example
```Python
"""
Example of matrix transposition 
"""

import pycubool as cb

#
#  Matrix initialization
#

a = cb.Matrix.empty(shape=(3, 3))           # Creating an empty matrix of a given shape
a[1, 0] = True
a[1, 1] = True
a[1, 2] = True
a[0, 1] = True

#
#  Matrix transpose operation
#

result = a.transpose()

print("Result of matrix transpose operation:")
print(result, sep='\n')                     # Matrix output
```

Output:
```
Result of matrix transpose operation:
        0   1   2 
  0 |   .   .   . |   0
  1 |   .   .   . |   1
  2 |   .   .   . |   2
        0   1   2
```

## Extract matrix values example
```Python
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
```

Output:
```
Rows - [0, 1, 1, 1]
Columns - [0, 0, 1, 2]
Values - [(0, 0), (1, 0), (1, 1), (1, 2)]
```

## Iteration over matrix example
```Python
"""
Example of iterating over matrix cells 
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
#  Iterating over matrix elements
#

print("Filled cell indices (row, column):")
for i, j in a:
    print(f"({i}, {j})", end=" ")
```

Output:
```
Filled cell indices (row, column):
(0, 1) (1, 0) (1, 1) (1, 2)
```

## Export matrix to file example
```Python
"""
An example of writing a matrix to a file in .mtx format
"""

import pycubool as cb

#
#  Matrix initialization
#

a = cb.Matrix.empty(shape=(3, 3))               # Creating an empty matrix of a given shape
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
```

Output:
```
Result matrix:
        0   1   2 
  0 |   1   .   . |   0
  1 |   1   1   1 |   1
  2 |   .   .   . |   2
        0   1   2
```

## Import matrix from file example
```Python
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
```

Output:
```
Result of import matrix from file:
        0   1   2 
  0 |   1   1   . |   0
  1 |   .   .   . |   1
  2 |   .   1   1 |   2
        0   1   2
```

## Logger example
```Python
"""
Example of start the logger 
"""

import pycubool as cb

#
#  Example of starting the logger
#

path = "my-example-log.textlog"                     # Set file name in current directory to logged messages
cb.setup_logger(path)

#
#  Matrix initialization
#

a = cb.Matrix.empty(shape=(3, 3))                   # Creating an empty matrix of a given shape
a[1, 0] = True
a[1, 1] = True
a[1, 2] = True
a[0, 0] = True

a.set_marker("a-log-example-matrix")                # Set debug marker for "a" matrix

b = cb.Matrix.empty(shape=(3, 3))                   # Creating an empty matrix of a given shape
b[0, 1] = True
b[0, 2] = True

#
#  Simple matrix multiplication
#

result = a.mxm(b)                                   # result = a * b

print("Simple matrix multiplication:")
print(result, sep='\n')                             # Matrix output

#
# Matrix multiplication with accumulate
#

result = a.mxm(b, out=a, accumulate=True)           # result = a + a * b

print("Matrix multiplication with accumulation:")
print(result, sep='\n')                             # Matrix output
```

Logger output:
```
[         0][         Level::Info] *** cuBool::Logger file ***
[         1][         Level::Info] Cuda device is not presented
[         2][         Level::Info] Create Matrix 0x1a7b3d0 (3,3)
[         3][         Level::Info] Create Matrix 0x1ab72a0 (3,3)
[         4][         Level::Info] Create Matrix 0x1acace0 (3,3)
[         5][         Level::Info] Release Matrix 0x1acace0
[         6][         Level::Info] Release Matrix 0x1ab72a0
[         7][         Level::Info] Release Matrix a-log-example-matrix (0x1a7b3d0)
[         8][         Level::Info] Enabled relaxed library finalize
[         9][         Level::Info] ** cuBool:Finalize backend **
```
