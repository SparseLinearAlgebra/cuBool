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

