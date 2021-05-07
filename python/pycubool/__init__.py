"""
Entry point for `pycubool` package.
pycubool is python wrapper for cuBool library.

The backend cuBool compiled library is initialized automatically
on package import into runtime. State of the library is managed
by the wrapper class. All resources are unloaded automatically on
python runtime exit.

Exported primitives:
- matrix (sparse matrix of boolean values)

For more information refer to:
- cuBool project: https://github.com/JetBrains-Research/cuBool
- bug tracker: https://github.com/JetBrains-Research/cuBool/issues

License:
- MIT
"""

from .wrapper import *
from .utils import *
from .matrix import *
from .vector import *
from .io import *
from .gviz import *

# Setup global module state
init_wrapper()
