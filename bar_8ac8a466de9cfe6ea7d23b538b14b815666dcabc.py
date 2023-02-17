# This file contains a reproducer for function `bar`. It consist of all the
# necessary code: imports, globals, Numba types, etc - to reproduce the issue.
# To execute, uncomment the last line: `bar.compile(sig)`, and replace `sig`
# by one of the available signatures

# Mandatory imports
import numpy as np
from numba import njit

# imports for signature to be eval
from numba.core import types
from numba.core.types import *
from numba.core.typing import signature

b = 3
@njit
def foo(x):
    return len(range(x))


@njit
def bar(a):
    return a + np.abs(b) + foo(123)

sig0 = eval("signature(None, types.float64)")
sig1 = eval("signature(None, types.int64)")
# bar.compile(sig.args)