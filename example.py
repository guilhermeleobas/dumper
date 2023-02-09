from pathlib import Path

import numpy as np
from numba import njit, types
from numba.core.typing import signature

from step import (AddSignature, Compile, FixMissingImports, FixMissingGlobals,
                  Pipeline, ReplacePlaceholders, Save)

add = """
@{decorator}
def add(a, b):
    sz = len(a)
    y = np.zeros(sz, dtype=np.int64)
    for i in range(sz):
        y[i] = a[i] + b - {c}
    return np.sum(y)
"""

bar = """
@{decorator}
def bar(a):
    return a + np.abs(b) + {c}
"""

b = 3

source = add
# ignore return type as it will be computed by Numba
sig = signature(None, types.int64)
sig2 = signature(None, types.float64)

pipe = Pipeline(
    ReplacePlaceholders({"decorator": njit, "c": 123}),
    Compile(),
    FixMissingGlobals(globals()),
    FixMissingImports(),
    AddSignature(sig),
    AddSignature(sig2),
    Save(Path("./")),
)

pipe.run(source)
