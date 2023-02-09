from pathlib import Path

import numpy as np
from numba import njit, types
from numba.core.typing import signature

from step import (AddSignature, Compile, FixMissingImports, FixUnusedGlobals,
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

source = bar
# ignore return type as it will be computed by Numba
sig = signature(None, types.int64)

pipe = Pipeline(
    ReplacePlaceholders({"decorator": njit, "c": 123}),
    Compile(),
    FixUnusedGlobals(globals()),
    FixMissingImports(),
    AddSignature(sig),
    Save(Path("./")),
)

print(pipe.run(source))
