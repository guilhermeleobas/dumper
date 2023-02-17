from pathlib import Path

import numpy as np
from numba import njit, types
from numba.core.typing import signature

from dumper import (AddSignature, Compile, Debug, FixMissingGlobalsVariables,
                    IncludeImports, Pipeline, ReplacePlaceholders, Save)


@njit
def foo(x):
    return len(range(x))


bar = """
@{decorator}
def bar(a):
    return a + np.abs(b) + foo({c})
"""

b = 3

source = bar
# ignore return type as it will be computed by Numba
sig = signature(None, types.int64)
sig2 = signature(None, types.float64)

pipe = Pipeline(
    ReplacePlaceholders({"decorator": njit, "c": 123}),
    Compile(),
    FixMissingGlobalsVariables(globals()),
    IncludeImports('import numpy as np', 'from numba import njit'),
    AddSignature(sig),
    AddSignature(sig2),
    Save(Path("./")),
    Debug(),
)

pipe.run(source)
