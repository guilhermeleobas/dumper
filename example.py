from pathlib import Path

import sys
import numpy as np
from numba import njit, types
from numba.core.typing import signature

from dumper import (AddSignature, Compile, Debug, AddMissingGlobalsVariables,
                    IncludeImports, Pipeline, ReplacePlaceholders, Save,
                    AddMissingInformation, Abort)

from file import func_1


class Config:
    @classmethod
    def get_jit_decorator(cls):
        return njit


@njit
def func(x, y):
    return x + y


@njit
def foo(x):
    return len(range(x)) + fn_0(x, 0) + fn_1(x, 1) + fn_2(x, 2) + abcd(x, 4)

source = """
@Config.get_jit_decorator()
def bar(a):
    return foo(a) + np.abs(b) + foo(a)
"""

ns = {
    # 'decorator': 'Config.get_jit_decorator()',
    'fn_0': func,
    'fn_1': func,
    'fn_2': func,
    'abcd': func,
    'foo': foo,
    'np': np,
    'b': 3,
}

# ignore return type as it will be computed by Numba
sig = signature(None, types.int64)
sig2 = signature(None, types.float64)

pipe = Pipeline(
    # ReplacePlaceholders({"decorator": Config.get_jit_decorator(), "c": 123, "baz": foo}),
    Compile(),
    AddMissingInformation(globals(), ns, sys.modules[__name__]),
    AddMissingGlobalsVariables(globals() | ns),
    IncludeImports('import numpy as np', 'from numba import njit', 'from example import Config'),
    AddSignature(sig),
    AddSignature(sig2),
    Save(Path("./")),
    Debug(),
).run(source)
