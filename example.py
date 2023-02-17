from pathlib import Path

import numpy as np
from numba import njit, types
from numba.core.typing import signature

from dumper import (AddSignature, Compile, Debug, AddMissingGlobalsVariables,
                    IncludeImports, Pipeline, ReplacePlaceholders, Save,
                    AddMissingInformation, Abort)


class Config:
    @classmethod
    def get_jit_decorator(cls):
        return njit


@njit
def func(x, y):
    return x + y


@njit
def foo(x):
    return len(range(x)) + func_0(x, 0) + func_1(x, 1) + func_2(x, 2) + abcd(x, 4)


bar = """
@{decorator}
def bar(a):
    return {baz}(a) + np.abs(b) + foo({c})
"""

ns = {
    'decorator': Config.get_jit_decorator(),
    'func_0': func,
    'func_1': func,
    'func_2': func,
    'abcd': func,
}

b = 3

source = bar
# ignore return type as it will be computed by Numba
sig = signature(None, types.int64)
sig2 = signature(None, types.float64)

pipe = Pipeline(
    ReplacePlaceholders({"decorator": Config.get_jit_decorator(), "c": 123, "baz": foo}),
    Compile(),
    AddMissingInformation(globals(), ns),
    AddMissingGlobalsVariables(globals()),
    IncludeImports('import numpy as np', 'from numba import njit'),
    AddSignature(sig),
    AddSignature(sig2),
    Save(Path("./")),
    Debug(),
)

pipe.run(source)
