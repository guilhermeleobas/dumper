import pytest
import numpy as np
from pathlib import Path
from numba import njit
from numba.core import types
from numba.core.typing import signature
from dumper import (AddSignature, Compile, FixMissingImports,
                    FixMissingGlobals, Pipeline, ReplacePlaceholders, Save)




def test_add(tmp_path):
    add = """
    @{decorator}
    def add(a, b):
        sz = len(a)
        y = np.zeros(sz, dtype=np.int64)
        for i in range(sz):
            y[i] = a[i] + b - {c}
        return np.sum(y)
    """

    sig0 = signature(None, types.Array(types.int64, 1, 'C'), types.int64)
    sig1 = signature(None, types.Array(types.float64, 1, 'C'), types.float64)

    pipe = Pipeline(
        ReplacePlaceholders({"decorator": njit, "c": 123}),
        Compile(),
        FixMissingGlobals(globals()),
        FixMissingImports(),
        AddSignature(sig0),
        AddSignature(sig1),
        Save(tmp_path),
    )

    pipe.run(add)

