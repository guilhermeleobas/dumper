from numba import njit
from file2 import func_2

# @njit
# def func_0(a):
#     return a + 1

@njit
def func_1(a):
    return func_2(a)