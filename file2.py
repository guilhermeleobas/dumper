from numba import njit


@njit
def func_2(a):
    return a + 1