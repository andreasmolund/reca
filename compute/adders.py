# Vectorizable operators/adders.
# Used in addition of state vector and input vector.

import random

import numpy as np

# rand.seed(20170131)


def normalized_addition(a, b):
    """"Normalized addition" according to Yilmaz:
    "entries with value 2 (i.e. 1 + 1) become 1,
    with value 0 stay 0 (i.e. 0 + 0),
    and with value 1 (i.e. 0 + 1) are decided randomly"
    0 0 | 0
    0 1 | 0 or 1
    1 0 | 0 or 1
    1 1 | 1

    :param a:
    :param b:
    :return: vector elements added
    """
    if a + b == 1:
        return random.randint(0, 1)
    else:
        return a | b


def bitwise_or(a, b):
    return a | b


def bitwise_xor(a, b):
    return a ^ b


def bitwise_left(a, b):
    return a


def bitwise_right(a, b):
    return b


combine = np.vectorize(bitwise_or)
