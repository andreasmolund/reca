import random as rand
import numpy as np


rand.seed(20161110)


def normalized_addition(a, b):
    """"Normalized addition" according to Yilmaz:
    "entries with value 2 (i.e. 1 + 1) become 1,
    with value 0 stay 0 (i.e. 0 + 0),
    and with value 1 (i.e. 0 + 1) are decided randomly"

    :param a:
    :param b:
    :return: vector elements added
    """
    s = a + b
    if s == 2:
        return 1
    elif s == 0:
        return 0
    elif s == 1:
        return rand.randint(0, 1)


def bitwise_or(a, b):
    """Bitwise OR

    :param a:
    :param b:
    :return:
    """
    return a | b


def bitwise_xor(a, b):
    """Bitwise XOR

    :param a:
    :param b:
    :return:
    """
    return a ^ b


def bitwise_left(a, b):
    return a


def bitwise_right(a, b):
    return b

combine = np.vectorize(normalized_addition)
