# Utilities related to reservoir handling
import numpy as np


def classify_operand(value):
    return 0 if value < 0.5 else 1


classify_output = np.vectorize(classify_operand)


def binarize_deterministic(value):
    return 0 if value < 0 else 1


def binarize_stochastic(value):
    """Hard sigmoid function

    :param value:
    :return:
    """
    tmp = (value+1)/2
    p = max(0, min(1, tmp))
    return 1 if np.random.rand() <= p else 0

print binarize_stochastic(-0.1)


binarize = np.vectorize(binarize_stochastic)
