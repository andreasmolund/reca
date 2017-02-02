# Utilities related to reservoir handling
import numpy as np


def classify_operand(value):
    return 0 if value < 0.5 else 1


classify_output = np.vectorize(classify_operand)

