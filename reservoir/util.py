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
    return 1 if np.random.rand() <= max(0, min(1, (value+1)/2)) else 0


binarize = np.vectorize(binarize_stochastic)


def extend_state_vectors(state_vectors, appendices):
    """
    Extends (in the beginning) each of the state vectors with the corresponding appendix.
    :param state_vectors:
    :param appendices:
    :return:
    """
    extends = []
    for appendix_sequence, state_sequence in zip(appendices, state_vectors):
        sequence = []
        for appendix, state_vector in zip(appendix_sequence, state_sequence):
            extended = appendix.tolist()
            extended.extend(state_vector)
            sequence.append(extended)
        extends.append(sequence)
    return extends
