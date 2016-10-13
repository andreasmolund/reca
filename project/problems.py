# Module to generate configurations corresponding to different problems
import random as rn


def density(size=5, on_probability=0.5):
    """Generates an initial configuration for the density classification,
    along with correct classification

    :param size: length
    :return: configuration, majority
    """
    config = []
    ones = 0
    zeros = 0
    for i in xrange(size):
        state = 1 if rn.random() < on_probability else 0
        if state == 1:
            ones += 1
        else:
            zeros += 1
        config.append(state)
    return config, 1 if ones > zeros else 0


def parity(size=4):
    """Generates an initial configuration for the parity problem,
    along with the solution

    :param size: length
    :return: configuration,
    """
    config = []
    ones = 0
    for i in xrange(size):
        state = rn.randint(0, 1)
        if state == 1:
            ones += 1
        config.append(state)
    return config, 1 if ones % 2 == 0 else 0


