# Module to generate configurations corresponding to different problems
# with labels/classes/fasit

import random as rn


def density(quantity, size=5, on_probability=0.5):
    """Generates initial configurations for the density classification,
    along with correct classifications

    :param quantity: number of configurations to generate
    :param size: length
    :return: configurations, labels
    """
    configs = []
    labels = []

    for c in xrange(quantity):
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
        configs.append(config)
        labels.append(1 if ones > zeros else 0)
    return configs, labels


def parity(quantity, size=4):
    """Generates initial configurations for the parity problem,
    along with the corresponding solutions

    :param quantity: number of configurations to generate
    :param size: length
    :return: configurations, labels
    """
    configs = []
    labels = []

    for c in xrange(quantity):
        config = []
        ones = 0
        for i in xrange(size):
            state = rn.randint(0, 1)
            if state == 1:
                ones += 1
            config.append(state)
        configs.append(config)
        labels.append(1 if ones % 2 == 0 else 0)
    return configs, labels


