import numpy as np


def before(element, from_n_parts):
    return element.reshape((1, from_n_parts * element.shape[1]))


def after(elements, n_random_mappings):
    """

    :param elements:
    :param n_random_mappings:
    :return:
    """
    return elements.reshape((1, n_random_mappings * elements.shape[1]))
