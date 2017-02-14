import numpy as np


def concat_before(elements, from_n_parts):
    return elements.reshape((elements.shape[0] / from_n_parts, elements.shape[1] * from_n_parts))


def concat_after(elements, n_random_mappings, intertwine_size):
    """

    :param elements:
    :param n_random_mappings:
    :param intertwine_size: really the size of one ECA (R * automaton area)
    :return:
    """

    # TODO: Implement numpy stuff here

    outputs = []
    for i in xrange(0, len(elements), n_random_mappings):
        a = elements[i]
        n_rows = a.shape[0] / intertwine_size
        a = a.reshape(n_rows, intertwine_size)
        for e in xrange(1, n_random_mappings):
            element = elements[i + e].reshape(n_rows, intertwine_size)
            a = np.insert(a, intertwine_size * e, element.T, axis=1)
        outputs.append(a.reshape(n_rows * intertwine_size * n_random_mappings).tolist())
    return np.array(outputs)
