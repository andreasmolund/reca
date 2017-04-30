import random as rn
import sys

import ca.util as cutil
import numpy as np

dump_path = "tmp/"
prefix = "dumpprocess"
file_type = "dump"


class Reservoir:
    def __init__(self, matter, iterations, verbose=1):
        """

        :param matter: the ECA object
        :param iterations: the number of iterations
        :param verbose: 1 prints basic information, 2 prints more
        """
        self.matter = matter
        self.iterations = iterations
        self.verbose = verbose

    @property
    def iterations(self):
        return self.iterations

    @iterations.setter
    def iterations(self, value):
        self.iterations = value


def transform(configurations, n_iterations, step):
    """Lets the reservoir digest the configuration.
    No training here.

    :param step: the function that transforms a CA config to another
    :param n_iterations: the number of iterations
    :param configurations: initial configuration
    :return: a list in which each element is the output of the translation of each configuration
    """
    configuration_size = configurations.shape[1]
    state_vector_len = n_iterations * configuration_size
    outputs = np.empty((configurations.shape[0], state_vector_len), dtype='int8')

    for i in xrange(configurations.shape[0]):
        concat = []
        config = configurations[i]
        # concat.extend(config)  # To include the initial configuration

        # Iterate
        for _ in xrange(n_iterations):
            new_config = step(config)
            # Concatenating this new configuration to the vector
            concat.extend(new_config)
            config = new_config
        outputs[i] = concat

    return outputs
