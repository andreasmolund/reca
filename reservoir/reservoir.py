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

    def transform(self, configurations):
        """Lets the reservoir digest the configuration.
        No training here.

        :param configurations: initial configuration
        :return: a list in which each element is the output of the translation of each configuration
        """
        configuration_size = configurations.shape[1]
        state_vector_len = self.iterations * configuration_size
        outputs = np.empty((configurations.shape[0], state_vector_len), dtype='int')

        for i in xrange(configurations.shape[0]):
            concat = []
            config = configurations[i]
            # concat.extend(config)  # To include the initial configuration

            # Iterate
            for _ in xrange(self.iterations):
                new_config = self.matter.step(config)
                # Concatenating this new configuration to the vector
                concat.extend(new_config)
                config = new_config
            outputs[i] = concat

        # conf_len = configuration.shape[0]
        # concat = np.empty(self.iterations * conf_len, dtype='int')
        # # Not including the initial configuration
        #
        # # Iterate
        # for t in xrange(concat.shape[0], step=conf_len):
        #     new_config = self.matter.step(configuration)
        #     # "Concatenating" this new configuration to the vector
        #     concat[t:t+conf_len] = new_config
        #     configuration = new_config

        return outputs

    @property
    def iterations(self):
        return self.iterations

    @iterations.setter
    def iterations(self, value):
        self.iterations = value
