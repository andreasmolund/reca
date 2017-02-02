import random as rn
import sys

import ca.util as cutil
import numpy as np

dump_path = "tmp/"
prefix = "dumpprocess"
filetype = "dump"


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
        """Lets the reservoir digest each of the configurations.
        No training here.

        :param configurations: a list of initial configurations,
                               and the quantity of it must be divisible on n_processes
        :return: a list in which each element is the output of the translation of each configuration
        """

        if self.verbose == 0:
            sys.stdout.write("Transforming... ")
            sys.stdout.flush()
        elif self.verbose > 1:
            print "Input:"
            for i, c in enumerate(configurations):
                cutil.print_config_1dim(c, postfix="(%d)" % i)

        state_vector_len = self.iterations * configurations.shape[1]
        outputs = np.empty((configurations.shape[0], state_vector_len), dtype='int')

        for i in xrange(configurations.shape[0]):
            concat = np.empty((self.iterations, configurations.shape[1]), dtype='int')
            config = configurations[i]
            # concat.extend(config)  # To include the initial configuration

            # Iterate
            for t in xrange(self.iterations):
                new_config = self.matter.step(config)
                # Concatenating this new configuration to the vector
                concat[t] = new_config
                config = new_config
            outputs[i] = concat.reshape(state_vector_len)

        if self.verbose == 0:
            sys.stdout.write("Done\n")
            sys.stdout.flush()

        return outputs

    @property
    def iterations(self):
        return self.iterations

    @iterations.setter
    def iterations(self, value):
        self.iterations = value


def make_random_mapping(input_size, input_area, input_offset=0):
    """Generates a pseudo-random mapping from inputs to outputs.
    The encoding stage.

    :param input_size: the size that the inputs come in
    :param input_area: the area/size that the inputs are to be mapped to
    :param input_offset: a number if an offset is wanted, default 0
    :return: an array of mapped indexes
    """
    input_indexes = []
    for i in xrange(input_size):
        # Going through all states in the reservoir
        # Might be possible to improve
        index = rn.randint(0, input_area - 1)
        while index in input_indexes:
            index = rn.randint(0, input_area - 1)
        input_indexes.append(index)
    return [i + input_offset for i in input_indexes]

