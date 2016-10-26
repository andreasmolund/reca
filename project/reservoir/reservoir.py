import sys
import numpy as np
import random as rn
import scipy as sp
import ca.util as cutil


class Reservoir:
    # TODO maybe make it capable of delay (?) like Bye did

    def __init__(self, matter, iterations, verbose=False):
        """

        :param matter: the CA object
        :param iterations: the number of iterations
        :param verbose: set to true if you want to print the configurations
        """
        self.matter = matter
        self.iterations = iterations
        self.random_mappings = []
        self.verbose = verbose

        print "%d iterations" \
              % self.iterations

    def transform(self, configurations):
        """Lets the reservoir digest each of the configurations.
        No training here.

        :param configurations: a list of initial configurations
        :return: a list in which each element is the output of the translation of each configuration
        """
        outputs = []
        n_configs = len(configurations)
        for i, config in enumerate(configurations):

            concat = []

            if self.verbose:
                print "NEW CONFIGURATION:"
                cutil.print_config_1dim(config)

            # concat.extend(config)

            # Iterate
            for step in xrange(self.iterations):

                new_config = self.matter.step(config)
                if self.verbose:
                    cutil.print_config_1dim(new_config)
                # if step >= (self.iterations - 5):
                # Concatenating this new configuration to the vector
                concat.extend(new_config)
                config = new_config
            outputs.append(concat)

            sys.stdout.write("\rTransformation progress: %d%%" % (100 * i / (n_configs - 1)))
            sys.stdout.flush()

        sys.stdout.write("\n")
        sys.stdout.flush()
        if self.verbose:
            print "OUTPUT VECTORS"
            for o in outputs:
                cutil.print_config_1dim(o)

        return outputs


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

