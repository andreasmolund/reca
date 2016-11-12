import marshal as dumper
import random as rn
import sys
from multiprocessing import Process, Queue

import ca.util as cutil

dump_path = "tmp/"
prefix = "dumpprocess"
filetype = "dump"


class Reservoir:
    # TODO maybe make it capable of delay (?) like Bye did

    def __init__(self, matter, iterations, verbose=1):
        """

        :param matter: the CA object
        :param iterations: the number of iterations
        :param verbose: 1 prints basic information, 2 prints the
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

        outputs = []
        for i in xrange(len(configurations)):
            concat = []
            config = configurations[i]
            # concat.extend(config)  # To include the initial configuration

            # Iterate
            for _ in xrange(self.iterations):
                new_config = self.matter.step(config)
                # Concatenating this new configuration to the vector
                concat.extend(new_config)
                config = new_config
            outputs.append(concat)

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

