import numpy as np
import random as rn
import scipy as sp
import ca.util as cutil


class Reservoir:
    # TODO maybe make it capable of delay (?) like Bye did

    def __init__(self, reservoir, iterations,  random_mappings, input_size, input_area, atomaton_area, verbose=False):
        """

        :param reservoir: the CA object
        :param iterations: the number of iterations
        :param random_mappings: the number of random mappings (0 is none)
        :param input_size: the size that the configurations come in
        :param input_area: the area/size that the inputs are to be mapped to
        :param size: the whole size of the CA
        """
        self.reservoir = reservoir
        self.iterations = iterations
        self.random_mappings = []
        self.size = atomaton_area
        self.input_area = input_area
        self.verbose = verbose
        if random_mappings > 0:
            for _ in xrange(random_mappings):
                self.random_mappings.append(make_random_mapping(input_size, input_area))
        else:
            self.input_area = input_size
            self.size = input_size
            self.random_mappings.append([i for i in xrange(input_size)])

    def transform(self, configs):
        """Lets the reservoir digest each of the configurations.
        No training here.

        :param configs: a list of initial configurations
        :param external_input: a dictionary where one can alter states of the reservoir at specific time steps
        :return: a list in which each element is the output of a configuration
        """
        outputs = []
        for ci in xrange(len(configs)):
            # For every initial configuration

            config = configs[ci]
            concat = []

            if self.verbose:
                print ""
                print "NEW CONFIG:"
                cutil.print_config_1dim(config)
                print ""

            for r in self.random_mappings:
                # For every random mapping, map the initial configuration ...
                mapped_config = sp.zeros([self.size], dtype=np.dtype(int))
                for ri in xrange(len(r)):
                    mapped_config[r[ri]] = config[ri]

                if self.verbose:
                    print "Random mapping:", r
                    cutil.print_config_1dim(mapped_config)

                # ... and iterate
                concat.extend(mapped_config)
                for step in xrange(self.iterations):
                    # edits = external_input.get(step, default=[])
                    # for key, val in edits:
                    #     mapped_config[key] = val

                    new_config = self.reservoir.step(mapped_config)
                    if self.verbose:
                        cutil.print_config_1dim(new_config)
                    # Concatenating this new configuration to the vector
                    concat.extend(new_config)
                    mapped_config = new_config
            outputs.append(concat)

        if self.verbose:
            print "OUTPUT VECTORS"
            for o in outputs:
                cutil.print_config_1dim(o)
        return outputs

    def set_seed(self, seed):
        rn.seed(seed)


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

