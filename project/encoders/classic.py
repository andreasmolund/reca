import random as rn
import scipy as sp
import numpy as np


class ClassicEncoder:
    """
    Creates a number of random mappings from a vector of size input_size
    on to a vector of size input_area
    with offset input_offset,
    and adds padding of 0's up to size automaton_area.

    If random_mappings=3, input_size=4,
    then translations of new size 4 configurations will be of size 3*4=12
    """

    def __init__(self, random_mappings, input_size, input_area, automaton_area, input_offset=0):
        """

        :param random_mappings: the number of random mappings (0 is none)
        :param input_size: the size that the configurations come in
        :param input_area: the area/size that the inputs are to be mapped to
        :param automaton_area: the size of the whole automaton
        :param input_offset: offset for the mapping
        """
        self.random_mappings = []
        self.input_area = max([input_size, input_area])
        self.automaton_area = max([self.input_area, automaton_area])

        if random_mappings > 0:
            for _ in xrange(random_mappings):
                self.random_mappings.append(make_random_mapping(input_size, self.input_area, input_offset))
        else:
            self.random_mappings.append([i for i in xrange(input_size)])

    def translate(self, configurations):
        translated_configs = []
        # print self.random_mappings
        for config in configurations:
            translated_config = sp.zeros([self.automaton_area * len(self.random_mappings)], dtype=np.dtype(int))
            for i, r in enumerate(self.random_mappings):
                for ri in xrange(len(r)):
                    translated_config[self.automaton_area * i + r[ri]] = config[ri]
            # cautil.print_config_1dim(config, prefix="From ")
            # cautil.print_config_1dim(translated_config, prefix="to   ")
            translated_configs.append(translated_config)
        return translated_configs


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

