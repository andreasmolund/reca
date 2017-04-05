import random as rand

import numpy as np
from compute.adders import combine

seed = 20170131
# rand.seed(seed)


class ClassicEncoder(object):
    """
    Creates a number of random mappings from a vector of size input_size
    on to a vector of size input_area
    with offset input_offset,
    and adds padding of 0's up to size automaton_area.

    If random_mappings=3, input_size=4,
    then translations of new size 4 configurations will be of size 3*4=12
    """

    def __init__(self,
                 n_random_mappings,
                 input_size,
                 input_area,
                 automaton_area,
                 input_offset=0,
                 verbose=0,
                 group_len=1):
        """

        :param n_random_mappings: the number of random mappings (0 is none)
        :param input_size: the size that the configurations come in
        :param input_area: the area/size that the inputs are to be mapped to
        :param automaton_area: the size of the whole automaton
        :param input_offset: offset for the mapping
        :param group_len: group elements in blocks
        """
        if input_size % group_len == 0:
            self.group_len = group_len
        else:
            self.group_len = 1
        self.R = n_random_mappings
        self.input_size = input_size
        self.input_area = max([input_size, input_area])
        self._automaton_area = max([self.input_area, automaton_area])
        self.verbose = verbose

        self.random_mappings = []
        if n_random_mappings > 0:
            for _ in xrange(n_random_mappings):
                self.random_mappings.extend(make_random_mapping(input_size / group_len,
                                                                self.input_area / group_len,
                                                                input_offset))
        else:
            self.random_mappings = [i for i in xrange(input_size / group_len)]
            # Even though there is no random mapping,
            # we set the integer to 1 for the rest of the system to work
            self.R = 1

        if self.verbose > 1:
            print "Random mappings:"
            print self.random_mappings

    def translate(self, configuration):
        zero_vector = np.zeros(self.n_random_mappings * self._automaton_area, dtype='int8')
        return self._separate(self._overwrite(configuration, zero_vector))

    def add(self, master, second):
        return self._separate(self._bitwise_add(master, second))

    def _bitwise_add(self, master, second):
        """
        A method of adding vectors.
        Yilmaz' (2015) normalized addition.

        :param master:
        :param second:
        :return:
        """
        zero_vector = np.zeros(self.R * self._automaton_area, dtype='int8')
        mapped_master = self._overwrite(master, zero_vector)
        return self._separate(combine(mapped_master, second))

    def _overwrite(self, master, second):
        """
        Writing master onto second according to the random mappings

        :param master:
        :param second:
        :return:
        """
        automaton_offset = 0  # Offset index
        for i, r in enumerate(self.random_mappings):
            master_i = i * self.group_len % self.input_size
            second_i = automaton_offset + r * self.group_len
            second[second_i:second_i + self.group_len] = master[master_i:master_i + self.group_len]

            if master_i + self.group_len == self.input_size:
                automaton_offset += self._automaton_area  # Adjusting offset
        return second

    def _separate(self, vector):
        """
        Splitting the vector up into separate pieces, with respect to the random mappings
        :param vector: an already mapped vector
        :return:
        """
        return vector.reshape((self.R, self._automaton_area))

    def pos(self, element):
        """Get what positions an element has

        :return: the position of all the element's mapping
        """
        return [pos[element] for pos in self.random_mappings]

    def mappings(self):
        return self.random_mappings

    @property
    def automaton_area(self):
        return self._automaton_area

    @automaton_area.setter
    def automaton_area(self, value):
        self._automaton_area = value

    @property
    def n_random_mappings(self):
        """The number of random mappings"""
        return self.R

    @property
    def total_area(self):
        """The area of all automata"""
        return self._automaton_area * self.n_random_mappings

    def encode_input(self, sequences):
        """
        Encode some input, for example to use as additional input to a subsequent layer
        :param sequences:
        :return:
        """
        encoded_inputs = []
        for sequence in sequences:
            sequence_len = len(sequence)
            encoded_coeffs = np.empty((sequence_len, self.automaton_area), dtype='int8')
            for t in xrange(sequence_len):
                # We only want one R, really, but this is for simplicity
                encoded_coeffs[t] = self._overwrite(sequence[t], [0b0] * self.total_area)[:self.automaton_area]
            encoded_inputs.append(encoded_coeffs)

        return encoded_inputs


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
        index = rand.randint(0, input_area - 1)
        while index in input_indexes:
            index = rand.randint(0, input_area - 1)
        input_indexes.append(index)
    return [i + input_offset for i in input_indexes]
