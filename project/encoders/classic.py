import random as rand


seed = 20161110
rand.seed(seed)


class ClassicEncoder:
    """
    Creates a number of random mappings from a vector of size input_size
    on to a vector of size input_area
    with offset input_offset,
    and adds padding of 0's up to size automaton_area.

    If random_mappings=3, input_size=4,
    then translations of new size 4 configurations will be of size 3*4=12
    """

    def __init__(self, n_random_mappings, input_size, input_area, automaton_area, input_offset=0, verbose=0):
        """

        :param n_random_mappings: the number of random mappings (0 is none)
        :param input_size: the size that the configurations come in
        :param input_area: the area/size that the inputs are to be mapped to
        :param automaton_area: the size of the whole automaton
        :param input_offset: offset for the mapping
        """
        self.random_mappings = []
        self.input_area = max([input_size, input_area])
        self.automaton_area = max([self.input_area, automaton_area])
        self.verbose = verbose

        if n_random_mappings > 0:
            for _ in xrange(n_random_mappings):
                self.random_mappings.append(make_random_mapping(input_size, self.input_area, input_offset))
        else:
            self.random_mappings.append([i for i in xrange(input_size)])

        if self.verbose > 1:
            print "Random mappings:"
            print self.random_mappings

    def translate(self, configurations):
        translated_configs = []
        for config in configurations:
            partial_translated_config = [0b0] * (self.n_random_mappings * self.automaton_area)
            translated_configs.extend(self.mapping_addition(config, partial_translated_config))

        return translated_configs

    def mapping_addition(self, master, second):
        """Master overwrites second according to the mapping/translation

        :param master: the unmapped master or input vector
        :param second: the already mapped vector that is to be overwritten
        :return: added vectors
        """
        mapped_vector = []
        automaton_offset = 0  # Offset index
        for r in self.random_mappings:
            partial_mapped = [None] * self.automaton_area
            for automaton_i in xrange(self.automaton_area):
                # print "i:%d,map_i:%d" % (i, master_i)
                if r.count(automaton_i) > 0:  # r[map_i] == automaton_i:  # automaton_i in r:
                    # If the area element has a mapping to it,
                    # then get from master
                    partial_mapped[automaton_i] = master[r.index(automaton_i)]
                else:
                    # Else, get from second
                    partial_mapped[automaton_i] = second[automaton_offset + automaton_i]
            mapped_vector.append(partial_mapped)
            automaton_offset += self.automaton_area  # Adjusting offset

        return mapped_vector

    def pos(self, element):
        """Get what positions an element has

        :return: the position of all the element's mapping
        """
        return [pos[element] for pos in self.random_mappings]

    @property
    def n_random_mappings(self):
        """The number of random mappings"""
        return len(self.random_mappings)

    @property
    def total_area(self):
        """The area of all automata"""
        return self.automaton_area * len(self.random_mappings)


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

