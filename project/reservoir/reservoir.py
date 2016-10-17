import numpy as np
import random as rn
import scipy as sp
from ca.ca import CA


class Reservoir:
    # TODO maybe make it capable of delay (?) like Bye did

    def __init__(self, reservoir, iterations,  random_mappings, input_area, size):
        self.reservoirs = []
        self.iterations = iterations
        if random_mappings > 0:
            for i in xrange(random_mappings):
                self.reservoirs.append(make_reservoir(size, reservoir, input_area, input_offset=0))
        else:
            self.reservoirs = [reservoir]
        self.input_area = input_area

    def transform(self, configs):
        """Fitting the regression model to the labels and what the reservoir outputs.
        Calls regression_model.fit().

        :param configs:
        :param labels:
        :param regression_model: a sklearn regression model
        :return: void
        """
        outputs = []
        for r in self.reservoirs:
            for i in xrange(len(configs)):
                state_vector = []
                config = configs[i]
                for step in xrange(self.iterations):
                    new_state = r.step(config)
                    state_vector.extend(new_state)
                    config = new_state
                outputs.append(state_vector)
        return outputs


def make_reservoir(length, raw_init_config, input_area, input_offset=0):
    input_indexes = []
    for i in xrange(length):
        # Going through all states in the reservoir
        # Might be possible to improve
        index = rn.randint(0, input_area)
        while index in input_indexes:
            index = rn.randint(0, input_area)
        input_indexes.append(index)
    init_config = sp.zeros([length], dtype=np.dtype(int))
    for i in xrange(len(raw_init_config)):
        init_config[input_indexes[i] + input_offset] = raw_init_config[i]
    return init_config

