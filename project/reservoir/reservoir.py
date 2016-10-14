import numpy as np


class Reservoir():

    def __init__(self, reservoir, iterations, random_mappings):
        self.reservoir = reservoir
        self.iterations = iterations
        self.random_mappings = random_mappings

    def fit(self, configs, labels, regression_model):
        """Fitting the regression model to the labels and what the reservoir outputs.
        Calls regression_model.fit().

        :param configs:
        :param labels:
        :param regression_model: a sklearn regression model
        :return: void
        """
        # Might take regression model arguments also as input
        training_sets = []
        for i in len(configs):
            state_vector = []
            config = configs[i]
            for step in xrange(self.iterations):
                new_state = self.reservoir.step(config)
                state_vector.append(new_state)
                config = new_state
            training_sets.append(state_vector)
        regression_model.fit(training_sets, labels)

    def spit_out(self):

    def compute(self):
        """Uses the reservoir provided, and lets it digest the initial configuration

        :return: all state vectors concatenated
        """
        concat = self.reservoir.config
        for i in xrange(self.reservoir.iterations):
            self.reservoir.step()
            concat = np.append(concat, self.reservoir.config)
        return concat
