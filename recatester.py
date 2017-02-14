# Module for testing the reservoir.
# No scientific testing, just visualizing.
# Hard coding everything, so to speak.

import numpy as np
from sklearn import linear_model

from ca.eca import ECA
from compute.computer import Computer
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir
from statistics.plotter import plot_temporal

input_size = 4
n_random_mappings = 3
n_iterations = 3
rule = 90
diffuse = 7
pad = 0
verbose = 0
concat_before = False

inputs = np.array([[[0, 1, 0, 0],
                    [1, 0, 0, 0]]])

labels = np.array([[[0, 1],
                    [0, 1]]])  # Doesn't matter

encoder = ClassicEncoder(n_random_mappings,
                         input_size,
                         diffuse,
                         pad,
                         verbose=verbose)
encoder.random_mappings = [6, 0, 2, 3,
                           0, 1, 6, 3,
                           5, 2, 1, 0]

automaton = ECA(rule)
reservoir = Reservoir(automaton,
                      n_iterations,
                      verbose=verbose)

estimator = linear_model.LinearRegression()

computer = Computer(encoder,
                    reservoir,
                    estimator,
                    concat_before=concat_before,
                    verbose=verbose)

# Preserve the state vectors
x = computer.train(inputs, labels)
plot_temporal(x,
              encoder.n_random_mappings,
              encoder.automaton_area,
              2,
              n_iterations,
              sample_nr=0)
