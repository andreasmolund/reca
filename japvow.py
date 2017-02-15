from sklearn import linear_model

from ca.eca import ECA
from compute.computer import Computer
from encoders.real import RealEncoder
from problemgenerator import japanese_vowels
from reservoir.reservoir import Reservoir

n_random_mappings = 8
n_iterations = 15
input_size = 12
rule = 90

encoder = RealEncoder(n_random_mappings,
                      input_size,
                      0,
                      0,
                      verbose=0)

estimator = linear_model.LinearRegression()

automaton = ECA(rule)

reservoir = Reservoir(automaton,
                      n_iterations,
                      verbose=0)

computer = Computer(encoder,
                    reservoir,
                    estimator,
                    concat_before=True,
                    verbose=0)

training_sets, labels = japanese_vowels()
