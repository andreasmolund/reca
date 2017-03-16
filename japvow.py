from itertools import count, izip

import numpy as np
from sklearn import svm
from sklearn import linear_model

from ca.eca import ECA
from compute.computer import Computer
from compute.distribute import flatten, distribute_and_collect, extend_state_vectors
from compute.jaegercomputer import JaegerComputer, jaeger_labels
from encoders.real import RealEncoder, quantize_l
from encoders.classic import ClassicEncoder
from problemgenerator import japanese_vowels
from reservoir.reservoir import Reservoir
from statistics.plotter import plot_temporal

n_iterations =      '8,4,4,4'
n_random_mappings = '8,4,4,4'
n_iterations = [int(value) for value in n_iterations.split(',')]
n_random_mappings = [int(value) for value in n_random_mappings.split(',')]
n_layers = min(len(n_random_mappings), len(n_iterations))
initial_input_size = 12
rule = 90

d = 6  # Jaeger's proposed D
training_sets, training_labels, testing_sets, testing_labels = japanese_vowels()
subsequent_training_labels = jaeger_labels(training_labels, d, 3)
subsequent_testing_labels = jaeger_labels(testing_labels, d, 3)

automaton = ECA(rule)
encoders = []
computers = []

for layer_i in xrange(n_layers):  # Setup

    if layer_i == 0:
        encoder = RealEncoder(n_random_mappings[layer_i],
                              initial_input_size,
                              initial_input_size,
                              initial_input_size)
    else:
        encoder = ClassicEncoder(n_random_mappings[layer_i],
                                 9,
                                 9,
                                 9)

    estimator = svm.SVC(kernel='linear')
    # estimator = linear_model.SGDClassifier()

    reservoir = Reservoir(automaton,
                          n_iterations[layer_i],
                          verbose=0)

    if layer_i == n_layers - 1:  # Last layer
        computer = JaegerComputer(encoder, reservoir, estimator, True, verbose=0, d=d, method=4)
    elif layer_i == 0:  # First layer
        computer = JaegerComputer(encoder, reservoir, estimator, True, verbose=0, d=d, method=3)
    else:  # Inter-layers
        computer = Computer(encoder, reservoir, estimator, True, verbose=0)

    encoders.append(encoder)
    computers.append(computer)


def classes_to_bits(predictions, n_elements, n_classes, d):
    """
    Assumes equal sequence length, e.g., from Jaeger's method 3
    """
    r = []
    man_shift = -1
    for total_i, p in enumerate(predictions):
        if total_i % d == 0:
            man_shift += 1
        translated_prediction = [0b0] * n_classes
        translated_prediction[p - 1] = 0b1
        r.append(translated_prediction)
    r = np.array(r).reshape((n_elements, d, n_classes))
    return r


o = None  # Output of one estimator

for layer_i in xrange(n_layers):  # Training

    x = computers[layer_i].train(training_sets if o is None else o,
                                 training_labels if layer_i == 0 else subsequent_training_labels)

    if layer_i < n_layers - 1:  # No need to test the last layer
        o, _ = computers[layer_i].test(training_sets, x)
        o = classes_to_bits(o, len(training_sets), 9, d)

correct = []
incorrect = []


def correctness(predicted, actual):
    n_correct = 0
    n_incorrect = 0

    for i, prediction, fasit_element in izip(count(), predicted, actual):
        if fasit_element == prediction:
            n_correct += 1
        else:
            n_incorrect += 1
    return n_correct, n_incorrect

o = None

for layer_i in xrange(n_layers):  # Testing

    o, x = computers[layer_i].test(testing_sets if o is None else o)

    n_correct, n_incorrect = correctness(o, flatten(jaeger_labels(testing_labels,
                                                                  d,
                                                                  3 if layer_i < n_layers - 1 else 4)))

    if layer_i < n_layers - 1:
        o = classes_to_bits(o, len(testing_sets), 9, d)

    print "Correct, incorrect, percent: %d, %d, %.1f" % (n_correct, n_incorrect, (100.0 * n_correct / (n_correct + n_incorrect)))

    sample_nr = 0
    if layer_i < n_layers - 1:
        time_steps = d
    else:
        time_steps = 1
    plot_temporal(x,
                  encoders[layer_i].n_random_mappings,
                  encoders[layer_i].automaton_area,
                  time_steps,
                  n_iterations[layer_i] * (1 if layer_i < n_layers - 1 else d),
                  sample_nr=sample_nr)
