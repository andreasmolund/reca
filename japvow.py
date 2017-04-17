from itertools import count, izip

import numpy as np
from sklearn import svm
from sklearn import linear_model

from ca.eca import ECA
from compute.computer import Computer
from compute.distribute import flatten
from compute.jaegercomputer import JaegerComputer, jaeger_labels, jaeger_method
from encoders.classic import ClassicEncoder
from encoders.real import RealEncoder, quantize_japvow, quantize_activation
from problemgenerator import japanese_vowels
from reservoir.reservoir import Reservoir
from statistics.plotter import plot_temporal

n_iterations =      '20,20,20,20,20'
n_random_mappings = '32,32,32,32,32'
n_iterations = [int(value) for value in n_iterations.split(',')]
n_random_mappings = [int(value) for value in n_random_mappings.split(',')]
n_layers = min(len(n_random_mappings), len(n_iterations))
initial_input_size = 12
diffuse = initial_input_size
pad = initial_input_size
rule = 90

d = 5  # Jaeger's proposed D
training_sets, training_labels, testing_sets, testing_labels = japanese_vowels()
jaeger_training_sets = jaeger_method(training_sets, d, 3, dtype=float)
jaeger_testing_sets = jaeger_method(testing_sets, d, 3, dtype=float)

tmpencoder = RealEncoder(0, 12, 12, 12, quantize=quantize_japvow)
encoded_jaeger_training_sets = jaeger_method(tmpencoder.encode_input(training_sets), d, 3)
encoded_jaeger_testing_sets = jaeger_method(tmpencoder.encode_input(testing_sets), d, 3)

sequence_len_training = [len(sequence) for sequence in training_sets]
sequence_len_testing = [len(sequence) for sequence in testing_sets]

subsequent_training_labels = jaeger_labels(training_labels, d, 3)
subsequent_testing_labels = jaeger_labels(testing_labels, d, 3)
final_layer_training_labels = jaeger_labels(training_labels, d, 4)
final_layer_testing_labels = jaeger_labels(testing_labels, d, 4)

automaton = ECA(rule)
encoders = []
computers = []

print "(I,R,D)=(%s,%s,%d)" % (n_iterations, n_random_mappings, d)

for layer_i in xrange(n_layers):  # Setup

    if layer_i == 0:
        encoder = RealEncoder(n_random_mappings[layer_i],
                              initial_input_size,
                              diffuse,
                              pad,
                              quantize=quantize_japvow)
    else:
        group_len = 3
        encoder = ClassicEncoder(n_random_mappings[layer_i],
                                 9 * group_len + tmpencoder.automaton_area,
                                 9 * group_len + tmpencoder.automaton_area,
                                 9 * group_len + tmpencoder.automaton_area,
                                 group_len=group_len)

    # estimator = svm.SVC(kernel='linear')
    # estimator = linear_model.SGDClassifier()
    # estimator = linear_model.Perceptron()
    estimator = linear_model.LinearRegression()
    # estimator = linear_model.SGDRegressor(loss='squared_loss')

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


def inter_process(predictions, n_elements, n_classes, d, activation_percentiles):
    r = []
    man_shift = -1
    for total_i, p in enumerate(predictions):
        if total_i % d == 0:
            man_shift += 1
        translated_prediction = quantize_activation(p, *activation_percentiles)
        # translated_prediction[p - 1] = 0b1
        # translated_prediction[p.argmax()] = 0b1
        r.append(translated_prediction)
    r = np.array(r).reshape((n_elements, d, len(r[0])))
    return r


activation_levels = []
o = None  # Output of one estimator
q = (66, 77, 88)

for layer_i in xrange(n_layers):  # Training

    layer_inputs = training_sets if o is None else o
    layer_labels = training_labels if layer_i == 0 else subsequent_training_labels
    x = computers[layer_i].train(layer_inputs, layer_labels, extensions=jaeger_training_sets)

    if layer_i < n_layers - 1:  # No need to test the last layer before the very real test
        o, _ = computers[layer_i].test(layer_inputs, extensions=jaeger_training_sets)
        percentiles = np.percentile(o, q)
        activation_levels.append(percentiles)
        print q, "percentiles (tr):", percentiles
        o = inter_process(o, len(training_sets), 9, d, percentiles)

        o = np.append(encoded_jaeger_training_sets, o, axis=2)


def correctness(predicted, actual):
    n_correct = 0
    n_incorrect = 0

    for i, prediction, fasit_element in izip(count(), predicted, actual):
        if fasit_element.argmax() == prediction.argmax():
        # if fasit_element == prediction:
            n_correct += 1
        else:
            n_incorrect += 1
    return n_correct, n_incorrect


o = None

for layer_i in xrange(n_layers):  # Testing

    o, x = computers[layer_i].test(testing_sets if o is None else o, extensions=jaeger_testing_sets)
    print q, "percent., new:   ", np.percentile(o, q)

    n_correct, n_incorrect = correctness(o, flatten(jaeger_labels(testing_labels,
                                                                  d,
                                                                  3 if layer_i < n_layers - 1 else 4)))

    print "Misprecictions, percent: %d, %.1f" % (n_incorrect, (100.0 * n_correct / (n_correct + n_incorrect)))

    if layer_i < n_layers - 1:
        percentiles = activation_levels[layer_i]
        o = inter_process(o, len(testing_sets), 9, d, percentiles)

        # o = np.append(x, o, axis=2)

        o = np.append(encoded_jaeger_testing_sets, o, axis=2)

        # sample_nr = 0
        # if layer_i < n_layers - 1:
        #     time_steps = sequence_len_testing[sample_nr]
        # else:
        #     time_steps = 1
        # plot_temporal(x,
        #               encoders[layer_i].n_random_mappings,
        #               encoders[layer_i].automaton_area,
        #               time_steps,
        #               n_iterations[layer_i] * (1 if layer_i < n_layers - 1 else d),
        #               sample_nr=sample_nr)
