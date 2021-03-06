import logging
import sys
from datetime import datetime
from itertools import count, izip

import numpy as np
from sklearn import linear_model

from ca.eca import ECA
from compute.computer import Computer
from compute.distribute import flatten
from compute.jaegercomputer import JaegerComputer, jaeger_labels, jaeger_method
from encoders.classic import ClassicEncoder
from encoders.real import RealEncoder, quantize_japvow, quantize_activation
from problemgenerator import japanese_vowels
from reservoir.reservoir import Reservoir
from util import digest_args

d = 4  # Jaeger's proposed D
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

start_time = datetime.now()
logit = False

q = (25, 50, 75)  # Activation quantiles


def main(initial_input_size, rules, n_iterations, n_random_mappings, diffuse, pad):
    initial_input_size = 12
    diffuse = initial_input_size
    pad = initial_input_size
    n_iterations = [int(value) for value in n_iterations.split(',')]
    n_random_mappings = [int(value) for value in n_random_mappings.split(',')]
    rules = [int(value) for value in rules.split(',')]
    encoders = []
    computers = []

    if not (len(n_iterations) == len(n_random_mappings)):
        raise ValueError("The number of iterations and random mappings do not match.")
    n_layers = len(n_random_mappings)  # The number of layers including the first

    for layer_i in xrange(n_layers):  # Setup
        automaton = ECA(rules[layer_i])

        if layer_i == 0:
            encoder = RealEncoder(n_random_mappings[layer_i],
                                  initial_input_size,
                                  diffuse,
                                  pad,
                                  quantize=quantize_japvow)
        else:
            group_len = len(quantize_activation([1]))
            encoder = ClassicEncoder(n_random_mappings[layer_i],
                                     9 * group_len + tmpencoder.automaton_area,
                                     9 * group_len + tmpencoder.automaton_area,
                                     9 * group_len + tmpencoder.automaton_area,
                                     group_len=group_len)

        # estimator = svm.SVC(kernel='linear')
        # estimator = linear_model.SGDClassifier()
        # estimator = linear_model.Perceptron()
        estimator = linear_model.LinearRegression(n_jobs=4)
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

    activation_levels = []
    o = None  # Output of one estimator
    tot_fit_time = 0

    for layer_i in xrange(n_layers):  # Training

        layer_inputs = training_sets if o is None else o
        layer_labels = training_labels if layer_i == 0 else subsequent_training_labels
        raw_training_input = None  # training_sets if layer_i == 0 else jaeger_training_sets

        x, fit_time = computers[layer_i].train(layer_inputs, layer_labels, extensions=raw_training_input)
        tot_fit_time += fit_time
        print fit_time, " fit time"

        if layer_i < n_layers - 1:  # No need to test the last layer before the very real test
            o, _ = computers[layer_i].test(layer_inputs, extensions=raw_training_input)
            percentiles = np.percentile(o, q)
            activation_levels.append(percentiles)
            print q, "percentiles (tr):", percentiles
            o = inter_process(o, len(training_sets), 9, d, percentiles)

            o = np.append(encoded_jaeger_training_sets, o, axis=2)
            # o = unflatten(o, sequence_len_training)

    o = None

    out_of = []
    misclassif = []

    for layer_i in xrange(n_layers):  # Testing

        raw_training_input = None  # testing_sets if layer_i == 0 else jaeger_testing_sets
        o, x = computers[layer_i].test(testing_sets if o is None else o, extensions=raw_training_input)
        print q, "percent., new:   ", np.percentile(o, q)

        n_correct, n_incorrect = correctness(o, flatten(
            subsequent_testing_labels if layer_i < n_layers - 1 else final_layer_testing_labels))
        out_of.append(n_correct + n_incorrect)
        misclassif.append(n_incorrect)

        # print "Misprecictions, percent: %d, %.1f" % (n_incorrect, (100.0 * n_correct / (n_correct + n_incorrect)))

        if layer_i < n_layers - 1:
            percentiles = activation_levels[layer_i]
            o = inter_process(o, len(testing_sets), 9, d, percentiles)

            # o = np.append(x, o, axis=2)

            o = np.append(encoded_jaeger_testing_sets, o, axis=2)
            # o = unflatten(o, sequence_len_testing)

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
    result = [j for i in zip(out_of, misclassif) for j in i]
    print ["%.1f" % (100 * float(result[i]) / result[i - 1]) for i in xrange(1, len(result), 2)], "% error"
    if logit:
        rep_string = "\"%s\",\"%s\",\"%s\",%d,%d,%d,%s,%s,%d,%.2f,%d" + ",%d" * n_layers * 2
        logging.info(rep_string,
                     ','.join(str(e) for e in n_iterations),
                     ','.join(str(e) for e in n_random_mappings),
                     ','.join(str(e) for e in rules),
                     initial_input_size,
                     diffuse,
                     pad,
                     'True',
                     estimator.__class__.__name__,
                     d,
                     tot_fit_time,
                     misclassif[-1],
                     *result)
    return 0


def inter_process(predictions, n_elements, n_classes, d, activation_percentiles):
    encoded = []
    man_shift = -1
    for total_i, p in enumerate(predictions):
        if total_i % d == 0:
            man_shift += 1
        translated_prediction = quantize_activation(p, *activation_percentiles)
        # translated_prediction[p - 1] = 0b1
        # translated_prediction[p.argmax()] = 0b1
        encoded.append(translated_prediction)
    encoded = np.array(encoded).reshape((n_elements, d, len(encoded[0])))
    return encoded


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


def init():
    if len(sys.argv) > 1:
        args = sys.argv
    else:
        args = ['japvow.py',
                '-I', '20,20,20',
                '-R', '20,20,20',
                '-r', '54,54,54'
                ]
    identifier, initial_input_size, rules, n_iterations, n_random_mappings, diffuse, pad = digest_args(args)

    if logit:
        file_name = 'rawresults/japvow-%s-%s-%s-part%s.csv' % (rules,
                                                               n_iterations,
                                                               n_random_mappings,
                                                               identifier)
        logging.basicConfig(format='"%(asctime)s",%(message)s',
                            filename=file_name,
                            level=logging.DEBUG)
        deep_spesific = ""
        for l in xrange(len([value for value in n_iterations.split(',')])):
            deep_spesific += ",%d out of,%d misclassif" % (l + 1, l + 1)
        logging.info("Is,Rs,Rule,Input size,Input area,Automaton size,Concat before,Estimator,"
                     "D,"
                     "Tot. fit time,Final misclassif." + deep_spesific)
    return initial_input_size, rules, n_iterations, n_random_mappings, diffuse, pad

if __name__ == '__main__':

    n_whole_runs = 2

    main_args = init()

    r = 0
    while r < n_whole_runs:
        print "Started run %d of %d" % (r + 1, n_whole_runs)
        response = main(*main_args)
        if response != 0:
            r -= 1  # Something went wrong. We need to run one more time

        r += 1

