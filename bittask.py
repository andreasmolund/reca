# Runnable 5 bit memory task
# In 5-bit, training and testing is on the same state vectors,
# or at the least, the same input.

import logging
import sys
import time
from datetime import datetime

import numpy as np
from numpy.linalg.linalg import LinAlgError
from sklearn import linear_model

import problemgenerator as problems
from ca.eca import ECA
from compute.computer import Computer
from compute.distribute import unflatten
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir
from reservoir.util import classify_output
from util import digest_args

start_time = datetime.now()
logit = False

dimensions = 2
n_memory_time_steps = 5
n_train = 32
n_test = 32
distractor_period = 50
inputs, labels = problems.memory_task_5_bit(n_train,
                                            distractor_period)


def main(size, rule, n_iterations, n_random_mappings, diffuse, pad):
    n_iterations = [int(value) for value in n_iterations.split(',')]
    n_random_mappings = [int(value) for value in n_random_mappings.split(',')]

    if not (len(n_iterations) == len(n_random_mappings)):
        return

    size = len(inputs[0][0])  # The size of the input, or
    concat_before = True  # Concat the automata before or after iterating
    verbose = 0  # How much information to print to consol
    n_layers = len(n_random_mappings)  # The number of layers including the first

    automaton = ECA(rule)

    encoders = []
    computers = []

    for layer_i in xrange(n_layers):
        encoder = ClassicEncoder(n_random_mappings[layer_i],
                                 size if layer_i == 0 else labels.shape[2],  # Input size if it's the first layer
                                 diffuse,
                                 pad,
                                 verbose=verbose)

        estimator = linear_model.LinearRegression(n_jobs=4)

        reservoir = Reservoir(automaton,
                              n_iterations[layer_i],
                              verbose=verbose)

        computer = Computer(encoder,
                            reservoir,
                            estimator,
                            concat_before=concat_before,
                            verbose=verbose)

        encoders.append(encoder)
        computers.append(computer)

    o = None  # Output of one estimator
    correct = []
    incorrect_predictions = []

    for layer_i in xrange(n_layers):
        # Training/fitting
        try:
            # Preserving the values of the output nodes
            # For the first layer, we transform/train with the very input,
            # and for the subsequent layers, the output from the previous layer is used
            x, _ = computers[layer_i].train(inputs if o is None else o, labels)
        except LinAlgError:
            # logging.error(linalgerrmessage)
            print "LinAlgError occured.", time.time()
            return 1

        o, _ = computers[layer_i].test(None, x)
        o = classify_output(o)
        o = unflatten(o, [2 * n_memory_time_steps + distractor_period] * n_train)

        n_correct = 0
        n_mispredicted_bits = 0
        for prediction_set, label_set in zip(o, labels):
            success = True
            for prediction_element, label_element in zip(prediction_set, label_set):
                if not np.array_equal(prediction_element, label_element):
                    success = False
                    n_mispredicted_bits += 1
            if success:
                n_correct += 1
        correct.append(n_correct)
        incorrect_predictions.append(n_mispredicted_bits)
        print "%d. corr. pred.:         %d" % (layer_i, n_correct)
        print "%d. incorr. pred.:       %d" % (layer_i, n_mispredicted_bits)

        # if n_whole_runs == 1:
        #     time_steps = 2 * n_memory_time_steps + distractor_period
        #     from statistics.plotter import plot_temporal
        #     plot_temporal(x,
        #                   encoders[layer_i].n_random_mappings,
        #                   encoders[layer_i].automaton_area,
        #                   time_steps,
        #                   n_iterations[layer_i],
        #                   sample_nr=2)

    if logit:
        logging.info("%d,%d,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d",
                     n_iterations[0],
                     encoders[0].n_random_mappings,
                     rule,
                     size,
                     encoders[0].input_area,
                     encoders[0].automaton_area,
                     concat_before,
                     linear_model.LinearRegression().__class__.__name__,
                     n_train,
                     n_train,
                     distractor_period,
                     1 if correct[-1] == n_train else 0,
                     correct[0],
                     incorrect_predictions[0],
                     correct[-1],
                     incorrect_predictions[-1])

    return 0  # The run went good


def init():
    if len(sys.argv) > 1:
        args = sys.argv
    else:
        args = ['bittask.py',
                '-I', '32',
                '-R', '30',
                '--diffuse', '0',
                '--pad', '0',
                '-r', '110'
                ]
    identifier, size, rule, n_iterations, n_random_mappings, diffuse, pad = digest_args(args)

    if logit:
        file_name = 'rawresults/bittask-%d-%s-%s-part%s.csv' % (rule,
                                                                n_iterations,
                                                                n_random_mappings,
                                                                identifier)
        logging.basicConfig(format='"%(asctime)s",%(message)s',
                            filename=file_name,
                            level=logging.DEBUG)
        logging.info("I,R,Rule,Input size,Input area,Automaton size,Concat before,Estimator,"
                     "Training sets,Testing sets,Distractor period,"
                     "Point (success),First res. correct,First res. wrong bits,"
                     "Last res. correct,Last res. wrong bits")
    return size, rule, n_iterations, n_random_mappings, diffuse, pad


if __name__ == '__main__':

    n_whole_runs = 1

    main_args = init()

    r = 0
    while r < n_whole_runs:
        response = main(*main_args)
        if response != 0:
            r -= 1  # Something went wrong. We need to run one more time

        r += 1
