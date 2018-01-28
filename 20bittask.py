# Runnable 20 bit memory task

import logging
import os.path
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

dimensions = 5
n_memory_time_steps = 10
distractor_period = 10
n_time_steps = 2 * n_memory_time_steps + distractor_period
n_train = 500
n_test = 100
inputs, labels = problems.memory_task_n_bit(dimensions, n_memory_time_steps, n_train + n_test, distractor_period)


def main(size, rules, n_iterations, n_random_mappings, diffuse, pad):
    n_iterations = [int(value) for value in n_iterations.split(',')]
    n_random_mappings = [int(value) for value in n_random_mappings.split(',')]
    rules = [int(value) for value in rules.split(',')]

    if not (len(n_iterations) == len(n_random_mappings)):
        raise ValueError("The number of iterations and random mappings do not match.")

    size = len(inputs[0][0])  # The size of the input, or
    diffuse = size
    pad = size
    concat_before = True  # Concat the automata before or after iterating
    verbose = 0  # How much information to print to consol
    n_layers = len(n_random_mappings)  # The number of layers including the first

    encoders = []
    computers = []

    for layer_i in xrange(n_layers):
        automaton = ECA(rules[layer_i])
        encoder = ClassicEncoder(n_random_mappings[layer_i],
                                 size if layer_i == 0 else labels.shape[2],  # Input size if it's the first layer
                                 diffuse,
                                 pad,
                                 verbose=verbose)

        estimator = linear_model.LinearRegression(n_jobs=12)

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

    time_checkpoint = time.time()

    o = None  # Output of one estimator
    correct = []
    incorrect_time_steps = []
    tot_fit_time = 0

    for layer_i in xrange(n_layers):
        # Training/fitting
        try:
            # Preserving the values of the output nodes
            # For the first layer, we transform/train with the very input,
            # and for the subsequent layers, the output from the previous layer is used
            x, fit_time = computers[layer_i].train(inputs[:n_train] if o is None else o, labels[:n_train])
            tot_fit_time += fit_time
        except LinAlgError:
            # logging.error(linalgerrmessage)
            print "LinAlgError occured.", time.time()
            return 1

        # Predicting with test data to get the very results
        o = computers[layer_i].test(inputs[n_train:])[0]
        o = classify_output(o)
        o = unflatten(o, [n_time_steps] * n_test)

        n_correct = 0
        n_mispredicted_time_steps = 0
        for predicted_seq, desired_seq in zip(o, labels[n_train:]):
            success = True
            for prediction_element, label_element in zip(predicted_seq, desired_seq):
                if not np.array_equal(prediction_element, label_element):
                    success = False
                    n_mispredicted_time_steps += 1
            if success:
                n_correct += 1
        correct.append(n_correct)
        incorrect_time_steps.append(n_mispredicted_time_steps)
        if not logit:
            print "Layer %d:\t%d correct seq.s\t%d mispred. time steps\t%.2fs" % (layer_i,
                                                                                  n_correct,
                                                                                  n_mispredicted_time_steps,
                                                                                  tot_fit_time)

        if layer_i < n_layers - 1:
            # Predicting with train data set in order to get input to the next layer
            o = computers[layer_i].test(inputs[:n_train])[0]
            o = classify_output(o)
            o = unflatten(o, [n_time_steps] * n_train)

            # if n_whole_runs == 1:
            #     from stats.plotter import plot_temporal
            #     plot_temporal(x,
            #                   encoders[layer_i].n_random_mappings,
            #                   encoders[layer_i].automaton_area,
            #                   n_time_steps,
            #                   n_iterations[layer_i],
            #                   sample_nr=2)

    # print "Time:                   %.1f (training, testing, binarizing)" % (time.time() - time_checkpoint)

    if logit:
        result = [j for i in zip(correct, incorrect_time_steps) for j in i]
        logging.info("\"%s\",\"%s\",\"%s\",%d,%d,%d,%s,%s,%.2f,%d,%d,%d,%d" + ",%d" * n_layers * 2,
                     ','.join(str(e) for e in n_iterations),
                     ','.join(str(e) for e in n_random_mappings),
                     ','.join(str(e) for e in rules),
                     size,
                     diffuse,
                     pad,
                     concat_before,
                     estimator.__class__.__name__,
                     tot_fit_time,
                     n_train,
                     n_test,
                     distractor_period,
                     1 if correct[-1] == n_test else 0,
                     *result)

    return 0  # The run went good


def init():
    if len(sys.argv) > 1:
        args = sys.argv
    else:
        args = ['20bittask.py',
                '-I', '1,1,1',
                '-R', '0,0,0',
                '--diffuse', '0',
                '--pad', '0',
                '-r', '0,0,0',
                '--id', 'ID'
                ]
    identifier, size, rules, n_iterations, n_random_mappings, diffuse, pad = digest_args(args)

    if logit:
        file_name = 'rawresults/20bittask-%s-%s-%s-part%s.csv' % (rules,
                                                                  n_iterations,
                                                                  n_random_mappings,
                                                                  identifier)
        logging.basicConfig(format='"%(asctime)s",%(message)s',
                            filename=file_name,
                            level=logging.DEBUG)
        deep_spesific = ""
        for l in xrange(len([value for value in n_iterations.split(',')])):
            deep_spesific += ",%d fully correct seq.,%d mispredicted time steps" % (l + 1, l + 1)
        logging.info("Is,Rs,Rule,Input size,Input area,Automaton size,Concat before,Estimator,"
                     "Tot. fit time,Training sets,Testing sets,Distractor period,"
                     "Point (success)" + deep_spesific)
    return size, rules, n_iterations, n_random_mappings, diffuse, pad


if __name__ == '__main__':

    n_whole_runs = 4

    main_args = init()
    r = 0
    while r < n_whole_runs:
        response = main(*main_args)
        if response != 0:
            r -= 1  # Something went wrong. We need to run one more time

        r += 1
