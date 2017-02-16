# Runnable 5 bit memory task

import getopt
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
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir
from reservoir.util import classify_output
from statistics.plotter import plot_temporal

start_time = datetime.now()
logit = False

n_whole_runs = 1
n_sets = 32
distractor_period = 199  # Because cue is within distr. period
inputs, labels = problems.bit_memory_task(n_sets,
                                          5,
                                          distractor_period)

#  TODO: Handle LinAlgError better. Run one more time per error occurrence


def main(raw_args):
    size, rule, n_iterations, n_random_mappings, diffuse, pad = digest_args(raw_args)

    n_iterations = [int(value) for value in n_iterations.split(',')]
    n_random_mappings = [int(value) for value in n_random_mappings.split(',')]

    if not (len(n_iterations) == len(n_random_mappings)):
        return

    size = 4  # The size of the input, or
    concat_before = True  # Concat the automata before or after iterating
    verbose = 0  # How much information to print to console
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

        estimator = linear_model.LinearRegression()

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
    incorrect_predictions = []

    for layer_i in xrange(n_layers):
        # Training/fitting
        try:
            # Preserving the values of the output nodes
            # For the first layer, we transform/train with the very input,
            # and for the subsequent layers, the output from the previous layer is used
            x = computers[layer_i].train(inputs if o is None else o, labels)
        except LinAlgError:
            logging.error(linalgerrmessage)
            return

        _, o = computers[layer_i].test(inputs, x)
        o = classify_output(o)

        n_correct = 0
        n_incorrect_predictions = 0
        for prediction_set, label_set in zip(o, labels):
            success = True
            for prediction_element, label_element in zip(prediction_set, label_set):
                if not np.array_equal(prediction_element, label_element):
                    success = False
                    n_incorrect_predictions += 1
            if success:
                n_correct += 1
        correct.append(n_correct)
        incorrect_predictions.append(n_incorrect_predictions)
        # print "%d. corr. pred.:         %d" % (layer_i, n_correct)
        # print "%d. incorr. pred.:       %d" % (layer_i, n_incorrect_predictions)

        if n_whole_runs < 1:
            time_steps = 2 * 5 + distractor_period + 1
            plot_temporal(x,
                          encoders[layer_i].n_random_mappings,
                          encoders[layer_i].automaton_area,
                          time_steps,
                          n_iterations[layer_i],
                          sample_nr=12)

    # print "Time:                   %.1f (training, testing, binarizing)" % (time.time() - time_checkpoint)

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
                     n_sets,
                     n_sets,
                     distractor_period,
                     1 if correct[-1] == n_sets else 0,
                     correct[0],
                     incorrect_predictions[0],
                     correct[-1],
                     incorrect_predictions[-1])


def digest_args(args):
    size = 5
    rule = 1
    iterations = '1'
    n_random_mappings = '1'
    diffuse = 0
    pad = 0
    opts, args = getopt.getopt(args[1:],
                               's:r:I:R:h?',
                               ['diffuse=', 'pad='])
    for o, a in opts:
        if o == '-s':
            size = int(a)
        elif o == '-r':
            rule = int(a)
        elif o == '-I':
            iterations = a
        elif o == '-R':
            n_random_mappings = a
        elif o == '--diffuse':
            diffuse = int(a)
        elif o == '--pad':
            pad = int(a)

    return size, rule, iterations, n_random_mappings, diffuse, pad


linalgerrmessage = ",,,,,,,LinAlgError occured: Skipping this run,,,,,,,,"


if __name__ == '__main__':
    if logit:
        file_name = 'preresults/%s-bitmem2res.csv' % start_time.isoformat().replace(":", "")
        logging.basicConfig(format='"%(asctime)s",%(message)s',
                            filename=file_name,
                            level=logging.DEBUG)
        logging.info("I,R,Rule,Input size,Input area,Automaton size,Concat before,Estimator,"
                     "Training sets,Testing sets,Distractor period,"
                     "Point (success),First res. correct,First res. wrong bits,Last res. correct,Last res. wrong bits")
    for r in xrange(n_whole_runs):
        # print "Run %d started" % r
        if len(sys.argv) > 1:
            main(sys.argv)
        else:
            main(['bittask.py',
                  '-I', '4',
                  '-R', '8',
                  '--diffuse', '40',
                  '--pad', '0',
                  '-r', '90'
                  ])
