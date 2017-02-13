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
from compute.temporalcomputer import TemporalComputer
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

#  TODO: Handle LinAlgError better. Run one more time per error occurence


def main(raw_args):
    size, rule, n_iterations, n_random_mappings, diffuse, pad = digest_args(raw_args)

    size = 4  # The size of the input, or
    concat_before = True  # Concat the automata before or after iterating
    verbose = 1  # How much information to print to console
    n_layers = 2  # The number of layers excluding the first

    encoder1 = ClassicEncoder(n_random_mappings,
                              size,
                              diffuse,
                              pad,
                              verbose=verbose)

    automaton = ECA(rule)
    reservoir = Reservoir(automaton, n_iterations, verbose=verbose)
    encoders = [None] * n_layers
    computers = [None] * n_layers
    estimators = [None] * n_layers

    for layer_i in xrange(n_layers):
        encoder = ClassicEncoder(n_random_mappings,
                                 3,
                                 diffuse,
                                 pad,
                                 verbose=verbose)

        estimator = linear_model.LinearRegression()

        computer = TemporalComputer(encoder,
                                    reservoir,
                                    estimator,
                                    concat_before=concat_before,
                                    verbose=verbose)

        encoders[layer_i] = encoder
        estimators[layer_i] = estimator
        computers[layer_i] = computer

    time_checkpoint = time.time()
    print "Complexity:             %d (I*R*L_d)" % (n_iterations * n_random_mappings * pad)

    # The first reservoir needs to be trained (fit)
    try:
        # Preserving the values of the output nodes
        x1 = computer1.train(inputs, labels)
    except LinAlgError:
        logging.error(linalgerrmessage)
        return

    # The first reservoir needs to predict the output (predict)
    _, o1 = computer1.test(inputs, x1)
    o1 = classify_output(o1)

    # Then, the second reservoir needs to be trained (fit)
    try:
        # Preserving the values of the output nodes
        x2 = computer2.train(o1, labels)
    except LinAlgError:
        logging.error(linalgerrmessage)
        return

    # Currently, the system is trained.
    # Now we need to test,
    # but it is no need to transform and predict the output of the first reservoir.

    _, o2 = computer2.test(o1, x2)
    o2 = classify_output(o2)

    try:
        # Preserving the values of the output nodes
        x3 = computer2.train(o2, labels)
    except LinAlgError:
        logging.error(linalgerrmessage)
        return
    _, o3 = computer2.test(o2, x3)
    o3 = classify_output(o3)

    print "Time:              %.1f (training, testing, binarizing)" % (time.time() - time_checkpoint)

    r1_n_correct = 0
    r1_n_incorrect_bits = 0
    for pred, set_labels in zip(o2, labels):
        correct = True
        for pred_element, label_element in zip(pred, set_labels):
            if not np.array_equal(pred_element, label_element):
                correct = False
                r1_n_incorrect_bits += 1
        if correct:
            r1_n_correct += 1
    print "1. Correct:       ", r1_n_correct
    print "1. Incorrect bits:", r1_n_incorrect_bits

    r2_n_correct = 0
    r2_n_incorrect_bits = 0
    for pred, set_labels in zip(o3, labels):
        correct = True
        for pred_element, label_element in zip(pred, set_labels):
            if not np.array_equal(pred_element, label_element):
                correct = False
                r2_n_incorrect_bits += 1
        if correct:
            r2_n_correct += 1
    print "2. Correct:       ", r2_n_correct
    print "2. Incorrect bits:", r2_n_incorrect_bits

    if logit:
        logging.info("%d,%d,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d",
                     n_iterations,
                     encoder1.n_random_mappings,
                     rule,
                     size,
                     encoder1.input_area,
                     encoder1.automaton_area,
                     concat_before,
                     estimator1.__class__.__name__,
                     n_sets,
                     n_sets,
                     distractor_period,
                     1 if r2_n_correct == n_sets else 0,
                     r1_n_correct,
                     r1_n_incorrect_bits,
                     r2_n_correct,
                     r2_n_incorrect_bits)

    if n_whole_runs < 1:
        time_steps = 2 * 5 + distractor_period + 1
        plot_temporal(x1,
                      encoder1.n_random_mappings,
                      encoder1.automaton_area,
                      time_steps,
                      n_iterations,
                      sample_nr=12)

        plot_temporal(x2,
                      encoder2.n_random_mappings,
                      encoder2.automaton_area,
                      time_steps,
                      n_iterations,
                      sample_nr=12)


def digest_args(args):
    size = 5
    rule = 1
    iterations = 1
    n_random_mappings = 0
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
            iterations = int(a)
        elif o == '-R':
            n_random_mappings = int(a)
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
                     "Point (success),R1 correct,R1 wrong bits,R2 correct,R2 wrong bits")
    for r in xrange(n_whole_runs):
        # print "Run %d started" % r
        if len(sys.argv) > 1:
            main(sys.argv)
        else:
            main(['bittask.py',
                  '-r', '90',
                  '-I', '32',
                  '-R', '36',
                  # '--diffuse', '4',
                  '--pad', '4'])
