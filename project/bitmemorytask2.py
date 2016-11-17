import logging
import sys
from datetime import datetime

import numpy as np
from numpy.linalg.linalg import LinAlgError
from sklearn import linear_model

import problemgenerator as problems
from bitmemorytask import digest_args
from ca.ca import CA
from compute.temporalcomputer import TemporalComputer
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir
from reservoir.util import classify_output

start_time = datetime.now()
logit = True

n_whole_runs = 1000
n_training_sets = 32
n_testing_sets = 32
bits = 5
distractor_period = 20
inputs, labels = problems.bit_memory_task(n_training_sets + n_testing_sets,
                                          bits,
                                          distractor_period)


def main(raw_args):
    size, rule, n_iterations, n_random_mappings, input_area, automaton_area = digest_args(raw_args)

    size = 4
    concat_before = True
    verbose = 1

    encoder1 = ClassicEncoder(n_random_mappings,
                              size,
                              input_area,
                              automaton_area,
                              verbose=verbose)
    encoder2 = ClassicEncoder(n_random_mappings,
                              3,
                              input_area,
                              automaton_area,
                              verbose=verbose)
    automaton = CA(rule, k=2, n=3)
    reservoir = Reservoir(automaton, n_iterations, verbose=verbose)
    estimator1 = linear_model.LinearRegression()
    estimator2 = linear_model.LinearRegression()
    computer1 = TemporalComputer(encoder1,
                                 reservoir,
                                 estimator1,
                                 concat_before=concat_before,
                                 verbose=verbose)
    computer2 = TemporalComputer(encoder2,
                                 reservoir,
                                 estimator2,
                                 concat_before=concat_before,
                                 verbose=verbose)

    try:
        computer1.train(inputs[:n_training_sets], labels[:n_training_sets])
    except LinAlgError as e:
        handle_linalgerror(e, computer1, encoder1, inputs[:n_training_sets], labels[:n_training_sets])
        return

    _, o1 = computer1.test(inputs[:n_training_sets])
    o1 = [[classify_output(t) for t in s] for s in o1]

    try:
        computer2.train(o1[:n_testing_sets], labels[:n_testing_sets])
    except LinAlgError as e:
        handle_linalgerror(e, computer2, encoder2, o1[:n_testing_sets], labels[:n_testing_sets])
        return

    _, o1 = computer1.test(inputs[n_training_sets:])
    o1 = [[classify_output(t) for t in s] for s in o1]
    _, o2 = computer2.test(o1)
    o2 = [[classify_output(t) for t in s] for s in o2]

    r1_n_correct = 0
    r1_n_incorrect_bits = 0
    for pred, set_labels in zip(o1, labels[n_training_sets:]):
        correct = True
        for pred_element, label_element in zip(pred, set_labels):
            if pred_element != label_element:
                correct = False
                r1_n_incorrect_bits += 1
        if correct:
            r1_n_correct += 1
    # print "1. Correct:       ", n_correct
    # print "1. Incorrect bits:", n_incorrect_bits

    r2_n_correct = 0
    r2_n_incorrect_bits = 0
    for pred, set_labels in zip(o2, labels[n_training_sets:]):
        correct = True
        for pred_element, label_element in zip(pred, set_labels):
            if pred_element != label_element:
                correct = False
                r2_n_incorrect_bits += 1
        if correct:
            r2_n_correct += 1
    # print "2. Correct:       ", n_correct
    # print "2. Incorrect bits:", n_incorrect_bits

    if logit:
        logging.info("%d,%d,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d",
                     n_iterations,
                     encoder1.n_random_mappings,
                     rule,
                     size,
                     encoder1.input_area,
                     encoder1.automaton_area,
                     concat_before,
                     estimator1.__class__.__name__,
                     n_training_sets,
                     n_testing_sets,
                     distractor_period,
                     r1_n_correct,
                     r1_n_incorrect_bits,
                     r2_n_correct,
                     r2_n_incorrect_bits)


def bitify_classes(sets, n_classes=3):
    bitified_sets = []
    for a_set in sets:
        bitified_set = []
        for a_class in a_set:
            bitified_class = [0] * n_classes
            bitified_class[a_class] = 1
            bitified_set.append(bitified_class)
        bitified_sets.append(bitified_set)
    return bitified_sets


def handle_linalgerror(e, computer, encoder, p_inputs, p_labels):
    print "ERROR OCCURED YO"
    print "YO DUDE"
    # Print encoder's mappings ...
    x = computer._distribute_and_collect(p_inputs)
    x = computer._post_process(x)
    np.set_printoptions(threshold='nan')
    logging.warning("\"Hmmmmm, a LinAlgError occured. Why is that? Have a look:\n"
                    "e: %s\n"
                    "e.args: %s\n"
                    "e.message: %s\n"
                    "Encoder mappings: %s\n"
                    "Sets: %s\n"
                    "Labels: %s\n"
                    "Outputs \"nodes\": %s\"",
                    e, e.args, e.message, encoder.mappings(), p_inputs, p_labels[:n_training_sets], x)
    np.set_printoptions(threshold=1000)


if __name__ == '__main__':
    if logit:
        logging.basicConfig(format='"%(asctime)s",%(message)s',
                            filename='preresults/bitmem2-%s.csv' % start_time.isoformat(),
                            level=logging.DEBUG)
        logging.info("I,R,Rule,Input size,Input area,Automaton size,Concat before,Estimator,"
                     "Training sets,Testing sets,Distractor period,"
                     "R1 successful,R1 wrong bits,R2 successful,R2 wrong bits")
    for r in xrange(n_whole_runs):
        # print "Run %d started" % r
        if len(sys.argv) > 1:
            main(sys.argv)
        else:
            main(['bitmemorytask.py',
                  '-r', '102',
                  '-i', '4',
                  '--random-mappings', '4',
                  '--input-area', '40',
                  '--automaton-area', '0'])
