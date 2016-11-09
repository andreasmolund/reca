import sys
import getopt
import scipy as sp
import numpy as np
import time
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error
import problemgenerator as problems
from ca.ca import CA
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir
from reservoir.reservoir import make_random_mapping
import reservoir.util as rutil
import ca.util as cutil
from compute.temporalcomputer import TemporalComputer
import marshal as dumper

path = "tmpresults/"
prefix = "results"
filetype = "dump"


def main(raw_args):
    size, rule, iterations, random_mappings, input_area, automaton_area = digest_args(raw_args)

    size = 4
    n_training_sets = 32
    n_testing_sets = 32
    bits = 5
    distractor_period = 200
    time_steps = 2 * bits + distractor_period + 1
    concat_before = True
    verbose = 1

    encoder = ClassicEncoder(random_mappings,
                             size,
                             input_area,
                             automaton_area,
                             verbose=verbose)
    automation = CA(rule, k=2, n=3)
    reservoir = Reservoir(automation, iterations, verbose=verbose)
    estimator = svm.SVC()
    # estimator = svm.SVC(kernel='linear')
    # estimator = linear_model.LinearRegression()
    computer = TemporalComputer(encoder,
                                reservoir,
                                estimator,
                                concat_before=concat_before,
                                verbose=verbose)

    inputs, labels = problems.bit_memory_task(n_training_sets + n_testing_sets,
                                              bits,
                                              distractor_period)
    # E.g. [[1000,1000,1000,0100,...], [1000,1000,1000,0100,...], [1000,1000,1000,0100,...]]

    # Training
    time_checkpoint = time.time()
    computer.train(inputs[:n_training_sets], labels[:n_training_sets])
    print "Training time: ", (time.time() - time_checkpoint)
    time_checkpoint = time.time()

    # Testing
    y_pred = computer.test(inputs[:n_training_sets])
    y_pred = y_pred.reshape(n_testing_sets, time_steps).tolist()
    out_file = open("%s%s.%s" % (path, prefix, filetype), 'wb')
    dumper.dump(y_pred, out_file)
    dumper.dump(labels[n_training_sets:], out_file)
    out_file.close()
    n_correct = 0
    for pred, labels in zip(y_pred, labels[n_training_sets:]):
        correct = True
        for pred_element, label_element in zip(pred, labels):
            if pred_element != label_element:
                correct = False
        if correct:
            n_correct += 1
    print "Correct: %d/%d" % (n_correct, n_testing_sets)

    # step = 1
    # print "Testing time: ", (time.time() - time_checkpoint)
    # print "TEST RESULTS (samples, every 50th value)"
    # # error = mean_squared_error(labels[n_testing_sets:], y_pred)
    # print "Predicted:         \n", np.array(y_pred).reshape(n_testing_sets, time_steps)
    # print "Actual:            \n", np.array(labels[n_training_sets:])

    # print "Mean squared error:", error
    # corrects = 0
    # for i in xrange(len(y_pred)):
    #     if y_pred[i] == labels[n_testing_sets + i]:
    #         corrects += 1
    # print "Correctness (%):   ", (100 * corrects / len(y_pred))


def digest_args(args):
    size = 5
    rule = 1
    iterations = 1
    random_mappings = 0
    input_area = 0
    automaton_area = 0
    opts, args = getopt.getopt(args[1:],
                               's:r:i:h?',
                               ['random-mappings=', 'input-area=', 'automaton-area='])
    for o, a in opts:
        if o == '-s':
            size = int(a)
        elif o == '-r':
            rule = int(a)
        elif o == '-i':
            iterations = int(a)
        elif o == '--random-mappings':
            random_mappings = int(a)
        elif o == '--input-area':
            input_area = int(a)
        elif o == '--automaton-area':
            automaton_area = int(a)

    print opts
    return size, rule, iterations, random_mappings, input_area, automaton_area


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv)
    else:
        main(['parity.py',
              '-s', '4',
              '-r', '90',
              '-i', '15',
              '--random-mappings', '2',
              '--input-area', '12',
              '--automaton-area', '12'])
