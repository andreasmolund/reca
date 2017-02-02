import getopt
import math
import sys
import time

from sklearn import svm
from sklearn.metrics import mean_squared_error

import ca.util as cutil
import problemgenerator as problems
import reservoir.util as rutil
from ca.eca import ECA
from compute.computer import Computer
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir


# This is the non-temporal parity problem;
# not an experiment that is to be included in the report

def main(raw_args):
    input_size, rule, iterations, random_mappings, input_area, automaton_area = digest_args(raw_args)

    if iterations == 0:
        iterations = int(math.ceil((input_size + 1) / 2))

    concat_before = True
    verbose = 0

    encoder = ClassicEncoder(random_mappings,
                             input_size,
                             input_area,
                             automaton_area,
                             verbose=verbose)
    automaton = ECA(rule)
    reservoir = Reservoir(automaton,
                          iterations,
                          verbose=verbose)
    estimator = svm.SVC()
    # estimator = svm.SVC(kernel='linear')
    # estimator = linear_model.LinearRegression()
    computer = Computer(encoder,
                        reservoir,
                        estimator,
                        concat_before=concat_before,
                        verbose=verbose)

    # Generating problem sets
    n_training_sets = 1000
    n_testing_sets = 200
    raw_inputs, labels = problems.parity(n_training_sets + n_testing_sets, input_size)

    if verbose > 1:
        print "Initial configurations:"
        for i, ri in enumerate(raw_inputs):
            cutil.print_config_1dim(ri, postfix="(%d)" % i)

    time_checkpoint = time.time()

    computer.train(raw_inputs[:n_training_sets], labels[:n_training_sets])

    print "Training time: ", (time.time() - time_checkpoint)
    time_checkpoint = time.time()

    # Testing
    y_pred = computer.test(raw_inputs[n_training_sets:])

    step = 50
    print "Testing time: ", (time.time() - time_checkpoint)
    print "TEST RESULTS (samples, every 50th value)"
    print "Predicted, raw:    ", y_pred[::step]
    error = mean_squared_error(labels[n_training_sets:], y_pred)

    y_pred = rutil.classify_output(y_pred)

    print "Predicted:         ", y_pred[::step]
    print "Actual:            ", labels[n_training_sets::step]
    print "Mean squared error:", error
    corrects = 0
    for i in xrange(len(y_pred)):
        if y_pred[i] == labels[n_training_sets + i]:
            corrects += 1
    print "Correctness (%):   ", (100 * corrects / len(y_pred))

    for i in xrange(10):
        print raw_inputs[n_training_sets + i * step], "->", y_pred[i * step]


def digest_args(args):
    size = 5
    rule = 1
    iterations = 0
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
              '-s', '3',
              '-r', '90',
              '-i', '4',
              '--random-mappings', '2'])
