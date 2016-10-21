import sys
import getopt
import scipy as sp
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import problemgenerator as problems
from ca.ca import CA
from reservoir.reservoir import Reservoir
import reservoir.reservoir as rdefs
import reservoir.util as rutil
import ca.util as cutil


def main(raw_args):
    size, rule, iterations, random_mappings, input_area, atomaton_area = digest_args(raw_args)

    print "Size %d, rule %d, %d iterations, %d random mappings, input area %d, atomaton area %d" \
          % (size, rule, iterations, random_mappings, input_area, atomaton_area)

    mappings = []

    if random_mappings < 1:
        input_area = size
        atomaton_area = size
    else:
        print "RANDOM MAPPINGS:"
        for _ in xrange(random_mappings):
            mapping = rdefs.make_random_mapping(size, input_area)
            print mapping
            mappings.append(mapping)

    automation = CA(rule, k=2, n=3, visual=False)
    reservoir = Reservoir(automation, iterations, 0, size*random_mappings, input_area, atomaton_area)

    raw_train_inputs, train_labels = problems.parity(2, size)
    train_inputs = []
    print "TRAIN INPUTS:"
    for raw_train_input in raw_train_inputs:
        train_input = []
        for mapping in mappings:
            configuration = sp.zeros([size], dtype=np.dtype(int))
            for ri in xrange(len(mapping)):
                configuration[mapping[ri]] = raw_train_input[ri]
            train_input.extend(configuration)
        train_inputs.append(train_input)
        cutil.print_config_1dim(raw_train_input, prefix="From\t")
        cutil.print_config_1dim(train_input, prefix="To\t")

    # Training
    train_outputs = reservoir.transform(train_inputs)
    regr = linear_model.LinearRegression()
    regr.fit(train_outputs, train_labels)

    # Testing
    # test_inputs, y_true = problems.parity(30, size)
    # test_outputs = reservoir.transform(test_inputs)
    # y_pred = regr.predict(test_outputs)
    # error = mean_squared_error(y_true, y_pred)
    #
    # print "TEST RESULTS"
    # print "Raw:\t\t", y_pred
    # print "Predicted:\t", rutil.classify_output(y_pred), "(2 means out of bounds)"
    # print "Actual:\t\t", y_true
    # print "Mean squared error is", error


def digest_args(args):
    size = 5
    rule = 1
    iterations = 1
    random_mappings = 0
    input_area = 0
    atomaton_area = 0
    opts, args = getopt.getopt(args[1:],
                               's:r:i:h?',
                               ['random-mappings=', 'input-area=', 'atomaton-area='])
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
        elif o == '--atomaton-area':
            atomaton_area = int(a)

    print opts
    return size, rule, iterations, random_mappings, input_area, atomaton_area


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv)
    else:
        main(['parity.py',
              '-s', 4,
              '-r', 90,
              '-i', 4,
              '--random-mappings', 2,
              '--input-area', 4,
              '--atomaton-area', 4])
