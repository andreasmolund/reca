import sys
import getopt
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import problemgenerator as problems
from ca.ca import CA
from reservoir.reservoir import Reservoir
import reservoir.util as rutil


def main(raw_args):
    input_size, rule, iterations, random_mappings, input_area, automaton_area = digest_args(raw_args)

    print "Size %d, rule %d, %d iterations, %d random mappings, input area %d, automaton area %d" \
          % (input_size, rule, iterations, random_mappings, input_area, automaton_area)

    if iterations == 0:
        iterations = int(math.ceil((input_size + 1) / 2))

    automation = CA(rule, k=2, n=3)
    reservoir = Reservoir(automation,
                          iterations,
                          random_mappings,
                          input_size,
                          input_area,
                          automaton_area,
                          verbose=True)

    # Training
    train_inputs, train_labels = problems.parity(2, input_size)
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
              '-s', 4,
              '-r', 90,
              '-i', 4,
              '--random-mappings', 2,
              '--input-area', 4,
              '--automaton-area', 4])
