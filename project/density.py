import sys
import getopt
import math
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import problemgenerator as problems
from ca.ca import CA
from reservoir.reservoir import Reservoir
import reservoir.util as rutil


def main(raw_args):
    length, rule, iterations = digest_args(raw_args)

    if iterations == 0:
        iterations = int(math.ceil((length + 1) / 2))

    automation = CA(rule, k=2, n=3, visual=False)
    reservoir = Reservoir(automation, iterations, 1)

    # Training
    train_inputs, train_labels = problems.density(80, length, on_probability=0.5)
    regr = linear_model.LinearRegression()
    reservoir.fit(train_inputs, train_labels, regr)

    # Testing
    test_inputs, y_true = problems.density(20, length, on_probability=0.5)
    test_outputs = []
    for config in test_inputs:
        automation = CA(1, rule, np.asarray(config), iterations)
        test_outputs.append(reservoir.compute(automation))
    y_pred = regr.predict(test_outputs)
    error = mean_squared_error(y_true, y_pred)

    print "TEST RESULTS"
    print "Raw:\t\t", y_pred
    print "Predicted:\t", rutil.classify_output(y_pred), "(2 means out of bounds)"
    print "Actual:\t\t", y_true
    print "Mean squared error is", error


def digest_args(args):
    length = 5
    rule = 1
    iterations = 0
    # length, rule, iterations
    opts, args = getopt.getopt(args[1:], "l:r:i:")
    for o, a in opts:
        if o == '-l':
            length = int(a)
        elif o == '-r':
            rule = int(a)
        elif o == '-i':
            iterations = int(a)
    return length, rule, iterations


if __name__ == '__main__':
    main(sys.argv)
