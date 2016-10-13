import sys
import getopt
import math
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from density import digest_args
import problemgenerator as problems
from ca.ca import CA
import reservoir.reservoir as reservoir
import reservoir.util as rutil


def main(raw_args):
    length, rule, iterations = digest_args(raw_args)

    if iterations == 0:
        iterations = int(math.ceil((length + 1) / 2))

    # Training
    train_inputs, train_labels = problems.parity(80, length)
    train_outputs = []
    for config in train_inputs:
        automation = CA(1, rule, np.asarray(config), iterations)
        train_outputs.append(reservoir.compute(automation))
    regr = linear_model.LinearRegression()
    regr.fit(train_outputs, train_labels)

    # Testing
    test_inputs, y_true = problems.parity(20, length)
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


if __name__ == '__main__':
    main(sys.argv)
