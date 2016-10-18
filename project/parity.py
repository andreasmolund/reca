import sys
import math
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from density import digest_args
import problemgenerator as problems
from ca.ca import CA
from reservoir.reservoir import Reservoir
import reservoir.util as rutil


def main(raw_args):
    length, rule, iterations = digest_args(raw_args)

    if iterations == 0:
        iterations = int(math.ceil((length + 1) / 2))

    automation = CA(rule, k=2, n=3, visual=False)
    reservoir = Reservoir(automation, iterations, 2, length, length, length)

    # Training
    train_inputs, train_labels = problems.parity(2, length)
    train_outputs = reservoir.transform(train_inputs)
    regr = linear_model.LinearRegression()
    regr.fit(train_outputs, train_labels)

    # Testing
    # test_inputs, y_true = problems.parity(1, length)
    # test_outputs = reservoir.transform(test_inputs)
    # y_pred = regr.predict(test_outputs)
    # error = mean_squared_error(y_true, y_pred)
    #
    # print "TEST RESULTS"
    # print "Raw:\t\t", y_pred
    # print "Predicted:\t", rutil.classify_output(y_pred), "(2 means out of bounds)"
    # print "Actual:\t\t", y_true
    # print "Mean squared error is", error


if __name__ == '__main__':
    main(sys.argv if len(sys.argv) > 1 else ['', '-l', 4, '-r', 90, '-i', 4])
