# This module combines the two packages "ca" and "reservoir"
# carc stands for cellular automata reservoir computing

from ca.ca import CA
import sys
import getopt
import math
import ca.util as util
import numpy as np
import reservoir.reservoir as reservoir
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def main(raw_args):
    length = 20
    rule = 90
    iterations = 0

    if len(raw_args) > 1:
        # length, simple, rule
        opts, args = getopt.getopt(raw_args[1:], "l:r:i:")
        for o, a in opts:
            if o == '-l':
                length = int(a)
            elif o == '-r':
                rule = int(a)
            elif o == '-i':
                iterations = int(a)

    if iterations == 0:
        iterations = int(math.ceil((length + 1) / 2))

    # Training
    train_inputs = []
    train_outputs = []
    train_labels = []
    for i in xrange(300):
        config, majority = util.config_rand(length)
        train_inputs.append(config)
        automation = CA(1, rule, np.asarray(train_inputs[i]), iterations)
        train_labels.append(majority)
        train_outputs.append(reservoir.compute(automation))
    regr = linear_model.LinearRegression()
    regr.fit(train_outputs, train_labels)

    # Testing
    test_inputs = []
    test_outputs = []
    y_true = []
    for i in xrange(10):
        config, majority = util.config_rand(length)
        test_inputs.append(config)
        automation = CA(1, rule, np.asarray(test_inputs[i]), iterations)
        y_true.append(majority)
        test_outputs.append(reservoir.compute(automation))
    y_pred = regr.predict(test_outputs)
    error = mean_squared_error(y_true, y_pred)
    print "Mean squared error is", error


if __name__ == '__main__':
    main(sys.argv)
