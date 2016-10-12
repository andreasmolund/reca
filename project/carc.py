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


def main(raw_args):
    length = 21
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

    train_inputs = []
    train_outputs = []
    train_labels = []
    for i in xrange(20):
        config, majority = util.config_rand(length)
        automation = CA(1, rule, np.asarray(train_inputs[i]), iterations)

        train_inputs.append(config)
        train_labels.append(majority)
        train_outputs.append(reservoir.compute(automation))

    regr = linear_model.LinearRegression()
    regr.fit(train_outputs, train_labels)


if __name__ == '__main__':
    main(sys.argv)
