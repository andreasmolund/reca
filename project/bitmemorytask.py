import sys
import getopt
import scipy as sp
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import problemgenerator as problems
from ca.ca import CA
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir
from reservoir.reservoir import make_random_mapping
import reservoir.util as rutil
import ca.util as cutil


def main(raw_args):
    size, rule, iterations, random_mappings, input_area, atomaton_area = digest_args(raw_args)

    mappings = []

    size = 4
    quantity = 2
    bits = 5
    distractor_period = 3
    time_steps = 2 * bits + distractor_period + 1

    # if random_mappings < 1:
    #     input_area = size
    #     atomaton_area = size
    # else:
    #     print "RANDOM MAPPINGS:"
    #     for _ in xrange(random_mappings):
    #         mapping = make_random_mapping(size, input_area)
    #         print mapping
    #         mappings.append(mapping)

    encoder = ClassicEncoder(random_mappings, size, input_area, atomaton_area)
    automation = CA(rule, k=2, n=3)
    reservoir = Reservoir(automation, iterations, verbose=False)

    raw_train_inputs, train_labels = problems.bit_memory_task(quantity, bits, distractor_period)
    # E.g. [[1000,1000,1000,0100,...], [1000,1000,1000,0100,...], [1000,1000,1000,0100,...]]

    # Training
    train_outputs = [[None]] * quantity
    for t in xrange(time_steps):
        # Get the input for the time step
        inputs_at_t = [i[t] for i in raw_train_inputs]

        # Translate it
        translations = encoder.translate(inputs_at_t)

        # Transform it
        outputs_at_t = np.array(reservoir.transform(translations, n_processes=1))

        for i, output in enumerate(outputs_at_t):
            train_outputs[i]=outputs_at_t[i]

    regr = linear_model.LinearRegression()
    regr.fit(train_outputs, train_labels)

    # # Testing
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
