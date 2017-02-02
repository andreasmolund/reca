import getopt
import math
import sys

from sklearn import svm
from sklearn.metrics import mean_squared_error

import problemgenerator as problems
import reservoir.util as rutil
from ca.eca import ECA
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir


# This is the non-temporal density classification task;
# not an experiment that is to be included in the report

def main(raw_args):
    input_size, rule, iterations, random_mappings, input_area, automaton_area = digest_args(raw_args)

    if iterations == 0:
        iterations = int(math.ceil((input_size + 1) / 2))

    encoder = ClassicEncoder(random_mappings, input_size, input_area, automaton_area)
    automaton = ECA(rule)
    reservoir = Reservoir(automaton, iterations, verbose=False)

    # Training
    raw_train_inputs, train_labels = problems.density(5000, input_size, on_probability=0.5)
    train_inputs = encoder.translate(raw_train_inputs)
    train_outputs = reservoir.transform(train_inputs)

    sys.stdout.write("Fitting... ")
    sys.stdout.flush()
    # regr = linear_model.LinearRegression()
    # regr.fit(train_outputs, train_labels)
    regr = svm.SVC(kernel='linear')
    regr.fit(train_outputs, train_labels)
    sys.stdout.write("Done\n")
    sys.stdout.flush()

    # Testing
    raw_test_inputs, y_true = problems.density(500, input_size, on_probability=0.5)
    test_inputs = encoder.translate(raw_test_inputs)
    test_outputs = reservoir.transform(test_inputs)
    y_pred = regr.predict(test_outputs)
    y_pred_class = rutil.classify_output(y_pred)
    error = mean_squared_error(y_true, y_pred)

    print "TEST RESULTS (samples, every 50th value)"
    print "Predicted, raw:    ", y_pred[::50]
    print "Predicted:         ", y_pred_class[::50]
    print "Actual:            ", y_true[::50]
    print "Mean squared error:", error
    corrects = 0
    for i in xrange(len(y_pred_class)):
        if y_pred_class[i] == y_true[i]:
            corrects += 1
    print "Correctness (%):   ", (100 * corrects / len(y_pred_class))


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
        main(['density.py',
              '-s', 3,
              '-r', 90,
              '-i', 16,
              '--random-mappings', 4,
              '--input-area', 4,
              '--automaton-area', 4])
