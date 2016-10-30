import sys
import getopt
import math
import itertools
import time
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import mean_squared_error
import problemgenerator as problems
from ca.ca import CA
from reservoir.reservoir import Reservoir
import reservoir.util as rutil
from encoders.classic import ClassicEncoder
import ca.util as cutil


def main(raw_args):
    input_size, rule, iterations, random_mappings, input_area, automaton_area = digest_args(raw_args)

    if iterations == 0:
        iterations = int(math.ceil((input_size + 1) / 2))

    concat_before = False
    concat_after = not concat_before

    encoder = ClassicEncoder(random_mappings,
                             input_size,
                             input_area,
                             automaton_area,
                             verbose=1,
                             concat=concat_before)
    automaton = CA(rule, k=2, n=3)
    reservoir = Reservoir(automaton, iterations, verbose=2)

    n_training_sets = 2
    n_testing_sets = 0
    time_checkpoint = time.time()
    raw_inputs, labels = problems.parity(n_training_sets + n_testing_sets, input_size)
    print "Initial configurations:"
    for i, ri in enumerate(raw_inputs):
        cutil.print_config_1dim(ri, postfix="(%d)" % i)
    inputs = encoder.translate(raw_inputs)
    raw_outputs = reservoir.transform(inputs, n_processes=2)
    print "Transforming time: ", (time.time() - time_checkpoint)

    if not concat_before:
        outputs = []
        for i in xrange(n_training_sets + n_testing_sets):
            span = i * random_mappings
            bleh = list(itertools.chain.from_iterable(raw_outputs[span:span+random_mappings]))
            cutil.print_config_1dim(bleh)
            outputs.append(bleh)
    else:
        outputs = raw_outputs


    # Training
    # time_checkpoint = time.time()
    # # regr = linear_model.LinearRegression()
    # # regr.fit(train_outputs, train_labels)
    # regr = svm.SVC(kernel='linear')
    # regr.fit(outputs[:n_training_sets], labels[:n_training_sets])
    # print "Fitting time time: ", (time.time() - time_checkpoint)
    #
    # # Testing
    # y_pred = regr.predict(outputs[n_testing_sets:])
    # y_pred_class = rutil.classify_output(y_pred)
    # error = mean_squared_error(labels[n_testing_sets:], y_pred)
    #
    # step = 50
    # print "TEST RESULTS (samples, every 50th value)"
    # print "Predicted, raw:    ", y_pred[::step]
    # print "Predicted:         ", y_pred_class[::step]
    # print "Actual:            ", labels[n_testing_sets::step]
    # print "Mean squared error:", error
    # corrects = 0
    # for i in xrange(len(y_pred_class)):
    #     if y_pred_class[i] == labels[n_testing_sets + i]:
    #         corrects += 1
    # print "Correctness (%):   ", (100 * corrects / len(y_pred_class))
    #
    # for i in xrange(10):
    #     print raw_inputs[n_testing_sets + i * step], "->", y_pred_class[i * step]


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
