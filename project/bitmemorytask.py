import getopt
import marshal as dumper
import sys
import time

from sklearn import svm

import problemgenerator as problems
from ca.ca import CA
from compute.temporalcomputer import TemporalComputer
from encoders.classic import ClassicEncoder
from plotter import plot_temporal
from reservoir.reservoir import Reservoir

path = "tmpresults/"
prefix = "bitmemoryresults"
file_type = "dump"


def main(raw_args):
    size, rule, n_iterations, n_random_mappings, input_area, automaton_area = digest_args(raw_args)

    size = 4
    n_training_sets = 32
    n_testing_sets = 32
    bits = 5
    distractor_period = 200
    time_steps = 2 * bits + distractor_period + 1
    concat_before = True
    verbose = 1

    encoder = ClassicEncoder(n_random_mappings,
                             size,
                             input_area,
                             automaton_area,
                             verbose=verbose)
    automation = CA(rule, k=2, n=3)
    reservoir = Reservoir(automation, n_iterations, verbose=verbose)
    estimator = svm.SVC()
    # estimator = svm.SVC(kernel='linear')
    # estimator = linear_model.LinearRegression()
    computer = TemporalComputer(encoder,
                                reservoir,
                                estimator,
                                concat_before=concat_before,
                                verbose=verbose)

    inputs, labels = problems.bit_memory_task(n_training_sets + n_testing_sets,
                                              bits,
                                              distractor_period)

    # Training
    time_checkpoint = time.time()
    computer.train(inputs[:n_training_sets], labels[:n_training_sets])
    print "Training time:       ", (time.time() - time_checkpoint)

    # Testing
    time_checkpoint = time.time()
    x, y_pred = computer.test(inputs[:n_training_sets])
    print "Testing time:        ", (time.time() - time_checkpoint)
    out_file = open("%s%s.%s" % (path, prefix, file_type), 'wb')
    dumper.dump(y_pred.tolist(), out_file)
    dumper.dump(labels[n_training_sets:], out_file)
    out_file.close()
    n_correct = 0
    for pred, labels in zip(y_pred, labels[n_training_sets:]):
        correct = True
        for pred_element, label_element in zip(pred, labels):
            if pred_element != label_element:
                correct = False
        if correct:
            n_correct += 1
    print "Correct:              %d/%d" % (n_correct, n_testing_sets)

    # Drawing
    print "a1 positions:        ", encoder.pos(0)
    print "a2 positions:        ", encoder.pos(1)
    print "Distractor positions:", encoder.pos(2)
    print "Cue positions:       ", encoder.pos(3)
    plot_temporal(x,
                  n_random_mappings,
                  encoder.automaton_area,
                  time_steps,
                  n_iterations)


def digest_args(args):
    size = 5
    rule = 1
    iterations = 1
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
              '-s', '4',
              '-r', '90',
              '-i', '4',
              '--random-mappings', '2',
              '--input-area', '8'])
