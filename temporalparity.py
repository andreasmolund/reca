import sys
import time

from sklearn import linear_model

import problemgenerator as problems
from bittask import digest_args
from ca.eca import ECA
from compute.computer import Computer
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir
from reservoir.util import classify_output
from statistics.plotter import plot_temporal


def main(raw_args):
    _, size, rule, n_iterations, n_random_mappings, input_area, automaton_area = digest_args(raw_args)

    n_training_sets = 30
    n_testing_sets = 20
    time_steps = 20
    delay = 0
    concat_before = True
    verbose = 1

    encoder = ClassicEncoder(n_random_mappings,
                             size,
                             input_area,
                             automaton_area,
                             verbose=verbose)
    automation = ECA(rule)
    reservoir = Reservoir(automation, n_iterations, verbose=verbose)
    # estimator = svm.SVC()
    # estimator = svm.SVC(kernel='linear')
    estimator = linear_model.LinearRegression()
    computer = Computer(encoder,
                        reservoir,
                        estimator,
                        concat_before=concat_before,
                        verbose=verbose)

    training_inputs, training_labels = problems.temporal_parity(n_training_sets,
                                                                time_steps,
                                                                window_size=size,
                                                                delay=delay)
    testing_inputs, testing_labels = problems.temporal_parity(n_testing_sets,
                                                              time_steps,
                                                              window_size=size,
                                                              delay=delay)

    sample_i = n_testing_sets / 2
    sample_quantity = 8
    print "Sample input, output:", zip(testing_inputs[sample_i][:sample_quantity],
                                       testing_labels[sample_i][:sample_quantity])

    time_checkpoint = time.time()
    computer.train(training_inputs, training_labels)
    x, y_pred_pre = computer.test(testing_inputs)
    y_pred = [classify_output(set_prediction) for set_prediction in y_pred_pre]
    print "Training and testing time:", (time.time() - time_checkpoint)

    n_correct = 0
    n_semi_correct = 0
    n_total_semi = 0
    print "Predicted (raw): [%s]" % ', '.join(["%.2f" % value for value in y_pred_pre[sample_i][:sample_quantity]])
    print "Predicted:      ", y_pred[sample_i][:sample_quantity]
    print "Factual:        ", testing_labels[sample_i][:sample_quantity]
    for predicted, actual in zip(y_pred, testing_labels):
        correct = True
        for y1, y2 in zip(predicted, actual):
            if y1 != y2:
                correct = False
            else:
                n_semi_correct += 1
            n_total_semi += 1
        n_correct += 1 if correct else 0
    print "Correct:       %d/%d" % (n_correct, n_testing_sets)
    print "Semi correct:  %d/%d" % (n_semi_correct, n_total_semi)

    plot_temporal(x,
                  n_random_mappings,
                  encoder.automaton_area,
                  time_steps + delay,
                  n_iterations,
                  sample_nr=sample_i)


if __name__ == '__main__':
    arg_size = '3'
    arg_rule = '204'
    arg_n_iterations = '4'
    arg_n_random_mappings = '4'
    arg_input_area = '30'

    if len(sys.argv) > 1:
        main(sys.argv)
    else:
        main(['parity.py',
              '-s', arg_size,
              '-r', arg_rule,
              '-i', arg_n_iterations,
              '--random-mappings', arg_n_random_mappings,
              '--input-area', arg_input_area])

