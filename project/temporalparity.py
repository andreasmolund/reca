import sys

import time
from sklearn import svm

import problemgenerator as problems
from bitmemorytask import digest_args
from ca.ca import CA
from compute.temporalcomputer import TemporalComputer
from encoders.classic import ClassicEncoder
from reservoir.reservoir import Reservoir


def main(raw_args):
    size, rule, n_iterations, n_random_mappings, input_area, automaton_area = digest_args(raw_args)

    n_training_sets = 200
    n_testing_sets = 50
    time_steps = 30
    delay = 0
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

    training_inputs, training_labels = problems.temporal_parity(n_training_sets,
                                                                time_steps,
                                                                window_size=size,
                                                                delay=delay)
    testing_inputs, testing_labels = problems.temporal_parity(n_testing_sets,
                                                              time_steps,
                                                              window_size=size,
                                                              delay=delay)
    print zip(training_inputs[0][:8], training_labels[0][:8])

    time_checkpoint = time.time()
    computer.train(training_inputs, training_labels)
    x, y_pred = computer.test(testing_inputs)
    print "Training and testing time:", (time.time() - time_checkpoint)

    n_correct = 0
    n_semi_correct = 0
    n_total_semi = 0
    print y_pred[0][:10].tolist()
    print testing_labels[0][:10]
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


if __name__ == '__main__':
    arg_size = '3'
    arg_rule = '90'
    arg_n_iterations = '4'
    arg_n_random_mappings = '8'
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

