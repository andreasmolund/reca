import getopt
import marshal as dumper
import sys
import time

from datetime import datetime
from sklearn import svm, linear_model
import logging

import problemgenerator as problems
from ca.ca import CA
from compute.temporalcomputer import TemporalComputer
from encoders.classic import ClassicEncoder
from plotter import plot_temporal
from reservoir.reservoir import Reservoir
from reservoir.util import classify_output

result_path = "preresults/"
result_prefix = "bitmemoryresults"
file_type = "dump"
start_time = datetime.now()
logit = True

n_whole_runs = 500
n_training_sets = 32
n_testing_sets = 32
bits = 5
distractor_period = 200
inputs, labels = problems.bit_memory_task(n_training_sets + n_testing_sets,
                                          bits,
                                          distractor_period)


def main(raw_args):
    size, rule, n_iterations, n_random_mappings, input_area, automaton_area = digest_args(raw_args)

    size = 4
    time_steps = 2 * bits + distractor_period + 1
    concat_before = True
    verbose = 1

    encoder = ClassicEncoder(n_random_mappings,
                             size,
                             input_area,
                             automaton_area,
                             verbose=verbose)
    automaton = CA(rule, k=2, n=3)
    reservoir = Reservoir(automaton, n_iterations, verbose=verbose)
    estimator = linear_model.LinearRegression()
    # estimator = svm.SVC()
    # estimator = svm.SVC(kernel='linear')
    computer = TemporalComputer(encoder,
                                reservoir,
                                estimator,
                                concat_before=concat_before,
                                verbose=verbose)

    # Training
    # time_checkpoint = time.time()
    x = computer.train(inputs[:n_training_sets], labels[:n_training_sets])
    # print "Training time:       ", (time.time() - time_checkpoint)

    # Testing
    # time_checkpoint = time.time()
    _, y_pred = computer.test(inputs[n_training_sets:], x)
    # print "Testing time:        ", (time.time() - time_checkpoint)
    y_pred = [[classify_output(t) for t in s] for s in y_pred]
    # out_file = open("%s%s.%s" % (result_path, result_prefix, file_type), 'wb')
    # dumper.dump(y_pred, out_file)
    # dumper.dump(labels[n_training_sets:], out_file)
    # out_file.close()
    n_correct = 0
    n_incorrect_bits = 0
    for pred, set_labels in zip(y_pred, labels[n_training_sets:]):
        correct = True
        for pred_element, label_element in zip(pred, set_labels):
            if pred_element != label_element:
                correct = False
                n_incorrect_bits += 1
        if correct:
            n_correct += 1
    # print "Correct:              %d/%d" % (n_correct, n_testing_sets)
    # print "Incorrect bits:       %d" % n_incorrect_bits
    if logit:
        logging.info("%d,%d,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d",
                     n_iterations,
                     encoder.n_random_mappings,
                     rule,
                     size,
                     encoder.input_area,
                     encoder.automaton_area,
                     concat_before,
                     estimator.__class__.__name__,
                     n_training_sets,
                     n_testing_sets,
                     distractor_period,
                     1 if n_correct == n_testing_sets else 0,
                     n_correct,
                     n_incorrect_bits)

    # Drawing
    # print "a1 positions:        ", encoder.pos(0)
    # print "a2 positions:        ", encoder.pos(1)
    # print "Distractor positions:", encoder.pos(2)
    # print "Cue positions:       ", encoder.pos(3)
    if n_whole_runs < 2:
        plot_temporal(x,
                      encoder.n_random_mappings,
                      encoder.automaton_area,
                      time_steps,
                      n_iterations,
                      sample_nr=12)


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

    return size, rule, iterations, random_mappings, input_area, automaton_area

if __name__ == '__main__':
    if logit:
        logging.basicConfig(format='"%(asctime)s",%(message)s',
                            filename='preresults/%s-bitmem1res.csv' % start_time.isoformat(),
                            level=logging.DEBUG)
        logging.info("I,R,Rule,Input size,Input area,Automaton size,Concat before,Estimator,"
                     "Training sets,Testing sets,Distractor period,"
                     "Point (success),Successful,Wrong bits")
    for r in xrange(n_whole_runs):
        # print "Run %d started" % r
        if len(sys.argv) > 1:
            main(sys.argv)
        else:
            main(['bitmemorytask.py',
                  '-r', '102',
                  '-i', '4',
                  '--random-mappings', '4',
                  '--input-area', '40',
                  '--automaton-area', '0'])
