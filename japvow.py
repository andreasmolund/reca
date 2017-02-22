from itertools import count, izip

import numpy as np
from sklearn import svm
from sklearn import linear_model

from ca.eca import ECA
from compute import distribute
from compute.computer import Computer
from encoders.real import RealEncoder
from problemgenerator import japanese_vowels
from reservoir.reservoir import Reservoir
from reservoir.util import classify_output
from statistics.plotter import plot_temporal

n_random_mappings = 8
n_iterations = 2
input_size = 12
rule = 110

encoder = RealEncoder(n_random_mappings,
                      input_size,
                      0,
                      3 * input_size,
                      verbose=0)

estimator = svm.SVC()

automaton = ECA(rule)

reservoir = Reservoir(automaton,
                      n_iterations,
                      verbose=0)

computer = Computer(encoder,
                    reservoir,
                    estimator,
                    concat_before=True,
                    verbose=0)

training_sets, training_labels, testing_sets, testing_labels = japanese_vowels()

# all_values = []
# for m in training_sets:
#     for n in m:
#         all_values.extend(n)
# print "[%.1f,%.1f]" % (min(all_values), max(all_values))
# print "Avg.: %.3f" % (sum(all_values) / len(all_values))
# q = (25, 50, 75)
# print "Quantiles:", np.percentile(all_values, q)

x = computer.train(training_sets, training_labels)
test_x = computer.train(testing_sets, testing_labels)
# _, predictions = computer.test(testing_sets)

processed_training_set = distribute.post_process(training_sets)


def extend_inputs_to_outputs(raw_inputs, outputs):
    extended = []
    for raw, output in zip(raw_inputs, outputs):
        concat = raw.tolist()
        concat.extend(output)
        extended.append(concat)
    return extended

x_with_raw_input = x
test_x_with_raw_input = test_x

estimator.fit(x_with_raw_input, distribute.post_process(training_labels))
predictions = estimator.predict(test_x_with_raw_input)

n_correct = 0
n_incorrect_predictions = 0

fasit = []
for m in testing_labels:
    for n in m:
        fasit.append(n)

# print predictions[0]
# print predictions[1]

# classified_prediction = classify_output(predictions)

for i, prediction, fasit_element in izip(count(), predictions, fasit):
    if fasit_element == prediction:
        n_correct += 1
    else:
        n_incorrect_predictions += 1

    # if np.array_equal(prediction, fasit_element):
    #     n_correct += 1
    #     print predictions[i]
    # else:
    #     n_incorrect_predictions += 1

print "Correct:", n_correct
print "Incorrect:", n_incorrect_predictions
print "%d percent" % (100*n_correct / (n_correct + n_incorrect_predictions))

sample_nr = 0
time_steps = len(testing_sets[sample_nr])
plot_temporal(x,
              encoder.n_random_mappings,
              encoder.automaton_area,
              time_steps,
              n_iterations,
              sample_nr=sample_nr)
