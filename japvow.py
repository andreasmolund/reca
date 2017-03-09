from itertools import count, izip

from sklearn import svm
from sklearn import linear_model

from ca.eca import ECA
from compute.computer import Computer
from compute.distribute import flatten, distribute_and_collect, extend_state_vectors
from encoders.real import RealEncoder, quantize_l
from problemgenerator import japanese_vowels
from reservoir.reservoir import Reservoir
from statistics.plotter import plot_temporal

n_random_mappings = 16
n_iterations = 16
input_size = 12
rule = 90

encoder = RealEncoder(n_random_mappings,
                      input_size,
                      0,
                      input_size,
                      verbose=0)

estimator = svm.SVC()
# estimator = linear_model.SGDClassifier()

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

# q = (25, 50, 75)
# print "Quantiles:", np.percentile(all_values, q)

d = 3  # Jaeger's proposed D


def jaeger_method(x1, y1):
    subtle_x = []
    subtle_y = []
    for m_i, m in enumerate(x1):
        l_i = len(m)
        snapshots = []
        for j in xrange(d):
            n_j = float((j + 1) * l_i) / d
            n_j = int(n_j + 0.5)  # Rounding off
            snapshots.extend(m[n_j - 1])  # Jaeger's method 4
        subtle_x.append(snapshots)
        subtle_y.append(y1[m_i][0])
    return subtle_x, subtle_y

training_x = distribute_and_collect(computer, training_sets)
training_x = extend_state_vectors(training_x, training_sets)
jaeger_x, jaeger_y = jaeger_method(training_x, training_labels)
estimator.fit(jaeger_x, jaeger_y)

testing_x = distribute_and_collect(computer, testing_sets)
testing_x = extend_state_vectors(testing_x, testing_sets)
jaeger_testing_x, jaeger_testing_y = jaeger_method(testing_x, testing_labels)
predictions = estimator.predict(jaeger_testing_x)

n_correct = 0
n_incorrect_predictions = 0

# fasit = flatten(testing_labels)

# print predictions[0]
# print predictions[1]

# classified_prediction = classify_output(predictions)

for i, prediction, fasit_element in izip(count(), predictions, jaeger_testing_y):
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
plot_temporal(testing_x[sample_nr],
              encoder.n_random_mappings,
              encoder.automaton_area,
              time_steps,
              n_iterations,
              sample_nr=sample_nr)
