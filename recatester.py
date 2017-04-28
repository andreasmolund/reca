# Module for testing the reservoir.
# No scientific testing, just visualizing.
# Hard coding everything, so to speak.

import marshal as dumper

from sklearn import linear_model

in_file = open('20bitinputclass', 'rb')
inputs = dumper.load(in_file)
in_file.close()
in_file = open('20bitlabelsclass', 'rb')
labels = dumper.load(in_file)
in_file.close()
model = linear_model.SGDClassifier(n_jobs=-1)
print "loaded"
model.fit(inputs, labels)
print "trained"

# input_size = 4
# n_random_mappings = 3
# n_iterations = 3
# rule = 90
# diffuse = 7
# pad = 0
# verbose = 0
# concat_before = False
#
# inputs = np.array([[[0, 1, 0, 0],
#                     [1, 0, 0, 0]]])
#
# labels = np.array([[[0, 1],
#                     [0, 1]]])  # Doesn't matter
#
# encoder = ClassicEncoder(n_random_mappings,
#                          input_size,
#                          diffuse,
#                          pad,
#                          verbose=verbose)
# encoder.random_mappings = [6, 0, 2, 3,
#                            0, 1, 6, 3,
#                            5, 2, 1, 0]
#
# automaton = ECA(rule)
# reservoir = Reservoir(automaton,
#                       n_iterations,
#                       verbose=verbose)
#
# estimator = linear_model.LinearRegression()
#
# computer = Computer(encoder,
#                     reservoir,
#                     estimator,
#                     concat_before=concat_before,
#                     verbose=verbose)
#
# # Preserve the state vectors
# x, _ = computer.train(inputs, labels)
# plot_temporal(x,
#               encoder.n_random_mappings,
#               encoder.automaton_area,
#               2,
#               n_iterations,
#               sample_nr=0)
