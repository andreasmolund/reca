import numpy as np
from itertools import count, izip

import time
from sklearn import svm
from sklearn import linear_model

from ca.eca import ECA
from compute.computer import Computer
from compute.distribute import flatten, distribute_and_collect, extend_state_vectors
from encoders.real import RealEncoder
from problemgenerator import japanese_vowels
from reservoir.reservoir import Reservoir
from stats.plotter import plot_temporal
from encoders.real import quantize_len


def unpickle(cifar_file):
    import cPickle
    fo = open(cifar_file, 'rb')
    dictionary = cPickle.load(fo)
    fo.close()
    return dictionary


n_random_mappings = 2
n_iterations = 14
rule = 90


def read_files(file_names):
    cifar_dir = "/home/andreas/Documents/cifar-10-batches-py/"
    t_dict = unpickle(cifar_dir + file_names[0])
    t_set = t_dict['data']
    t_labels = t_dict['labels']
    for file_name in file_names[1:]:
        t_dict = unpickle(cifar_dir + file_name)
        t_set = np.append(t_set, t_dict['data'], axis=0)
        t_labels.extend(t_dict['labels'])
    return t_set, t_labels


training_sets, training_labels = read_files(["data_batch_1",
                                             "data_batch_2",
                                             "data_batch_3",
                                             "data_batch_4",
                                             "data_batch_5"])
testing_sets, testing_labels = read_files(["test_batch"])

training_sets = training_sets.reshape((50000, 1, 3072))
training_labels = np.array(training_labels).reshape((50000, 1))
testing_sets = testing_sets.reshape((10000, 1, 3072))
testing_labels = np.array(testing_labels).reshape((10000, 1))

# # q = (25, 50, 75)
# # print np.percentile(training_set.ravel(), q)

# classifier = svm.SVC()
classifier = linear_model.SGDClassifier()
ca = ECA(rule)
reservoir = Reservoir(ca, n_iterations, verbose=0)
encoder = RealEncoder(n_random_mappings, 3072, 0, 3072, verbose=0)
computer = Computer(encoder, reservoir, classifier, True, False, verbose=0)

up_to = 100
training_sets = training_sets[0:up_to]
training_labels = training_labels[0:up_to]
testing_sets = testing_sets[0:100]
testing_labels = testing_labels[0:100]

time_checkpoint = time.time()
computer.train(training_sets, training_labels)
predictions, _ = computer.test(testing_sets)
print "Train + test time:", (time.time() - time_checkpoint)

n_correct = 0
for predicted, actual in zip(predictions, testing_labels.ravel()):
    if predicted == actual:
        n_correct += 1

print "Correct (%):", (float(n_correct) / len(predictions))
print predictions[0:36]
