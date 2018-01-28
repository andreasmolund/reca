import numpy as np
import time
from sklearn.utils import shuffle

from distribute import flatten, distribute_and_collect
from computer import Computer
from reservoir.util import extend_state_vectors


class JaegerComputer(Computer):
    """
    Trains and test a reservoir according to Jaeger's methods 3/4
    (http://dx.doi.org/10.1016/j.neunet.2007.04.016).

    The basics is to take snapshots of the sequences to generate equally spaced samples of the state vectors.
    Handles sequences of unequal length, which is the main motivation for the method.
    Nevertheless, this very implemented method differs from Jaeger's
    in interpolation when D is not divisible on the sequence length.
    The implemented method chooses the closest vector instead of interpolation.

    method: if 4, then the reservoir generates one hypothesis per sequence,
            if 3, then the reservoir generates 'd' hypotheses per sequence
    """

    def __init__(self,
                 encoder,
                 reservoir,
                 estimator,
                 concat_before=True,
                 verbose=1,
                 d=3,
                 method=4):
        Computer.__init__(self, encoder, reservoir, estimator, concat_before, verbose)
        self.d = d
        self.method = method
        self.n_pieces = 1

    def train(self, sets, labels, extensions=None):

        x = distribute_and_collect(self, sets)

        if extensions is not None:
            x = extend_state_vectors(x, extensions)

        x = jaeger_method(x, self.d, self.method)
        labels = jaeger_labels(labels, self.d, self.method)

        x = flatten(x)
        labels = flatten(labels)

        time_checkpoint = time.time()
        self.estimator.fit(x, labels)
        fit_time = time.time() - time_checkpoint

        return x, fit_time

        # len_x = len(x)
        # x, labels = shuffle(x, labels)
        # piece_len = float(len_x) / self.n_pieces
        # pieces = np.empty((len_x, self.d), dtype='int8')
        # classes = np.unique(labels)
        # for i in xrange(self.n_pieces):
        #     training_start_i = int(round(i * piece_len))
        #     training_end_i = int(round(i * piece_len + piece_len))
        #     testing_start_i = training_end_i % len_x
        #     testing_end_i = (int(round(i * piece_len + piece_len + piece_len)) - 1) % len_x + 1
        #     x_piece = flatten(x[training_start_i:training_end_i])
        #     y_piece = flatten(labels[training_start_i:training_end_i])
        #     self.estimator.partial_fit(x_piece,
        #                                y_piece,
        #                                classes)
        #     computed = self.estimator.predict(flatten(x[testing_start_i:testing_end_i]))
        #     pieces[testing_start_i:testing_end_i] = computed.reshape((testing_end_i - testing_start_i, self.d))
        # return pieces

    def test(self, sets, x=None, extensions=None):
        if x is None:
            x = distribute_and_collect(self, sets)
            if extensions is not None:
                x = extend_state_vectors(x, extensions)
            x = jaeger_method(x, self.d, self.method)
            x = flatten(x)
        predictions = self.estimator.predict(x)
        return predictions, x


def jaeger_method(x, d, method, dtype='int8'):
    n = d if method == 3 else 1
    state_vector_len = len(x[0][0])
    o = state_vector_len if method == 3 else d * state_vector_len

    j_x = np.empty((len(x), n, o), dtype=dtype)
    for i, sequence in enumerate(x):
        l_i = len(sequence)
        snapshots = []
        for j in xrange(d):
            n_j = int(((j + 1.0) * l_i / d) + 0.5)
            if method == 4:
                snapshots.extend(sequence[n_j - 1])
            elif method == 3:
                snapshots.append(sequence[n_j - 1])
            else:
                raise ValueError("No Jaeger method with that number. Must be element in {3,4}")
        j_x[i] = snapshots
    return j_x


def jaeger_labels(y, d, method):
    m = len(y)
    n = d if method == 3 else 1  # How many labels per sequence
    o = 1  # Label length
    if len(y[0].shape) > 1:
        new_y_shape = (m, n, y[0].shape[1])
    else:
        new_y_shape = (m, n)
    j_y = np.empty(new_y_shape, dtype='int8')
    for i in xrange(m):
        j_y[i] = [y[i][0]] * n
    return j_y

