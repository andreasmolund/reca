import numpy as np

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
                 extended_state_vector=False,
                 verbose=1,
                 d=3,
                 method=4):
        Computer.__init__(self, encoder, reservoir, estimator, concat_before, extended_state_vector, verbose)
        self.d = d
        self.method = method

    def train(self, sets, labels):
        x = distribute_and_collect(self, sets)
        if self.extended_state_vector:
            x = extend_state_vectors(x, sets)
        x = jaeger_method(x, self.d, self.method)
        labels = jaeger_labels(labels, self.d, self.method)
        x = flatten(x)
        labels = flatten(labels)
        self.estimator.fit(x, labels)
        return x

    def test(self, sets, x=None):
        if x is None:
            x = distribute_and_collect(self, sets)
            if self.extended_state_vector:
                x = extend_state_vectors(x, sets)
            x = jaeger_method(x, self.d, self.method)
            x = flatten(x)
        predictions = self.estimator.predict(x)
        return predictions, x


def jaeger_method(x, d, method):
    state_vector_len = len(x[0][0])
    n = d if method == 3 else 1
    o = state_vector_len if method == 3 else d * state_vector_len

    j_x = np.empty((len(x), n, o), dtype='int8')
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
    j_y = []
    n_labels = d if method == 3 else 1
    for sequence in y:
        j_y.append([sequence[0]] * n_labels)
    return j_y

