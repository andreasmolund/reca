import time

import numpy as np

from distribute import flatten, distribute_and_collect
from reservoir.util import extend_state_vectors


class Computer:
    """
    Constitutes the recurrent architecture

    Combines the encoder, reservoir, and estimator,
    so that one can more easily train and use a reservoir computing system.
    """

    def __init__(self,
                 encoder,
                 reservoir,
                 estimator,
                 concat_before=True,
                 verbose=1):
        """

        :param encoder:
        :param reservoir: must have a function transform(sets)
        :param estimator: must have functions fit(X,y) and predict(X)
        :param concat_before: whether to concatenate before or after iterations
        """
        self.encoder = encoder
        self.reservoir = reservoir
        self.estimator = estimator
        self.concat_before = concat_before
        self.verbose = verbose

    def train(self, sets, labels, extensions=None):
        """

        :param sets: a list of training sets with shape (m,n,o),
                     each containing a list of their chronological input vectors
        :param labels: a list/array with the same shape as sets, so that it corresponds
        :param extensions: appended to state vectors if is not None, shape (m,n,p)
        :return: x, the values of the output nodes
        :rtype: python list
        """
        time_checkpoint = time.time()

        x = distribute_and_collect(self, sets)
        # x = np.array(x, dtype='int8')
        # sequence_lengths = [len(m) for m in x]
        if extensions is not None:
            x = extend_state_vectors(x, extensions)
        x = flatten(x)

        labels = flatten(labels)

        if self.verbose > 0:
            print "Transforming time:      %.1f" % (time.time() - time_checkpoint)
            time_checkpoint = time.time()

        self.estimator.fit(x, labels)

        if self.verbose > 0:
            print "Estimator fitting time: %.1f" % (time.time() - time_checkpoint)

        return x

    def test(self, sets, x=None, extensions=None):
        if x is None:
            x = distribute_and_collect(self, sets)
            # x = np.array(x, dtype='int8')
            if extensions is not None:
                x = extend_state_vectors(x, extensions)
            x = flatten(x)
        predictions = self.estimator.predict(x)
        return predictions, x
