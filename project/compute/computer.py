import itertools
import ca.util as cutil


class Computer:

    def __init__(self, encoder, reservoir, estimator, concat_before=True, verbose=1):
        """

        :param encoder:
        :param reservoir:
        :param estimator: must have functions fit(X,y) and predict(X)
        :param concat_before: whether to concatenate before or after iterations
        """
        self.encoder = encoder
        self.reservoir = reservoir
        self.estimator = estimator
        self.concat_before = concat_before
        self.verbose = verbose

    def train(self, sets, labels):
        """Encodes and transforms the sets, and fits the estimator to it

        :param sets: training sets
        :param labels: labels/classes corresponding to the training sets
        :return: void
        """
        outputs = self._translate_and_transform(sets)

        self.estimator.fit(outputs, labels)

    def test(self, sets):
        outputs = self._translate_and_transform(sets)

        return self.estimator.predict(outputs)

    def _translate_and_transform(self, sets):
        inputs = self.encoder.translate(sets)

        inputs = self._concat(inputs) if self.concat_before else inputs

        n_processes = self._n_processes(len(sets))

        outputs = self.reservoir.transform(inputs, n_processes=n_processes)

        return outputs if self.concat_before else self._concat(outputs)

    def _concat(self, elements):
        if self.verbose > 1:
            print "Concats:"
        n_random_mappings = self.encoder.n_random_mappings
        outputs = []
        for i in xrange(len(elements) / n_random_mappings):
            span = i * n_random_mappings
            concat = list(itertools.chain.from_iterable(elements[span:span+n_random_mappings]))
            outputs.append(concat)

            if self.verbose > 1:
                cutil.print_config_1dim(concat, postfix="(%d)" % i)
        return outputs

    @staticmethod
    def _n_processes(n_sets):
        if n_sets % 4 == 0:
            return 4
        elif n_sets % 2 == 0:
            return 2
        else:
            return 1


