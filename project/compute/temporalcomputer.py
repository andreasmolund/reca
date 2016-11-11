import numpy as np

import ca.util as cutil
from compute.computer import Computer
from operators import bitwise_or


class TemporalComputer(Computer):

    def train(self, sets, labels):
        """

        :param sets: a list of training sets, each containing a list of their chronological input vectors
        :param labels: a list/array with the same shape as sets, so that it corresponds
        :return: void
        """
        x = self._translate_and_transform(sets)
        Y = np.array(labels).flatten()
        self.estimator.fit(x, Y)

    def test(self, sets):
        x = self._translate_and_transform(sets)
        return x, self.estimator.predict(x).reshape(len(sets), len(sets[0]))

    def _translate_and_transform(self, sets):
        """You need to reshape it to get a list of sets again

        :return: a list of all time steps to every set
        """
        n_processes = self._n_processes(len(sets))
        n_time_steps = len(sets[0])
        size = self.encoder.total_area

        if self.verbose > 1:
            print "Each time step in every set:"
            for i, the_set in enumerate(sets):
                print "Set %d:" % i
                for time_step in the_set:
                    cutil.print_config_1dim(time_step)

        # Outputs, used for regression/classification
        outputs = [None] * n_time_steps

        for t in xrange(n_time_steps):
            if self.verbose > 1:
                print "Time step %d:" % t

            sets_at_t = [s[t] for s in sets]  # Input at time t
            if t == 0:
                # Translating the initial input
                sets_at_t = self.encoder.translate(sets_at_t)
            else:
                # Adding input with parts of the previous output
                new_sets_at_t = []
                for set_at_t, prev_output in zip(sets_at_t, outputs[t - 1]):
                    new_sets_at_t.extend(self.encoder.mapping_addition(set_at_t, prev_output[-size:]))
                sets_at_t = new_sets_at_t
            sets_at_t = self._concat(sets_at_t) if self.concat_before else sets_at_t
            outputs_at_t = self.reservoir.transform(sets_at_t, n_processes=n_processes)
            outputs[t] = outputs_at_t if self.concat_before else self._concat(outputs_at_t)

        # "outputs" is currently a list of time steps (each containing all outputs at that time step),
        # but we want a list of outputs so that it corresponds with the labels (transposition)
        x = np.transpose(outputs, (1, 0, 2))

        # We also want to concatenate/flatten, which must be done through reshaping
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        if self.verbose > 1:
            print "Final outputs:"
            for time_step in x:
                cutil.print_config_1dim(time_step)

        return x


combine = np.vectorize(bitwise_or)
