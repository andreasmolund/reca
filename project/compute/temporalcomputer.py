import numpy as np
from operators import bitwise_or
import ca.util as cutil

from compute.computer import Computer


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
        return self.estimator.predict(x)

    def _translate_and_transform(self, sets):
        n_processes = self._n_processes(len(sets))
        n_time_steps = len(sets[0])
        size = self.encoder.to_area

        if self.verbose > 1:
            for the_set in sets:
                for time_step in the_set:
                    cutil.print_config_1dim(time_step)

        # Outputs, used for regression/classification
        outputs = [None] * n_time_steps

        for t in xrange(n_time_steps):
            if self.verbose > 1:
                print "Time step %d:" % t

            sets_at_t = [s[t] for s in sets]  # Input
            if t == 0:
                sets_at_t = self.encoder.translate(sets_at_t)  # Input, translated/encoded
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
        # but we want a list of outputs so that it corresponds with the labels
        x = np.transpose(outputs, (1, 0, 2))
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
        if self.verbose > 1:
            print "Final outputs:"
            for time_step in x:
                cutil.print_config_1dim(time_step)
        return x


combine = np.vectorize(bitwise_or)