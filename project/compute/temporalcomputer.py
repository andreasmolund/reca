import marshal as dumper

import numpy as np
import time

from compute.computer import Computer, file_name, _n_processes
from operators import bitwise_or


class TemporalComputer(Computer):

    def train(self, sets, labels):
        """

        :param sets: a list of training sets, each containing a list of their chronological input vectors
        :param labels: a list/array with the same shape as sets, so that it corresponds
        :return: void
        """
        time_checkpoint = time.time()
        x = self._distribute_and_collect(sets, n_processes=4)
        x = self._post_process(x)
        if self.verbose > 0:
            print "Transformed training sets:", time.time() - time_checkpoint

        labels = np.array(labels).flatten()
        # self.estimator.fit(x, labels)

    def test(self, sets):
        time_checkpoint = time.time()
        x = self._distribute_and_collect(sets, n_processes=4)
        x = self._post_process(x)
        if self.verbose > 0:
            print "Transformed testing sets: ", time.time() - time_checkpoint

        return x, self.estimator.predict(x).reshape(len(sets), len(sets[0]))

    @staticmethod
    def _post_process(outputs):
        # We want to concatenate/flatten, which must be done through reshaping
        outputs = np.array(outputs)
        shape = outputs.shape
        return outputs.reshape(shape[0] * shape[1], shape[2])

    @staticmethod
    def _translate_and_transform(sets, reservoir, encoder, concat_function, concat_before, identifier, queue):
        out_file = open(file_name(identifier), 'wb')
        n_time_steps = len(sets[0])
        size = encoder.total_area
        n_random_mappings = encoder.n_random_mappings

        outputs = [None] * n_time_steps

        for t in xrange(n_time_steps):

            # Input at time t
            sets_at_t = [s[t] for s in sets]

            if t == 0:
                # Translating the initial input
                sets_at_t = encoder.translate(sets_at_t)
            else:
                # Adding input with parts of the previous output
                new_sets_at_t = []
                for set_at_t, prev_output in zip(sets_at_t, outputs[t - 1]):
                    new_sets_at_t.extend(encoder.mapping_addition(set_at_t, prev_output[-size:]))
                sets_at_t = new_sets_at_t

            # Concatenating before if that is to be done
            sets_at_t = concat_function(sets_at_t, n_random_mappings) if concat_before else sets_at_t

            # Transforming in the reservoir
            outputs_at_t = reservoir.transform(sets_at_t)

            # Concatenating after if it wasn't done before
            if not concat_before:
                outputs_at_t = concat_function(outputs_at_t, n_random_mappings)

            # Saving
            outputs[t] = outputs_at_t

        # "outputs" is currently a list of time steps (each containing all outputs at that time step),
        # but we want a list of outputs so that it corresponds with what came as argument to this method
        outputs = np.transpose(outputs, (1, 0, 2))

        # Writing to file
        dumper.dump(outputs.tolist(), out_file)
        out_file.close()

        # Telling whomever invoked this function that we're done!
        queue.put(identifier)


combine = np.vectorize(bitwise_or)
