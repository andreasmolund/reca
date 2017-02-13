import marshal as dumper
import time
from itertools import izip, count

import numpy as np

from computer import Computer, file_name


class TemporalComputer(Computer):
    """
    Constitutes the recurrent architecture
    """

    def train(self, sets, labels):
        """

        :param sets: a list of training sets, each containing a list of their chronological input vectors
        :param labels: a list/array with the same shape as sets, so that it corresponds
        :return: x, the values of the output nodes
        """
        time_checkpoint = time.time()
        x = self._translate_and_transform(sets,
                                          self.reservoir,
                                          self.encoder,
                                          self._concat_before,
                                          self._concat_after,
                                          self.concat_before,
                                          (0,),
                                          None)

        x = self._post_process(x)

        labels = np.array(labels)
        shape = labels.shape
        if len(shape) < 3:
            new_shape = shape[0] * shape[1]
        else:
            new_shape = (shape[0] * shape[1], shape[2])
        labels = labels.reshape(new_shape)

        print "Transforming time:      %.1f" % (time.time() - time_checkpoint)
        time_checkpoint = time.time()

        self.estimator.fit(x, labels)
        print "Estimator fitting time: %.1f" % (time.time() - time_checkpoint)

        return x

    def test(self, sets, x=None):
        if x is None:
            x = self._distribute_and_collect(sets)
            x = self._post_process(x)
        predictions = self.estimator.predict(x)
        new_shape = (len(sets), len(sets[0]))
        if len(predictions.shape) > 1:
            new_shape = new_shape + (predictions.shape[1],)
        return x, predictions.reshape(new_shape)

    @staticmethod
    def _post_process(outputs):
        # We want to concatenate/flatten, which must be done through reshaping
        outputs = np.array(outputs)
        shape = outputs.shape
        return outputs.reshape(shape[0] * shape[1], shape[2])

    @staticmethod
    def _translate_and_transform(sets,
                                 reservoir,
                                 encoder,
                                 concat_before_function,
                                 concat_after_function,
                                 concat_before,
                                 identifier,
                                 queue):
        n_sets = sets.shape[0]
        n_time_steps = sets.shape[1]
        size = encoder.total_area
        n_random_mappings = encoder.n_random_mappings
        automaton_area = size / n_random_mappings

        outputs = np.empty((n_time_steps, n_sets, size * reservoir.iterations), dtype='int')

        for t in xrange(n_time_steps):

            # Input at time t
            sets_at_t = sets.transpose((1, 0, 2))[t]

            if t == 0:
                # Translating the initial input
                sets_at_t = encoder.translate(sets_at_t)
            else:
                # Adding input with parts of the previous output

                new_sets_at_t = np.empty((n_sets, n_random_mappings, automaton_area), dtype='int')
                for i, set_at_t, prev_output in izip(count(), sets_at_t, outputs[t - 1]):
                    prev_state_vector = prev_output[-size:].copy()
                    new_sets_at_t[i] = encoder.normalized_addition(set_at_t, prev_state_vector)

                sets_at_t = new_sets_at_t.reshape((n_sets * n_random_mappings, automaton_area))

            # Concatenating before if that is to be done
            if concat_before:
                sets_at_t = concat_before_function(sets_at_t, n_random_mappings)

            # Transforming in the reservoir
            outputs_at_t = reservoir.transform(sets_at_t)

            # Concatenating after if it wasn't done before
            if not concat_before:
                outputs_at_t = concat_after_function(outputs_at_t, n_random_mappings, automaton_area)

            # Saving
            outputs[t] = outputs_at_t

        # "outputs" is currently a list of time steps (each containing all outputs at that time step),
        # but we want a list of outputs so that it corresponds with what came as argument to this method
        outputs = outputs.transpose(1, 0, 2)

        # Writing to file
        # out_file = open(file_name(identifier), 'wb')
        outputs = outputs.tolist()
        # dumper.dump(outputs, out_file)
        # out_file.close()

        # Telling whomever invoked this function that we're done!
        # queue.put(identifier)
        return outputs

