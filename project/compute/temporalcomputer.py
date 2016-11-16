import marshal as dumper

import numpy as np
from computer import Computer, file_name

from operators import bitwise_or


class TemporalComputer(Computer):

    def train(self, sets, labels):
        """

        :param sets: a list of training sets, each containing a list of their chronological input vectors
        :param labels: a list/array with the same shape as sets, so that it corresponds
        :return: void
        """
        x = self._distribute_and_collect(sets)
        x = self._post_process(x)

        labels = np.array(labels)
        shape = labels.shape
        if len(shape) < 3:
            new_shape = shape[0] * shape[1]
        else:
            new_shape = (shape[0] * shape[1], shape[2])
        labels = labels.reshape(new_shape)
        self.estimator.fit(x, labels)
        return x

    def test(self, sets):
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
            if concat_before:
                sets_at_t = concat_before_function(sets_at_t, n_random_mappings)

            # Transforming in the reservoir
            outputs_at_t = reservoir.transform(sets_at_t)

            # Concatenating after if it wasn't done before
            if not concat_before:
                outputs_at_t = concat_after_function(outputs_at_t, n_random_mappings, size / n_random_mappings)

            # Saving
            outputs[t] = outputs_at_t

        # "outputs" is currently a list of time steps (each containing all outputs at that time step),
        # but we want a list of outputs so that it corresponds with what came as argument to this method
        outputs = np.transpose(outputs, (1, 0, 2))

        # Writing to file
        out_file = open(file_name(identifier), 'wb')
        dumper.dump(outputs.tolist(), out_file)
        out_file.close()

        # Telling whomever invoked this function that we're done!
        queue.put(identifier)


combine = np.vectorize(bitwise_or)
