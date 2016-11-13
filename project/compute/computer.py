import copy
import itertools
import marshal as dumper
import time
from multiprocessing import Process, Queue

dump_path = "tmp/"
prefix = "dump-process"
file_type = "dump"


class Computer:

    def __init__(self, encoder, reservoir, estimator, concat_before=True, verbose=1):
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

    def train(self, sets, labels):
        """Encodes and transforms the sets, and fits the estimator to it

        :param sets: training sets
        :param labels: labels/classes corresponding to the training sets
        :return: void
        """
        outputs = self._distribute_and_collect(sets)

        self.estimator.fit(outputs, labels)

    def test(self, sets):
        """Encodes and transforms the sets, and uses the estimator to predict outputs

        :param sets:
        :return:
        """
        outputs = self._distribute_and_collect(sets)

        return self.estimator.predict(outputs)

    @staticmethod
    def _translate_and_transform(sets, reservoir, encoder, concat_function, concat_before, identifier, queue):
        """What the computer desires to to with the sets.
        How it wants to translate and transform them.

        :param sets: a subset of the whole
        :param reservoir: reservoir
        :param encoder: encoder
        :param concat_function: concat function
        :param concat_before: true if concat_function is to be applied before transformation, false otherwise
        :param identifier: id that makes the work unique
        :param queue: to put the identifier on when done
        :return:
        """
        inputs = encoder.translate(sets)

        inputs = concat_function(inputs, encoder.n_random_mappings) if concat_before else inputs

        outputs = reservoir.transform(inputs)

        if not concat_before:
            outputs = concat_function(outputs, encoder.n_random_mappings)

        out_file = open(file_name(identifier), 'wb')
        dumper.dump(outputs, out_file)
        out_file.close()
        queue.put(identifier)

    def _distribute_and_collect(self, sets):
        """Divides the sets into parts,
        distributes them to processes,
        and collects the results afterwards

        :param sets:
        :return:
        """
        # Starting processes to distribute work
        out_q = Queue()
        processes = []
        n_processes = _n_processes(sets)

        for n in xrange(n_processes):
            start, end = custom_range(sets, n, n_processes)
            process = Process(target=self._translate_and_transform,
                              args=(sets[start:end],
                                    copy.deepcopy(self.reservoir),
                                    copy.deepcopy(self.encoder),
                                    self._concat,
                                    self.concat_before,
                                    (start, end),
                                    out_q))
            processes.append(process)
            process.start()

        # Collecting data from the different processes
        outputs = [None] * len(sets)
        for _ in xrange(n_processes):
            identifier = out_q.get()  # Process identifier
            in_file = open(file_name(identifier), 'rb')

            data = dumper.load(in_file)
            outputs[identifier[0]:identifier[1]] = data

            in_file.close()

        for process in processes:
            process.join()

        return outputs

    @staticmethod
    def _concat(elements, from_n_parts):
        from_n_parts = from_n_parts
        outputs = []
        for i in xrange(len(elements) / from_n_parts):
            span = i * from_n_parts
            concat = list(itertools.chain.from_iterable(elements[span:span + from_n_parts]))
            outputs.append(concat)
        return outputs


def custom_range(sets, part, n_parts):
    """Calculates a start index and an end index from the arguments

    :param sets:
    :param part:
    :param n_parts:
    :return: start, end
    """
    if part >= n_parts:
        raise ValueError("Size of part must be less than n_parts")

    n_sets = len(sets)
    sizes = sizes_of(n_parts)

    start = int(round(sum(sizes[:part]) * n_sets))
    end = int(round(sum(sizes[:part + 1]) * n_sets))

    return start, end


def sizes_of(n_parts):
    """Finds different sizes of distribution.

    :param n_parts: the number of parts you want to distribute work to
    :return: a list of length n_parts, with values summing to 1
    """
    if n_parts == 4:
        return [0.22, 0.24, 0.26, 0.28]
    elif n_parts == 2:
        return [0.45, 0.55]
    elif n_parts == 1:
        return [1]
    else:
        raise NotImplementedError("n_parts of value %d is not implemented" % n_parts)


def _n_processes(sets):
    """

    :param sets:
    :return: a number of processes in which you can divide the sets
    """
    n_sets = len(sets)
    if n_sets >= 4:
        return 4
    elif n_sets >= 2:
        return 2
    else:
        return 1


def file_name(identifier):
    return "%s%s%s.%s" % (dump_path, prefix, identifier, file_type)
