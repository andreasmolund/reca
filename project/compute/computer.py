import copy
import itertools
import marshal as dumper
from multiprocessing import Process, Queue
from math import floor, ceil

import time

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
        outputs = self._distribute_and_collect(sets, n_processes=_n_processes(sets))

        self.estimator.fit(outputs, labels)

    def test(self, sets):
        """Encodes and transforms the sets, and uses the estimator to predict outputs

        :param sets:
        :return:
        """
        outputs = self._distribute_and_collect(sets, n_processes=_n_processes(sets))

        return self.estimator.predict(outputs)

    @staticmethod
    def _translate_and_transform(sets, reservoir, encoder, concat_function, concat_before, identifier, queue):
        out_file = open(file_name(identifier), 'wb')
        inputs = encoder.translate(sets)

        inputs = concat_function(inputs, encoder.n_random_mappings) if concat_before else inputs

        outputs = reservoir.transform(inputs)

        if not concat_before:
            outputs = concat_function(outputs, encoder.n_random_mappings)

        dumper.dump(outputs, out_file)
        out_file.close()
        queue.put(identifier)

    def _distribute_and_collect(self, sets, n_processes):
        # Starting processes to distribute work
        out_q = Queue()
        sets_per_thread = len(sets) / n_processes
        processes = []
        for n in xrange(n_processes):
            start, end = range_from(sets, n, n_processes)
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
        print "Started to collect"
        time_checkpoint = time.time()
        for _ in xrange(n_processes):
            print "Work time %f. Now waiting for work" % (time.time() - time_checkpoint)
            time_checkpoint = time.time()
            identifier = out_q.get()  # Process index; from where the process began
            print "Working on %s. Idle time %f" % (identifier, time.time() - time_checkpoint)
            time_checkpoint = time.time()
            in_file = open(file_name(identifier), 'rb')

            # for i in xrange(sets_per_thread):
            #     outputs[process_i + i] = dumper.load(in_file)
            data = dumper.load(in_file)
            outputs[identifier[0]:identifier[1]] = data

            in_file.close()
        print "Work time %f" % (time.time() - time_checkpoint)
        print "Collected"

        for process in processes:
            process.join()
        print "Joined processes"

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


def range_from(sets, part, n_parts):
    n_sets = len(sets)

    if n_sets > 9:
        part_sizes = custom_range(n_parts)
        start = int(round(sum(part_sizes[:part]) * n_sets))
        end = int(round(sum(part_sizes[:part + 1]) * n_sets))
        # end = n_sets if part == len(part_sizes) else start + size
    else:
        # Just dividing into n_parts equal parts
        size = n_sets / n_parts
        start = part * size
        end = start + size

    return start, end


def custom_range(n_parts):
    if n_parts == 4:
        # return [0.16, 0.22, 0.28, 0.34]
        # return [0.19, 0.23, 0.27, 0.31]
        return [0.22, 0.24, 0.26, 0.28]
    else:
        raise NotImplementedError("n_parts of value %d is not implemented" % n_parts)


def custom_range2(n_parts):
    return [2**i for i in xrange(n_parts)]


def _n_processes(sets):
    n_sets = len(sets)
    if n_sets % 4 == 0:
        return 4
    elif n_sets % 2 == 0:
        return 2
    else:
        return 1


def file_name(nr):
    return "%s%s%s.%s" % (dump_path, prefix, nr, file_type)
