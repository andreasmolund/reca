# Functions for multiprocessing, for distributing work

import copy
import marshal as dumper
import os
import random
from multiprocessing import Process, Queue

import numpy as np

import concat
import ca.util as cutil

dump_path = "tmp/"
prefix = "dump-process"
file_type = "dump"


def distribute_and_collect(computer, sets):
    """Divides the sets into parts,
    distributes them to processes,
    and collects the preresults afterwards

    :param computer:
    :param sets:
    :return:
    """
    # Starting processes to distribute work
    out_q = Queue()
    processes = []
    n_parts = n_processes(sets)

    for n in xrange(n_parts):
        start, end = custom_range(sets, n, n_parts)
        process = Process(target=translate_and_transform,
                          args=(sets[start:end],
                                copy.deepcopy(computer.reservoir),
                                copy.deepcopy(computer.encoder),
                                computer.concat_before,
                                (start, end, '%032x' % random.getrandbits(128)),
                                out_q))
        processes.append(process)
        process.start()

    # Collecting data from the different processes
    outputs = [None] * len(sets)
    for _ in xrange(n_parts):
        identifier = out_q.get()  # Process identifier
        in_file = open(file_name(identifier), 'rb')

        outputs[identifier[0]:identifier[1]] = dumper.load(in_file)

        in_file.close()
        os.remove(file_name(identifier))

    for process in processes:
        process.join()

    return outputs


def translate_and_transform(sets,
                            reservoir,
                            encoder,
                            concat_before,
                            identifier,
                            queue):
    n_random_mappings = encoder.n_random_mappings
    total_area = encoder.total_area

    # "outputs" is a python list which shall contain ndarrays
    outputs = []

    for i, a_set in enumerate(sets):
        n_time_steps = a_set.shape[0]
        output = np.empty((n_time_steps, total_area * reservoir.iterations), dtype='int8')
        for t in xrange(n_time_steps):  # Time steps
            if t == 0:
                # Translating the initial input
                a_set_at_t = encoder.translate(a_set[0])
            else:
                # Adding input with parts of the previous output
                prev_state_vector = output[t - 1][-total_area:].copy()
                a_set_at_t = encoder.add(a_set[t], prev_state_vector)

            # Concatenating before if that is to be done
            if concat_before:
                a_set_at_t = concat.before(a_set_at_t, n_random_mappings)

            # Transforming in the reservoir
            output_at_t = reservoir.transform(a_set_at_t)

            # Concatenating after if it wasn't done before
            if not concat_before:
                output_at_t = concat.after(output_at_t, n_random_mappings)

            # Saving/recording
            output[t] = output_at_t
        outputs.append(output)

    # Writing to file
    out_file = open(file_name(identifier), 'wb')
    dumper.dump([o.tolist() for o in outputs], out_file)
    out_file.close()

    # Telling whomever invoked this function that we're done!
    queue.put(identifier)


def flatten(unflattened):
    """
    From a (m,n,o) list to a (m*n,o) list.
    :param unflattened:
    :return:
    """
    processed = []
    for m in unflattened:
        for n in m:
            processed.append(n)
    return processed


def custom_range(sets, part, n_parts):
    """Calculates a start index and an end index from the arguments.
    Used in multithreading.

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
    Used when splitting up a set into parts.

    :param n_parts: the number of parts you want to distribute work to
    :return: a list of length n_parts, with values summing to 1
    """
    if n_parts == 8:
        return [0.055, 0.075, 0.095, 0.115, 0.135, 0.155, 0.175, 0.195]
    elif n_parts == 4:
        return [0.22, 0.24, 0.26, 0.28]
    elif n_parts == 2:
        return [0.45, 0.55]
    elif n_parts == 1:
        return [1]
    else:
        raise NotImplementedError("n_parts of value %d is not implemented" % n_parts)


def n_processes(sets):
    """

    :param sets:
    :return: a number of processes in which you can divide the sets
    """
    n_sets = len(sets)
    n = 1

    if n_sets >= 32:
        n = 8
    elif n_sets >= 4:
        n = 4
    elif n_sets >= 2:
        n = 2
    return n


def file_name(identifier):
    return "%s%s%s.%s" % (dump_path, prefix, identifier, file_type)


def unflatten(flattened, sequence_lengths):
    reshaped = []
    offset = 0
    for sequence_len in sequence_lengths:
        reshaped.append(np.array(flattened[offset:offset + sequence_len]))
        offset += sequence_len
    return reshaped