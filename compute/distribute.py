# Functions for multiprocessing, for distributing work

import copy
import marshal as dumper
import os
import random
from itertools import izip, count
from multiprocessing import Process, Queue
import concat

import numpy as np

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
                                concat.before,
                                concat.after,
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


def post_process(outputs):
    # We want to concatenate/flatten, which must be done through reshaping
    outputs = np.array(outputs)
    shape = outputs.shape
    return outputs.reshape(shape[0] * shape[1], shape[2])


def translate_and_transform(sets,
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

    # TODO: make it robust against arbitrary number of time steps

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
                new_sets_at_t[i] = encoder.overwrite(set_at_t, prev_state_vector)

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
    out_file = open(file_name(identifier), 'wb')
    dumper.dump(outputs.tolist(), out_file)
    out_file.close()

    # Telling whomever invoked this function that we're done!
    queue.put(identifier)


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

    if n_sets >= 8:
        n = 8
    elif n_sets >= 4:
        n = 4
    elif n_sets >= 2:
        n = 2

    return n


def file_name(identifier):
    return "%s%s%s.%s" % (dump_path, prefix, identifier, file_type)