# Module to generate configurations corresponding to different problems
# with labels/classes/fasit

import random as rn

import numpy as np
# np.random.seed(20161112)


def bit_memory_task(quantity, bits, distractor_period):
    """

    :param quantity: the number of tasks to generate
    :param bits:
    :param distractor_period:
    :return: tasks, labels
    """
    tasks = []
    labels = []

    for i in xrange(quantity):
        y = 2  # y being 0, 1, 2 means which of the output nodes y1, y2, y3 that is activated, respectively
        task = []
        label = []
        for t in xrange(bits + distractor_period + 1 + bits):
            if t < bits:
                bit = (i & 2**t) / 2**t
                a1 = - bit + 1
                a2 = bit
            else:
                a1 = 0
                a2 = 0

            if bits <= t < bits + distractor_period or t > bits + distractor_period:
                # Distractor signal
                a3 = 1
            else:
                a3 = 0

            if bits + distractor_period == t:
                # Cue signal
                a4 = 1
            else:
                a4 = 0

            task.append(np.asarray([a1, a2, a3, a4]))

            if t > bits + distractor_period:
                y1 = task[t - (bits + distractor_period) - 1][0]
                y = 0 if y1 == 1 else 1

            # if i == 7:
            #     print "%d %d %d %d -> %d" % (a1, a2, a3, a4, y)
            label.append(y)
        tasks.append(task)
        labels.append(label)
    return tasks, labels


def temporal_parity(quantity, size, window_size=2, delay=0):
    tasks = []
    labels = []
    for _ in xrange(quantity):
        task = []
        label = []

        bit_stream = np.random.randint(2, size=size, dtype='int')

        for i in xrange(size + delay):
            from_i = i
            to_i = i + window_size
            task_element = bit_stream[from_i:to_i]
            if to_i > size:
                # Adding zeros if we have moved beyond the original bit_stream
                task_element = np.append(task_element, np.zeros(min(to_i - size, window_size), dtype='int'))

            task.append(task_element.tolist())
            label_element = False if from_i < delay else sum(task[i - delay]) % 2 == 1
            label_element = 1 if label_element else 0
            label.append(label_element)
        tasks.append(task)
        labels.append(label)
    return tasks, labels


def density(quantity, size=5, on_probability=0.5):
    """Generates initial configurations for the density classification,
    along with correct classifications

    :param quantity: number of configurations to generate
    :param size: length
    :return: configurations, labels
    """
    configs = []
    labels = []

    for c in xrange(quantity):
        config = []
        ones = 0
        zeros = 0
        for i in xrange(size):
            state = 1 if rn.random() < on_probability else 0
            if state == 1:
                ones += 1
            else:
                zeros += 1
            config.append(state)
        configs.append(config)
        labels.append(1 if ones > zeros else 0)
    return configs, labels


def parity(quantity, size=4):
    """Generates initial configurations for the parity problem,
    along with the corresponding solutions

    :param quantity: number of configurations to generate
    :param size: length
    :return: configurations, labels
    """
    configs = []
    labels = []

    for c in xrange(quantity):
        config = []
        ones = 0
        for i in xrange(size):
            state = rn.randint(0, 1)
            if state == 1:
                ones += 1
            config.append(state)
        configs.append(config)
        labels.append(1 if ones % 2 == 0 else 0)
    return configs, labels

